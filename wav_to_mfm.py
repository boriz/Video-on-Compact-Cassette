#!/usr/bin/env python3
"""
mfm_from_wav.py — Decode an MFM-encoded mono WAV (16-bit PCM) into the original payload.

Robustness features:
  - Zero-crossing transitions with linear interpolation
  - Debounce of near-duplicate crossings
  - Bit period T estimated from Δt[i+2] (every-other transition), robust to 0.5T mid transitions
  - Boundary rail detection (even/odd parity) using leader behavior (gap ≈ T)
  - Decode on boundary-locked grid; then try all 8 bit offsets and both MSB/LSB byte orders
  - Small timing nudges if needed

Usage:
  python mfm_from_wav.py input.wav output.bin --bitrate 2000
  # or omit --bitrate and let it auto-estimate from the signal
"""

import argparse, bisect, struct, wave, zlib, statistics, sys
from array import array
from typing import List, Tuple, Optional

# ---------- WAV I/O ----------

def read_wav_mono_16(path: str) -> Tuple[List[int], int]:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1: sys.exit("WAV must be mono.")
        if wf.getsampwidth() != 2: sys.exit("WAV must be 16-bit PCM.")
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = array("h"); a.frombytes(raw)
    return list(a), sr

# ---------- Transition extraction ----------

def find_zero_crossings(samples: List[int], sr: int) -> List[float]:
    times: List[float] = []
    prev = samples[0]
    for i in range(1, len(samples)):
        x = samples[i]
        if prev == 0:
            prev = -1 if x > 0 else 1
        if (prev < 0 and x > 0) or (prev > 0 and x < 0):
            denom = (x - prev)
            frac = 0.0 if denom == 0 else (0 - prev) / denom
            t = (i - 1 + frac) / sr
            times.append(t)
        prev = x
    return times

def debounce_transitions(trans: List[float], min_sep: float) -> List[float]:
    if not trans: return trans
    out = [trans[0]]
    last = trans[0]
    for t in trans[1:]:
        if t - last >= min_sep:
            out.append(t)
            last = t
    return out

# ---------- Period & rail estimation ----------

def robust_median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    return xs[n//2] if n % 2 else 0.5*(xs[n//2 - 1] + xs[n//2])

def estimate_T_every_other(trans: List[float], use_first_n: int = 2000) -> float:
    n = min(len(trans) - 2, use_first_n)
    if n <= 0: sys.exit("Not enough transitions to estimate T.")
    gaps2 = [trans[i+2] - trans[i] for i in range(n)]
    med = robust_median(gaps2)
    keep = [g for g in gaps2 if 0.5*med <= g <= 1.5*med]
    if not keep: sys.exit("Unable to estimate T (noisy or too short).")
    return robust_median(keep)

def choose_boundary_parity(trans: List[float], T: float, leader_window_s: float = 1.0) -> Tuple[int, List[float]]:
    if not trans: sys.exit("No transitions found.")
    t_end = trans[0] + leader_window_s
    cut = len([t for t in trans if t <= t_end])
    cut = max(cut, 100)

    candidates = []
    for parity in (0, 1):
        rail = trans[parity:cut:2]
        gaps = [rail[i+1] - rail[i] for i in range(len(rail)-1)]
        if gaps:
            med = statistics.median(gaps)
            spread = statistics.median([abs(g - med) for g in gaps]) if gaps else 0.0
        else:
            med, spread = float("inf"), float("inf")
        candidates.append((parity, rail, med, spread))

    # Choose rail whose median gap is closest to T (tie-break: smaller spread)
    candidates.sort(key=lambda r: (abs(r[2] - T), r[3]))
    parity, rail, med, spread = candidates[0]
    return parity, rail

# ---------- Grid decode ----------

def nearest_in_window(sorted_vals: List[float], center: float, tol: float) -> Optional[float]:
    lo, hi = center - tol, center + tol
    idx = bisect.bisect_left(sorted_vals, lo)
    best, best_abs = None, tol + 1e-9
    while idx < len(sorted_vals) and sorted_vals[idx] <= hi:
        d = abs(sorted_vals[idx] - center)
        if d < best_abs:
            best_abs, best = d, sorted_vals[idx]
        idx += 1
    return best

def decode_bits_on_grid(trans: List[float], T: float, t0: float, tol_frac: float, limit_bits: Optional[int]=None) -> List[int]:
    tol = tol_frac * T
    last_t = trans[-1] if trans else 0.0
    max_bits = max(0, int((last_t - t0) / T) - 1)
    if limit_bits is not None:
        max_bits = min(max_bits, limit_bits)
    bits = []
    for i in range(max_bits):
        mid = t0 + i*T + 0.5*T
        bits.append(1 if nearest_in_window(trans, mid, tol) is not None else 0)
    return bits

# ---------- Byte packing & frame ----------

def bits_to_bytes_offset(bits: List[int], offset: int, order: str) -> bytes:
    out = bytearray()
    i = offset
    while i + 8 <= len(bits):
        chunk = bits[i:i+8]
        b = 0
        if order == 'msb':
            for j, bit in enumerate(chunk): b |= (bit & 1) << (7 - j)
        else:
            for j, bit in enumerate(chunk): b |= (bit & 1) << j
        out.append(b)
        i += 8
    return bytes(out)

def parse_frame(bytestream: bytes) -> bytes:
    magic = b"MFM1"
    idx = bytestream.find(magic)
    if idx < 0:
        raise ValueError("MAGIC not found.")
    if idx + 8 > len(bytestream):
        raise ValueError("Truncated at length.")
    length = struct.unpack(">I", bytestream[idx+4:idx+8])[0]
    end = idx + 8 + length + 4
    if end > len(bytestream):
        raise ValueError("Truncated payload.")
    payload = bytestream[idx+8:idx+8+length]
    crc_got = struct.unpack(">I", bytestream[idx+8+length:end])[0]
    crc_exp = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_got != crc_exp:
        raise ValueError("CRC mismatch.")
    return payload

def try_offsets_orders(bits: List[int]) -> bytes:
    last_err = None
    for order in ("msb", "lsb"):
        for off in range(8):
            bs = bits_to_bytes_offset(bits, off, order)
            try:
                return parse_frame(bs)
            except Exception as e:
                last_err = e
    raise SystemExit(f"Failed to find MAGIC (order/offset sweep): {last_err}")

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Decode MFM-encoded WAV to binary payload.")
    ap.add_argument("input_wav")
    ap.add_argument("output_bin")
    ap.add_argument("--bitrate", type=float, default=0.0, help="Known bitrate (bps). If 0, auto-estimate.")
    ap.add_argument("--tol", type=float, default=0.25, help="Timing tolerance as fraction of bit period (0.20–0.35).")
    ap.add_argument("--debounce", type=float, default=0.25, help="Debounce min separation as fraction of T (default 0.25).")
    ap.add_argument("--maxbits", type=int, default=0, help="Optional cap on decoded bits (0=auto).")
    args = ap.parse_args()

    samples, sr = read_wav_mono_16(args.input_wav)
    trans = find_zero_crossings(samples, sr)
    if len(trans) < 10:
        sys.exit("Too few transitions; check recording/cabling.")

    # First rough T (or fixed if provided)
    T = (1.0 / args.bitrate) if args.bitrate > 0 else estimate_T_every_other(trans)

    # Debounce using T scale (keep both boundary and 0.5T mid transitions)
    trans = debounce_transitions(trans, min_sep=args.debounce * T)

    # Re-estimate T if not provided
    if args.bitrate <= 0:
        T = estimate_T_every_other(trans)

    # Choose boundary rail (leader is zeros -> boundaries spaced ~T)
    parity, rail = choose_boundary_parity(trans, T, leader_window_s=1.0)
    if len(rail) < 2:
        sys.exit("Could not form a stable boundary rail (leader too short?).")

    t0 = rail[0]  # first boundary → grid origin

    # Decode bits on this grid
    bits = decode_bits_on_grid(
        trans, T, t0, tol_frac=args.tol,
        limit_bits=(args.maxbits if args.maxbits > 0 else None)
    )

    # Try byte packing across offsets & bit orders; if fail, nudge timing a bit
    try:
        payload = try_offsets_orders(bits)
    except SystemExit as e:
        payload = None
        for nud in (-0.25, 0.25, -0.125, 0.125):
            bits2 = decode_bits_on_grid(
                trans, T, t0 + nud*T, tol_frac=args.tol,
                limit_bits=(args.maxbits if args.maxbits > 0 else None)
            )
            try:
                payload = try_offsets_orders(bits2)
                bits = bits2
                break
            except SystemExit:
                continue
        if payload is None:
            raise e

    with open(args.output_bin, "wb") as f:
        f.write(payload)

    print(f"Recovered {len(payload)} bytes -> {args.output_bin}")
    print(f"T≈{T*1e3:.3f} ms  |  bitrate≈{1.0/T:.1f} bps  |  chosen_rail_parity={parity}")

if __name__ == "__main__":
    main()
