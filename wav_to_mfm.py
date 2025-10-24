#!/usr/bin/env python3
"""
mfm_from_wav_pll.py — MFM decoder with a simple digital PLL for wow/flutter.

Pipeline:
  1) Read mono 16-bit PCM WAV, find zero-crossing transition times (linear interp).
  2) Debounce near-duplicate crossings.
  3) Estimate bit period T from every-other gap (Δt[i+2]) to ignore 0.5T mids.
  4) Detect boundary "rail" parity from leader (even/odd transitions).
  5) Run a 2-tap PLL (phase + slow frequency) that locks to boundary/mid events.
  6) Sample bits at moving mid-points; sweep byte offsets + MSB/LSB to find "MFM1".
  7) Verify CRC32 and write payload.

Usage:
  python mfm_from_wav_pll.py input.wav output.bin --bitrate 2000
  # or omit --bitrate to auto-estimate T
"""

import argparse, bisect, struct, wave, zlib, statistics, sys
from array import array
from typing import List, Tuple, Optional

# ---------- WAV I/O ----------

def read_wav_mono_16(path: str):
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
            out.append(t); last = t
    return out

# ---------- Period & rail estimation ----------

def robust_median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    return xs[n//2] if n % 2 else 0.5*(xs[n//2 - 1] + xs[n//2])

def choose_boundary_parity(trans: List[float], T: float, leader_window_s: float = 1.0):
    """Pick even/odd parity whose within-parity gaps cluster near T over the leader."""
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
            spread = statistics.median([abs(g - med) for g in gaps])
        else:
            med, spread = float("inf"), float("inf")
        candidates.append((parity, rail, med, spread))

    candidates.sort(key=lambda r: (abs(r[2] - T), r[3]))
    parity, rail, _, _ = candidates[0]
    return parity, rail

# ---------- Helpers ----------

def has_transition_in_window(trans: List[float], center: float, tol: float) -> bool:
    lo = bisect.bisect_left(trans, center - tol)
    return lo < len(trans) and trans[lo] <= center + tol

# ---------- PLL-driven bit recovery ----------

def decode_bits_pll(
    trans: List[float],
    T_init: float,
    t0_init: float,
    tol_sample_frac: float = 0.30,  # bit=1 if mid has a transition within ±0.30T
    tol_lock_frac: float   = 0.45,  # PLL will lock to events within ±0.45T
    alpha: float = 0.08,            # phase gain  (0.05–0.15)
    beta: float  = 0.003,           # frequency gain (0.001–0.01)
    max_bits: Optional[int] = None
) -> List[int]:
    """
    Simple 2-tap digital PLL:
      - Tracks phase 't_ref' and period 'T' using timing errors from nearest event
        (boundary or mid) to observed transitions.
      - Samples bits at moving midpoints: t_mid = t_ref + 0.5*T for each bit.
    """
    bits: List[int] = []
    t_ref = t0_init
    T = T_init
    T0 = T_init

    last_t = trans[-1] if trans else 0.0
    # Transition index used for PLL lock updates (to avoid reusing same transition)
    p = bisect.bisect_left(trans, t_ref - 0.5*T)

    n = 0
    while (t_ref + 1.2*T) < last_t and (max_bits is None or n < max_bits):
        # Current tolerances
        tol_samp = tol_sample_frac * T
        tol_lock = tol_lock_frac  * T

        # ---- PLL phase/frequency correction using nearest upcoming transition ----
        if p < len(trans):
            tau = trans[p]
            # Make sure p points to a transition near/after the current boundary
            if tau < t_ref - 0.3*T:
                p = bisect.bisect_left(trans, t_ref - 0.3*T, p)
                if p < len(trans):
                    tau = trans[p]
            if p < len(trans):
                b_time = t_ref
                m_time = t_ref + 0.5*T
                # Use whichever expected event (boundary/mid) is closer
                predicted = b_time if abs(tau - b_time) <= abs(tau - m_time) else m_time
                e = tau - predicted
                if abs(e) <= tol_lock:
                    # Clip huge errors; apply small corrections
                    lim = 0.5*T
                    if e >  lim: e =  lim
                    if e < -lim: e = -lim
                    t_ref += alpha * e
                    T     += beta  * e
                    # Keep T sane
                    Tmin, Tmax = 0.5*T0, 2.0*T0
                    if T < Tmin: T = Tmin
                    if T > Tmax: T = Tmax
                    p += 1  # consume this transition for the lock loop

        # ---- Bit decision: is there a mid-bit transition? ----
        t_mid = t_ref + 0.5*T
        bit1 = has_transition_in_window(trans, t_mid, tol_samp)
        bits.append(1 if bit1 else 0)

        # Advance to next bit boundary
        t_ref += T
        n += 1

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
    ap = argparse.ArgumentParser(description="Decode MFM-encoded WAV with PLL.")
    ap.add_argument("input_wav")
    ap.add_argument("output_bin")
    ap.add_argument("--bitrate", type=float, default=1000, help="Bitrate (bps)")
    ap.add_argument("--debounce", type=float, default=0.25, help="Debounce min-sep as fraction of T (default 0.25).")
    ap.add_argument("--tol_sample", type=float, default=0.30, help="Mid-bit detection window as fraction of T.")
    ap.add_argument("--tol_lock",   type=float, default=0.45, help="PLL lock window as fraction of T.")
    ap.add_argument("--alpha", type=float, default=0.08, help="PLL phase gain (0.05–0.15).")
    ap.add_argument("--beta",  type=float, default=0.003, help="PLL frequency gain (0.001–0.01).")
    ap.add_argument("--maxbits", type=int, default=0, help="Optional cap on decoded bits (0=auto).")
    args = ap.parse_args()

    # 1) Transitions
    samples, sr = read_wav_mono_16(args.input_wav)
    trans = find_zero_crossings(samples, sr)
    if len(trans) < 10:
        sys.exit("Too few transitions; check recording/cabling.")

    # 2) Initial T (fixed or estimate)
    T = (1.0/args.bitrate) if args.bitrate > 0 else estimate_T_every_other(trans)

    # 3) Debounce with T scale
    trans = debounce_transitions(trans, min_sep=args.debounce * T)

    # 5) Find boundary rail (leader zeros)
    parity, rail = choose_boundary_parity(trans, T, leader_window_s=1.0)
    if len(rail) < 2:
        sys.exit("Could not form a stable boundary rail (leader too short?).")
    t0 = rail[0]

    # 6) PLL-driven decode
    bits = decode_bits_pll(
        trans, T, t0,
        tol_sample_frac=args.tol_sample,
        tol_lock_frac=args.tol_lock,
        alpha=args.alpha, beta=args.beta,
        max_bits=(args.maxbits if args.maxbits > 0 else None)
    )

    # 7) Frame search
    try:
        payload = try_offsets_orders(bits)
    except SystemExit as e:
        # Last-ditch: small phase nudges and re-decode once or twice
        payload = None
        for nud in (-0.25, 0.25, -0.125, 0.125):
            bits2 = decode_bits_pll(
                trans, T, t0 + nud*T,
                tol_sample_frac=args.tol_sample,
                tol_lock_frac=args.tol_lock,
                alpha=args.alpha, beta=args.beta,
                max_bits=(args.maxbits if args.maxbits > 0 else None)
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
    print(f"PLL: alpha={args.alpha}, beta={args.beta}, tol_sample={args.tol_sample}, tol_lock={args.tol_lock}")
    print(f"T≈{T*1e3:.3f} ms  |  bitrate≈{1.0/T:.1f} bps  |  rail_parity={parity}")

if __name__ == "__main__":
    main()
