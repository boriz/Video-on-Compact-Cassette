#!/usr/bin/env python3
"""
mfm_from_wav_v3.py — Robust MFM decoder for WAV made by mfm_to_wav.py

Key upgrades:
  - Bit period T estimated from Δt[i+2] (every-other transition) to ignore 0.5T mid-bit gaps
  - Boundary rail detection: pick even/odd transition parity whose gaps cluster near T (leader=all zeros)
  - Uses that rail to set t0 (bit-boundary phase) before sampling mids
  - Still searches byte bit-offsets (0..7) and both bit orders (MSB/LSB)
  - Optional tiny timing nudges if needed
  - Optional --selftest: synthesize a WAV with the same rules and verify end-to-end

Usage:
  python mfm_from_wav_v3.py in.wav out.bin --bitrate 2000
  python mfm_from_wav_v3.py --selftest
"""

import argparse, bisect, struct, wave, zlib, os, tempfile, math, statistics
from array import array
from typing import List, Tuple, Optional

# ---------------- WAV I/O ----------------

def read_wav_mono_16(path: str) -> Tuple[List[int], int]:
    with wave.open(path, "rb") as wf:
        if wf.getnchannels() != 1: raise SystemExit("WAV must be mono.")
        if wf.getsampwidth() != 2: raise SystemExit("WAV must be 16-bit PCM.")
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    a = array("h"); a.frombytes(raw)
    return list(a), sr

def write_wav(path: str, pcm: bytes, sr: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(pcm)

# ---------------- Transition extraction ----------------

def find_zero_crossings(samples: List[int], sr: int) -> List[float]:
    """Linear-interpolated zero-cross times."""
    times: List[float] = []
    p = samples[0]
    for i in range(1, len(samples)):
        x = samples[i]
        if p == 0: p = -1 if x > 0 else 1
        if (p < 0 and x > 0) or (p > 0 and x < 0):
            denom = (x - p)
            frac = 0.0 if denom == 0 else (0 - p) / denom
            t = (i - 1 + frac) / sr
            times.append(t)
        p = x
    return times

def debounce_transitions(trans: List[float], min_sep: float) -> List[float]:
    """Drop spurious extra crossings closer than min_sep."""
    if not trans: return trans
    out = [trans[0]]
    last = trans[0]
    for t in trans[1:]:
        if t - last >= min_sep:
            out.append(t); last = t
    return out

# ---------------- T & rail estimation ----------------

def robust_median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0: return 0.0
    return xs[n//2] if n % 2 else 0.5*(xs[n//2-1] + xs[n//2])

def estimate_T_every_other(trans: List[float], use_first_n: int = 2000) -> float:
    """Estimate T from Δt[i+2] to avoid 0.5T gaps."""
    n = min(len(trans) - 2, use_first_n)
    if n <= 0: raise SystemExit("Not enough transitions to estimate T.")
    gaps2 = [trans[i+2] - trans[i] for i in range(n)]
    # Δt[i+2] ideally equals T (boundary→next boundary) in leader
    med = robust_median(gaps2)
    # Filter outliers to [0.5*med, 1.5*med] then re-median
    keep = [g for g in gaps2 if 0.5*med <= g <= 1.5*med]
    if not keep: raise SystemExit("Unable to estimate T (noisy or too short).")
    return robust_median(keep)

def choose_boundary_parity(trans: List[float], T: float, leader_window_s: float = 1.0) -> Tuple[int, List[float]]:
    """
    Pick parity (0 for even indices, 1 for odd) whose within-parity gaps cluster near T over the
    first leader_window_s seconds. Return (parity, chosen_rail_times).
    """
    if not trans: raise SystemExit("No transitions found.")
    t_end = trans[0] + leader_window_s
    cut = len([t for t in trans if t <= t_end])
    cut = max(cut, 100)  # ensure some samples

    rails = []
    for parity in (0, 1):
        rail = trans[parity:cut:2]  # every other transition of given parity
        gaps = [rail[i+1] - rail[i] for i in range(len(rail)-1)]
        if gaps:
            med = statistics.median(gaps)
            spread = statistics.median([abs(g - med) for g in gaps]) if gaps else 0.0
            rails.append((parity, rail, med, spread))
        else:
            rails.append((parity, rail, float("inf"), float("inf")))

    # Score: closeness of median to T + small spread
    scored = sorted(rails, key=lambda r: (abs(r[2] - T), r[3]))
    parity, rail, med, spread = scored[0]
    return parity, rail

# ---------------- Grid helpers ----------------

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
    if limit_bits is not None: max_bits = min(max_bits, limit_bits)
    bits = []
    for i in range(max_bits):
        mid = t0 + i*T + 0.5*T
        bits.append(1 if nearest_in_window(trans, mid, tol) is not None else 0)
    return bits

# ---------------- Byte packing & frame ----------------

def bits_to_bytes_offset(bits: List[int], offset: int, order: str) -> bytes:
    out = bytearray(); i = offset
    while i + 8 <= len(bits):
        chunk = bits[i:i+8]; b = 0
        if order == 'msb':
            for j, bit in enumerate(chunk): b |= (bit & 1) << (7 - j)
        else:
            for j, bit in enumerate(chunk): b |= (bit & 1) << j
        out.append(b); i += 8
    return bytes(out)

def parse_frame(bytestream: bytes) -> Tuple[bytes, int, int]:
    magic = b"MFM1"
    idx = bytestream.find(magic)
    if idx < 0: raise ValueError("MAGIC not found.")
    if idx + 8 > len(bytestream): raise ValueError("Truncated at length.")
    length = struct.unpack(">I", bytestream[idx+4:idx+8])[0]
    end = idx + 8 + length + 4
    if end > len(bytestream): raise ValueError("Truncated payload.")
    payload = bytestream[idx+8:idx+8+length]
    crc_got = struct.unpack(">I", bytestream[idx+8+length:end])[0]
    crc_exp = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_got != crc_exp: raise ValueError("CRC mismatch.")
    return payload, idx, end

def try_offsets_orders(bits: List[int]) -> bytes:
    last_err = None
    for order in ("msb", "lsb"):
        for off in range(8):
            bs = bits_to_bytes_offset(bits, off, order)
            try:
                payload, _, _ = parse_frame(bs)
                print(f"Found frame: order={order}, bit_offset={off}")
                return payload
            except Exception as e:
                last_err = e
    raise SystemExit(f"Failed to find MAGIC (order/offset sweep): {last_err}")

# ---------------- Optional self-test synthesizer ----------------

def bytes_to_bits_msb(data: bytes) -> List[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def mfm_transition_times(bits: List[int], T: float) -> List[float]:
    times: List[float] = []; prev = 0; t = 0.0
    for b in bits:
        if prev == 0 and b == 0: times.append(t)              # boundary
        if b == 1: times.append(t + 0.5*T)                    # mid
        prev = b; t += T
    return times

def synth_square(trans_times: List[float], sr: int, duration: float, amp: int=12000) -> bytes:
    ns = int(round(duration * sr))
    idxs = sorted(set(int(round(t * sr)) for t in trans_times if 0 <= t < duration))
    level = amp; p = 0; nxt = idxs[p] if idxs else None
    samples = [0]*ns
    for n in range(ns):
        while nxt is not None and n == nxt:
            level = -level; p += 1
            nxt = idxs[p] if p < len(idxs) else None
        samples[n] = level
    return array("h", samples).tobytes()

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_wav", nargs="?")
    ap.add_argument("output_bin", nargs="?")
    ap.add_argument("--bitrate", type=float, default=0.0, help="Known bitrate (bps). If 0, auto.")
    ap.add_argument("--tol", type=float, default=0.25, help="Timing tolerance as fraction of T.")
    ap.add_argument("--selftest", action="store_true", help="Generate a WAV, decode it, verify.")
    ap.add_argument("--samplerate", type=int, default=48000, help="Used only with --selftest.")
    ap.add_argument("--leader", type=float, default=2.0, help="Used only with --selftest.")
    ap.add_argument("--trailer", type=float, default=0.5, help="Used only with --selftest.")
    args = ap.parse_args()

    if args.selftest:
        payload = b"Hello, cassette world! \x00\x01\x02\xFE\xFF"
        framed = b"MFM1" + struct.pack(">I", len(payload)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
        bitrate = 2000.0
        T = 1.0 / bitrate
        bits = [0]*int(args.leader*bitrate) + bytes_to_bits_msb(framed) + [0]*int(args.trailer*bitrate)
        trans_times = mfm_transition_times(bits, T)
        duration = len(bits) * T
        pcm = synth_square(trans_times, args.samplerate, duration, amp=12000)
        tmpdir = tempfile.mkdtemp(prefix="mfm_v3_")
        wavpath = os.path.join(tmpdir, "loop.wav")
        write_wav(wavpath, pcm, args.samplerate)
        print("Wrote", wavpath)
        # decode the file we just made
        args.input_wav = wavpath
        args.output_bin = os.path.join(tmpdir, "out.bin")
        args.bitrate = bitrate  # known for the selftest

    if not args.input_wav or not args.output_bin:
        raise SystemExit("Provide input_wav and output_bin, or use --selftest.")

    samples, sr = read_wav_mono_16(args.input_wav)
    trans = find_zero_crossings(samples, sr)
    if len(trans) < 10: raise SystemExit("Too few transitions; check recording.")

    # First estimate a coarse T using every-other gaps
    T = (1.0/args.bitrate) if args.bitrate > 0 else estimate_T_every_other(trans)

    # Debounce AFTER we have T (to set a scale). Keep both boundary/mid transitions (0.5T apart).
    trans = debounce_transitions(trans, min_sep=0.25 * T)  # allow 0.5T mids to pass

    # Re-estimate T on de-bounced set
    if args.bitrate <= 0:
        T = estimate_T_every_other(trans)

    # Choose even/odd rail that behaves like leader boundaries (gap ~ T)
    parity, rail = choose_boundary_parity(trans, T, leader_window_s=1.0)
    if len(rail) < 2:
        raise SystemExit("Could not form a stable boundary rail.")

    # Set t0 to the first boundary on that rail (start of a bit cell)
    t0 = rail[0]

    # Decode bits on that grid
    bits = decode_bits_on_grid(trans, T, t0, tol_frac=args.tol)

    # Try byte packing across offsets / orders
    try:
        payload = try_offsets_orders(bits)
    except SystemExit as e:
        # Small timing nudges if grid is slightly off
        for nud in (-0.25, 0.25, -0.125, 0.125):
            bits2 = decode_bits_on_grid(trans, T, t0 + nud*T, tol_frac=args.tol)
            try:
                payload = try_offsets_orders(bits2)
                bits = bits2
                break
            except SystemExit:
                payload = None
        if payload is None:
            raise e

    # Success
    with open(args.output_bin, "wb") as f:
        f.write(payload)
    print(f"Recovered {len(payload)} bytes -> {args.output_bin}")
    print(f"Chosen rail parity={parity}, T≈{T*1e3:.3f} ms, bitrate≈{1.0/T:.1f} bps")

if __name__ == "__main__":
    main()
