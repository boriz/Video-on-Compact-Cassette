#!/usr/bin/env python3
"""
mfm_to_wav.py — Encode a binary file as MFM into a mono WAV (16-bit PCM).

Framing:  MAGIC "MFM1" + LEN(4, big-endian) + DATA + CRC32(4, IEEE/ZIP)
Bit order: MSB-first per byte
MFM rule:  - boundary transition at t=i*T if (prev==0 and bit==0)
           - mid-bit transition at t=i*T + T/2 if (bit==1)

Recommended for cassette: 1200–2400 bps, leader >= 2s, Dolby/NR OFF.
"""

import argparse, struct, wave, zlib
from typing import List
import sys

def bytes_to_bits_msb(data: bytes) -> List[int]:
    bits = []
    for b in data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return bits

def pack_frame(payload: bytes) -> bytes:
    magic = b"MFM1"
    length = struct.pack(">I", len(payload))
    crc = struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    return magic + length + payload + crc

def mfm_transition_times(bits: List[int], T: float) -> List[float]:
    times = []
    prev = 0
    t = 0.0
    for b in bits:
        if prev == 0 and b == 0:
            times.append(t)              # boundary
        if b == 1:
            times.append(t + 0.5 * T)    # mid
        prev = b
        t += T
    return times

def build_leader_bits(seconds: float, bitrate: int) -> List[int]:
    return [0] * max(0, int(round(seconds * bitrate)))

def synth_square_from_transitions(trans_times, sr: int, duration: float, amp: int) -> bytes:
    # Generate mono 16-bit PCM square wave toggling at each transition time
    import struct as _struct
    ns = int(round(duration * sr))
    samples = [0] * ns
    idxs = sorted(set(int(round(t * sr)) for t in trans_times if 0 <= t < duration))

    level = amp
    p = 0
    nxt = idxs[p] if idxs else None
    for n in range(ns):
        while nxt is not None and n == nxt:
            level = -level
            p += 1
            nxt = idxs[p] if p < len(idxs) else None
        samples[n] = level
    return _struct.pack("<" + "h" * ns, *samples)

def preemphasis(pcm: bytes, coeff: float) -> bytes:
    # y[n] = x[n] - a*x[n-1], simple 1st-order
    if coeff <= 0.0:
        return pcm
    import array
    a = array.array("h"); a.frombytes(pcm)
    y = array.array("h")
    prev = 0.0
    for x in a:
        xn = float(x)
        yn = xn - coeff * prev
        prev = xn
        if yn > 32767: yn = 32767
        if yn < -32768: yn = -32768
        y.append(int(yn))
    return y.tobytes()

def write_wav(path: str, pcm: bytes, sr: int):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm)

def main():
    ap = argparse.ArgumentParser(description="Encode binary to MFM WAV.")
    ap.add_argument("input", help="Input binary file")
    ap.add_argument("output", help="Output WAV path")
    ap.add_argument("--bitrate", type=int, default=1000, help="Bits per second")
    ap.add_argument("--samplerate", type=int, default=48000, help="WAV sample rate (Hz)")
    ap.add_argument("--leader", type=float, default=2.0, help="Leader seconds (zeros before frame)")
    ap.add_argument("--trailer", type=float, default=0.5, help="Trailer seconds (zeros after frame)")
    ap.add_argument("--amplitude", type=int, default=12000, help="Square amplitude (<=30000)")
    ap.add_argument("--preemph", type=float, default=0.0, help="Pre-emphasis coeff (0 disables; try 0.7–0.9 for tape)")
    args = ap.parse_args()

    if args.amplitude > 30000:
        print("Amplitude too high; use <= 30000.", file=sys.stderr); sys.exit(1)

    with open(args.input, "rb") as f:
        payload = f.read()

    frame = pack_frame(payload)
    bits = []
    bits += build_leader_bits(args.leader, args.bitrate)
    bits += bytes_to_bits_msb(frame)
    bits += build_leader_bits(args.trailer, args.bitrate)

    T = 1.0 / float(args.bitrate)
    trans = mfm_transition_times(bits, T)
    duration = len(bits) * T

    pcm = synth_square_from_transitions(trans, args.samplerate, duration, args.amplitude)
    if args.preemph > 0.0:
        pcm = preemphasis(pcm, args.preemph)

    write_wav(args.output, pcm, args.samplerate)

    print(f"Wrote {args.output}")
    print(f"Payload: {len(payload)} bytes  |  Bitrate: {args.bitrate} bps  |  Duration: {duration:.2f} s")
    print("Record tips: conservative levels, Dolby/NR OFF, good azimuth.")

if __name__ == "__main__":
    main()
