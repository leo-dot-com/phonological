#!/usr/bin/env python3
"""
Generate phoneme audio files using eSpeak‑NG.
Place this script in the same directory as app.py.
"""

import os
import json
import subprocess
from pathlib import Path

# Define all possible phonemes from your P3–P6 tasks (add more as needed)
PHONEMES = [
    's', 't', 'a', 'm', 'p', 'b', 'l', 'e', 'n', 'd', 'c', 'r', 'i', 's', 'p',
    't', 'w', 'i', 's', 't', 'g', 'r', 'a', 's', 'p', 's', 'p', 'l', 'i', 't',
    's', 'k', 'r', 'e', 't', 'ch', 'th', 'r', 'i', 'l', 'l', 's', 'p', 'r', 'i', 'n', 't',
    'k', 'r', 'u', 'n', 'ch', 's', 'k', 'w', 'e', 'r', 'th', 'r', 'o', 't', 'l',
    's', 't', 'r', 'e', 'n', 'g', 'th', 's', 'k', 'r', 'a', 'p', 't', 't', 'w', 'e', 'l', 'f', 'th',
    's', 'p', 'l', 'i', 'n', 't', 's', 'th', 'r', 'e', 'sh', 'h', 'o', 'l', 'd',
    's', 'k', 'r', 'u', 'p', 'ə', 'l', 's', 'k', 'w', 'ɪ', 'n', 't', 'ɪ', 'ŋ',
    's', 'p', 'l', 'ɪ', 'n', 't', 'ə', 'z', 'f', 'oʊ', 'n', 'iː', 'm'
]
# Remove duplicates while preserving order
unique_phonemes = list(dict.fromkeys(PHONEMES))

# Directory to save MP3 files
OUTPUT_DIR = Path("static/phonemes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Map phoneme symbols to eSpeak‑NG IPA or phoneme codes
# This mapping ensures correct pronunciation (add more as needed)
PHONEME_TO_ESPEAK = {
    's': 's',
    't': 't',
    'a': 'æ',        # as in "cat"
    'm': 'm',
    'p': 'p',
    'b': 'b',
    'l': 'l',
    'e': 'ɛ',        # as in "bed"
    'n': 'n',
    'd': 'd',
    'c': 'k',        # hard c
    'r': 'r',
    'i': 'ɪ',        # as in "sit"
    'w': 'w',
    'g': 'g',
    'k': 'k',
    'ch': 'tʃ',
    'th': 'θ',       # unvoiced th
    'sh': 'ʃ',
    'ng': 'ŋ',
    'ə': 'ə',        # schwa
    'oʊ': 'oʊ',
    'iː': 'iː',
    'ɪ': 'ɪ',
    'ŋ': 'ŋ',
    'z': 'z',
    'f': 'f',
    'u': 'u',
    'p': 'p',
    'l': 'l',
    'e': 'ɛ',
    'n': 'n',
    'd': 'd',
    't': 't',
    'r': 'r',
    'a': 'æ',
    's': 's',
    'c': 'k',
    'g': 'g',
    'k': 'k',
    'w': 'w',
    'h': 'h',
    'o': 'ɒ',        # as in "hot"
    'th': 'θ',
    'dh': 'ð',       # voiced th (if needed)
    'zh': 'ʒ',
}

def generate_phoneme_audio(phoneme, ipa_symbol):
    """Generate MP3 for a single phoneme using eSpeak‑NG."""
    output_wav = OUTPUT_DIR / f"{phoneme}.wav"
    output_mp3 = OUTPUT_DIR / f"{phoneme}.mp3"

    # Skip if already exists
    if output_mp3.exists():
        print(f"✔ {phoneme}.mp3 already exists")
        return

    # eSpeak command to speak the IPA symbol
    cmd = [
        "espeak-ng",
        "-v", "en-us",
        "--ipa", ipa_symbol,
        "-w", str(output_wav)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Convert WAV to MP3 using ffmpeg
        subprocess.run([
            "ffmpeg", "-i", str(output_wav),
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",  # high quality
            str(output_mp3)
        ], check=True, capture_output=True)
        output_wav.unlink()  # remove WAV
        print(f"✓ Generated {phoneme}.mp3")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate {phoneme}: {e.stderr.decode()}")

def main():
    print("Generating phoneme audio files...")
    for phoneme in unique_phonemes:
        ipa = PHONEME_TO_ESPEAK.get(phoneme, phoneme)  # fallback to phoneme itself
        generate_phoneme_audio(phoneme, ipa)
    print("Done!")

if __name__ == "__main__":
    main()
