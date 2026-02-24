# analysis/audio_analysis.py

import librosa
import numpy as np


def analyze_audio(file_path: str) -> dict:
    """
    Loads an audio file and extracts core sound features.
    Works for full songs or single sounds.
    """

    y, sr = librosa.load(file_path, sr=None)

    # Trim silence
    y, _ = librosa.effects.trim(y)

    # Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Spectral flux (movement)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = np.mean(onset_env)

    # Fundamental frequency (pitch)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7')
    )

    fundamental_freq = np.nanmean(f0) if f0 is not None else 110

    # Harmonic vs percussive ratio
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_ratio = np.sum(np.abs(y_harm)) / (
        np.sum(np.abs(y_harm)) + np.sum(np.abs(y_perc)) + 1e-6
    )

    # Attack time estimation
    rms = librosa.feature.rms(y=y)[0]
    peak_index = np.argmax(rms)
    attack_time = peak_index / sr

    # Sustain energy
    sustain_energy = np.mean(rms)

    return {
        "spectral_centroid": float(spectral_centroid),
        "spectral_flux": float(spectral_flux),
        "fundamental_freq": float(fundamental_freq),
        "harmonic_ratio": float(harmonic_ratio),
        "attack_time": float(attack_time),
        "sustain_energy": float(sustain_energy)
    }