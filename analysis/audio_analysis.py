# analysis/audio_analysis.py

import math
import librosa
import numpy as np


def _finite_float(x, default=0.0):
    """Ensure value is a finite float for downstream (Vital presets must have no NaN/Inf)."""
    if isinstance(x, (int, float)) and math.isfinite(x):
        return float(x)
    return default


def analyze_audio(file_path: str) -> dict:
    """
    Loads an audio file and extracts core sound features.
    Works for full songs or single sounds.
    """

    # Load mono at native samplerate
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y)

    # Guard against empty after trim
    if len(y) == 0:
        # Fallback to a tiny dummy signal to avoid NaNs
        y = np.zeros(int(0.5 * sr), dtype=np.float32)

    # Spectral centroid (brightness)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # Spectral rolloff (upper frequency at which 85% of energy lies)
    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    )

    # Spectral flatness (noisiness vs tonality)
    spectral_flatness = float(
        np.mean(librosa.feature.spectral_flatness(y=y))
    )

    # Spectral flux / onset envelope (movement)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = float(np.mean(onset_env))

    # Fundamental frequency (pitch)
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
    )
    fundamental_freq = float(np.nanmean(f0)) if f0 is not None else 110.0

    # Harmonic vs percussive ratio (tonal vs percussive)
    y_harm, y_perc = librosa.effects.hpss(y)
    harm_energy = np.sum(np.abs(y_harm))
    perc_energy = np.sum(np.abs(y_perc))
    harmonic_ratio = float(
        harm_energy / (harm_energy + perc_energy + 1e-6)
    )

    # RMS envelope for loudness and full ADSR-style envelope estimation
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    if len(rms) == 0:
        rms = np.array([0.0])

    peak_index = int(np.argmax(rms))
    peak_rms = float(rms[peak_index]) if peak_index < len(rms) else 1e-6
    attack_time = float(peak_index * hop_length / sr)

    # Sustain: mean level after the transient (from ~30% past peak to end)
    post_peak_start = min(peak_index + max(1, len(rms) // 20), len(rms) - 1)
    post_peak_rms = rms[post_peak_start:]
    sustain_energy = float(np.mean(rms))
    sustain_mean = float(np.mean(post_peak_rms)) if len(post_peak_rms) > 0 else sustain_energy
    sustain_ratio = float(np.clip(sustain_mean / (peak_rms + 1e-9), 0.0, 1.0))

    # Decay time: from peak to first frame where level <= sustain target
    sustain_target = peak_rms * 0.1 + sustain_mean * 0.9
    decay_frame = peak_index
    for i in range(peak_index + 1, len(rms)):
        if rms[i] <= sustain_target:
            decay_frame = i
            break
    decay_time = float((decay_frame - peak_index) * hop_length / sr)

    # Release time: from end backwards, how long until level was above threshold
    release_threshold = peak_rms * 0.2
    release_start_idx = len(rms) - 1
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] >= release_threshold:
            release_start_idx = i
            break
    release_time = float((len(rms) - 1 - release_start_idx) * hop_length / sr)

    # Zero crossing rate (helps with noisiness/brightness)
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    # Transient vs body brightness: centroid at attack vs later (piano is bright at hit, duller in tail)
    centroid_frames = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    n_frames = len(centroid_frames)
    transient_win = 3
    transient_start = max(0, peak_index - transient_win)
    transient_end = min(n_frames, peak_index + transient_win + 1)
    transient_centroid = float(np.mean(centroid_frames[transient_start:transient_end])) if transient_end > transient_start else spectral_centroid
    body_start = min(peak_index + max(1, n_frames // 10), n_frames - 1)
    body_end = min(peak_index + int(n_frames * 0.6), n_frames)
    body_centroid = float(np.mean(centroid_frames[body_start:body_end])) if body_end > body_start else spectral_centroid

    # Harmonic character: odd vs even harmonic balance, and harmonic richness
    # Use spectrum around the transient (peak) for timbre
    n_fft = 2048
    frame_start = max(0, peak_index * hop_length - n_fft // 2)
    frame = y[frame_start:frame_start + n_fft]
    if len(frame) < n_fft:
        frame = np.pad(frame.astype(np.float64), (0, n_fft - len(frame)), mode="constant")
    windowed = frame * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    f0_hz = fundamental_freq if fundamental_freq > 0 else 110.0
    odd_energy = 0.0
    even_energy = 0.0
    harmonic_count = 0
    for h in range(1, 32):
        harm_freq = h * f0_hz
        if harm_freq >= sr / 2:
            break
        bin_idx = int(harm_freq * len(spec) / (sr / 2))
        bin_idx = min(bin_idx, len(spec) - 1)
        mag = float(spec[bin_idx])
        if mag > 1e-8:
            harmonic_count += 1
        if h % 2 == 1:
            odd_energy += mag
        else:
            even_energy += mag
    total_harm = odd_energy + even_energy + 1e-9
    odd_ratio = float(odd_energy / total_harm)
    # Richness: how many harmonics present, normalized (e.g. 0-1 for 0-20 harmonics)
    harmonic_richness = float(np.clip(harmonic_count / 20.0, 0.0, 1.0))

    return {
        "spectral_centroid": _finite_float(spectral_centroid, 2000.0),
        "transient_centroid": _finite_float(transient_centroid, 2000.0),
        "body_centroid": _finite_float(body_centroid, 2000.0),
        "spectral_rolloff": _finite_float(spectral_rolloff, 4000.0),
        "spectral_flatness": _finite_float(spectral_flatness, 0.1),
        "spectral_flux": _finite_float(spectral_flux, 0.5),
        "fundamental_freq": _finite_float(fundamental_freq, 110.0),
        "harmonic_ratio": _finite_float(harmonic_ratio, 0.5),
        "attack_time": _finite_float(attack_time, 0.05),
        "decay_time": _finite_float(np.clip(decay_time, 0.01, 4.0), 0.2),
        "sustain_ratio": _finite_float(sustain_ratio, 0.3),
        "release_time": _finite_float(np.clip(release_time, 0.02, 4.0), 0.3),
        "sustain_energy": _finite_float(sustain_energy, 0.1),
        "zero_crossing_rate": _finite_float(zcr, 0.05),
        "harmonic_odd_ratio": _finite_float(odd_ratio, 0.5),
        "harmonic_richness": _finite_float(harmonic_richness, 0.5),
        "sample_rate": _finite_float(sr, 44100.0),
    }