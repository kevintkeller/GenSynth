# mapping/rules_engine.py

from math import log2


def _classify_envelope(features: dict) -> str:
    """
    Very rough envelope class:
    - 'pluck' : fast attack, low sustain
    - 'pad'   : slower attack, high sustain
    - 'standard' otherwise
    """
    attack_speed = features["attack_speed"]       # 0..1, 1 = super fast
    sustain = features["sustain_amount"]          # 0..1

    if attack_speed > 0.7 and sustain < 0.35:
        return "pluck"
    if attack_speed < 0.4 and sustain > 0.6:
        return "pad"
    return "standard"


def _choose_wave_frame(brightness: float, noisiness: float) -> float:
    """
    Map brightness/noisiness into a wavetable frame index.
    Assumes osc_1_wave_frame ranges roughly 0..255 in your template.

    - Dark & clean -> lower frames (sine/triangle territory)
    - Bright & tonal -> mid/high frames (saw-ish)
    - Bright & noisy -> even higher (more aggressive textures)
    """
    # Base from brightness
    base = brightness

    # Push noisy sounds a bit brighter / more complex
    base = base * (0.7 + 0.3 * noisiness)

    return float(base * 255.0)


def _env_params(env_class: str, features: dict):
    sustain = features["sustain_amount"]
    attack_speed = features["attack_speed"]

    if env_class == "pluck":
        # Very fast attack, short decay, almost no sustain, shortish release
        env_attack = 0.003
        env_decay = 0.25
        env_sustain = 0.05
        env_release = 0.2

    elif env_class == "pad":
        # Slower attack, long decay, high sustain, long release
        env_attack = 0.4 + (1.0 - attack_speed) * 0.4   # 0.4..0.8
        env_decay = 1.0 + sustain * 1.0                 # 1..2
        env_sustain = 0.7 + 0.3 * sustain               # 0.7..1
        env_release = 1.0 + sustain * 1.0               # 1..2

    else:  # "standard"
        env_attack = 0.01 + (1.0 - attack_speed) * 0.3  # 0.01..0.31
        env_decay = 0.2 + (1.0 - sustain) * 0.8         # 0.2..1.0
        env_sustain = 0.3 + 0.5 * sustain               # 0.3..0.8
        env_release = 0.25 + sustain * 0.8              # ~0.25..1.05

    return env_attack, env_decay, env_sustain, env_release


def _filter_params(brightness: float, noisiness: float):
    """
    Map brightness/noisiness into filter cutoff / resonance.

    In Vital, filter_1_cutoff is a frequency-like control (roughly 0..120-ish).
    We'll keep it in that ballpark, but strongly driven by brightness.
    """
    # Brighter sounds -> higher cutoff, darker -> lower
    cutoff = 20.0 + brightness * 90.0  # 20..110

    # More resonant for mid-bright signals
    resonance = 0.2 + 0.8 * max(0.0, min(1.0, brightness * (1.0 - 0.5 * noisiness)))

    return cutoff, resonance


def _lfo_params(movement: float):
    """
    Simple mapping: more movement -> faster LFO.
    Vital uses a log-style rate control; -4..+4 is a nice musical range.
    """
    lfo_freq = -4.0 + movement * 8.0  # -4..+4
    return lfo_freq


def _pitch_params(fundamental_freq: float):
    """
    Map detected fundamental freq into coarse transpose + fine tune for osc 1.

    If your template already has osc pitch set sensibly, this is optional;
    but it helps get bass vs lead ranges a bit closer.
    """
    if fundamental_freq <= 0:
        # Fallback: leave template pitch as-is
        return None, None

    # Middle A (A4) = 440 Hz
    semitones_from_a4 = 12.0 * log2(fundamental_freq / 440.0)

    # Split into coarse (integer) + fine (fractional)
    coarse = round(semitones_from_a4)
    fine = semitones_from_a4 - coarse

    # Vital defaults:
    # - coarse probably around 0 at base note
    # - fine typically -1..+1 range
    coarse = float(coarse)
    fine = float(max(-1.0, min(1.0, fine)))

    return coarse, fine


def build_vital_parameters(features: dict) -> dict:
    """
    Map high-level features into a set of Vital parameter changes.
    This assumes:
      - osc_1_wave_frame, env_1_*, filter_1_*, lfo_1_frequency, etc. exist
        in your template.vital under 'settings'.
    """

    brightness = features["brightness"]
    noisiness = features["noisiness"]
    tonal_vs_perc = features["tonal_vs_perc"]
    movement = features["movement"]
    f0 = features["fundamental_freq"]

    # Envelope classification and params
    env_class = _classify_envelope(features)
    env_attack, env_decay, env_sustain, env_release = _env_params(env_class, features)

    # Oscillator wavetable frame
    wave_frame = _choose_wave_frame(brightness, noisiness)

    # Filter shape
    filter_cutoff, filter_resonance = _filter_params(brightness, noisiness)

    # LFO rate
    lfo_freq = _lfo_params(movement)

    # FX amounts
    # More sustain & tonal -> more reverb
    reverb_amount = 0.1 + 0.7 * features["sustain_amount"] * tonal_vs_perc

    # More movement & noisiness -> more chorus
    chorus_amount = 0.15 + 0.6 * movement * (0.5 + 0.5 * noisiness)

    # Brighter + noisier -> more distortion drive
    distortion_drive = 10.0 + 50.0 * (0.6 * brightness + 0.4 * noisiness)

    params = {
        # Oscillator 1
        "osc_1_on": 1.0,
        "osc_1_wave_frame": wave_frame,
        "osc_1_unison_voices": 4.0 if brightness > 0.6 else 1.0,
        "osc_1_unison_detune": brightness * 0.35,

        # Amp envelope (env 1)
        "env_1_attack": env_attack,
        "env_1_decay": env_decay,
        "env_1_sustain": env_sustain,
        "env_1_release": env_release,

        # Filter
        "filter_1_on": 1.0,
        "filter_1_cutoff": filter_cutoff,
        "filter_1_resonance": filter_resonance,

        # Movement (assuming LFO 1 is already routed somewhere in template)
        "lfo_1_frequency": lfo_freq,

        # Reverb
        "reverb_on": 1.0,
        "reverb_dry_wet": reverb_amount,

        # Chorus
        "chorus_on": 1.0,
        "chorus_dry_wet": chorus_amount,

        # Distortion
        "distortion_on": 1.0,
        "distortion_drive": distortion_drive,
    }

    # Special-case: very tonal, low-noise pluck-like sounds (e.g. pianos)
    # We keep these cleaner and closer to the template pitch.
    if tonal_vs_perc > 0.8 and noisiness < 0.35 and env_class == "pluck":
        params["osc_1_unison_voices"] = 1.0
        params["osc_1_unison_detune"] = 0.05 * brightness
        params["chorus_dry_wet"] = 0.05 + movement * 0.15
        params["distortion_drive"] = 5.0 + 20.0 * brightness

        # For now, do NOT override oscillator pitch for these sounds.
        # We rely on the template's base pitch so we don't force
        # everything into a bass region when analyzing low notes.

    return params