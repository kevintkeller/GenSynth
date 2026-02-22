import numpy as np

PARAM_NAMES = [
    # OSC
    "osc_waveform",        # 0=sine,1=tri,2=square,3=saw
    "osc_unison",
    "osc_detune",

    # AMP ENV
    "env_attack",
    "env_decay",
    "env_sustain",
    "env_release",

    # FILTER
    "filter_cutoff",
    "filter_resonance",

    # LFO
    "lfo_rate",
    "lfo_amount",

    # FX
    "dist_amount",
    "reverb_amount",
    "delay_amount",
]

OUTPUT_SIZE = len(PARAM_NAMES)


class SynthParameterVector:
    def __init__(self, vector):
        assert len(vector) == OUTPUT_SIZE
        self.vector = np.clip(vector, 0.0, 1.0)

    def to_dict(self):
        return dict(zip(PARAM_NAMES, self.vector))