# GenSynth
Generate vital synth presets based on audio file uploads. The flow of the application essentially goes like this: Audio -> AI analysis -> Full synth preset (.vital/ .fxp). Note this is not just wavetable extraction, it outputs entire presets–oscillators, envelopes, filters, modulation routing, FX, etc.

# Audio Analysis Engine
Detects components of a sound file so we have the parameters to map to a vital preset

# A vital preset looks like this:
{
  "settings": { ... },
  "osc_1": { ... },
  "env_1": { ... },
  "filter_1": { ... },
  "lfo_1": { ... },
  "effects": { ... }
}