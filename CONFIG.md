# GenSynth configuration

## Approach

- **Rule-based mapping** from audio features to Vital params. One pipeline: analyze → features → sound_class → rules → patch export.
- **Per–sound_class config** makes piano, bass, pluck, pad, and lead clearly different (wave ranges, FX caps, which oscillators/filters are on).
- **Preset bank** (`mapping/preset_bank.py`) can derive default ranges from your Vital preset examples; use `get_per_type_defaults()` to tune.

## Where to tune (no code changes)

1. **`mapping/rules_engine.py` → `RULES_CONFIG["sound_class"]`**  
   For each class (`piano_pluck`, `bass_pluck`, `pluck`, `pad`, `lead`, `standard`):
   - `wave_min`, `wave_max`: osc 1 wavetable frame range (0–255). Piano 32–52, bass 8–38, pluck 38–85, lead 70–180.
   - `reverb_wet_max`, `delay_wet_max`, `chorus_wet_max`: cap FX amount so e.g. piano stays light.
   - `osc_2`, `filter_2`, `lfo`: turn blocks on/off.

2. **`mapping/rules_engine.py` → `RULES_CONFIG["acoustic_like"]`**  
   Piano-only: filter cutoff range, envelope caps, wave_frame (overrides generic osc for piano_pluck).

3. **`analysis/feature_vector.py` → `_classify_sound()`**  
   Thresholds for assigning sound_class (attack_speed, sustain_amount, tonal_vs_perc, noisiness, fundamental_freq).

4. **`mapping/vital_writer.py`**  
   `PATCH_SAFE_RANGES`, `FX_SAFE_RANGES`: clamp any param so Vital never sees out-of-range values.

## Preset bank (optional)

Put full-format Vital presets in `vital_templates/Presets/`. Filenames hint type (e.g. "bass", "pluck", "piano"):

```python
from mapping.preset_bank import get_per_type_defaults
defaults = get_per_type_defaults()
# Use defaults["bass_pluck"], defaults["pluck"], etc. to seed RULES_CONFIG or blend.
```

## Export mode

- **Core + FX for pad/lead only** (default): patches oscillators, envelopes, filters, LFOs for all; FX (reverb, delay, chorus, compressor) are patched only when sound_class is **pad** or **lead**. Piano, bass, and pluck have effects forced off to avoid Vital crashes.
- Set `PATCH_CORE_ONLY=0` to patch every controlled param (may hit Vital bugs on some effect params).
