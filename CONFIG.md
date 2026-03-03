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

   **`RULES_CONFIG["envelope_pluck_floor"]`** (pluck, piano_pluck, bass_pluck): `decay_min` (default 0.9 s), `decay_max`, `release_min` (0.6 s), `release_max`. When `decay_scale_by_wave_tilt` is true, decay is scaled by the same tilt used for wavetable position (brighter/richer waves get slightly longer decay).

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

## Macros

- **macro_control_1..4** are patched from analysis: brightness → 1, movement → 2, harmonic_richness → 3, noisiness → 4 (all 0–1).
- Template search order prefers `vital_templates/Presets/Some crazy synthy shit 11212025.vital` when present; that preset has these keys and labels (SYNC, WAVES, DAMP, SPACE). Assign macros in Vital’s Matrix to filter/osc/LFO as needed.
- Any full-format preset with `macro_control_1`–`macro_control_4` in `settings` will get these values; macro label keys (`macro1`–`macro4`) are left unchanged when using patch export.

## Validation & research (accurate product)

Settings and ranges are chosen so generated presets load correctly in Vital and reflect the analyzed audio.

- **Brightness → filter / osc**  
  Spectral centroid (and rolloff) are standard measures of perceived brightness; higher centroid ⇒ brighter sound. We map normalized brightness to filter cutoff (cutoff_min + brightness × range) and to wavetable frame (wave_frame 0–255: lower = more sine-like, higher = more saw-like), so brighter analysis yields a more open filter and brighter oscillator position.

- **Envelope (ADSR) from analysis**  
  Attack, decay, sustain, and release are taken from the analyzed amplitude envelope and written directly into Vital’s env 1–4. Pluck/piano get a decay floor so the tail isn’t too short. Vital’s documented ranges are respected: delay/hold 0–4 s, attack/decay/release 0–32 s, sustain 0–1 (see `mapping/vital_writer.py` PATCH_SAFE_RANGES).

- **Macros**  
  macro_control_1–4 are 0–1 and map to brightness, movement (spectral flux), harmonic_richness, and noisiness so the template’s Matrix assignments (e.g. SYNC, WAVES, DAMP, SPACE) receive sensible values.

- **Vital ranges we enforce**  
  Filter/EQ cutoffs 0–100; envelope times as above; osc wave_frame 0–255; unison voices 1–16; FX dry/wet and feedback in 0–1 (or documented caps). This avoids out-of-range values that could crash Vital or produce invalid presets.

- **Sanity check**  
  Load a generated `.vital` in Vital; the preset should load without errors and the played note should reflect the source (e.g. plucky vs pad-like, brighter vs darker). Tweak `RULES_CONFIG` and `analysis/feature_vector.py` thresholds if a class or timbre is off.
