# Vital presets for GenSynth

- **Use as templates (full Vital format):** `blank-template.vital`, `TEMPLATE.vital`, and any preset that opens in Vital and has the full synth state (author, preset_name, settings with 700+ keys). GenSynth will load these and only overwrite numeric parameters.
- **Not loadable by Vital (simplified format):** Files like `WideSawLead.vital`, `TechSub.vital`, `SyncScreamer.vital`, `TechBass01.vital`, `TechHook.vital`, `SubHammer.vital` use a reduced JSON schema (e.g. `osc1_waveframe`, `filter1_cutoff`) and are **not** real .vital presets. Do not use them as the GenSynth template; use `blank-template.vital` or `TEMPLATE.vital` instead.

GenSynth picks the first existing file in this order: `blank-template.vital` → `TEMPLATE.vital` → `mapping/template.vital`.
