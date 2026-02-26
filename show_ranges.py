import re
with open("vital_templates/Presets/blank-template.vital") as f:
    text = f.read()
for name in ["eq_band_cutoff", "eq_high_cutoff", "eq_low_cutoff", "chorus_voices", "compressor_band_gain", "filter_1_cutoff"]:
    m = re.search(r'"' + name + r'":\s*([-\d.eE+]+)', text)
    print(name, m.group(1) if m else "not found")
