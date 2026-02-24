# mapping/vital_writer.py

import json


def export_vital_preset(preset_data: dict, output_path: str):
    with open(output_path, "w") as f:
        json.dump(preset_data, f, indent=2)