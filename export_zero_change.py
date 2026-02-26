"""Copy the blank template to a file without any parameter changes.
Run this and open the output in Vital. If it opens, the crash is caused by our patched values.
Then run main.py again (we fixed number formatting); if it still crashes we can narrow down which param(s).
"""
from mapping.vital_schema import get_default_template_path
from mapping.vital_writer import export_zero_change

if __name__ == "__main__":
    template_path = get_default_template_path()
    export_zero_change(template_path, "generated-zero-change.vital")
    print("Created generated-zero-change.vital (identical to template). Open it in Vital to confirm it loads.")
