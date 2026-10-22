import json
import os
import sys

dirs = sys.argv[1]
dirs = dirs.split(",")

for d in dirs:
    exp_dirs = [x[0] for x in os.walk(d)][1:]
    for exp in exp_dirs:
        try:
            with open(os.path.join(exp, "instances.log"), "r", encoding='utf-8') as f:
                content = f.read()
            decoded_content = content.encode().decode('unicode_escape')
            with open(os.path.join(exp, "decoded_instances.log"), "w", encoding='utf-8') as f:
                f.write(decoded_content)
        except:
            Warning(f"No instances.log found in {exp}")
            pass
        try:
            with open(os.path.join(exp, "xcomet_results.json"), "r") as f:
                results = json.load(f)
            with open(os.path.join(exp, "xcomet_metadata.json"), "r") as f:
                metadata = json.load(f)

            # Re-save
            with open(os.path.join(exp, "xcomet_results.json"), "w", encoding='utf8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            with open(os.path.join(exp, "xcomet_metadata.json"), "w", encoding='utf8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
        except:
            Warning(f"No json files found in {exp}")
            pass