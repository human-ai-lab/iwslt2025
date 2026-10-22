import importlib
import os

# Get the directory path of this file
current_dir = os.path.dirname(__file__)

# Dynamically import all Python files in this directory
for file in sorted(os.listdir(current_dir)):
    if file.endswith(".py") and not file.startswith("_"):
        module_name = file[: file.find(".py")]  # Strip ".py" extension
        importlib.import_module(f"iwslt.criterions.{module_name}")
