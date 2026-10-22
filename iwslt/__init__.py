import importlib
import os

# Define subdirectories to import from
SUBDIRS = ["criterions", "tasks"]

# Get the current directory (iwslt)
current_dir = os.path.dirname(__file__)

# Dynamically import all Python files in the specified subdirectories
for subdir in SUBDIRS:
    subdir_path = os.path.join(current_dir, subdir)
    if os.path.exists(subdir_path):  # Ensure the subdirectory exists
        for file in sorted(os.listdir(subdir_path)):
            if file.endswith(".py") and not file.startswith("_"):
                # Compute the module name relative to the iwslt package
                module_name = f"iwslt.{subdir}.{file[:-3]}"  # Strip ".py" extension
                importlib.import_module(module_name)

