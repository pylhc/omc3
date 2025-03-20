from pathlib import Path

this_dir = Path(__file__).parent

# Welcome ---
print("Welcome to omc3!\n")

# Scripts ---
scripts = this_dir.glob("[a-zA-Z]*.py")
print("Available entrypoints:\n")
for script in scripts:
    print(f"omc3.{script.stem}")
print()

# Modules ---
modules = this_dir.glob("*/**/__main__.py")
print("Other Modules:\n")
for module in modules:
    print(f"omc3.{module.parent.stem}")
print()