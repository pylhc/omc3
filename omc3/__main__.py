from pathlib import Path
this_dir = Path(__file__).parent
scripts = this_dir.glob("[a-zA-Z]*.py")
print("Welcome to omc3!")
print("Available entrypoints:\n")
for script in scripts:
    print(f"omc3.{script.stem}")
print()