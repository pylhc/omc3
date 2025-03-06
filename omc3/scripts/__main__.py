from pathlib import Path
this_dir = Path(__file__).parent
scripts = this_dir.glob("[a-zA-Z]*.py")
print("Available scripts:\n")
for script in scripts:
    print(f"omc3.{this_dir.name}.{script.stem}")
print()