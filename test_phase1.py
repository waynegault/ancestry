from phase1_cleanup import get_all_python_files

files = get_all_python_files()
print(f"Would process {len(files)} files:")
for f in files[:10]:
    print(f"  - {f.name}")
if len(files) > 10:
    print(f"  ... and {len(files)-10} more")
