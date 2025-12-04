import pathlib

with pathlib.Path("simple_result.txt").open("w", encoding="utf-8") as f:
    f.write("Hello from python")
