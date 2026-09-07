import glob
import json
import sys


def python_only(source):
    if isinstance(source, str):
        source = source.splitlines(keepends=True)
    out = []
    in_shell = False
    for line in source:
        stripped = line.lstrip()
        indent = " " * (len(line) - len(stripped))
        if in_shell:
            in_shell = stripped.rstrip("\n").endswith("\\")
            continue
        if stripped.startswith(("!", "%")):
            out.append(indent + "pass\n")
            in_shell = stripped.rstrip("\n").endswith("\\")
            continue
        out.append(line)
    return "".join(out)


def main():
    paths = sorted(glob.glob("notebooks/*.ipynb"))
    if not paths:
        print("no notebooks found")
        return 1
    failed = 0
    for path in paths:
        try:
            nb = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{path}: INVALID JSON - {exc}")
            failed += 1
            continue
        errors = []
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                compile(python_only(cell["source"]), f"{path}:cell{i}", "exec")
            except SyntaxError as exc:
                errors.append(f"cell {i}: {exc.msg} (line {exc.lineno})")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        if errors:
            failed += 1
            print(f"{path}: {n_code} code cells, {len(errors)} FAILED")
            for e in errors:
                print(f"  {e}")
        else:
            print(f"{path}: {n_code} code cells, ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
