import subprocess
import json
import re
import sys
from pathlib import Path

def get_tracked_files():
    result = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, encoding="utf-8"
    )
    return [f for f in result.stdout.splitlines() if f.strip()]

def extract_py_comments(filepath):
    comments = []
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return [(0, f"[COULD NOT READ FILE: {e}]")]
    in_docstring = False
    docstring_delim = None
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not in_docstring:
            for delim in ('"""', "'''"):
                if stripped.startswith(delim):
                    if stripped.count(delim) >= 2 and len(stripped) > 3:
                        comments.append((i, stripped))
                    else:
                        in_docstring = True
                        docstring_delim = delim
                        comments.append((i, stripped))
                    break
            else:
                if "#" in line:
                    comments.append((i, stripped))
        else:
            comments.append((i, stripped))
            if docstring_delim in stripped:
                in_docstring = False
    return comments

def extract_js_comments(filepath):
    comments = []
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return [(0, f"[COULD NOT READ FILE: {e}]")]
    in_block = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if in_block:
            comments.append((i, stripped))
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            comments.append((i, stripped))
            if "*/" not in stripped:
                in_block = True
        elif "//" in stripped:
            comments.append((i, stripped))
    return comments

def extract_ipynb_comments(filepath):
    results = []
    try:
        with open(filepath, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        return [(0, f"[COULD NOT PARSE NOTEBOOK: {e}]")]
    for cell_idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        if isinstance(source, list):
            source_text = "".join(source)
        else:
            source_text = source
        for line_idx, line in enumerate(source_text.splitlines(), 1):
            stripped = line.strip()
            if "#" in stripped:
                results.append((f"cell {cell_idx}, line {line_idx}", stripped))
    return results

def main():
    files = get_tracked_files()
    report_lines = []
    total_comments = 0

    for f in files:
        path = Path(f)
        if not path.exists():
            continue
        ext = path.suffix.lower()
        comments = []
        if ext == ".py":
            comments = extract_py_comments(f)
        elif ext == ".js":
            comments = extract_js_comments(f)
        elif ext == ".ipynb":
            comments = extract_ipynb_comments(f)
        else:
            continue

        if comments:
            report_lines.append(f"\n{'='*80}\nFILE: {f}  ({len(comments)} comment lines)\n{'='*80}")
            for loc, text in comments:
                report_lines.append(f"  [{loc}] {text}")
            total_comments += len(comments)

    report_lines.insert(0, f"COMMENT AUDIT REPORT\nTotal files with comments scanned. Total comment lines found: {total_comments}\n")

    output_path = "comment_audit_report.txt"
    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(report_lines))

    print(f"Done. Report written to {output_path}")
    print(f"Total comment lines found: {total_comments}")

if __name__ == "__main__":
    main()