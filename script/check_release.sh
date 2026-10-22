#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

git diff --check
git diff --cached --check

mapfile -t shell_files < <(
    git ls-files 'script/*.sh' 'iwslt/*.sh' 'iwslt/**/*.sh'
)
for shell_file in "${shell_files[@]}"; do
    bash -n "${shell_file}"
done

python_bin="${PYTHON:-python3}"
"${python_bin}" - <<'PY'
import ast
from pathlib import Path

files = sorted(Path("iwslt").rglob("*.py"))
for path in files:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
print(f"Checked {len(files)} Python files and all project shell scripts.")
PY
