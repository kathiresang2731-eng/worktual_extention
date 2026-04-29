from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_env_files(project_root: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for name in (".env", ".env.example", "settings.env"):
        env_path = Path(project_root) / name
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def _detect_package_manager(project_root: str) -> str:
    root = Path(project_root)
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "yarn.lock").exists():
        return "yarn"
    return "npm"


def _script_command(manager: str, script: str) -> str:
    return f"npm run {script}" if manager == "npm" else f"{manager} {script}"


def _is_usable_script(value: str) -> bool:
    return bool(value and value.strip() and "no test specified" not in value.lower())


def _build_node_commands_for_root(project_root: str, package_json: Path) -> List[Dict[str, str]]:
    pkg = _read_json(package_json)
    scripts = pkg.get("scripts", {}) if isinstance(pkg, dict) else {}
    manager = _detect_package_manager(str(package_json.parent))
    commands: List[Dict[str, str]] = []
    has_playwright_specs = _has_playwright_specs(package_json.parent)

    if _is_usable_script(str(scripts.get("compile", ""))):
        commands.append({
            "label": "TypeScript compile check",
            "command": _script_command(manager, "compile"),
            "cwd": str(package_json.parent),
            "kind": "compile",
        })
    if _is_usable_script(str(scripts.get("build", ""))):
        commands.append({
            "label": "Build check",
            "command": _script_command(manager, "build"),
            "cwd": str(package_json.parent),
            "kind": "build",
        })
    if _is_usable_script(str(scripts.get("test", ""))):
        commands.append({
            "label": "Automated tests",
            "command": _script_command(manager, "test"),
            "cwd": str(package_json.parent),
            "kind": "test",
        })
    e2e_script = next(
        (
            name for name in ("test:e2e", "e2e", "playwright", "test:e2e:headed")
            if _is_usable_script(str(scripts.get(name, "")))
        ),
        None,
    )
    if e2e_script and has_playwright_specs:
        commands.append({
            "label": "End-to-end tests",
            "command": _script_command(manager, e2e_script),
            "cwd": str(package_json.parent),
            "kind": "e2e",
        })
    elif (
        has_playwright_specs and (
        (package_json.parent / "playwright.config.js").exists()
        or (package_json.parent / "playwright.config.ts").exists()
        or pkg.get("dependencies", {}).get("@playwright/test")
        or pkg.get("devDependencies", {}).get("@playwright/test")
        )
    ):
        commands.append({
            "label": "Playwright end-to-end tests",
            "command": "npx playwright test",
            "cwd": str(package_json.parent),
            "kind": "e2e",
        })

    return commands


def _has_playwright_specs(project_root: Path) -> bool:
    skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "out", "coverage", "__pycache__", ".venv", "venv", "playwright-report", "test-results"}
    queue: List[tuple[Path, int]] = [(project_root, 0)]
    test_file_pattern = re.compile(r".*\.(spec|test)\.(ts|tsx|js|jsx|mjs|cjs)$", re.IGNORECASE)

    while queue:
        current, depth = queue.pop(0)
        if depth > 4:
            continue
        try:
            for entry in current.iterdir():
                if entry.is_dir():
                    if entry.name in skip_dirs:
                        continue
                    queue.append((entry, depth + 1))
                    continue
                if entry.is_file() and test_file_pattern.match(entry.name):
                    return True
        except Exception:
            continue

    return False


def _score_node_test_root(project_root: str, package_json: Path) -> int:
    candidate_root = package_json.parent
    rel = candidate_root.relative_to(Path(project_root)).as_posix() if candidate_root != Path(project_root) else ""
    name = candidate_root.name.lower()
    depth = len([part for part in rel.split("/") if part]) if rel else 0
    score = 0

    if name == "frontend":
        score += 45
    if name in {"client", "web", "ui"}:
        score += 30
    if rel and any(segment in rel.lower() for segment in ("frontend", "client", "web", "ui")):
        score += 20
    if (candidate_root / "playwright.config.js").exists() or (candidate_root / "playwright.config.ts").exists():
        score += 80
    if (candidate_root / "e2e").exists() or (candidate_root / "tests" / "e2e").exists():
        score += 35

    pkg = _read_json(package_json)
    scripts = pkg.get("scripts", {}) if isinstance(pkg, dict) else {}
    if _is_usable_script(str(scripts.get("test:e2e", ""))):
        score += 60
    if _is_usable_script(str(scripts.get("e2e", ""))):
        score += 50
    if _is_usable_script(str(scripts.get("playwright", ""))):
        score += 40
    if pkg.get("dependencies", {}).get("@playwright/test") or pkg.get("devDependencies", {}).get("@playwright/test"):
        score += 35
    if pkg.get("dependencies", {}).get("react") or pkg.get("devDependencies", {}).get("react"):
        score += 10
    if pkg.get("dependencies", {}).get("vite") or pkg.get("devDependencies", {}).get("vite"):
        score += 10

    score -= depth * 2
    return score


def _find_node_test_roots(project_root: str) -> List[Path]:
    root = Path(project_root)
    if not root.exists():
        return []

    discovered: List[tuple[int, Path]] = []
    skip_dirs = {"node_modules", ".git", "dist", "build", ".next", "out", "coverage", "__pycache__", ".venv", "venv"}
    queue: List[tuple[Path, int]] = [(root, 0)]
    seen: set[str] = set()

    while queue:
        current, depth = queue.pop(0)
        normalized = str(current.resolve())
        if normalized in seen:
            continue
        seen.add(normalized)

        package_json = current / "package.json"
        if package_json.exists():
            discovered.append((_score_node_test_root(project_root, package_json), package_json))

        if depth >= 4:
            continue

        try:
            for entry in current.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name in skip_dirs:
                    continue
                queue.append((entry, depth + 1))
        except Exception:
            continue

    discovered.sort(key=lambda item: item[0], reverse=True)
    return [package_json for _, package_json in discovered]


def _build_node_commands(project_root: str) -> List[Dict[str, str]]:
    node_roots = _find_node_test_roots(project_root)
    if not node_roots:
        return []

    return _build_node_commands_for_root(project_root, node_roots[0])


def _collect_python_files(project_root: str, written_files: List[str], full_project: bool) -> List[str]:
    root = Path(project_root)
    if not full_project:
        result: List[str] = []
        for file_path in written_files:
            if not str(file_path).endswith(".py"):
                continue
            candidate = Path(file_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            if candidate.exists():
                result.append(candidate.relative_to(root).as_posix())
        return result

    result: List[str] = []
    for path in root.rglob("*.py"):
        if any(part in {"__pycache__", ".venv", "venv", "node_modules", "dist", "build", "out"} for part in path.parts):
            continue
        result.append(path.relative_to(root).as_posix())
    return result[:200]


def _build_python_commands(project_root: str, written_files: List[str], full_project: bool) -> List[Dict[str, str]]:
    root = Path(project_root)
    has_python = (
        any(str(item).endswith(".py") for item in written_files)
        or (root / "requirements.txt").exists()
        or (root / "pyproject.toml").exists()
    )
    if not has_python:
        return []

    if (root / "pytest.ini").exists() or (root / "conftest.py").exists() or (root / "tests").exists():
        return [{
            "label": "Python tests",
            "command": "python -m pytest -q",
            "cwd": project_root,
            "kind": "test",
        }]

    python_files = _collect_python_files(project_root, written_files, full_project)
    if not python_files:
        return []

    return [{
        "label": "Project Python syntax check" if full_project else "Changed-file Python syntax check",
        "command": "python -m py_compile " + " ".join(f'"{item}"' for item in python_files),
        "cwd": project_root,
        "kind": "syntax",
    }]


def _find_sqlite_path(project_root: str) -> Optional[str]:
    env_values = _parse_env_files(project_root)
    sqlite_url = env_values.get("DATABASE_URL") or env_values.get("SQLITE_URL")
    if sqlite_url and sqlite_url.startswith("sqlite:///"):
        return str((Path(project_root) / sqlite_url.replace("sqlite:///", "", 1)).resolve())

    for key in ("SQLITE_PATH", "SQLITE_DB", "SQLITE_FILE", "DATABASE"):
        value = env_values.get(key)
        if value and value.endswith((".db", ".sqlite", ".sqlite3")):
            return str((Path(project_root) / value).resolve())
    return None


def _build_sqlite_smoke_test(project_root: str) -> List[Dict[str, str]]:
    sqlite_path = _find_sqlite_path(project_root)
    if not sqlite_path:
        return []

    script = "; ".join([
        "import sqlite3",
        "from pathlib import Path",
        f'db_path = Path(r"{sqlite_path}")',
        "db_path.parent.mkdir(parents=True, exist_ok=True)",
        "conn = sqlite3.connect(db_path)",
        "cur = conn.cursor()",
        'cur.execute("CREATE TEMP TABLE IF NOT EXISTS worktual_dev_testing_agent (value TEXT)")',
        'cur.execute("INSERT INTO worktual_dev_testing_agent(value) VALUES (?)", ("ok",))',
        'row = cur.execute("SELECT value FROM worktual_dev_testing_agent ORDER BY rowid DESC LIMIT 1").fetchone()',
        'print("DB_OK:" + (row[0] if row else "missing"))',
        "conn.close()",
    ])
    escaped_script = script.replace('"', '\\"')

    return [{
        "label": "SQLite insert/select smoke test",
        "command": f'python -c "{escaped_script}"',
        "cwd": project_root,
        "kind": "db",
    }]


def build_test_plan(
    project_root: str,
    written_files: Optional[List[str]] = None,
    user_request: str = "",
    full_project: bool = False,
) -> Dict[str, Any]:
    written_files = list(written_files or [])
    commands: List[Dict[str, str]] = []
    seen = set()

    for command in (
        _build_node_commands(project_root)
        + _build_python_commands(project_root, written_files, full_project)
        + _build_sqlite_smoke_test(project_root)
    ):
        key = (command["cwd"], command["command"])
        if key in seen:
            continue
        seen.add(key)
        commands.append(command)

    detected_stacks: List[str] = []
    root = Path(project_root)
    if _find_node_test_roots(project_root):
        detected_stacks.append("node")
    if (root / "requirements.txt").exists() or any(str(item).endswith(".py") for item in written_files):
        detected_stacks.append("python")
    if _find_sqlite_path(project_root):
        detected_stacks.append("sqlite")

    return {
        "summary": "Full-project verification plan" if full_project else "Post-write verification plan",
        "project_root": project_root,
        "commands": commands,
        "todo_path": str(root / "TODO.md"),
        "todo_exists": (root / "TODO.md").exists(),
        "detected_stacks": detected_stacks,
        "user_request": user_request,
    }
