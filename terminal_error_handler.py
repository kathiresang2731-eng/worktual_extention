from __future__ import annotations

import re
from typing import Dict


ERROR_PATTERNS = [
    ("missing_dependency", re.compile(r"ModuleNotFoundError|Cannot find module|No module named|ERR_MODULE_NOT_FOUND|Cannot find package", re.IGNORECASE)),
    ("no_tests", re.compile(r"No tests found|No test files found|no tests matched|No files matched", re.IGNORECASE)),
    ("syntax_error", re.compile(r"SyntaxError|IndentationError|Unexpected token|Parsing error", re.IGNORECASE)),
    ("database_error", re.compile(r"sqlite|postgres|mysql|database|sqlalchemy|psycopg|pymysql", re.IGNORECASE)),
    ("test_failure", re.compile(r"FAILED|AssertionError|expected .* but got", re.IGNORECASE)),
    ("port_conflict", re.compile(r"EADDRINUSE|address already in use|port .* in use", re.IGNORECASE)),
]


def classify_terminal_error(output: str) -> Dict[str, str]:
    text = str(output or "").strip()
    if not text:
        return {
            "error_kind": "unknown",
            "root_cause": "No terminal output was available.",
            "recommended_fix": "Run the command again and capture stderr/stdout.",
        }

    for kind, pattern in ERROR_PATTERNS:
        if pattern.search(text):
            return {
                "error_kind": kind,
                "root_cause": _root_cause_for(kind),
                "recommended_fix": _fix_for(kind),
            }

    return {
        "error_kind": "unknown",
        "root_cause": "The command failed but did not match a known issue pattern.",
        "recommended_fix": "Inspect the failing module, dependency graph, and runtime configuration before retrying.",
    }


def _root_cause_for(kind: str) -> str:
    mapping = {
        "missing_dependency": "A required runtime dependency or module import is missing.",
        "no_tests": "The project has a test runner configured, but no matching test files were found.",
        "syntax_error": "The generated or updated source contains invalid syntax.",
        "database_error": "The database configuration or DB-facing code path failed during execution.",
        "test_failure": "At least one automated assertion failed during verification.",
        "port_conflict": "The target runtime port is already occupied by another process.",
    }
    return mapping.get(kind, "Unknown failure.")


def _fix_for(kind: str) -> str:
    mapping = {
        "missing_dependency": "Install the missing dependency or fix the incorrect import path, then rerun the tests.",
        "no_tests": "Create or point the runner to real test files before rerunning E2E verification. Until then, use build/smoke checks only.",
        "syntax_error": "Repair the syntax issue in the reported file before rerunning verification.",
        "database_error": "Verify database URLs, credentials, migrations, and round-trip query logic.",
        "test_failure": "Inspect the failing test output, repair the affected code path, and rerun the full check.",
        "port_conflict": "Stop the process using the port or switch to an available port and rerun.",
    }
    return mapping.get(kind, "Investigate the failing command and retry after applying a targeted fix.")
