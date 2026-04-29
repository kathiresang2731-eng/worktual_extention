from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_GENERATE_MODEL = "gemini-2.5-flash"
DEFAULT_DB_NAME = "project_rag.sqlite3"
DEFAULT_DB_DIR_NAME = ".project_rag_store"
DEFAULT_TEMP_UPDATE_DIR = "tmp_updates"
LEGACY_DB_NAME = ".project_rag.sqlite3"


def load_dotenv(dotenv_path: str = ".env") -> Dict[str, str]:
    env_values: Dict[str, str] = {}
    env_file = Path(dotenv_path)
    if not env_file.exists():
        return env_values

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env_values[key] = value
        os.environ.setdefault(key, value)
    return env_values


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def extract_json_block(raw_text: str) -> Dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Gemini response did not contain a JSON object.")
    return json.loads(text[start : end + 1])


@dataclass
class RagConfig:
    api_key: str
    project_root: str
    db_path: Optional[str] = None
    embed_model: str = DEFAULT_EMBED_MODEL
    generate_model: str = DEFAULT_GENERATE_MODEL
    chunk_size_lines: int = 120
    chunk_overlap_lines: int = 30
    max_file_bytes: int = 512_000
    max_notebook_bytes: int = 2_000_000
    top_k: int = 8
    max_context_chars: int = 24_000
    max_context_chunks: int = 10
    include_extensions: Tuple[str, ...] = (
        ".ipynb",
        ".py",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".cs",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".json",
        ".yml",
        ".yaml",
        ".toml",
        ".md",
        ".html",
        ".css",
        ".scss",
        ".sql",
        ".sh",
    )
    exclude_dirs: Tuple[str, ...] = (
        ".git",
        "node_modules",
        "dist",
        "build",
        ".next",
        ".turbo",
        ".idea",
        ".vscode",
        "__pycache__",
        ".venv",
        "venv",
        "coverage",
        ".project_rag",
    )

    def resolved_db_path(self) -> str:
        if self.db_path:
            return self.db_path
        project_hash = hashlib.sha256(self.project_root.encode("utf-8")).hexdigest()[:16]
        db_dir = Path(self.project_root) / DEFAULT_DB_DIR_NAME
        db_dir.mkdir(parents=True, exist_ok=True)
        return str(db_dir / f"{project_hash}_{DEFAULT_DB_NAME}")

    def legacy_project_db_path(self) -> str:
        return str(Path(self.project_root) / LEGACY_DB_NAME)


class GeminiClient:
    def __init__(self, api_key: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc
        return json.loads(raw)

    def embed_text(self, model: str, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:embedContent?key={self.api_key}"
        )
        payload = {
            "model": f"models/{model}",
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
        }
        response = self._post_json(url, payload)
        values = response.get("embedding", {}).get("values")
        if not values:
            raise RuntimeError("Gemini embedding response did not contain vector values.")
        return values

    def generate_json(self, model: str, prompt: str) -> Dict[str, Any]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        response = self._post_json(url, payload)
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini generate response did not contain candidates: {response}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            raise RuntimeError("Gemini generate response did not contain text.")
        return extract_json_block(text)

    def generate_text(self, model: str, prompt: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
            },
        }
        response = self._post_json(url, payload)
        candidates = response.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini generate response did not contain candidates: {response}")
        parts = candidates[0].get("content", {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)
        if not text.strip():
            raise RuntimeError("Gemini generate response did not contain text.")
        return text.strip()


@dataclass
class CodeChunk:
    path: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    vector: Optional[List[float]] = None
    file_mtime: float = 0.0


class SQLiteVectorStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                file_mtime REAL NOT NULL,
                UNIQUE(path, chunk_index)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def upsert_chunks(self, chunks: Iterable[CodeChunk]) -> int:
        rows = [
            (
                chunk.path,
                chunk.chunk_index,
                chunk.start_line,
                chunk.end_line,
                chunk.content,
                json.dumps(chunk.vector or []),
                chunk.file_mtime,
            )
            for chunk in chunks
        ]
        self.conn.executemany(
            """
            INSERT INTO chunks (
                path, chunk_index, start_line, end_line, content, vector_json, file_mtime
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path, chunk_index) DO UPDATE SET
                start_line = excluded.start_line,
                end_line = excluded.end_line,
                content = excluded.content,
                vector_json = excluded.vector_json,
                file_mtime = excluded.file_mtime
            """,
            rows,
        )
        self.conn.commit()
        return len(rows)

    def delete_path(self, path: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        self.conn.commit()

    def list_indexed_paths(self) -> Dict[str, float]:
        cursor = self.conn.execute("SELECT path, MAX(file_mtime) FROM chunks GROUP BY path")
        return {path: mtime for path, mtime in cursor.fetchall()}

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self.conn.commit()

    def get_metadata(self, key: str) -> Optional[str]:
        cursor = self.conn.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else None

    def count_paths(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(DISTINCT path) FROM chunks")
        row = cursor.fetchone()
        return int(row[0] or 0)

    def count_chunks(self) -> int:
        cursor = self.conn.execute("SELECT COUNT(*) FROM chunks")
        row = cursor.fetchone()
        return int(row[0] or 0)

    def similarity_search(self, query_vector: Sequence[float], top_k: int = 8) -> List[Dict[str, Any]]:
        cursor = self.conn.execute(
            "SELECT path, chunk_index, start_line, end_line, content, vector_json FROM chunks"
        )
        scored: List[Dict[str, Any]] = []
        for path, chunk_index, start_line, end_line, content, vector_json in cursor.fetchall():
            vector = json.loads(vector_json)
            score = cosine_similarity(query_vector, vector)
            scored.append(
                {
                    "path": path,
                    "chunk_index": chunk_index,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": content,
                    "score": score,
                }
            )
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


class ProjectChunker:
    def __init__(self, config: RagConfig) -> None:
        self.config = config

    def iter_project_files(self) -> Iterable[Path]:
        root = Path(self.config.project_root)
        for current_root, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in self.config.exclude_dirs]
            for file_name in files:
                file_path = Path(current_root) / file_name
                if file_path.suffix.lower() not in self.config.include_extensions:
                    continue
                try:
                    size_limit = (
                        self.config.max_notebook_bytes
                        if file_path.suffix.lower() == ".ipynb"
                        else self.config.max_file_bytes
                    )
                    if file_path.stat().st_size > size_limit:
                        continue
                except OSError:
                    continue
                yield file_path

    def chunk_file(self, file_path: Path) -> List[CodeChunk]:
        if file_path.suffix.lower() == ".ipynb":
            text = self._extract_notebook_text(file_path)
        else:
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines()
        if not lines:
            return []

        rel_path = str(file_path.relative_to(self.config.project_root))
        mtime = file_path.stat().st_mtime
        chunks: List[CodeChunk] = []
        size = self.config.chunk_size_lines
        overlap = self.config.chunk_overlap_lines
        step = max(1, size - overlap)

        for chunk_index, start in enumerate(range(0, len(lines), step)):
            end = min(len(lines), start + size)
            slice_lines = lines[start:end]
            if not slice_lines:
                continue
            content = "\n".join(slice_lines).strip()
            if not content:
                continue
            chunks.append(
                CodeChunk(
                    path=rel_path,
                    chunk_index=chunk_index,
                    start_line=start + 1,
                    end_line=end,
                    content=content,
                    file_mtime=mtime,
                )
            )
            if end >= len(lines):
                break
        return chunks

    def _extract_notebook_text(self, file_path: Path) -> str:
        try:
            notebook = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""

        blocks: List[str] = []
        for index, cell in enumerate(notebook.get("cells", []), start=1):
            cell_type = cell.get("cell_type", "unknown")
            source = cell.get("source") or []
            if isinstance(source, list):
                source_text = "".join(source)
            else:
                source_text = str(source)
            source_text = source_text.strip()
            if not source_text:
                continue
            blocks.append(f"# CELL {index} [{cell_type}]\n{source_text}")
        return "\n\n".join(blocks)


class ProjectRagEngine:
    def __init__(self, config: RagConfig) -> None:
        self.config = config
        self.client = GeminiClient(config.api_key)
        self.store = SQLiteVectorStore(config.resolved_db_path())
        self.chunker = ProjectChunker(config)

    def build_or_update_index(self, force: bool = False) -> Dict[str, Any]:
        indexed = self.store.list_indexed_paths()
        embedded_chunks = 0
        embedded_files = 0
        deleted_files = 0
        seen_paths = set()

        for file_path in self.chunker.iter_project_files():
            rel_path = str(file_path.relative_to(self.config.project_root))
            seen_paths.add(rel_path)
            mtime = file_path.stat().st_mtime
            if not force and indexed.get(rel_path) == mtime:
                continue

            chunks = self.chunker.chunk_file(file_path)
            if not chunks:
                self.store.delete_path(rel_path)
                continue

            self.store.delete_path(rel_path)
            for chunk in chunks:
                task_type = "RETRIEVAL_DOCUMENT"
                chunk.vector = self.client.embed_text(
                    model=self.config.embed_model,
                    text=f"{chunk.path}\nLines {chunk.start_line}-{chunk.end_line}\n{chunk.content}",
                    task_type=task_type,
                )
            self.store.upsert_chunks(chunks)
            embedded_chunks += len(chunks)
            embedded_files += 1

        for rel_path in indexed:
            if rel_path not in seen_paths:
                self.store.delete_path(rel_path)
                deleted_files += 1

        self.store.set_metadata("last_indexed_at", str(time.time()))
        self.store.set_metadata("embed_model", self.config.embed_model)

        return {
            "embedded_files": embedded_files,
            "embedded_chunks": embedded_chunks,
            "deleted_files": deleted_files,
            "total_indexed_files": self.store.count_paths(),
            "total_indexed_chunks": self.store.count_chunks(),
            "db_path": self.config.resolved_db_path(),
        }

    def search_related_code(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        query_vector = self.client.embed_text(
            model=self.config.embed_model,
            text=query,
            task_type="RETRIEVAL_QUERY",
        )
        return self.store.similarity_search(query_vector, top_k or self.config.top_k)

    def _extract_requested_filename(self, user_message: str) -> Optional[str]:
        match = re.search(r"([A-Za-z0-9_.\-/]+\.[A-Za-z0-9_]+)", user_message)
        return match.group(1) if match else None

    def _resolve_project_file(self, file_name: str) -> Optional[Path]:
        candidate = Path(self.config.project_root) / file_name
        if candidate.exists() and candidate.is_file():
            return candidate

        matches = []
        for file_path in self.chunker.iter_project_files():
            rel_path = str(file_path.relative_to(self.config.project_root))
            if rel_path == file_name or file_path.name == file_name:
                matches.append(file_path)
        return matches[0] if matches else None

    def _inspect_python_syntax(self, file_path: Path) -> Dict[str, Any]:
        source = file_path.read_text(encoding="utf-8", errors="ignore")
        try:
            completed = subprocess.run(
                ["python3", "-m", "py_compile", str(file_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return {
                    "has_syntax_error": False,
                    "error_count": 0,
                    "issues": [],
                    "source": source,
                }

            stderr_text = (completed.stderr or "").strip()
            line_match = re.search(r"line (\d+)", stderr_text)
            message_line = stderr_text.splitlines()[-1] if stderr_text else "Syntax error"
            issue = {
                "line": int(line_match.group(1)) if line_match else 0,
                "offset": 0,
                "message": message_line,
                "text": "",
            }
            return {
                "has_syntax_error": True,
                "error_count": 1,
                "issues": [issue],
                "source": source,
            }
        except Exception as exc:
            issue = {
                "line": 0,
                "offset": 0,
                "message": f"Unable to inspect syntax: {exc}",
                "text": "",
            }
            return {
                "has_syntax_error": True,
                "error_count": 1,
                "issues": [issue],
                "source": source,
            }

    def _handle_python_syntax_request(self, user_message: str) -> Optional[Dict[str, Any]]:
        lower_message = user_message.lower()
        file_name = self._extract_requested_filename(user_message)
        if not file_name or not file_name.endswith(".py"):
            return None
        if not any(keyword in lower_message for keyword in ("error", "errors", "syntax", "fix")):
            return None

        file_path = self._resolve_project_file(file_name)
        if not file_path:
            return {
                "answer": f"File not found: {file_name}",
                "retrieved_context": [],
            }

        inspection = self._inspect_python_syntax(file_path)
        rel_path = str(file_path.relative_to(self.config.project_root))

        if "how many" in lower_message or "count" in lower_message:
            if inspection["error_count"] == 0:
                answer = f"{rel_path}: 0 syntax errors"
            else:
                issue = inspection["issues"][0]
                answer = (
                    f"{rel_path}: 1 syntax error\n"
                    f"line {issue['line']}: {issue['message']}"
                )
            return {
                "answer": answer,
                "retrieved_context": [
                    {
                        "path": rel_path,
                        "start_line": 1,
                        "end_line": len(inspection["source"].splitlines()),
                        "score": 1.0,
                        "content": inspection["source"],
                    }
                ],
            }

        if "fix" in lower_message:
            if inspection["error_count"] == 0:
                return {
                    "answer": f"{rel_path}: 0 syntax errors",
                    "retrieved_context": [
                        {
                            "path": rel_path,
                            "start_line": 1,
                            "end_line": len(inspection["source"].splitlines()),
                            "score": 1.0,
                            "content": inspection["source"],
                        }
                    ],
                }

            issue = inspection["issues"][0]
            prompt = f"""
You are fixing Python syntax only.

File path:
{rel_path}

Syntax error:
Line {issue['line']}, column {issue['offset']}: {issue['message']}
Problematic text:
{issue['text']}

Return valid JSON only in this shape:
{{
  "error_count": 1,
  "issues": ["short syntax issue"],
  "updated_code": "full corrected file content"
}}

Rules:
- Fix only the syntax issue.
- Keep the rest of the code unchanged unless required for syntax correctness.
- Do not add explanation outside JSON.

Original code:
{inspection["source"]}
""".strip()
            result = self.client.generate_json(self.config.generate_model, prompt)
            answer_lines = [f"{rel_path}: {result.get('error_count', 1)} syntax error"]
            for item in result.get("issues", []):
                answer_lines.append(str(item))
            answer_lines.append("")
            answer_lines.append(result.get("updated_code", ""))
            return {
                "answer": "\n".join(answer_lines).strip(),
                "retrieved_context": [
                    {
                        "path": rel_path,
                        "start_line": 1,
                        "end_line": len(inspection["source"].splitlines()),
                        "score": 1.0,
                        "content": inspection["source"],
                    }
                ],
            }

        return None

    def _detect_project_stack(self) -> Dict[str, Any]:
        indexed_paths = sorted(self.store.list_indexed_paths().keys())
        lower_paths = [path.lower() for path in indexed_paths]

        signals: List[str] = []
        detected = {
            "frontend": [],
            "backend": [],
            "tooling": [],
        }

        if any(path.endswith(("vite.config.ts", "vite.config.js")) for path in lower_paths):
            detected["tooling"].append("vite")
            signals.append("vite config detected")
        if any(path.endswith(("package.json",)) for path in lower_paths):
            detected["tooling"].append("npm")
            signals.append("package.json detected")
        if any(path.endswith((".tsx", ".jsx")) for path in lower_paths):
            detected["frontend"].append("react-like-ui")
            signals.append("tsx/jsx files detected")
        if any("/src/" in f"/{path}" for path in lower_paths):
            signals.append("src directory detected")
        if any(path.endswith(".py") for path in lower_paths):
            detected["backend"].append("python")
            signals.append("python files detected")
        if any(path.endswith((".ts", ".js")) for path in lower_paths):
            detected["backend"].append("javascript-typescript")

        for key in detected:
            detected[key] = sorted(set(detected[key]))

        return {
            "detected_stack": detected,
            "signals": signals,
            "indexed_file_count": len(indexed_paths),
            "sample_files": indexed_paths[:25],
        }

    def _build_context_blocks(self, related_chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
        selected_chunks: List[Dict[str, Any]] = []
        context_blocks: List[str] = []
        current_chars = 0

        for item in related_chunks[: self.config.max_context_chunks]:
            block = "\n".join(
                [
                    f"FILE: {item['path']}",
                    f"LINES: {item['start_line']}-{item['end_line']}",
                    f"SCORE: {item['score']:.4f}",
                    item["content"],
                ]
            )
            if selected_chunks and current_chars + len(block) > self.config.max_context_chars:
                break
            selected_chunks.append(item)
            context_blocks.append(block)
            current_chars += len(block)

        return selected_chunks, context_blocks

    def build_update_plan(
        self,
        user_request: str,
        prompt_instructions: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        related_chunks = self.search_related_code(user_request, top_k=top_k)
        stack_summary = self._detect_project_stack()
        if not related_chunks:
            return {
                "summary": "No related project code was found in the index.",
                "assumptions": [
                    "The project may not be indexed yet, PROJECT_ROOT may be incorrect, or supported source files were not found."
                ],
                "files": [],
                "retrieved_context": [],
            }
        selected_chunks, context_blocks = self._build_context_blocks(related_chunks)

        prompt = f"""
You are updating a source code project using only the supplied retrieved context.

User change request:
{user_request}

Additional prompt instructions from caller:
{prompt_instructions}

Detected project stack summary:
{json.dumps(stack_summary, indent=2)}

Retrieved project context:
{chr(10).join(context_blocks)}

Return valid JSON only with this shape:
{{
  "summary": "short summary",
  "assumptions": ["..."],
  "files": [
    {{
      "path": "relative/path/to/file",
      "reason": "why this file should change",
      "change_type": "replace_block|rewrite_file|create_file|update_notebook_cell",
      "notebook_cell_index": 6,
      "target_lines": {{
        "start": 1,
        "end": 10
      }},
      "updated_code": "full replacement code for the block or full file content",
      "validation_notes": ["what to test"]
    }}
  ]
}}

Rules:
- Prefer the most relevant files from retrieved context.
- You may create new files when the user requests a new backend, API, service, config, or cross-language feature.
- When the project is frontend-heavy, infer the data flow and create backend files in the requested language if the user asks for it.
- New files must fit the existing folder structure and naming style as much as possible.
- Do not invent unrelated files.
- If context is insufficient, say so in assumptions.
- For replace_block, return the replacement code only for the target line range.
- For .ipynb notebook edits, prefer change_type="update_notebook_cell" and set notebook_cell_index as a 1-based cell number.
- Keep code production-ready.
""".strip()
        result = self.client.generate_json(self.config.generate_model, prompt)
        result["retrieved_context"] = selected_chunks
        result["project_stack"] = stack_summary
        return result

    def chat_about_project(
        self,
        user_message: str,
        system_instruction: str = "Answer based on the retrieved project code. Be precise and practical.",
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        syntax_result = self._handle_python_syntax_request(user_message)
        if syntax_result is not None:
            return syntax_result

        related_chunks = self.search_related_code(user_message, top_k=top_k)
        stack_summary = self._detect_project_stack()
        if not related_chunks:
            return {
                "answer": (
                    "I could not find any indexed project code for this question. "
                    "Check whether PROJECT_ROOT is correct, whether the project was indexed successfully, "
                    "and whether the folder contains supported source files."
                ),
                "retrieved_context": [],
            }
        selected_chunks, context_blocks = self._build_context_blocks(related_chunks)

        prompt = f"""
You are a project-aware coding assistant.

System instruction:
{system_instruction}

User message:
{user_message}

Detected project stack summary:
{json.dumps(stack_summary, indent=2)}

Retrieved project code:
{chr(10).join(context_blocks)}

Answer in normal chat style.
If the user is asking for a code update, explain what files should change and why.
If the user asks for a backend in Python or another language, use the frontend context to propose or generate matching backend files.
Do not claim anything that is not supported by the retrieved code.
Keep the answer concise. Do not give long summaries unless the user explicitly asks for detail.
""".strip()

        answer = self.client.generate_text(self.config.generate_model, prompt)
        return {
            "answer": answer,
            "retrieved_context": selected_chunks,
            "project_stack": stack_summary,
        }

    def describe_workspace(self) -> Dict[str, Any]:
        stack_summary = self._detect_project_stack()
        return {
            "workspace_path": self.config.project_root,
            "index_summary": {
                "total_indexed_files": self.store.count_paths(),
                "total_indexed_chunks": self.store.count_chunks(),
            },
            "project_stack": stack_summary,
        }

    def _server_temp_root(self) -> Path:
        temp_root = Path(self.config.project_root) / DEFAULT_DB_DIR_NAME / DEFAULT_TEMP_UPDATE_DIR
        temp_root.mkdir(parents=True, exist_ok=True)
        return temp_root

    def _create_temp_session_dir(self) -> Path:
        session_name = f"session_{int(time.time() * 1000)}"
        session_dir = self._server_temp_root() / session_name
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _prepare_temp_targets(self, update_plan: Dict[str, Any], temp_root: Path) -> None:
        for file_change in update_plan.get("files", []):
            rel_path = file_change.get("path", "").strip()
            if not rel_path:
                continue
            source_path = Path(self.config.project_root) / rel_path
            temp_path = temp_root / rel_path
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.exists() and source_path.is_file():
                shutil.copy2(source_path, temp_path)

    def _apply_update_plan_to_root(self, update_plan: Dict[str, Any], root_path: Path) -> Dict[str, Any]:
        applied_files: List[str] = []
        skipped_files: List[Dict[str, Any]] = []

        for file_change in update_plan.get("files", []):
            rel_path = file_change.get("path", "").strip()
            change_type = file_change.get("change_type", "").strip()
            updated_code = file_change.get("updated_code", "")
            target_lines = file_change.get("target_lines") or {}

            if not rel_path or not change_type:
                skipped_files.append(
                    {"path": rel_path, "reason": "Missing path or change_type in update plan."}
                )
                continue

            absolute_path = root_path / rel_path
            absolute_path.parent.mkdir(parents=True, exist_ok=True)

            if absolute_path.suffix.lower() == ".ipynb" and change_type in {
                "update_notebook_cell",
                "replace_block",
            }:
                if not absolute_path.exists():
                    skipped_files.append(
                        {"path": rel_path, "reason": "Notebook target file does not exist."}
                    )
                    continue
                cell_index = self._infer_notebook_cell_index(file_change, absolute_path)
                if cell_index is None:
                    skipped_files.append(
                        {
                            "path": rel_path,
                            "reason": "Could not determine notebook cell number for update.",
                        }
                    )
                    continue
                notebook_result = self._update_notebook_cell(
                    absolute_path,
                    cell_index,
                    updated_code,
                )
                if notebook_result:
                    applied_files.append(rel_path)
                else:
                    skipped_files.append(
                        {
                            "path": rel_path,
                            "reason": f"Notebook cell {cell_index} does not exist.",
                        }
                    )
                continue

            if change_type == "create_file":
                absolute_path.write_text(updated_code, encoding="utf-8")
                applied_files.append(rel_path)
                continue

            if not absolute_path.exists():
                skipped_files.append(
                    {
                        "path": rel_path,
                        "reason": f"Target file does not exist for change_type={change_type}.",
                    }
                )
                continue

            existing_text = absolute_path.read_text(encoding="utf-8", errors="ignore")

            if change_type == "rewrite_file":
                absolute_path.write_text(updated_code, encoding="utf-8")
                applied_files.append(rel_path)
                continue

            if change_type == "replace_block":
                start = int(target_lines.get("start", 0))
                end = int(target_lines.get("end", 0))
                if start <= 0 or end < start:
                    skipped_files.append(
                        {"path": rel_path, "reason": "Invalid target_lines for replace_block."}
                    )
                    continue
                lines = existing_text.splitlines()
                replacement_lines = updated_code.splitlines()
                new_lines = lines[: start - 1] + replacement_lines + lines[end:]
                if existing_text.endswith("\n"):
                    new_text = "\n".join(new_lines) + "\n"
                else:
                    new_text = "\n".join(new_lines)
                absolute_path.write_text(new_text, encoding="utf-8")
                applied_files.append(rel_path)
                continue

            skipped_files.append(
                {"path": rel_path, "reason": f"Unsupported change_type={change_type}."}
            )

        return {
            "summary": update_plan.get("summary", ""),
            "applied_files": applied_files,
            "skipped_files": skipped_files,
        }

    def _infer_notebook_cell_index(
        self,
        file_change: Dict[str, Any],
        notebook_path: Path,
    ) -> Optional[int]:
        raw_index = file_change.get("notebook_cell_index")
        if isinstance(raw_index, int) and raw_index > 0:
            return raw_index
        if isinstance(raw_index, str) and raw_index.isdigit() and int(raw_index) > 0:
            return int(raw_index)

        searchable = json.dumps(file_change, ensure_ascii=False)
        match = re.search(r"cell\\s*#?\\s*(\\d+)", searchable, re.IGNORECASE)
        if match:
            return int(match.group(1))

        target_lines = file_change.get("target_lines") or {}
        start_line = int(target_lines.get("start", 0) or 0)
        if start_line > 0:
            return self._notebook_cell_index_from_extracted_line(notebook_path, start_line)

        return None

    def _notebook_cell_index_from_extracted_line(
        self,
        notebook_path: Path,
        extracted_line_number: int,
    ) -> Optional[int]:
        extracted_text = self.chunker._extract_notebook_text(notebook_path)
        current_cell: Optional[int] = None
        for line_number, line in enumerate(extracted_text.splitlines(), start=1):
            marker = re.match(r"# CELL\\s+(\\d+)\\s+\\[", line)
            if marker:
                current_cell = int(marker.group(1))
            if line_number >= extracted_line_number:
                return current_cell
        return current_cell

    def _update_notebook_cell(self, notebook_path: Path, cell_index: int, updated_code: str) -> bool:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = notebook.get("cells", [])
        zero_based = cell_index - 1
        if zero_based < 0 or zero_based >= len(cells):
            return False

        lines = updated_code.splitlines()
        if not lines:
            source: List[str] = []
        else:
            source = [line + "\n" for line in lines]
            if not updated_code.endswith("\n"):
                source[-1] = source[-1].rstrip("\n")

        cells[zero_based]["source"] = source
        notebook_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return True

    def _validate_temp_updates(self, temp_root: Path, changed_files: List[str]) -> List[Dict[str, Any]]:
        issues: List[Dict[str, Any]] = []
        for rel_path in changed_files:
            temp_path = temp_root / rel_path
            if not temp_path.exists():
                continue
            if temp_path.suffix.lower() == ".py":
                completed = subprocess.run(
                    ["python3", "-m", "py_compile", str(temp_path)],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0:
                    issues.append(
                        {
                            "path": rel_path,
                            "reason": (completed.stderr or "Python syntax validation failed").strip(),
                        }
                    )
            elif temp_path.suffix.lower() == ".json":
                try:
                    json.loads(temp_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    issues.append({"path": rel_path, "reason": f"JSON validation failed: {exc}"})
        return issues

    def _promote_temp_updates(self, temp_root: Path, changed_files: List[str]) -> None:
        for rel_path in changed_files:
            temp_path = temp_root / rel_path
            target_path = Path(self.config.project_root) / rel_path
            if not temp_path.exists():
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_path, target_path)

    def apply_update_plan(self, update_plan: Dict[str, Any]) -> Dict[str, Any]:
        temp_session_dir = self._create_temp_session_dir()
        self._prepare_temp_targets(update_plan, temp_session_dir)
        staged_result = self._apply_update_plan_to_root(update_plan, temp_session_dir)
        validation_issues = self._validate_temp_updates(
            temp_session_dir, staged_result.get("applied_files", [])
        )

        if validation_issues:
            shutil.rmtree(temp_session_dir, ignore_errors=True)
            return {
                "summary": "Temporary validation failed. Original files were not changed.",
                "applied_files": [],
                "skipped_files": staged_result.get("skipped_files", []) + validation_issues,
                "staged_files": staged_result.get("applied_files", []),
                "used_temp_directory": str(self._server_temp_root()),
            }

        self._promote_temp_updates(temp_session_dir, staged_result.get("applied_files", []))
        shutil.rmtree(temp_session_dir, ignore_errors=True)
        return {
            "summary": update_plan.get("summary", ""),
            "applied_files": staged_result.get("applied_files", []),
            "skipped_files": staged_result.get("skipped_files", []),
            "used_temp_directory": str(self._server_temp_root()),
        }

    def update_project(
        self,
        user_request: str,
        prompt_instructions: str,
        top_k: Optional[int] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        index_result = self.build_or_update_index(force=force_reindex)
        update_plan = self.build_update_plan(
            user_request=user_request,
            prompt_instructions=prompt_instructions,
            top_k=top_k,
        )
        apply_result = self.apply_update_plan(update_plan)
        reindex_result = self.build_or_update_index(force=False)
        return {
            "index_result": index_result,
            "update_plan": update_plan,
            "apply_result": apply_result,
            "reindex_result": reindex_result,
        }

    def chat_and_update_project(
        self,
        user_message: str,
        prompt_instructions: str,
        top_k: Optional[int] = None,
        force_reindex: bool = False,
    ) -> Dict[str, Any]:
        index_result = self.build_or_update_index(force=force_reindex)
        chat_result = self.chat_about_project(
            user_message=user_message,
            system_instruction=prompt_instructions,
            top_k=top_k,
        )
        update_plan = self.build_update_plan(
            user_request=user_message,
            prompt_instructions=prompt_instructions,
            top_k=top_k,
        )
        apply_result = self.apply_update_plan(update_plan)
        reindex_result = self.build_or_update_index(force=False)
        return {
            "index_result": index_result,
            "chat_result": chat_result,
            "update_plan": update_plan,
            "apply_result": apply_result,
            "reindex_result": reindex_result,
        }


def build_engine(
    api_key: Optional[str] = None,
    workspace_path: Optional[str] = None,
    project_root: Optional[str] = None,
    db_path: Optional[str] = None,
    embed_model: Optional[str] = None,
    generate_model: Optional[str] = None,
) -> ProjectRagEngine:
    load_dotenv()
    resolved_api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()
    resolved_project_root = workspace_path or project_root or os.environ.get("PROJECT_ROOT", "").strip()
    resolved_embed_model = embed_model or os.environ.get("GEMINI_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    resolved_generate_model = generate_model or os.environ.get(
        "GEMINI_GENERATE_MODEL", DEFAULT_GENERATE_MODEL
    )

    if not resolved_api_key:
        raise ValueError("Missing GEMINI_API_KEY. Provide it directly or via .env.")
    if not resolved_project_root:
        raise ValueError("Missing PROJECT_ROOT. Provide it directly or via .env.")

    config = RagConfig(
        api_key=resolved_api_key,
        project_root=resolved_project_root,
        db_path=db_path,
        embed_model=resolved_embed_model,
        generate_model=resolved_generate_model,
    )
    return ProjectRagEngine(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project RAG + Gemini code updater")
    parser.add_argument("--workspace-path", dest="workspace_path", help="Absolute path of the current workspace")
    parser.add_argument("--project-root", dest="project_root", help="Absolute path of the target project")
    parser.add_argument("--request", dest="user_request", help="Feature/update request from the user")
    parser.add_argument(
        "--instructions",
        dest="prompt_instructions",
        default="Preserve the existing architecture and coding style.",
        help="Additional prompt instructions for Gemini",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="How many related chunks to send to Gemini",
    )
    parser.add_argument(
        "--force-reindex",
        dest="force_reindex",
        action="store_true",
        help="Rebuild embeddings for all supported files",
    )
    parser.add_argument(
        "--search-only",
        dest="search_only",
        action="store_true",
        help="Only build embeddings and retrieve related code, do not update files",
    )
    parser.add_argument(
        "--chat",
        dest="chat",
        action="store_true",
        help="Get a project-aware chat response using retrieved code",
    )
    parser.add_argument(
        "--chat-update",
        dest="chat_update",
        action="store_true",
        help="Get a chat response and then apply project updates",
    )
    return parser.parse_args()


def demo() -> None:
    load_dotenv()
    args = parse_args()
    engine = build_engine(workspace_path=args.workspace_path, project_root=args.project_root)

    if args.search_only:
        engine.build_or_update_index(force=args.force_reindex)
        if not args.user_request:
            raise SystemExit("Provide --request when using --search-only.")
        results = engine.search_related_code(args.user_request, top_k=args.top_k)
        print(json.dumps({"related_code": results}, indent=2))
        return

    if args.chat:
        engine.build_or_update_index(force=args.force_reindex)
        if not args.user_request:
            raise SystemExit("Provide --request when using --chat.")
        result = engine.chat_about_project(
            user_message=args.user_request,
            system_instruction=args.prompt_instructions,
            top_k=args.top_k,
        )
        print(json.dumps(result, indent=2))
        return

    if args.chat_update:
        if not args.user_request:
            raise SystemExit("Provide --request when using --chat-update.")
        result = engine.chat_and_update_project(
            user_message=args.user_request,
            prompt_instructions=args.prompt_instructions,
            top_k=args.top_k,
            force_reindex=args.force_reindex,
        )
        print(json.dumps(result, indent=2))
        return

    if not args.user_request:
        raise SystemExit("Provide --request for project update flow.")

    result = engine.update_project(
        user_request=args.user_request,
        prompt_instructions=args.prompt_instructions,
        top_k=args.top_k,
        force_reindex=args.force_reindex,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()
