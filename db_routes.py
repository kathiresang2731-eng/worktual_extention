# ─────────────────────────────────────────────────────────────────────────────
# db_routes.py  —  Worktual Dev DB Feature
#
# Provides two FastAPI routes:
#   POST /generate_sql   → AI generates SQL from natural language
#   POST /execute_sql    → Executes SQL on real DB, returns results as JSON
#
# Wire into backend.py:
#   from db_routes import db_router
#   app.include_router(db_router)
#
# Add to requirements.txt:
#   psycopg2-binary>=2.9.0
#   pymysql>=1.1.0
#   pymongo>=4.6.0
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import time
import warnings

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from google import genai
from pydantic import BaseModel
from typing import Optional, List, Any, Dict

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="google")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

db_router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class SQLGenerationRequest(BaseModel):
    user_message: str
    connection_summary: str
    db_type: str
    schema_hint: Optional[str] = ""
    conversation_history: Optional[str] = ""

class SQLGenerationResponse(BaseModel):
    sql: str
    explanation: str
    warnings: List[str]
    db_type: str
    is_destructive: bool = False

class ExecuteRequest(BaseModel):
    sql: str
    db_type: str
    host:              Optional[str] = None
    port:              Optional[int] = None
    database:          Optional[str] = None
    username:          Optional[str] = None
    password:          Optional[str] = None
    file_path:         Optional[str] = None
    connection_string: Optional[str] = None
    confirmed:         bool = False

class ExecuteResponse(BaseModel):
    model_config = {"populate_by_name": True}
    success: bool
    columns:      List[str]       = []
    rows:         List[List[Any]] = []
    row_count:    int             = 0
    affected:     int             = 0
    exec_time_ms: float           = 0
    error_message: Optional[str]  = None   # renamed from 'error' — avoids Pydantic v2 BaseModel conflict
    is_select:    bool            = True
    summary:      str             = ""

# ─────────────────────────────────────────────────────────────────────────────
# SQL generation prompt
# ─────────────────────────────────────────────────────────────────────────────

DB_SYNTAX_RULES = {
    "postgresql": "Use SERIAL/BIGSERIAL PKs, TEXT strings, TIMESTAMPTZ, ILIKE, LIMIT/OFFSET, RETURNING clause.",
    "mysql":      "Use AUTO_INCREMENT PKs, backtick quoting, VARCHAR(255), DATETIME, LIMIT offset,count.",
    "sqlite":     "Use INTEGER PRIMARY KEY AUTOINCREMENT, TEXT/REAL/INTEGER types, no RIGHT JOIN, ? placeholders.",
    "mongodb":    "Output PyMongo Python calls: collection.find(), insert_one(), update_many(), aggregate([...])."
}

SQL_SYSTEM_PROMPT = """\
You are a senior database engineer. Generate production-ready {db_type} queries.

RULES:
1. Output valid {db_type} only. No pseudo-code, no <placeholder> tokens.
2. Use the exact schema provided; if none, state assumptions clearly in warnings.
3. Add SQL comments on non-obvious logic.
4. Use LIMIT unless the user explicitly wants all rows.
5. For DELETE/DROP/TRUNCATE/UPDATE without WHERE: add a prominent WARNING comment and set is_destructive=true.
6. Handle NULLs safely (COALESCE, IS NULL).
7. {db_syntax}

Return ONLY valid JSON (no markdown fences):
{{
  "sql":            "complete query",
  "explanation":    "1-3 sentence plain-English explanation",
  "warnings":       ["list of important notes, assumptions, or performance tips"],
  "is_destructive": false
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# /generate_sql
# ─────────────────────────────────────────────────────────────────────────────

@db_router.post("/generate_sql", response_model=SQLGenerationResponse)
async def generate_sql(req: SQLGenerationRequest):
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API unavailable")

    db_type = req.db_type.lower()
    system  = SQL_SYSTEM_PROMPT.format(
        db_type=db_type.upper(),
        db_syntax=DB_SYNTAX_RULES.get(db_type, "")
    )

    parts = [f"CONNECTION:\n{req.connection_summary}"]
    if req.schema_hint and req.schema_hint.strip():
        parts.append(f"SCHEMA:\n{req.schema_hint.strip()}")
    if req.conversation_history:
        parts.append(f"RECENT CONTEXT:\n{req.conversation_history[-800:]}")
    parts.append(f"USER REQUEST:\n{req.user_message}")

    raw = ""
    try:
        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents="\n\n".join(parts),
            config={"system_instruction": system, "temperature": 0.1, "max_output_tokens": 2048}
        )
        raw = resp.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
        raw = raw.strip()
        parsed = json.loads(raw)
        return SQLGenerationResponse(
            sql=parsed.get("sql", "-- generation failed"),
            explanation=parsed.get("explanation", ""),
            warnings=parsed.get("warnings", []),
            db_type=db_type,
            is_destructive=parsed.get("is_destructive", False)
        )
    except json.JSONDecodeError:
        return SQLGenerationResponse(
            sql=_extract_sql(raw), explanation="", db_type=db_type,
            warnings=["Response format unexpected — review carefully."]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SQL generation failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# /execute_sql
# ─────────────────────────────────────────────────────────────────────────────

ROW_CAP = 500

@db_router.post("/execute_sql", response_model=ExecuteResponse)
async def execute_sql(req: ExecuteRequest):
    if _is_destructive(req.sql) and not req.confirmed:
        return ExecuteResponse(
            success=False,
            error_message="DESTRUCTIVE_UNCONFIRMED",
            summary="Query contains a destructive operation. Confirm to proceed.",
            is_select=False
        )
    try:
        t0 = time.monotonic()
        db_type = req.db_type.lower()
        if db_type == "postgresql":
            result = _exec_postgres(req)
        elif db_type == "mysql":
            result = _exec_mysql(req)
        elif db_type == "sqlite":
            result = _exec_sqlite(req)
        elif db_type == "mongodb":
            result = _exec_mongo(req)
        else:
            raise ValueError(f"Unsupported db_type: {db_type}")
        result["exec_time_ms"] = round((time.monotonic() - t0) * 1000, 1)
        return ExecuteResponse(**result)
    except Exception as e:
        return ExecuteResponse(
            success=False,
            error_message=_friendly_error(str(e), req.db_type),
            summary=f"Execution failed: {type(e).__name__}"
        )


def _exec_postgres(req):
    import psycopg2, psycopg2.extras
    dsn = req.connection_string or _pg_dsn(req)
    conn = psycopg2.connect(dsn, connect_timeout=10)
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(req.sql)
            if cur.description:
                rows_raw = cur.fetchmany(ROW_CAP)
                cols = [d.name for d in cur.description]
                rows = [list(r.values()) for r in rows_raw]
                conn.rollback()
                cap = f" (capped at {ROW_CAP})" if len(rows) == ROW_CAP else ""
                return dict(success=True, is_select=True, columns=cols,
                            rows=_ser(rows), row_count=len(rows),
                            summary=f"Returned {len(rows)} row(s){cap}")
            else:
                aff = cur.rowcount; conn.commit()
                return dict(success=True, is_select=False, affected=aff,
                            summary=f"{aff} row(s) affected")
    except:
        conn.rollback(); raise
    finally:
        conn.close()


def _exec_mysql(req):
    import pymysql, pymysql.cursors
    cfg = _my_cfg(req)
    conn = pymysql.connect(**cfg, cursorclass=pymysql.cursors.DictCursor,
                           connect_timeout=10, autocommit=False)
    try:
        with conn.cursor() as cur:
            cur.execute(req.sql)
            if cur.description:
                rows_raw = cur.fetchmany(ROW_CAP)
                cols = [d[0] for d in cur.description]
                rows = [list(r.values()) for r in rows_raw]
                conn.rollback()
                cap = f" (capped at {ROW_CAP})" if len(rows) == ROW_CAP else ""
                return dict(success=True, is_select=True, columns=cols,
                            rows=_ser(rows), row_count=len(rows),
                            summary=f"Returned {len(rows)} row(s){cap}")
            else:
                aff = cur.rowcount; conn.commit()
                return dict(success=True, is_select=False, affected=aff,
                            summary=f"{aff} row(s) affected")
    except:
        conn.rollback(); raise
    finally:
        conn.close()


def _exec_sqlite(req):
    import sqlite3
    path = req.file_path or req.database or ":memory:"
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.execute(req.sql)
        if cur.description:
            rows_raw = cur.fetchmany(ROW_CAP)
            cols = [d[0] for d in cur.description]
            rows = [list(r) for r in rows_raw]
            cap  = f" (capped at {ROW_CAP})" if len(rows) == ROW_CAP else ""
            return dict(success=True, is_select=True, columns=cols,
                        rows=_ser(rows), row_count=len(rows),
                        summary=f"Returned {len(rows)} row(s){cap}")
        else:
            conn.commit()
            return dict(success=True, is_select=False, affected=cur.rowcount,
                        summary=f"{cur.rowcount} row(s) affected")
    except:
        conn.rollback(); raise
    finally:
        conn.close()


def _exec_mongo(req):
    from pymongo import MongoClient
    import bson
    uri    = req.connection_string or _mongo_uri(req)
    client = MongoClient(uri, serverSelectionTimeoutMS=9000)
    db_obj = client[req.database or uri.rsplit("/",1)[-1].split("?")[0] or "test"]
    expr   = req.sql.strip().rstrip(";")
    try:
        result = eval(expr, {"__builtins__": {}}, {"db": db_obj, "bson": bson})
    except Exception as e:
        client.close()
        raise RuntimeError(f"Mongo eval failed: {e}")
    rows: List[List[Any]] = []
    cols: List[str] = []
    if hasattr(result, "__iter__") and not isinstance(result, dict):
        docs = []
        try:
            for _, d in zip(range(ROW_CAP), result):
                docs.append(d)
        except Exception:
            pass
        if docs:
            key_set: dict = {}
            for d in docs:
                for k in d: key_set[k] = None
            cols = list(key_set)
            rows = [[str(d.get(c, "")) for c in cols] for d in docs]
        client.close()
        return dict(success=True, is_select=True, columns=cols, rows=rows,
                    row_count=len(rows), summary=f"Returned {len(rows)} document(s)")
    elif isinstance(result, dict):
        info = {k: str(v) for k, v in result.items() if not k.startswith("_")}
        client.close()
        return dict(success=True, is_select=False,
                    columns=list(info), rows=[list(info.values())],
                    summary="Operation completed")
    client.close()
    return dict(success=True, is_select=False, summary=str(result))


# ─── Connection builders ─────────────────────────────────────────────────────

def _pg_dsn(req):
    parts = []
    if req.host:     parts.append(f"host={req.host}")
    if req.port:     parts.append(f"port={req.port}")
    if req.database: parts.append(f"dbname={req.database}")
    if req.username: parts.append(f"user={req.username}")
    if req.password: parts.append(f"password={req.password}")
    return " ".join(parts)

def _my_cfg(req):
    if req.connection_string:
        from urllib.parse import urlparse
        u = urlparse(req.connection_string)
        return dict(host=u.hostname, port=u.port or 3306,
                    user=u.username, password=u.password,
                    database=u.path.lstrip("/"))
    return dict(host=req.host or "localhost", port=req.port or 3306,
                user=req.username or "", password=req.password or "",
                database=req.database or "")

def _mongo_uri(req):
    if req.connection_string: return req.connection_string
    h, p = req.host or "localhost", req.port or 27017
    if req.username and req.password:
        from urllib.parse import quote_plus
        return f"mongodb://{quote_plus(req.username)}:{quote_plus(req.password)}@{h}:{p}/{req.database or ''}"
    return f"mongodb://{h}:{p}/{req.database or ''}"


# ─── Utilities ───────────────────────────────────────────────────────────────

_DESTRUCTIVE = re.compile(
    r"\b(DELETE\s+FROM|DROP\s+(TABLE|DATABASE|INDEX|VIEW)|TRUNCATE|"
    r"UPDATE\s+\w[\w.]*\s+SET(?![\s\S]*\bWHERE\b))\b",
    re.IGNORECASE | re.DOTALL
)

def _is_destructive(sql):
    return bool(_DESTRUCTIVE.search(sql))

def _ser(rows):
    import decimal, datetime
    out = []
    for row in rows:
        r = []
        for c in row:
            if isinstance(c, (decimal.Decimal, datetime.date, datetime.datetime,
                               datetime.time, bytes, bytearray)):
                r.append(str(c))
            else:
                r.append(c)
        out.append(r)
    return out

def _extract_sql(text):
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.I)
    if m: return m.group(1).strip()
    m = re.search(r"((?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH)\b[\s\S]+)", text, re.I)
    if m: return m.group(1).strip()
    return text.strip()

def _friendly_error(msg, db_type):
    lo = msg.lower()
    if "connection refused" in lo or "could not connect" in lo:
        return f"Cannot reach {db_type} server. Check host/port and confirm the server is running."
    if "authentication" in lo or "access denied" in lo or "password" in lo:
        return "Authentication failed. Check your username and password."
    if "does not exist" in lo or "unknown database" in lo or "no such table" in lo:
        return f"Database or table not found: {msg}"
    if "permission denied" in lo or "insufficient privilege" in lo:
        return "Permission denied. The DB user may lack the required privileges."
    if "syntax error" in lo:
        return f"SQL syntax error — check the query: {msg}"
    if "timeout" in lo:
        return "Connection timed out. The database may be unreachable."
    return msg[:400]
