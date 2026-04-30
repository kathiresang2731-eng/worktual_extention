import os
import warnings
import requests
import json
import ast
import sys
import io
import subprocess
import threading
import time
import fnmatch

import re
import json as pyjson
from pathlib import Path
from datetime import datetime
from google import genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Tuple
from contextlib import redirect_stdout
from dotenv import load_dotenv
from embedd_The_project import build_engine
from project_update_with_llm import get_project_update_response
from db_routes import db_router
from integration_routes import integration_router
from terminal_runner import build_test_plan
from terminal_error_handler import classify_terminal_error

# Silence harmless Pydantic warnings from the google-genai SDK
# (Field name shadowing in internal Operation model — not our code)
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="google")

load_dotenv()

# ------------------------------------------------------------
# Server mode: when True, no local file writes are performed.
# All file operations are returned as messages to the extension.
# ------------------------------------------------------------
SERVER_MODE = True

# ------------------------------------------------------------
# Gemini API setup
# ------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Gemini API key not configured")
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")  # override via .env

# Separate fast model just for inline completions.
# Gemini 2.0 Flash has been deprecated; default to the current Flash model.
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL", "gemini-2.5-flash")  # override via .env

# ------------------------------------------------------------
# Low-cost context + logging defaults
# ------------------------------------------------------------
CHAT_HISTORY_CHAR_LIMIT = int(os.getenv("CHAT_HISTORY_CHAR_LIMIT", "6000"))
CHAT_WORKSPACE_SUMMARY_CHAR_LIMIT = int(os.getenv("CHAT_WORKSPACE_SUMMARY_CHAR_LIMIT", "2500"))
UPDATE_PROJECT_CONTEXT_CHAR_LIMIT = int(os.getenv("UPDATE_PROJECT_CONTEXT_CHAR_LIMIT", "28000"))
UPDATE_PROJECT_PER_FILE_CHAR_LIMIT = int(os.getenv("UPDATE_PROJECT_PER_FILE_CHAR_LIMIT", "4500"))
UPDATE_PROJECT_HISTORY_CHAR_LIMIT = int(os.getenv("UPDATE_PROJECT_HISTORY_CHAR_LIMIT", "7000"))
ANALYZE_FRONTEND_CONTEXT_CHAR_LIMIT = int(os.getenv("ANALYZE_FRONTEND_CONTEXT_CHAR_LIMIT", "70000"))
ANALYZE_FRONTEND_PER_FILE_CHAR_LIMIT = int(os.getenv("ANALYZE_FRONTEND_PER_FILE_CHAR_LIMIT", "5000"))

_LLM_LOG_LOCK = threading.Lock()
_LLM_LOG_ROOT = Path(__file__).resolve().parent.parent


def _current_vscode_log_path() -> Path:
    return _LLM_LOG_ROOT / f"VsCode_Log_{datetime.now():%d_%m_%Y}.log"


def _clip_text_for_model(text: str, max_chars: int, keep: str = "tail") -> str:
    raw = str(text or "")
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    note = "\n[...truncated for low-cost context...]\n"
    budget = max(0, max_chars - len(note))
    if keep == "head":
        return raw[:budget] + note
    return note + raw[-budget:]


def _compact_text_for_log(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _extract_usage_metadata(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage_metadata", None)

    def _to_int(field_name: str) -> int:
        try:
            value = getattr(usage, field_name, 0)
            return int(value or 0)
        except Exception:
            return 0

    return {
        "prompt": _to_int("prompt_token_count"),
        "completion": _to_int("candidates_token_count"),
        "total": _to_int("total_token_count"),
        "cached": _to_int("cached_content_token_count"),
        "thoughts": _to_int("thoughts_token_count"),
    }


def _log_model_usage(
    label: str,
    model_name: str,
    user_query: str,
    response: Any,
    duration_ms: int,
    extra: Optional[str] = None,
) -> None:
    usage = _extract_usage_metadata(response)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = (
        f"[{timestamp}] label={label} | model={model_name} | "
        f"prompt_tokens={usage['prompt']} | completion_tokens={usage['completion']} | "
        f"total_tokens={usage['total']} | cached_tokens={usage['cached']} | "
        f"thought_tokens={usage['thoughts']} | duration_ms={duration_ms} | "
        f"query=\"{_compact_text_for_log(user_query)}\""
    )
    if extra:
        log_line += f" | extra={extra}"

    print(f"[VSCode LLM] {log_line}")

    try:
        log_path = _current_vscode_log_path()
        with _LLM_LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")
    except Exception as log_error:
        print(f"[VSCode LLM] Failed to write log file: {log_error}")


def _log_model_failure(label: str, model_name: str, user_query: str, err: Exception) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = (
        f"[{timestamp}] label={label} | model={model_name} | error={_compact_text_for_log(str(err), 240)} | "
        f"query=\"{_compact_text_for_log(user_query)}\""
    )
    print(f"[VSCode LLM] {log_line}")
    try:
        log_path = _current_vscode_log_path()
        with _LLM_LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(log_line + "\n")
    except Exception:
        pass


def generate_content_logged(
    *,
    label: str,
    contents: Any,
    model: Optional[str] = None,
    config: Optional[dict] = None,
    user_query: str = "",
    extra: Optional[str] = None,
):
    chosen_model = model or GEMINI_MODEL
    started = time.time()
    try:
        response = client.models.generate_content(
            model=chosen_model,
            contents=contents,
            config=config,
        )
    except Exception as err:
        _log_model_failure(label, chosen_model, user_query, err)
        raise

    duration_ms = int((time.time() - started) * 1000)
    _log_model_usage(label, chosen_model, user_query, response, duration_ms, extra=extra)
    return response

# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Smart Dev AI Backend")
app.include_router(db_router)
app.include_router(integration_router)

# ------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    conversation_history: str = ""
    pending_action: Optional[dict] = None
    files: Optional[List[dict]] = None
    workspace_summary: str = ""
    response_mode: str = "chat"

class ChatResponse(BaseModel):
    messages: List[dict]


class TestingPlanRequest(BaseModel):
    project_root: str
    written_files: List[str] = []
    user_request: str = ""
    full_project: bool = False


class TestingBlueprintRequest(BaseModel):
    project_root: str = ""
    user_request: str
    workspace_summary: str = ""


SYSTEM_PROMPT = """
ROLE:
You are the Worktual VS Code Extension AI Assistant - an expert-level AI software architect and senior developer.
You act as BOTH:
1) a developer assistant explaining advanced concepts with production-ready examples
2) an execution agent that issues structured JSON actions for the extension.

3) website Builder: You can create complete websites with modern frameworks like React, Next.js, Vue, Angular, Svelte, etc. You generate all necessary files and configurations.

website building Rules:
- When user asks to create a website, you generate ALL necessary files (HTML, CSS, JS, config files) in ONE response. Do NOT create files one by one.
- Use modern frameworks and best practices for the requested website type.
- Always include proper error handling, security considerations, and optimizations in the generated code.

- The generated website must be have all the necessary files to run immediately after creation (no placeholders, no missing dependencies).

Your primary goal is to help users build, modify, debug, and run projects with PRODUCTION-QUALITY, ADVANCED-LEVEL code only.

If asked about yourself, reply:
"I am an AI assistant integrated into VS Code through the Worktual extension. I provide advanced, production-quality code and help create files, debug code, run programs, and manage projects using structured actions."



IMPORTANT OUTPUT CONTINUATION RULE:

If the generated code exceeds the response length limit:

- Continue generating the remaining code automatically.
- Do NOT stop mid-file.
- Do NOT summarize code.
- Do NOT shorten the implementation.

If a file is cut off due to response limits:
Continue exactly from the last line of the previous response.

NEVER output:
"..."
"# rest of code"
"code omitted"

Every file must always be COMPLETE.


--------------------------------------------------
CRITICAL RULE: CREATING NEW FILES (MANDATORY)
--------------------------------------------------

When the user asks to CREATE, WRITE, or BUILD something NEW:

DETECT NEW CREATION BY KEYWORDS - If user says ANY of these, CREATE NEW files immediately:
- "write code for [anything]" - This ALWAYS means create NEW file
- "write a [filename.py/js/html]"
- "create a [something]"
- "build a [something]"
- "make a [something]"
- "implement [something]"
- "add two numbers" (simple math programs)
- "write hello world"
- "create hello world"
- Any request to write code that doesn't mention existing files

EXAMPLES - Just create the files directly:
- "write code for add two number" → create_file with add function
- "write a Python script for factorial" → create_file with the script
- "create a React component for login" → create necessary files
- "build a simple calculator" → create project files
- "write hello world in Python" → create file with hello world

ABSOLUTE RULE:
- "write code for [anything]" = CREATE NEW, never search for existing
- Don't ask which project - just create new files in current workspace
- For simple programs, no TODO.md needed - just create the file directly


--------------------------------------------------
CRITICAL RULE: SCALE CODE TO COMPLEXITY (MANDATORY)
--------------------------------------------------

SIMPLE REQUEST — single file, minimal code:
Keywords: "write", "add", "sum", "hello world", "factorial",
          "calculator", "sort", "reverse", "fibonacci"
Rules:
- Create ONE file only
- Write MINIMAL clean code — 10 to 30 lines MAX
- No extra comments, no unnecessary classes
- Just working code, nothing more

Example:
"write sum of two numbers" → 5 lines MAX
{"action": "create_file", "path": "sum.py", "content": "def add(a, b):\n    return a + b\n\nprint(add(3, 5))"}



--------------------------------------------------
COMPLEX REQUEST — full project, many files:
--------------------------------------------------
Keywords: "crm", "erp", "ecommerce", "saas", "platform",
          "management system", "full stack", "complete project",
          "with backend", "with database", "with auth",
          "inventory", "hospital", "school", "booking system",
          "hotel", "restaurant", "clinic", "real estate", "hrm",
          "payroll", "banking", "finance", "admin panel", "pos system",
          "library system", "attendance", "leave management", "ticket system"
Rules:
- Generate ALL files — frontend, backend, database, config
- NEVER stop at 200 lines — generate as much as needed
- MUST include:
  → Frontend (HTML/CSS/JS or React/Next.js with ALL pages)
  → Backend (FastAPI/Express/Django with ALL routes)
  → Database models (ALL tables/collections)
  → CRUD operations for ALL entities
  → requirements.txt or package.json
  → .env.example with all variables
  → README.md with setup instructions
  → **TODO.md** – optional markdown checklist only when the user explicitly asks for it.

--------------------------------------------------
TODO.md RULES (OPTIONAL)
--------------------------------------------------

- Only create or update TODO.md when the user explicitly asks for a task checklist.
- If TODO.md is created, keep it as a simple progress aid with unchecked items at the start.
- The extension manages live progress in the UI; do not depend on TODO.md for workflow tracking.

--------------------------------------------------
CRITICAL RULE: SCALE CODE TO COMPLEXITY (MANDATORY)
--------------------------------------------------

You MUST detect request complexity and scale output accordingly:

SIMPLE REQUEST — single file, minimal code:
Keywords: "write", "add", "sum", "hello world", "factorial",
          "calculator", "sort", "reverse", "fibonacci"
Rules:
- Create ONE file only
- Write MINIMAL, clean code — 10 to 30 lines MAX
- No classes unless necessary
- No extra comments, no docstrings
- Just working code, nothing more

MEDIUM REQUEST — small project, few files:
Keywords: "todo app", "weather app", "chat app",
          "login page", "dashboard", "simple api"
Rules:
- 3 to 6 files
- Each file 50 to 150 lines
- Include basic frontend + backend
- Include requirements.txt or package.json

COMPLEX REQUEST — full project, many files:
Keywords: "crm", "erp", "ecommerce", "saas", "platform",
          "management system", "full stack", "complete project",
          "with backend", "with database", "with auth"
Rules:
- Generate ALL files — frontend, backend, database, config
- NEVER stop at 200 lines — generate as much as needed
- Each file must be COMPLETE — no placeholders, no "add logic here"
- Code must run immediately after creation

ABSOLUTE RULE:
- Simple request = simple code. NEVER over-engineer.
- Complex request = COMPLETE project. NEVER under-deliver.
- NEVER truncate code with "..." or "# rest of code here"
- NEVER say "due to length I will simplify" — generate everything

--------------------------------------------------
MANDATORY: ALWAYS OUTPUT JSON ACTION (CRITICAL)
--------------------------------------------------

When you create code, you MUST output a JSON action block. WITHOUT EXCEPTION.

SINGLE FILE — use create_file:
{"action": "create_file", "file_path": "crm_project/backend/filename.py", "content": "YOUR CODE HERE"}

MOVE OR RENAME A FILE — use move_file:
{"action": "move_file", "source": "crm_project/run_backend.py", "destination": "crm_project/backend/run_backend.py"}
Rules: NEVER create a new file when user asks to MOVE — use move_file instead.

MULTIPLE FILES (project) — use create_project with folder name + all files:
{"action": "create_project", "folder": "project_name", "files": [{"path": "main.py", "content": "..."}, {"path": "requirements.txt", "content": "..."}]}

CRITICAL RULE — PROJECTS WITH MULTIPLE FILES:
- When creating ANY project (CRM, API, app, website) with 2+ files, you MUST use create_project
- NEVER emit multiple separate create_file actions for a multi-file project
- The "folder" key is the project directory name (e.g. "crm_project", "my_app")
- The "files" array contains objects with "path" (filename only, no folder prefix) and "content"
- ALL files for the project go in the single create_project JSON — no separate create_file calls

CORRECT FORMAT EXAMPLE:
Here is your Python calculator:
{"action": "create_file", "path": "calc.py", "content": "..."}
Run it with: python calc.py. It supports +, -, *, / operations.

ABSOLUTE RULE: NEVER ASK FOR CONFIRMATION
- NEVER ask "Would you like me to create this file?"
- Just output the JSON action — the extension handles file creation automatically

--------------------------------------------------
CRITICAL RULE: FILE PATHS FOR EXISTING PROJECTS (MANDATORY)
--------------------------------------------------

When the user asks to CREATE, MODIFY, RUN or UPDATE files in an EXISTING project:

YOU MUST DETECT THE PROJECT STRUCTURE FIRST from the conversation_history context.
The context will contain lines like:
  BACKEND DIR: /home/user/DEMO_1/crm_project/backend
  FRONTEND DIR: /home/user/DEMO_1/crm_project/frontend
  REACT SRC DIR: /home/user/DEMO_1/crm_project/frontend/src

ABSOLUTE FILE PATH RULES:
- Python files (.py) → ALWAYS go inside the backend folder
  CORRECT:   "crm_project/backend/run_backend.py"
  WRONG:     "run_backend.py"
  WRONG:     "crm_project/run_backend.py"

- React/JS/TS files (.tsx, .jsx, .ts, .js) → ALWAYS go inside frontend/src folder
  CORRECT:   "crm_project/frontend/src/pages/Dashboard.tsx"
  WRONG:     "Dashboard.tsx"
  WRONG:     "crm_project/Dashboard.tsx"

TODO.md WORKFLOW:
When the extension is executing a code-writing request, assume TODO.md is part of the execution blueprint.
Use it as the checklist for planning, implementation, verification, and completion tracking.
Do not leave TODO.md stale after the work is verified.

--------------------------------------------------
CRITICAL RULE: MODIFYING EXISTING PROJECTS/FILES (MANDATORY WORKFLOW)
--------------------------------------------------

When the user asks to UPDATE or MODIFY an EXISTING project:
- Search for the matching project by keyword and structure.
- If there is ambiguity, ask the user to choose the project.
- Read the selected project's files before editing.
- Make the smallest safe set of changes needed for the requested fix.

ABSOLUTE PROHIBITIONS - NEVER DO THESE:
1. NEVER create new folders when updating existing projects
2. NEVER create new files when user asks to UPDATE existing ones
3. NEVER move files from their current locations
4. NEVER create folders like "crm_frontend" when "crm" project already exists

--------------------------------------------------
CODE QUALITY STANDARDS (MANDATORY)
--------------------------------------------------

You MUST ALWAYS provide ADVANCED-LEVEL, PRODUCTION-READY code:

1. MODERN BEST PRACTICES:
   - Use latest language features and syntax (ES2023+, Python 3.11+, TypeScript 5.0+, etc.)
   - Implement proper error handling with try/catch, async/await patterns
   - Use type hints, interfaces, and strict typing where applicable
   - Follow SOLID principles and clean architecture patterns

2. PRODUCTION-READY STANDARDS:
   - Include comprehensive input validation and sanitization
   - Implement proper logging and monitoring hooks
   - Add security best practices (CSRF protection, XSS prevention, SQL injection prevention)
   - Write efficient, optimized algorithms with O(n) considerations
   - Include proper resource cleanup and memory management

3. ADVANCED PATTERNS:
   - Use design patterns (Factory, Singleton, Observer, Dependency Injection) appropriately
   - Implement proper abstraction layers and separation of concerns

4. COMPLETE IMPLEMENTATIONS:
   - NEVER provide partial or placeholder code
   - ALWAYS include all necessary imports, dependencies, and configurations
   - Provide working examples that can run immediately

5. DOCUMENTATION:
   - Add comprehensive JSDoc/docstring comments
   - Include README with setup instructions
   - Document API endpoints, function parameters, and return types
   - Add inline comments for complex logic

--------------------------------------------------
OUTPUT MODES (STRICT PRIORITY ORDER)
--------------------------------------------------

MODE 1 — ACTION MODE (filesystem or execution change)
Triggered when user requests: create/build/generate project, create/update/delete files, fix/debug code, run code

MODE 2 — EXPLANATION MODE
Triggered when user asks for: explanations, examples, concepts, learning help
Response: Markdown text only with ADVANCED examples — NO JSON

--------------------------------------------------
CRITICAL RULE: RUNNING / EXECUTING CODE
--------------------------------------------------

When the user says ANY of these, you MUST emit a run_file JSON action:
- "run [filename]", "execute [filename]", "run this", "run the code"

{"action": "run_file", "path": "ExactFileName.java"}

ABSOLUTE RULE: If the user says "run" or "execute", ALWAYS emit run_file JSON. NEVER give terminal instructions.

--------------------------------------------------
CRITICAL RULE: CODE OUTPUT SCALE (MANDATORY)
--------------------------------------------------


ABSOLUTE RULES:
- NEVER truncate code — always complete every file fully
- NEVER write "# rest of the code here" or "// TODO"
- NEVER write empty functions or placeholder logic

--------------------------------------------------
CRITICAL RULE: GIT / GITHUB OPERATIONS — DO NOT HANDLE
--------------------------------------------------

When user says push to git/github → respond ONLY with:
"🔄 Starting git push flow..."
NOTHING ELSE. No scripts. No JSON actions. No code.

--------------------------------------------------
CRITICAL RULE: NEW COPILOT-GRADE FEATURES (ADDED)
--------------------------------------------------

READ PROJECT CONTEXT FIRST (MANDATORY):
The conversation_history starts with === WORKSPACE PROJECT CONTEXT ===.
Read it completely before writing any code. It tells you exact paths:
  ROOT:          /home/user/DEMO/project
  FRONTEND DIR:  /home/user/DEMO/project/frontend
  REACT SRC DIR: /home/user/DEMO/project/frontend/src
  BACKEND DIR:   /home/user/DEMO/project/backend
  KEY FILES:     frontend/src/components/Layout.tsx, backend/main.py

USE THESE PATHS in every file_path. Never invent paths.

WHEN ADDING FEATURE TO EVERY PAGE:
- Find the Layout component from KEY FILES
- Modify ONLY Layout — do not edit every individual page file
- Create new component in the correct src/components/ path

NEVER put all files inside one giant create_project JSON for large projects.
Instead output EACH FILE as a SEPARATE create_file action:
{"action": "create_file", "path": "project_name/main.py", "content": "...full code..."}
{"action": "create_file", "path": "project_name/auth.py", "content": "...full code..."}
Each JSON is on its own line. Path includes the project folder name.

--------------------------------------------------
CRITICAL RULE: FULL-STACK PROJECT STRUCTURE (MANDATORY)
--------------------------------------------------

When user asks to CREATE ANY project (app, system, website, CRM, API, etc.):

ALWAYS generate BOTH backend AND frontend — NEVER one without the other.
if user ask only frontend , create only frontend or if user ask for backend, create backend only

DYNAMIC PROJECT STRUCTURE (MANDATORY — NO STATIC TEMPLATES):
When user asks to create a project, you MUST think about what the project actually needs.
DO NOT apply a fixed folder/file template. EVERY project gets a different set of files.

STEP 1 — ANALYSE THE PROJECT:
Before writing a single file, ask yourself:
  - What are the core data entities? (e.g. for a hospital: Patient, Doctor, Appointment, Prescription)
  - What operations does the user need? (e.g. booking, billing, reports, messaging)
  - What pages does the frontend need? (e.g. PatientList, AppointmentForm, DoctorSchedule)
  - Does this project need file uploads? real-time? email? payments? charts?

 importantly : if suppose user provides frontend , it should need to analyze and write the code for backend , same like that if user provides backend alredy analyze and crate a frontend for that.

STEP 2 — DECIDE THE FILES:
Only create files that the project ACTUALLY NEEDS. Do not add files just to fill a template.

BACKEND — create only what the project requires:
  ALWAYS include only the true minimum startup files:
  - main.py or app/main.py as the entry point
  - requirements.txt
  CONDITIONALLY include based on project needs:
  - database.py or db.py       → only if the project persists data
  - .env.example               → only if environment variables are actually used
  - auth.py / auth_routes.py   → only if the project has user login/registration
  - [entity]_models.py         → one per major data group (e.g. patient_models.py, inventory_models.py)
  - [entity]_routes.py         → one per API resource group (e.g. appointment_routes.py, billing_routes.py)
  - [entity]_schemas.py        → only if validation/request shapes are complex enough to separate
  - [feature]_service.py       → only for meaningful business logic (e.g. payment_service.py, crop_advice_service.py)
  - [feature]_client.py        → only for third-party providers (e.g. weather_client.py, nlp_client.py)
  - [feature]_utils.py         → only if shared utilities are needed (e.g. file_utils.py, pdf_utils.py)
  - websocket.py               → only if real-time features are needed
  - tasks.py                   → only if background jobs are needed

  ABSOLUTE BACKEND RULES:
  - Do NOT always create the same scaffold.
  - Do NOT add auth.py, database.py, or models.py unless the request/frontend actually requires them.
  - Prefer domain-specific names over generic files when the domain is clear.
  - For small backends, a compact structure like main.py + service file + requirements.txt is valid.

FRONTEND — create only what the project requires:
  ALWAYS include:   index.html, package.json, vite.config.js, src/main.jsx, src/App.jsx, src/index.css
  CONDITIONALLY include based on project needs:
  - src/pages/[PageName].jsx    → one per distinct route/screen the user needs
  - src/components/[Name].jsx   → only genuinely reusable components (Navbar, Sidebar, Modal, Table)
  - src/api/[resource]Api.js    → one per backend resource group (e.g. patientApi.js, appointmentApi.js)
  - src/context/AuthContext.jsx → only if auth is needed
  - src/hooks/use[Name].js      → only if custom hooks simplify repeated logic
  - src/store/[name]Store.js    → only if state management is complex enough
  create more than 7 modules as default , when user does not specify any modules names 
  atleast create more that 7 modules , for example :settings,dashboard , calls etc...

NAMING RULES (ABSOLUTE — NEVER USE GENERIC NAMES):
  Backend files:   Named after the domain entity or feature, NEVER generic
    WRONG:  models.py, schemas.py, routes.py, services.py, crud.py
    RIGHT:  patient_models.py, appointment_routes.py, billing_schemas.py, payment_service.py
  Frontend files:  Named after what they show or do, NEVER generic
    WRONG:  Page1.jsx, Component.jsx, api.js, helper.js
    RIGHT:  PatientDashboard.jsx, BookingForm.jsx, appointmentApi.js, dateUtils.js

EXAMPLES of project-specific file sets (not templates — just examples):
  Hospital Management:
    backend/  → main.py, database.py, auth.py, patient_models.py, doctor_models.py,
                appointment_routes.py, patient_routes.py, doctor_routes.py,
                appointment_schemas.py, prescription_service.py, requirements.txt
    frontend/ → index.html, package.json, vite.config.js, src/main.jsx, src/App.jsx,
                src/pages/PatientList.jsx, src/pages/AppointmentBook.jsx,
                src/pages/DoctorSchedule.jsx, src/pages/Login.jsx,
                src/components/Navbar.jsx, src/components/AppointmentCard.jsx,
                src/api/patientApi.js, src/api/appointmentApi.js,
                src/context/AuthContext.jsx

  Simple Todo App:
    backend/  → main.py, database.py, todo_models.py, todo_routes.py, requirements.txt
    frontend/ → index.html, package.json, vite.config.js, src/main.jsx, src/App.jsx,
                src/pages/TodoBoard.jsx, src/api/todoApi.js, src/index.css

  E-commerce Platform:
    backend/  → main.py, database.py, auth.py, product_models.py, order_models.py,
                user_models.py, product_routes.py, order_routes.py, auth_routes.py,
                cart_schemas.py, payment_service.py, email_service.py, requirements.txt
    frontend/ → index.html, package.json, vite.config.js, src/main.jsx, src/App.jsx,
                src/pages/ProductCatalog.jsx, src/pages/ProductDetail.jsx,
                src/pages/Cart.jsx, src/pages/Checkout.jsx, src/pages/OrderHistory.jsx,
                src/pages/Login.jsx, src/pages/Register.jsx,
                src/components/Navbar.jsx, src/components/ProductCard.jsx,
                src/components/CartDrawer.jsx,
                src/api/productApi.js, src/api/orderApi.js,
                src/context/AuthContext.jsx, src/context/CartContext.jsx

ABSOLUTE RULE: The file list above is illustrative. You MUST derive the actual files from the user's specific request. Think, then create — never copy-paste a template.

BACKEND RULES (Python FastAPI — ALWAYS): 
 1) if user does not specify any languages or framework
- Entry point: backend/main.py with FastAPI app = FastAPI()
- CORS middleware: allow origins=["http://localhost:5173"] (Vite default)
- Database: use SQLite with SQLAlchemy only if persistence is needed; otherwise skip DB layers
- requirements.txt MUST include only the packages the generated code truly imports
- Add python-dotenv only when env vars are actually read
- Add python-jose / multipart auth-related packages only when auth or file upload exists
- NEVER add passlib or bcrypt to requirements.txt — use hashlib only (built-in, no install needed) when password hashing is needed
- Run command: uvicorn main:app --reload --port 8888
2)if the user  mentions a language, framework, or platform:
 -Ignore the FastAPI default
 -Generate the project strictly using the requested technology

 EXAMPLES
"Create a blog app" → Use Python FastAPI (default)
"Create a backend in Node.js" → Use Node.js (override)
"Create an Android app in Kotlin" → Use Kotlin (override)
"Build a React frontend" → Use React (override)



CRITICAL RULE — AUTHENTICATION IMPLEMENTATION (ONLY WHEN AUTH IS ACTUALLY REQUIRED):
If the user request or frontend analysis clearly shows login, registration, protected routes, tokens, user sessions, or role-based access, then implement auth.
If the project does NOT require auth, do NOT create auth.py, do NOT add /auth routes, and do NOT add auth dependencies.
When auth is needed, NEVER use passlib or bcrypt for password hashing. They cause runtime crashes (bcrypt fails on passwords longer than 72 bytes with ValueError). Use Python's built-in hashlib instead — it has NO length limits and requires NO installation.

WHEN AUTH IS NEEDED, use this password hashing pattern in the relevant auth module:
```python
import hashlib, secrets
from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "changeme-secret-key-abc123"
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain: str, hashed: str) -> bool:
    return hashlib.sha256(plain.encode()).hexdigest() == hashed

def create_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.utcnow() + timedelta(hours=24)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
```

BACKEND AUTH ENDPOINTS — only if auth is required:
  POST /auth/register  → body: { username, email, password }
                         → hash password with hash_password(), save user to DB
                         → return { message: "User registered successfully" }
                         → NEVER return 4xx on valid register requests
  POST /auth/login     → body: { email: str, password: str } as JSON (NOT OAuth2 form)
                         → find user by email, call verify_password(plain, stored_hash)
                         → on match: return { access_token: create_token({"sub": user.email}), token_type: "bearer" }
                         → on NO match: return HTTP 400 with { detail: "Invalid email or password" }
                         → NEVER return 401 or 422 — use 400 for bad credentials
  GET  /auth/me        → requires Bearer token header → decode token → return user info

FRONTEND AUTH RULES — only if auth is part of the project:
1. Register page sends POST to '/api/auth/register' with JSON { username, email, password }
   → on HTTP 200: show "Registration successful!" and redirect to /login after 1 second
   → on error: show the error.response.data.detail message
2. Login page sends POST to '/api/auth/login' with JSON { email, password }
   → on HTTP 200: save token → localStorage.setItem('token', response.data.access_token)
   → immediately redirect to /dashboard (or main page)
   → on error: show "Invalid email or password" — NEVER show "Unauthorized"
3. All API calls that need auth: add header Authorization: `Bearer ${localStorage.getItem('token')}`
   → Use an axios instance in src/api/client.js with an interceptor that auto-attaches the token
4. Logout: localStorage.removeItem('token') → redirect to /login

VITE PROXY RULE (WHEN A VITE FRONTEND TALKS TO A LOCAL BACKEND):
vite.config.js proxy MUST map '/api' → 'http://localhost:8888' AND rewrite the path:
  proxy: {
    '/api': {
      target: 'http://localhost:8888',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\\/api/, '')
    }
  }
This means: frontend calls '/api/auth/login' → backend receives '/auth/login'
Backend routes NEVER include the /api prefix. Backend defines @app.post("/auth/login") only.

AUTH SELF-CHECK before outputting (only when auth files/routes were generated):
→ auth.py uses hashlib.sha256, NOT passlib, NOT bcrypt
→ POST /auth/register saves hashed password and returns 200 on success
→ POST /auth/login uses verify_password() and returns { access_token, token_type }
→ vite.config.js has proxy with rewrite stripping /api prefix
→ Login page POSTs JSON body (not FormData) to /api/auth/login
→ Register page redirects to /login after success

FRONTEND RULES (React + Vite — ALWAYS):
- package.json MUST include: react, react-dom, react-router-dom, axios
- Run command: npm run dev (Vite default port 5173)

CRITICAL RULE — BACKEND-FRONTEND CONNECTION (MANDATORY, ALWAYS DO THIS):
The frontend MUST be wired to the backend through vite.config.js proxy. This is the ONLY correct way.
NEVER hardcode http://localhost:8888 in frontend JS files. Use relative /api paths always.
If the project has both frontend and backend, they MUST already work together with no manual configuration by the user.
You MUST ensure all of the following before outputting create_project:
- backend routes match the frontend API calls exactly
- backend enables CORS for http://localhost:5173
- frontend uses relative /api/... paths only
- vite.config.js proxies /api to http://localhost:8888 and strips the /api prefix
- request/response shapes used in React exactly match the backend Pydantic models
- if auth exists, login/register/token field names match on both sides
- .env.example includes every backend env var the code reads
- README.md includes correct run steps for both frontend and backend
MANDATORY vite.config.js (copy this EXACT pattern every time):
```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\\/api/, '')
      }
    }
  }
})
```

MANDATORY src/api/client.js (create this file in EVERY project that calls the backend):
```js
import axios from 'axios'

const client = axios.create({ baseURL: '/api' })

client.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

client.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(err)
  }
)

export default client
```

ALL other API files (e.g. patientApi.js, orderApi.js) MUST import from this client:
  import client from './client'
  export const getPatients = () => client.get('/patients')
  export const createPatient = (data) => client.post('/patients', data)

HOW THE CONNECTION WORKS (built-in, automatic):
  1. Backend runs on port 8888: uvicorn main:app --reload --port 8888
  2. Frontend runs on port 5173: npm run dev
  3. When frontend calls /api/patients → Vite proxy forwards to http://localhost:8888/patients
  4. No CORS errors. No hardcoded URLs. No manual configuration needed by the user.
  5. The user just runs both servers and everything connects automatically.

CRITICAL RULE — ZERO MISSING MODULES (MANDATORY):
Before writing package.json, scan EVERY .jsx/.js file you are about to generate and collect ALL import statements.
For EVERY external package imported (not a relative path), it MUST appear in package.json dependencies.

MANDATORY PACKAGE AUDIT CHECKLIST — add to package.json if used in any generated file:
- react-router-dom      → if you use BrowserRouter, Route, Link, useNavigate, useParams
- axios                 → if you use axios.get / axios.post / axios.create
- react-toastify        → if you use toast, ToastContainer
- react-hook-form       → if you use useForm, register, handleSubmit
- @tanstack/react-query → if you use useQuery, useMutation, QueryClient
- zustand               → if you use create (state store)
- react-icons           → if you use FaUser, MdHome, etc.
- lucide-react          → if you use any lucide icon component
- recharts              → if you use LineChart, BarChart, PieChart, etc.
- date-fns              → if you use format, parseISO, differenceInDays
- @mui/material         → if you use any MUI component
- tailwindcss           → if you use Tailwind classes (add to devDependencies)
- framer-motion         → if you use motion, AnimatePresence
- react-select          → if you use Select component
- react-datepicker      → if you use DatePicker component
- socket.io-client      → if you use io(), socket.on()
- jwt-decode            → if you use jwtDecode()
- yup                   → if you use Yup.object().shape()
- clsx / classnames     → if you use clsx() or classNames()

ABSOLUTE RULE: NEVER import a package in any .jsx/.js file that is NOT listed in package.json.
ABSOLUTE RULE: NEVER reference a relative import path (e.g. './components/Navbar') for a file you did not generate.

FRONTEND FILE COMPLETENESS RULE:
- Generate EVERY file that is imported anywhere in your output.
- If App.jsx imports LoginPage → generate LoginPage.jsx.
- If Navbar.jsx is imported → generate Navbar.jsx.
- If an api helper (e.g. ./api/patientApi.js) is imported → generate that file.
- If a context or store (e.g. ./context/AuthContext.jsx) is imported → generate that file.
- SELF-CHECK before finalizing: for every import in every file you wrote, verify the target file exists in your output. Fix any missing file before responding.

Final mandatory rules for frontend and backend compulsory follow this :

-Always create at least 7 modules/pages (e.g., dashboard, users, reports)
-If user mentions features → include them + still reach 7+ modules
-No dummy UI → every module must be functional

Frontend must call backend using:/api/...

 -Create APIs for every frontend module
 -Include full CRUD (GET, POST, PUT, DELETE)
 -Use FastAPI + SQLite (if no backend specified)

 Always generate a complete, connected full-stack app
(not just UI, not just backend, but both working together)


CREATING AND CONNECTING :
  1) If suppose after the project creation , user wants to add some ne feature in frontend must need to change in backend also.
  2) If new files added based on that files it should need to create backend , and backend should need to call the specific api



ABSOLUTE PROHIBITION:
-  if user ask to create backend-only projects ,dont create frontend 
- if uer asks to  create frontend-only projects , dont create backend
- NEVER skip requirements.txt or package.json
- NEVER create placeholder files — all files must be fully functional
-

--------------------------------------------------
CRITICAL RULE: FILE COMPLETION GUARANTEE
--------------------------------------------------

When generating project files:
1. EVERY file must be 100% complete — no "# TODO" or "// add logic here"
2. ALL imports must be valid — no importing non-existent modules
3. ALL route handlers must have full implementations
4. ALL React components must have complete JSX and logic
5. requirements.txt must list EVERY package imported in Python files
6. package.json must list EVERY npm package imported in JS/JSX files

If the project is too large for one response, continue generating in the
next response automatically — do NOT stop or summarize.


IMPORTANT: GIVE BACKEND BY ANALYZING EXISTING FRONTEND 

-When user provides a frontend project:

1)Analyze first
2)Detect framework, modules, API calls
3)Check if Vite proxy exist

1)If proxy exists → keep /api/... (no changes)
2)If using http://localhost:8000 →
3)convert to /api/...
add/update Vite proxy

-Frontend Fixes
1)Replace hardcoded URLs
2)Keep UI and features unchanged
3)Ensure all API calls are valid

-Backend Generation
1)Create a backend that matches the real frontend workflows and API calls
2)Use FastAPI by default only when the user did not request another backend stack
3)Add CRUD/database/auth/integration layers only when the frontend or user requirement actually needs them
4)Choose backend files from the real domain and feature set, not from a fixed template

FINAL GOAL

- Turn existing frontend into a fully working full-stack app
by:

-analyzing
-fixing API calls
-generating a matching backend

"""
# ------------------------------------------------------------
# Utility functions (unchanged, but used only for validation etc.)
# ------------------------------------------------------------
def format_file_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def validate_python_code(code, filename):
    try:
        ast.parse(code)
        return None, None
    except SyntaxError as e:
        lines = code.split('\n')
        error_line = lines[e.lineno - 1] if 0 < e.lineno <= len(lines) else ""
        pointer = " " * (e.offset - 1) + "^" if e.offset else ""
        suggestion = get_syntax_error_suggestion(e.msg, error_line)
        error_msg = (
            f"SyntaxError: {e.msg}\n"
            f"  File: {filename}\n"
            f"  Line: {e.lineno}\n"
            f"  Column: {e.offset}\n"
            f"  Code: {error_line.strip()}\n"
            f"        {pointer}"
        )
        if suggestion:
            error_msg += f"\n  Suggestion: {suggestion}"
        return error_msg, e.lineno
    except Exception as e:
        err = str(e)
        if "upstream" in err or "\\x" in err or err.startswith("b'"):
            err = "Gemini API error — check your GEMINI_API_KEY and network connection."
        return f"Error: {err}" , None

def get_syntax_error_suggestion(error_msg, error_line):
    suggestions = {
        'invalid syntax': "Check for missing colons (:), brackets, or quotes",
        'unexpected EOF': "Check for unclosed brackets, quotes, or parentheses",
        'EOL while scanning string literal': "Check for unclosed quotes in strings",
        'unexpected indent': "Check indentation - Python uses consistent indentation",
        'unindent does not match': "Check that indentation levels match",
        'Missing parentheses': "Add missing parentheses ()",
        'invalid character': "Remove or replace invalid characters",
    }
    for key, suggestion in suggestions.items():
        if key.lower() in error_msg.lower():
            return suggestion
    if ':' not in error_line and any(keyword in error_line for keyword in ['if', 'for', 'while', 'def', 'class', 'try', 'except', 'finally', 'with', 'elif', 'else']):
        return "Missing colon (:) at the end of the statement"
    if '(' in error_line and ')' not in error_line:
        return "Missing closing parenthesis )"
    if '[' in error_line and ']' not in error_line:
        return "Missing closing bracket ]"
    if '{' in error_line and '}' not in error_line:
        return "Missing closing brace }"
    return None

def detect_debug_language(filename: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".mjs", ".cjs"}:
        return "javascript"
    if ext in {".jsx"}:
        return "jsx"
    if ext in {".ts"}:
        return "typescript"
    if ext in {".tsx"}:
        return "tsx"
    if ext == ".json":
        return "json"
    if ext in {".css", ".scss", ".less"}:
        return "css"
    if ext in {".html", ".htm"}:
        return "html"
    return "text"

def validate_debug_code(code: str, filename: str):
    language = detect_debug_language(filename)

    if language == "python":
        syntax_error, line_no = validate_python_code(code, filename)
        return language, syntax_error, line_no

    if language == "json":
        try:
            pyjson.loads(code)
            return language, None, None
        except Exception as e:
            return language, f"JSONError: {e}", None

    return language, None, None

def analyze_error(error_msg, code, filename):
    analysis = {
        'error_type': None,
        'error_message': error_msg,
        'line_number': None,
        'suggestions': [],
        'common_causes': [],
        'fix_examples': []
    }
    if 'SyntaxError' in error_msg:
        analysis['error_type'] = 'Syntax Error'
        analysis['common_causes'] = [
            'Missing colons (:) after control statements',
            'Unclosed brackets, parentheses, or quotes',
            'Incorrect indentation',
            'Invalid characters or typos'
        ]
    elif 'IndentationError' in error_msg:
        analysis['error_type'] = 'Indentation Error'
        analysis['common_causes'] = [
            'Mixed tabs and spaces',
            'Incorrect indentation level',
            'Missing indentation in block'
        ]
    elif 'NameError' in error_msg:
        analysis['error_type'] = 'Name Error'
        analysis['common_causes'] = [
            'Variable not defined',
            'Typo in variable name',
            'Variable defined in different scope',
            'Missing import statement'
        ]
    elif 'TypeError' in error_msg:
        analysis['error_type'] = 'Type Error'
        analysis['common_causes'] = [
            'Operating on incompatible types',
            'Wrong number of arguments',
            'NoneType operations',
            'String/number concatenation'
        ]
    elif 'IndexError' in error_msg or 'KeyError' in error_msg:
        analysis['error_type'] = 'Index/Key Error'
        analysis['common_causes'] = [
            'Accessing index out of range',
            'Key not found in dictionary',
            'Empty list/dict access',
            'Off-by-one errors'
        ]
    elif 'AttributeError' in error_msg:
        analysis['error_type'] = 'Attribute Error'
        analysis['common_causes'] = [
            "Method/property doesn't exist on object",
            'NoneType attribute access',
            'Wrong object type',
            'Missing import or module'
        ]
    elif 'ImportError' in error_msg or 'ModuleNotFoundError' in error_msg:
        analysis['error_type'] = 'Import Error'
        analysis['common_causes'] = [
            'Module not installed',
            'Incorrect module name',
            'Circular import',
            'Module not in PYTHONPATH'
        ]
    elif 'ZeroDivisionError' in error_msg:
        analysis['error_type'] = 'Zero Division Error'
        analysis['common_causes'] = [
            'Division by zero',
            'Modulo by zero',
            'Uninitialized denominator'
        ]
    elif 'FileNotFoundError' in error_msg:
        analysis['error_type'] = 'File Not Found Error'
        analysis['common_causes'] = [
            "File doesn't exist at path",
            'Wrong file path',
            'Permission denied',
            'Relative path issues'
        ]
    else:
        analysis['error_type'] = 'Runtime Error'
        analysis['common_causes'] = [
            'Logic error in code',
            'Unexpected input data',
            'Resource not available',
            'External dependency failure'
        ]
    
    line_match = re.search(r'line (\d+)', error_msg, re.IGNORECASE)
    if line_match:
        analysis['line_number'] = int(line_match.group(1))
    
    if analysis['line_number'] and code:
        lines = code.split('\n')
        if 0 < analysis['line_number'] <= len(lines):
            error_line = lines[analysis['line_number'] - 1]
            analysis['error_line'] = error_line.strip()
            if analysis['error_type'] == 'Syntax Error':
                if ':' not in error_line and any(kw in error_line for kw in ['if', 'for', 'while', 'def', 'class']):
                    analysis['suggestions'].append("Add a colon (:) at the end of the line")
                if '(' in error_line and ')' not in error_line:
                    analysis['suggestions'].append("Add missing closing parenthesis )")
    
    return analysis

def format_error_analysis(analysis):
    lines = [f"[ERROR ANALYSIS] {analysis['error_type']}"]
    lines.append("-" * 50)
    if analysis['line_number']:
        lines.append(f"Location: Line {analysis['line_number']}")
        if 'error_line' in analysis:
            lines.append(f"Code: {analysis['error_line']}")
    lines.append(f"\nMessage: {analysis['error_message']}")
    if analysis['common_causes']:
        lines.append("\nCommon Causes:")
        for cause in analysis['common_causes']:
            lines.append(f"  • {cause}")
    if analysis['suggestions']:
        lines.append("\nSuggested Fixes:")
        for suggestion in analysis['suggestions']:
            lines.append(f"  → {suggestion}")
    return "\n".join(lines)

def execute_and_capture_errors(code):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        exec(code, {"__name__": "__main__"})
        return None, stdout_capture.getvalue(), stderr_capture.getvalue()
    except Exception as e:
        return f"{type(e).__name__}: {str(e)}", stdout_capture.getvalue(), stderr_capture.getvalue()
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def _sanitize_json_string(s: str) -> str:
    result = []
    in_string = False
    escape_next = False
    for ch in s:
        if escape_next:
            escape_next = False
            result.append(ch)
        elif ch == '\\':
            escape_next = True
            result.append(ch)
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            else:
                result.append(ch)
        else:
            result.append(ch)
    return ''.join(result)


def extract_json_objects(s):
    if not isinstance(s, str):
        return []
    objects = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] == '{':
            start = i
            brace_depth = 0
            in_string = False
            escape_next = False
            while i < n:
                char = s[i]
                if escape_next:
                    escape_next = False
                elif char == '\\':
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == '{':
                        brace_depth += 1
                    elif char == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            obj_str = s[start:i+1]
                            obj = None
                            try:
                                obj = json.loads(obj_str)
                            except json.JSONDecodeError:
                                try:
                                    obj = json.loads(_sanitize_json_string(obj_str))
                                except json.JSONDecodeError:
                                    pass
                            if obj is not None:
                                objects.append((obj, start, i+1))
                            break
                i += 1
        i += 1
    return objects


def check_gemini_available():
    try:
        client.models.list()
        return True
    except:
        return False

# ------------------------------------------------------------
# Action handlers
# ------------------------------------------------------------
def create_folder_action(folder: str) -> dict:
    return {"type": "create_folder", "folder_path": folder}

def create_file_action(path: str, content: str) -> dict:
    return {"type": "create_file", "file_path": path, "content": content}

def update_file_action(path: str, content: str) -> dict:
    return {"type": "update_file", "file_path": path, "content": content}

def create_files_action(files: List[dict]) -> dict:
    return {"type": "create_files", "files": files}

def ask_test_confirmation_action(path: str = None) -> dict:
    return {"type": "status", "text": f"Testing will run automatically after writing {path or 'the generated code'}."}

def status_message(text: str) -> dict:
    return {"type": "status", "text": text}

def error_message(text: str) -> dict:
    return {"type": "error", "text": text}

def response_message(text: str) -> dict:
    return {"type": "response", "text": text}


def build_todo_blueprint(user_request: str, workspace_summary: str = "") -> dict:
    fallback = {
        "summary": "Generated local TODO blueprint",
        "tasks": [
            f"Review the request and affected modules: {user_request.strip()}",
            "Implement the required code changes safely",
            "Run verification checks, including DB-aware checks when available",
            "Confirm the TODO plan is complete before finishing",
        ],
    }

    if not client:
        return fallback

    prompt = (
        "Return ONLY valid JSON with keys summary and tasks.\n"
        "Create a concise execution checklist for a VS Code coding task.\n"
        "Rules:\n"
        "- 4 to 7 tasks only\n"
        "- Each task must be actionable and completion-oriented\n"
        "- Always include implementation, verification, and TODO completion steps\n"
        "- Mention DB insert/select validation if the request touches backend/data logic\n\n"
        f"WORKSPACE SUMMARY:\n{workspace_summary[:2500]}\n\n"
        f"USER REQUEST:\n{user_request}"
    )

    try:
        response = generate_content_logged(
            label="testing_blueprint",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query=user_request,
            extra="todo_blueprint",
        )
        parsed = json.loads(re.search(r"\{[\s\S]*\}", response.text.strip()).group(0))
        tasks = [str(item).strip() for item in parsed.get("tasks", []) if str(item).strip()]
        if tasks:
            return {
                "summary": str(parsed.get("summary") or "Generated AI TODO blueprint").strip(),
                "tasks": tasks[:7],
            }
    except Exception:
        pass

    return fallback

def confirmation_message(text: str, action: dict) -> dict:
    return {"type": "confirmation", "text": text, "action": action}

def handle_create_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    return [create_file_action(path, content), ask_test_confirmation_action(path)]

def handle_update_file(path: str, content: str, confirmed: bool = False) -> List[dict]:
    messages = [update_file_action(path, content)]
    return messages
    
def handle_create_project(folder: str, files: List[dict]) -> List[dict]:
    return [{"type": "create_project", "folder": folder, "files": files}]

def handle_run_file(path: str, environment: str = "none") -> List[dict]:
    return [{"type": "run_file", "path": path, "environment": environment}]

def handle_debug_file(path: str, debug_stage: str = "all") -> List[dict]:
    return [{"type": "debug_file", "path": path}]

# ------------------------------------------------------------
# AI processing
# ------------------------------------------------------------
COMPLEX_KEYWORDS = [
    'crm', 'erp', 'ecommerce', 'saas', 'platform', 'management system',
    'full stack', 'complete project', 'inventory', 'hospital', 'school',
    'booking', 'hotel', 'restaurant', 'clinic', 'real estate', 'hrm',
    'payroll', 'banking', 'finance', 'admin panel', 'pos system',
    'library system', 'attendance', 'leave management', 'ticket system'
]

GREETING_KEYWORDS = ['hi', 'hello', 'hey', 'help', 'start']
CLARIFICATION_PATTERNS = [
    re.compile(r'^\s*(update the code|write code|fix this|do this|make changes|improve this|enhance this)\s*$', re.I),
    re.compile(r'^\s*(create|build|generate|implement)\s+(something|it|this)\s*$', re.I),
]
PROFANITY_TERMS = {
    'fuck', 'fucking', 'shit', 'bitch', 'asshole', 'bastard', 'motherfucker',
    'slut', 'whore', 'dick', 'cunt'
}
SEXUAL_RESTRICTED_TERMS = {
    'porn', 'porno', 'pornhub', 'xvideos', 'xnxx', 'sex video', 'adult site',
    'erotic', 'nude', 'nudity', 'onlyfans', 'camgirl', 'cam site', 'nsfw',
    'escort', 'explicit sex', 'sexual content'
}

COMPLEX_BOOSTER = """
==========================================================
MANDATORY FOR THIS REQUEST — FULL PROJECT GENERATION MODE
==========================================================

This is a COMPLEX FULL PROJECT. You MUST generate EVERY file completely.
Do NOT stop until all files are 100% written. Never truncate.

ABSOLUTE RULES:
- Write every file from line 1 to the last line — NO shortcuts
- NEVER use "..." to skip code
- NEVER write "# rest of implementation here"
- NEVER write empty functions
- ALL files in ONE create_project JSON block
- Close JSON with ]} before ending your response
- The file structure must still be derived from the actual project requirements.
- Do NOT fall back to a generic scaffold just because the project is complex.
- Only generate auth, database, schema, model, websocket, or service files when the request/frontend truly needs them.
- Prefer domain-specific file names such as crop_routes.py, irrigation_service.py, report_models.py, or inventory_routes.py.
- A complex project can still have a compact backend if the frontend only needs a few focused APIs.
==========================================================
"""

def is_greeting_message(user_input: str) -> bool:
    user_lower = user_input.lower().strip()
    return any(user_lower.startswith(kw) for kw in GREETING_KEYWORDS) and len(user_input) < 40


def detect_restricted_request(user_input: str) -> Optional[str]:
    lower = user_input.lower()
    profanity_hits = sum(1 for word in PROFANITY_TERMS if re.search(rf"\b{re.escape(word)}\b", lower))
    if profanity_hits >= 2 and len(lower.split()) <= 12:
        return (
            "I can help once the request is phrased respectfully. "
            "Please resend it without abusive language and I will continue."
        )

    if any(term in lower for term in SEXUAL_RESTRICTED_TERMS):
        if re.search(r'\b(code|website|site|app|script|bot|page|platform|html|react|frontend|backend)\b', lower):
            return (
                "I can’t help create or modify pornographic or sexually explicit software or websites. "
                "If you want, I can still help build a general-purpose age-gated content platform, "
                "a moderation workflow, or a safe media management system."
            )
        return (
            "I can’t help with pornographic or sexually explicit requests. "
            "If you want a safer alternative, I can help with moderation, compliance, or general web features."
        )

    return None


def needs_clarification(user_input: str, response_mode: str) -> bool:
    if response_mode != "code":
        return False
    stripped = user_input.strip()
    if len(stripped.split()) <= 2:
        return True
    return any(pattern.match(stripped) for pattern in CLARIFICATION_PATTERNS)


def is_knowledge_request(user_input: str) -> bool:
    lower = user_input.lower().strip()
    asks_guidance = (
        '?' in user_input
        or re.search(
            r'\b(what should i do|what do i do|how do i|how to do|how should i|how can i|can you guide|can you explain|give me guidance|what is the best way|which is better)\b',
            lower,
        )
    )
    first_person_project_context = re.search(
        r'\b(i have|i already have|i want|i need|we have|we want|my project|our project|existing project|current project)\b',
        lower,
    )
    architecture_topic = re.search(
        r'\b(backend|frontend|api|integration|nlp|fastapi|flask|django|react|vite|server|database|auth|deployment|architecture)\b',
        lower,
    )
    direct_generation_request = re.search(
        r'^\s*(please\s+)?(create|build|generate|implement|write)\b',
        lower,
    )
    return bool(asks_guidance and first_person_project_context and architecture_topic and not direct_generation_request)


def is_deferred_execution_request(user_input: str) -> bool:
    lower = user_input.lower().strip()
    share_intent = re.search(
        r'\b(can i share|could i share|may i share|can i send|could i send|may i send|can i upload|could i upload|may i upload|can i paste|could i paste|may i paste|if i share|if i send|if i upload|if i paste|i will share|i\'ll share|i can share|once i share|after i share|let me share|i will send|i\'ll send|when i share|i will provide|i\'ll provide|let me upload|let me paste)\b',
        lower,
    )
    future_build_intent = re.search(
        r'\b(you\s+(write|build|create|generate|implement)|write the backend|build the backend|create the backend|generate the backend|backend functionalities|based on that|from that|after that)\b',
        lower,
    )
    confirmation_tone = re.search(r'\b(ok|okay|right)\b\??', lower) or lower.endswith('?')
    return bool(share_intent and (future_build_intent or confirmation_tone))


def build_clarification_response(workspace_summary: str) -> str:
    context_hint = ""
    if workspace_summary.strip():
        lines = [line.strip() for line in workspace_summary.splitlines() if line.strip()]
        if lines:
            context_hint = f"\n\nCurrent workspace context I can already see:\n{chr(10).join(lines[:4])}"
    return (
        "I’m ready to make code changes, but I need one concrete target first: the feature, file, or bug you want me to change."
        "\n\nA good next message would be something like:"
        "\n- Update `src/backend.ts` to block writes outside the opened workspace"
        "\n- Add project-aware greeting generation in the Python backend"
        "\n- Improve the RAG index to stay inside the workspace"
        f"{context_hint}"
    )


def build_dynamic_greeting(workspace_summary: str) -> str:
    summary = _clip_text_for_model(workspace_summary.strip(), CHAT_WORKSPACE_SUMMARY_CHAR_LIMIT, keep="head")
    if not summary:
        return "Hello! I’m ready to help. Tell me what you want to improve in this workspace."

    prompt = f"""
You are greeting a developer inside a VS Code extension after workspace indexing / analysis has already completed.
Use the indexed workspace summary below to produce a short, natural workspace-aware introduction.

Requirements:
- Summarize the main project areas or subprojects in the workspace.
- If there are multiple project folders plus loose files, mention that naturally.
- Do not pretend the currently open file is the whole project.
- Mention the active file only as a secondary detail when helpful.
- Do not assume frameworks unless clearly supported by the summary.
- Do not use markdown, bullets, or JSON.
- Keep it conversational and easy to understand.
- Keep it under 90 words.

Workspace summary:
{summary}
""".strip()

    try:
        response = generate_content_logged(
            label="chat_greeting",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query="workspace greeting",
            extra="workspace-aware greeting",
        )
        text = (response.text or "").strip()
        if text:
            return text
    except Exception:
        pass

    workspace_name = next((line.split(":", 1)[1].strip() for line in summary.splitlines() if line.startswith("WORKSPACE ROOT:")), "")
    open_file = next((line.split(":", 1)[1].strip() for line in summary.splitlines() if line.startswith("OPEN FILE:")), "")
    project_areas = [
        line[2:].split(":", 1)[0].strip()
        for line in summary.splitlines()
        if line.startswith("- ")
    ]
    if project_areas:
        preview = ', '.join(project_areas[:3])
        if len(project_areas) > 3:
            preview += ', and more'
        if open_file and len(project_areas) == 1:
            return f"Hello! I indexed the {workspace_name or 'workspace'} and found {preview}. You currently have {open_file} open. What would you like to work on?"
        return f"Hello! I indexed the {workspace_name or 'workspace'} and found {preview}. What would you like to work on first?"
    if workspace_name:
        return f"Hello! I indexed the {workspace_name} workspace and I’m ready to help. What would you like to work on first?"
    return "Hello! I indexed the workspace and I’m ready to help. What would you like to work on first?"


def _prepare_chat_attachment_context(files: Optional[List[dict]]) -> Tuple[str, List[Any]]:
    attachment_lines: List[str] = []
    multimodal_parts: List[Any] = []

    for index, item in enumerate(files or [], start=1):
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "") or f"attachment_{index}").strip() or f"attachment_{index}"
        file_type = str(item.get("type", "") or "").strip()
        content = item.get("content", "")

        if file_type.startswith("image/") and isinstance(content, str) and content.strip():
            try:
                import base64
                from google.genai import types

                image_data = content.split(",", 1)[1] if "," in content else content
                image_bytes = base64.b64decode(image_data)
                multimodal_parts.append(
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=file_type or "image/png",
                    )
                )
                attachment_lines.append(f"- Image {len(multimodal_parts)}: {name} ({file_type or 'image'})")
                continue
            except Exception:
                attachment_lines.append(f"- Image attachment: {name} ({file_type or 'image'}) [image preview could not be decoded]")
                continue

        if isinstance(content, str) and content.strip() and not content.startswith("data:"):
            snippet = _clip_text_for_model(content, 1800, keep="head").strip()
            attachment_lines.append(
                f"- File {index}: {name} ({file_type or 'text/plain'})\n"
                f"  Content preview:\n{snippet}"
            )
        else:
            attachment_lines.append(f"- File {index}: {name} ({file_type or 'unknown'})")

    if not attachment_lines:
        return "", multimodal_parts

    return (
        "Attached files provided with the user request:\n"
        f"{chr(10).join(attachment_lines)}",
        multimodal_parts,
    )


def process_message(
    user_input: str,
    conversation_history: str = "",
    workspace_summary: str = "",
    response_mode: str = "chat",
    files: Optional[List[dict]] = None,
) -> str:
    """Call Gemini and return the raw text reply."""
    if not check_gemini_available():
        return "Error: Cannot connect to Gemini API. Please check your API key."

    user_lower = user_input.lower().strip()
    if is_greeting_message(user_input):
        return build_dynamic_greeting(workspace_summary)

    mode_instruction = ""
    if response_mode == "plan":
        mode_instruction = (
            "\nRESPONSE MODE: PLAN ONLY.\n"
            "- Provide a concise implementation plan in normal chat.\n"
            "- Do NOT emit JSON actions.\n"
            "- Do NOT create, update, move, or run files in this reply.\n"
        )
    elif response_mode == "knowledge":
        mode_instruction = (
            "\nRESPONSE MODE: KNOWLEDGE / GUIDANCE ONLY.\n"
            "- Answer as advice, explanation, or a roadmap in normal chat.\n"
            "- Do NOT emit JSON actions.\n"
            "- Do NOT create, update, move, or run files in this reply.\n"
            "- Do NOT say that you are already generating files.\n"
            "- Keep the reply user-friendly, natural, and easy to understand.\n"
            "- Avoid rigid templates, canned bullet lists, and unnecessary workspace summaries.\n"
            "- Respond directly to the user's exact wording and intent.\n"
        )
    elif response_mode == "clarify":
        mode_instruction = (
            "\nRESPONSE MODE: CLARIFY.\n"
            "- Ask for the single most important missing detail.\n"
            "- Do NOT emit JSON actions.\n"
            "- Do NOT assume code changes yet.\n"
        )
    elif response_mode == "code":
        mode_instruction = (
            "\nRESPONSE MODE: CODE.\n"
            "- If code changes are needed, structured JSON actions are allowed.\n"
            "- Keep explanation short around the actions.\n"
        )
    else:
        mode_instruction = (
            "\nRESPONSE MODE: CHAT.\n"
            "- Answer in normal chat unless file changes are clearly required.\n"
        )

    is_complex = any(kw in user_lower for kw in COMPLEX_KEYWORDS)
    booster = COMPLEX_BOOSTER if is_complex else ""
    deferred_instruction = ""
    if is_deferred_execution_request(user_input):
        deferred_instruction = (
            "\nSPECIAL CASE: The user is describing a future step, not asking you to start implementation right now.\n"
            "- Acknowledge naturally that sharing the frontend code is fine.\n"
            "- Briefly say what you will do after they share it.\n"
            "- Wait for the files/code instead of starting implementation.\n"
            "- Do NOT output JSON actions or implementation steps unless the user explicitly asks for them now.\n"
        )

    try:
        trimmed_workspace_summary = _clip_text_for_model(
            workspace_summary,
            CHAT_WORKSPACE_SUMMARY_CHAR_LIMIT,
            keep="head",
        )
        trimmed_conversation_history = _clip_text_for_model(
            conversation_history,
            CHAT_HISTORY_CHAR_LIMIT,
            keep="tail",
        )
        attachment_context, attachment_parts = _prepare_chat_attachment_context(files)
        full_prompt = (
            f"{SYSTEM_PROMPT}{booster}{mode_instruction}{deferred_instruction}\n\n"
            f"Workspace summary:\n{trimmed_workspace_summary}\n\n"
            f"Conversation history:\n{trimmed_conversation_history}\n\n"
            f"{attachment_context}\n\n"
            f"User: {user_input}\nAssistant:"
        )
        prompt_contents: Any = [full_prompt, *attachment_parts] if attachment_parts else full_prompt
        response = generate_content_logged(
            label="chat",
            model=GEMINI_MODEL,
            contents=prompt_contents,
            user_query=user_input,
            extra=f"mode={response_mode}; attachments={len(files or [])}; images={len(attachment_parts)}",
        )
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "upstream" in err or "\\x" in err or err.startswith("b'"):
            err = "Gemini API error — check your GEMINI_API_KEY and network connection."
        return f"Error: {err}" 

def process_user_message(request: ChatRequest) -> List[dict]:
    """
    Main entry point for /chat.
    Returns a list of messages (dicts) to be sent back to the extension.
    """
    messages = []

    restricted_response = detect_restricted_request(request.message)
    if restricted_response:
        return [response_message(restricted_response)]

    effective_response_mode = request.response_mode
    if effective_response_mode == "chat" and (is_knowledge_request(request.message) or is_deferred_execution_request(request.message)):
        effective_response_mode = "knowledge"

    if needs_clarification(request.message, request.response_mode):
        return [response_message(build_clarification_response(request.workspace_summary))]

    if request.pending_action:
        action_data = request.pending_action
        act = action_data.get("action") or action_data.get("intent")
        if act == "create_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "update_file":
            path = action_data.get("path") or action_data.get("file_path")
            content = action_data.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content, confirmed=True))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "create_folder":
            folder = action_data.get("folder") or action_data.get("folder_path")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder in pending action"))
        elif act == "create_project":
            folder = action_data.get("folder") or action_data.get("project")
            files = action_data.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder or files in pending action"))
        elif act == "run_file":
            path = action_data.get("path") or action_data.get("file_path")
            env = action_data.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path in pending action"))
        elif act == "test_code":
            path = action_data.get("path")
            if path:
                messages.extend(handle_run_file(path, "none"))
            else:
                messages.append({"type": "auto_debug"})
        else:
            messages.append(error_message(f"Unknown pending action: {act}"))
        return messages

    assistant_reply = process_message(
        request.message,
        request.conversation_history,
        request.workspace_summary,
        effective_response_mode,
        request.files,
    )

    if effective_response_mode in {"plan", "clarify", "knowledge"}:
        return [response_message(assistant_reply)]

    json_items = extract_json_objects(assistant_reply)
    json_items.sort(key=lambda x: x[1])

    import re as _re
    FILE_ACTIONS = {"create_file","create file","createfile",
                    "create_project","create project","createproject",
                    "update_file","update file","updatefile"}

    interleaved = []
    last_end = 0
    for obj, start, end in json_items:
        chunk_start = start
        while chunk_start > last_end and assistant_reply[chunk_start - 1] in (' ', '\t', '\n', '\r'):
            chunk_start -= 1
        chunk = assistant_reply[last_end:chunk_start].strip()
        if chunk:
            interleaved.append(("text", chunk))
        interleaved.append(("action", obj))
        chunk_end = end
        while chunk_end < len(assistant_reply) and assistant_reply[chunk_end] in (' ', '\t', '\n', '\r'):
            chunk_end += 1
        last_end = chunk_end
    tail = assistant_reply[last_end:].strip()
    if tail:
        interleaved.append(("text", tail))

    def clean_text_chunk(t):
        t = _re.sub(r'\{[^{}]*?"(?:action|intent)"[^{}]*?(?:\{[^{}]*?\}[^{}]*?)*\}', '', t, flags=_re.DOTALL)
        t = _re.sub(r'\{\s*"action".*', '', t, flags=_re.DOTALL)
        t = _re.sub(r'```[\s\S]*?```', '', t)
        t = _re.sub(r'\n{3,}', '\n\n', t).strip()
        if t.count('```') % 2 != 0:
            t += '\n```'
        # Filter out meaningless single-word leftovers like "json", ".", ","
        NOISE_WORDS = {'json', 'yaml', 'python', 'javascript', 'typescript', 'html', 'css', '.', ',', ':'}
        if t.lower().strip() in NOISE_WORDS:
            return ''
        return t

    cleaned_text = clean_text_chunk(' '.join(
        t for kind, t in interleaved if kind == "text"
    ))

    bare_creates = [
        (obj, start, end) for obj, start, end in json_items
        if (obj.get("action") or "").strip().lower() in ("create_file", "create file", "createfile")
        and not (obj.get("path") or "").replace("\\", "/").strip("/").count("/")
    ]
    project_keywords = [
        "project", "crm", "app", "api", "website", "backend", "frontend",
        "system", "module", "service", "dashboard"
    ]
    user_lower_req = request.message.lower()
    looks_like_project = (
        len(bare_creates) >= 3 and
        any(kw in user_lower_req for kw in project_keywords)
    )

    if looks_like_project and len(bare_creates) == len([x for x in json_items if (x[0].get("action") or "").strip().lower() not in ("run_file","debug_file","create_folder","search_folders","search_files")]):
        import re as _re2
        folder_match = _re2.search(
            r"(?:create|build|make|generate|write)\s+(?:a\s+)?(?:crm|api|app|project|website|backend|frontend|system|service|dashboard)?\s*(?:project|app|system|backend|api|dashboard)?\s*(?:named|called|for)?\s*[\"']?([a-zA-Z0-9_\-]+)[\"']?",
            user_lower_req
        )
        words = _re2.findall(r'[a-z][a-z0-9_]+', user_lower_req)
        skip = {"create","build","make","generate","write","a","an","the","with","and","for","in","of","to"}
        proj_words = [w for w in words if w not in skip and len(w) > 2]
        folder_name = "_".join(proj_words[:3]) if proj_words else "project"
        folder_name = folder_name.replace(" ", "_").replace("-", "_")[:40]

        grouped_files = [{"path": obj.get("path",""), "content": obj.get("content","")}
                         for obj, _, _ in bare_creates]
        messages.extend(handle_create_project(folder_name, grouped_files))
        bare_paths = {obj.get("path") for obj, _, _ in bare_creates}
        json_items = [(obj, s, e) for obj, s, e in json_items
                      if not ((obj.get("action") or "").strip().lower() in ("create_file","create file","createfile")
                              and obj.get("path") in bare_paths)]

    for obj, _, _ in json_items:
        action = obj.get("action") or obj.get("intent")
        if not action:
            continue
        act = action.strip().lower()

        if act in ("create_folder", "create folder", "createfolder"):
            folder = obj.get("folder") or obj.get("name")
            if folder:
                messages.append(create_folder_action(folder))
            else:
                messages.append(error_message("Missing folder name"))

        elif act in ("create_project", "create project", "createproject"):
            folder = obj.get("folder") or obj.get("name") or obj.get("project")
            files = obj.get("files", [])
            if folder and files:
                messages.extend(handle_create_project(folder, files))
            else:
                messages.append(error_message("Missing folder name or files list"))

        elif act in ("create_file", "create file", "createfile"):
            path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_create_file(path, content))
            else:
                messages.append(error_message("Missing path: create_file needs 'file_path' or 'path' field"))

        elif act in ("update_file", "update file", "updatefile"):
            path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
            content = obj.get("content", "")
            if path:
                messages.extend(handle_update_file(path, content))
            else:
                messages.append(error_message("Missing path: update_file needs 'file_path' or 'path' field"))

        elif act in ("move_file", "move file", "movefile", "rename_file", "rename file"):
            src  = obj.get("source") or obj.get("src") or obj.get("from") or obj.get("old_path")
            dst  = obj.get("destination") or obj.get("dst") or obj.get("to") or obj.get("new_path")
            if src and dst:
                messages.append({"type": "move_file", "source": src, "destination": dst})
            else:
                messages.append(error_message("move_file needs 'source' and 'destination' fields"))

        elif act in ("write_todo", "update_todo", "write todo", "update todo"):
            todo_content = obj.get("content") or obj.get("text") or obj.get("todo")
            todo_path    = obj.get("path") or obj.get("file_path") or "TODO.md"
            if todo_content:
                messages.extend(handle_update_file(todo_path, todo_content))
            else:
                messages.append(error_message("write_todo needs a 'content' field"))

        elif act in ("run_file", "run file", "runfile", "test_file", "test file", "testfile"):
            path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
            env = obj.get("environment", "none")
            if path:
                messages.extend(handle_run_file(path, env))
            else:
                messages.append(error_message("Missing path: run_file needs 'path' or 'file_path'"))

        elif act in ("debug_file", "debug file", "debugfile"):
            path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
            stage = obj.get("stage", "all")
            if path:
                messages.extend(handle_debug_file(path, stage))
            else:
                messages.append(error_message("Missing path: debug_file needs 'path' or 'file_path'"))
        elif act in ("auto_debug", "auto debug", "autodebug"):
            messages.append({"type": "auto_debug"})
        elif act in ("search_files", "search files", "searchfiles"):
            messages.append(status_message("File search is handled locally by the extension."))
        elif act in ("search_folders", "search folders", "searchfolders"):
            messages.append(status_message("Folder search is handled locally by the extension."))
        elif act in ("search_in_files", "search in files", "searchinfiles", "grep"):
            messages.append(status_message("Content search is handled locally by the extension."))
        elif act in ("get_file_info", "get file info", "getfileinfo", "file_info"):
            messages.append(status_message("File info is handled locally by the extension."))
        else:
            messages.append(error_message(f"Unknown action: {act}"))

    messages_ordered = []

    for kind, item in interleaved:
        if kind == "text":
            chunk = clean_text_chunk(item)
            if chunk:
                messages_ordered.append(response_message(chunk))
        else:
            obj = item
            action = (obj.get("action") or obj.get("intent") or "").strip().lower()
            if action in ("create_file", "create file", "createfile"):
                path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
                if path:
                    messages_ordered.extend(handle_create_file(path, obj.get("content", "")))
            elif action in ("update_file", "update file", "updatefile"):
                path = obj.get("file_path") or obj.get("path") or obj.get("filename") or obj.get("file")
                if path:
                    messages_ordered.extend(handle_update_file(path, obj.get("content", "")))
            elif action in ("create_project", "create project", "createproject"):
                folder = obj.get("folder") or obj.get("name") or obj.get("project")
                files = obj.get("files", [])
                if folder and files:
                    messages_ordered.extend(handle_create_project(folder, files))
            elif action in ("create_folder", "create folder", "createfolder"):
                folder = obj.get("folder") or obj.get("name")
                if folder:
                    messages_ordered.append(create_folder_action(folder))

    file_action_types = {"create_file","update_file","create_files","create_project","create_folder"}
    for m in messages:
        if m.get("type") not in file_action_types and m.get("type") != "response":
            messages_ordered.append(m)

    return messages_ordered


# ------------------------------------------------------------
# Smart Project Update Endpoint
# ------------------------------------------------------------

class UpdateProjectRequest(BaseModel):
    project_path: str = ""    # kept for backward compat, ignored on Render
    files: Dict[str, str] = {}  # ← filename → content, sent from VS Code
    user_query: str
    conversation_history: str = ""
    target_directory: str = ""
    planned_files: List[str] = []
    generation_mode: str = "update"

UPDATE_READABLE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx',
    '.html', '.css', '.scss',
    '.json', '.env', '.txt', '.md', '.yaml', '.yml',
    '.toml', '.cfg', '.ini', '.sql'
}
UPDATE_SKIP_DIRS = {
    'node_modules', '__pycache__', '.git', 'venv', '.venv',
    'dist', 'build', 'coverage', '.mypy_cache', '.pytest_cache'
}

def _read_project_files(project_path: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    base = Path(project_path)
    if not base.exists():
        return files
    for entry in base.rglob('*'):
        rel_parts = entry.relative_to(base).parts
        parts = set(rel_parts)
        if parts & UPDATE_SKIP_DIRS:
            continue
        if len(rel_parts) >= 2 and rel_parts[0] in {'frontend', 'backend'} and rel_parts[1] in {'frontend', 'backend'}:
            continue
        entry_name = entry.name.lower()
        is_readable_special = entry_name in {'.env.example', 'dockerfile'} or entry_name.endswith('.env.example')
        if entry.is_file() and (entry.suffix.lower() in UPDATE_READABLE_EXTENSIONS or is_readable_special):
            rel = str(entry.relative_to(base))
            try:
                files[rel] = entry.read_text(encoding='utf-8', errors='ignore')
            except Exception:
                pass
    return files


def _normalize_rel_path(path_value: str) -> str:
    return str(path_value or "").replace("\\", "/").strip().lstrip("./").strip("/")


def _extract_query_terms(text: str) -> List[str]:
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "your", "have",
        "want", "need", "write", "build", "create", "update", "fix", "make", "add",
        "backend", "frontend", "file", "files", "project", "code", "folder", "inside",
        "there", "their", "what", "when", "where", "which", "while", "using", "based",
        "please", "must", "only", "into", "like", "just", "some", "more", "main",
    }
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_/-]{2,}", str(text or "").lower())
    ordered: List[str] = []
    seen = set()
    for token in tokens:
        cleaned = token.strip("/-_")
        if len(cleaned) < 3 or cleaned in stop_words or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
    return ordered[:18]


def _project_file_priority(
    rel_path: str,
    content: str,
    *,
    user_query: str,
    target_directory: str = "",
    planned_files: Optional[List[str]] = None,
    generation_mode: str = "update",
) -> Tuple[int, int]:
    normalized = _normalize_rel_path(rel_path)
    lower_rel = normalized.lower()
    lower_text = str(content or "").lower()
    name = Path(normalized).name.lower()
    planned = [_normalize_rel_path(item) for item in (planned_files or []) if item]
    score = 0
    tier = 9

    target = _normalize_rel_path(target_directory)
    backend_like_roots = (
        "backend/", "server/", "api/", "routes/", "routers/", "models/",
        "schemas/", "services/", "config/", "core/", "db/", "tests/"
    )
    root_priority_files = {
        "package.json", "vite.config.js", "vite.config.ts", ".env", ".env.example",
        "requirements.txt", "pyproject.toml", "readme.md", "dockerfile", "docker-compose.yml",
    }
    frontend_integration_paths = (
        "src/api/", "src/services/", "src/hooks/", "src/lib/", "src/utils/"
    )
    query_terms = _extract_query_terms(user_query)

    if target and (normalized == target or normalized.startswith(target + "/")):
        score += 900
        tier = min(tier, 0)

    if any(normalized == item or normalized.endswith("/" + item) for item in planned):
        score += 700
        tier = min(tier, 0)

    if lower_rel in root_priority_files or name in root_priority_files:
        score += 260
        tier = min(tier, 1)

    if any(lower_rel.startswith(prefix) or ("/" + prefix) in lower_rel for prefix in backend_like_roots):
        score += 220
        tier = min(tier, 1)

    if generation_mode.startswith("backend"):
        if any(lower_rel.startswith(prefix) for prefix in frontend_integration_paths):
            score += 160
            tier = min(tier, 2)
        elif lower_rel.startswith("src/"):
            score -= 60

    for term in query_terms:
        if term in lower_rel:
            score += 45
            tier = min(tier, 2)
        elif term in lower_text:
            score += 10

    if name in {"main.py", "app.py", "database.py", "schemas.py", "models.py", "auth.py"}:
        score += 80
        tier = min(tier, 2)

    if lower_rel.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".env", ".md", ".toml")):
        score += 15

    return (score, tier)


def _build_low_cost_project_files_block(
    project_files: Dict[str, str],
    *,
    user_query: str,
    target_directory: str = "",
    planned_files: Optional[List[str]] = None,
    generation_mode: str = "update",
    total_char_limit: int = UPDATE_PROJECT_CONTEXT_CHAR_LIMIT,
    per_file_char_limit: int = UPDATE_PROJECT_PER_FILE_CHAR_LIMIT,
) -> Tuple[str, List[str]]:
    ranked_with_scores = []
    for rel_path, content in project_files.items():
        score, tier = _project_file_priority(
            rel_path,
            content,
            user_query=user_query,
            target_directory=target_directory,
            planned_files=planned_files,
            generation_mode=generation_mode,
        )
        ranked_with_scores.append((score, tier, rel_path, content))

    ranked_files = sorted(ranked_with_scores, key=lambda item: (-item[0], item[1], item[2]))

    selected_parts: List[str] = []
    selected_paths: List[str] = []
    total_chars = 0

    for _, _, rel_path, content in ranked_files:
        normalized_content = str(content or "")
        clipped = normalized_content[:per_file_char_limit]
        suffix = ""
        if len(normalized_content) > per_file_char_limit:
            suffix = f"\n[...{len(normalized_content) - per_file_char_limit} chars truncated for low-cost context...]"
        entry = f"\n\n===== FILE: {rel_path} =====\n{clipped}{suffix}"
        if total_chars + len(entry) > total_char_limit:
            continue
        selected_parts.append(entry)
        selected_paths.append(rel_path)
        total_chars += len(entry)
        if total_chars >= total_char_limit:
            break

    if not selected_parts:
        return "", []

    return "".join(selected_parts), selected_paths


def _normalize_update_output_path(
    raw_path: str,
    target_directory: str = "",
    planned_files: Optional[List[str]] = None,
    project_folder: str = "",
) -> str:
    normalized = str(raw_path or "").replace("\\", "/").strip()
    normalized = re.sub(r"^\./+", "", normalized).lstrip("/")
    if not normalized:
        return normalized

    target = str(target_directory or "").replace("\\", "/").strip().strip("/")
    project = str(project_folder or "").replace("\\", "/").strip().strip("/")
    planned = []
    for item in planned_files or []:
        if not item:
            continue
        cleaned = str(item).replace("\\", "/").strip().lstrip("./").strip("/")
        if cleaned:
            planned.append(cleaned)

    if project and (normalized == project or normalized.startswith(project + "/")):
        normalized = normalized[len(project):].lstrip("/")

    if target and project:
        duplicated_prefixes = [
            f"{target}/{project}/{target}/",
            f"{target}/{project}/{target}",
        ]
        for prefix in duplicated_prefixes:
            if normalized.startswith(prefix):
                remainder = normalized[len(prefix):].lstrip("/")
                normalized = f"{target}/{remainder}" if remainder else target

    if target:
        doubled_target = f"{target}/{target}/"
        if normalized.startswith(doubled_target):
            normalized = f"{target}/{normalized[len(doubled_target):].lstrip('/')}"

    if target and (normalized == target or normalized.startswith(target + "/")):
        return normalized

    if normalized in planned:
        return normalized

    for prefix in ("backend/", "server/", "api/"):
        if target and normalized.startswith(prefix):
            remainder = normalized[len(prefix):].lstrip("/")
            return f"{target}/{remainder}" if remainder else target

    exact_suffix_matches = [path for path in planned if path == normalized or path.endswith("/" + normalized)]
    if len(exact_suffix_matches) == 1:
        return exact_suffix_matches[0]

    frontend_like_roots = {
        "src", "public", "app", "pages", "components", "hooks",
        "store", "assets", "styles", "frontend", "client", "web", "ui"
    }
    backend_like_roots = {
        "routes", "route", "routers", "router", "models", "model",
        "schemas", "schema", "services", "service", "utils", "config",
        "core", "db", "database", "auth", "middleware", "middlewares",
        "repositories", "repository", "migrations", "alembic", "tests"
    }
    first_segment = normalized.split("/", 1)[0].lower()
    backendish_ext = bool(re.search(r"\.(py|txt|env|cfg|ini|toml|sql|md|example)$", normalized, re.I))

    if target:
        if "/" not in normalized and backendish_ext:
            return f"{target}/{normalized}"
        if first_segment in backend_like_roots:
            return f"{target}/{normalized}"
        if backendish_ext and first_segment not in frontend_like_roots:
            return f"{target}/{normalized}"

    return normalized

UPDATE_PROJECT_SYSTEM = """
You are an expert full-stack developer performing a TARGETED UPDATE on an existing project.

The user will tell you what to fix or improve.
You will receive the COMPLETE contents of every file in the project.

YOUR TASK:
- Read every file carefully.
- Make only the smallest set of changes needed so the user requested functionality works 100%.
- Fix broken API calls, missing routes, incorrect JS logic, broken HTML, CSS issues - but keep unrelated code unchanged.
- Do NOT leave any placeholder, TODO, or incomplete code.
- Do NOT truncate any file with "..." - every file must be COMPLETE.
- Do NOT rewrite an entire file when a small patch is enough.
- If a fix is logical or behavioral, prefer surgical edits in the affected file(s) rather than replacing the full app.
- For backend generation, every planned route/model/schema/service/config file must contain REAL working code.
- Do NOT output placeholder comments such as "models are defined elsewhere", "routes are defined in ...", "implementation goes here", or files that only contain imports plus `router = APIRouter()`.
- If a route file is requested, include at least one real endpoint in that file unless the file is explicitly an aggregator module.
- If a model/schema file is requested, include the actual class definitions in that file instead of pointing to another file.

FRONTEND-SPECIFIC RULES (when the request involves generating or repairing React/Vite UI):
- Every visible module/page must be fully functional, not just presentational.
- If the backend exposes CRUD routes for an entity, the matching frontend page must include real GET, POST, PUT, and DELETE wiring, not dead buttons.
- Use exact backend route paths discovered in the project files. If a FastAPI router defines @router.get("/") under prefix "/agents", call "/agents/" in the frontend to avoid redirect issues.
- If the backend uses Bearer auth, attach Authorization: `Bearer ${localStorage.getItem('token')}` to every protected request.
- Treat 401 and 403 as auth failures in the API client and redirect to /login when appropriate.
- Do not create read-only list pages when the project requires management actions such as add/edit/delete.
- Do not invent API paths or simplify the app into static UI when backend endpoints already exist.
- If the backend provides auth, all protected module pages must check auth state and fail gracefully instead of silently swallowing API errors.
- If the request is to ADD a backend to an existing frontend game/app, preserve the existing frontend shell and routing unless the user explicitly asks for a UI redesign.
- If the frontend contains gameplay, scoring, form, or button-driven interactions, generate backend endpoints that match those actions and update the frontend network calls so the UI actually talks to the backend.
- If the frontend has action buttons or workflow actions, those actions must trigger backend requests and must not stay local-only.
- If the frontend already exists, patch the existing frontend files and connection config instead of creating a second frontend shell or duplicate app scaffold.
- When adding backend to an existing frontend, only create backend files and narrowly-scoped integration/support files; do not create a fresh frontend app scaffold.
- When generating a frontend for an existing backend, keep backend logic on the server. Do not rebuild persistence, auth checks, workflow engines, or server-side business rules in React.
- When a backend already exists, do not replace API-backed flows with mock data, browser-only arrays, fake repositories, or localStorage-only state unless the user explicitly asks for an offline demo.
- For game projects, persist player name, score, and session/game state in the backend instead of leaving all state in the browser.
- Do not alter frontend entry files (index.html, src/main.*, src/App.*) unless a specific frontend wiring fix is required.
- Generate the backend so the existing frontend can keep opening at its current Vite route without becoming a 404 page.
- Preserve the existing file creation structure. Do not rename, move, or reorganize files or folders unless the user explicitly asks for that change.
- When adding backend files to an existing frontend, create only the backend files and the minimum config files needed; do not reshuffle frontend files into new locations.
- If the request includes a planned backend file list, you must create every planned file unless the user explicitly asked to simplify the structure.
- Do not silently collapse a 15-file backend plan into a 5-file scaffold.
- If a target backend directory is provided, all new backend files must be created inside that directory.
- Paths like routes/x.py, models/y.py, services/z.py, config/db.py are backend files and belong under the target backend directory unless the prompt explicitly says otherwise.

OUTPUT FORMAT (mandatory):
For every file you change, output EXACTLY this JSON on its own line:
{"action": "update_file", "path": "RELATIVE/PATH/TO/FILE", "content": "COMPLETE FILE CONTENT HERE"}

Rules:
- Output ONLY the JSON lines for changed files - no prose, no markdown fences.
- The "path" must match the exact relative path given to you.
- The "content" must be the FULL updated file - never partial.
- If a file needs no changes, do NOT include it in the output.
- If you need to CREATE a new file, use {"action": "create_file", "path": "...", "content": "..."}
"""

@app.post("/update_project")
async def update_project_endpoint(req: UpdateProjectRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        if req.files:
            project_files = req.files          # sent directly from VS Code
        else:
            if SERVER_MODE:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Hosted backend cannot read the VS Code workspace from project_path. "
                        "Send the project files inline in the request body."
                    ),
                )
            project_files = _read_project_files(req.project_path)  # local fallback
        if not project_files:
            raise HTTPException(status_code=404, detail="No readable files found")

        target_directory = str(req.target_directory or "").replace("\\", "/").strip().strip("/")
        planned_files = [
            str(item).replace("\\", "/").strip().lstrip("./").strip("/")
            for item in (req.planned_files or [])
            if str(item or "").strip()
        ]
        
        
        files_block, selected_paths = _build_low_cost_project_files_block(
            project_files,
            user_query=req.user_query,
            target_directory=target_directory,
            planned_files=planned_files,
            generation_mode=req.generation_mode,
        )

        conversation_block = (
            f"CONVERSATION / PROJECT CONTEXT:\n{_clip_text_for_model(req.conversation_history, UPDATE_PROJECT_HISTORY_CHAR_LIMIT, keep='tail')}\n\n"
            if req.conversation_history.strip()
            else ""
        )
        target_directory_block = (
            f"TARGET DIRECTORY FOR NEW BACKEND FILES:\n{target_directory}\n\n"
            if target_directory
            else ""
        )
        planned_files_block = (
            "PLANNED FILES THAT MUST BE PRESENT AFTER THIS GENERATION:\n"
            + "\n".join(f"- {item}" for item in planned_files)
            + "\n\n"
            if planned_files
            else ""
        )
        selected_files_block = (
            "LOW-COST PROJECT CONTEXT FILES INCLUDED IN THIS REQUEST:\n"
            + "\n".join(f"- {item}" for item in selected_paths)
            + "\n\n"
            if selected_paths
            else ""
        )

        prompt = (
            f"{UPDATE_PROJECT_SYSTEM}\n\n"
            f"GENERATION MODE: {req.generation_mode}\n\n"
            "CONTEXT BUDGET RULE:\n"
            "- This request intentionally uses a LOW-COST context budget.\n"
            "- Prioritize the target backend directory, planned files, integration files, and root config files.\n"
            "- Do not invent placeholder wrappers because some unrelated files were omitted from context.\n"
            "- If a requested file depends on another module, write the import path and the actual implementation needed for the requested file.\n\n"
            f"{conversation_block}"
            f"{target_directory_block}"
            f"{planned_files_block}"
            f"{selected_files_block}"
            f"USER REQUEST: {req.user_query}\n\n"
            f"PROJECT FILES:{files_block}\n\n"
            f"Now output ONLY the JSON update lines for every file that needs to change:"
        )

        response = generate_content_logged(
            label="update_project",
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 32000, "temperature": 0.15},
            user_query=req.user_query,
            extra=f"mode={req.generation_mode}; selected_files={len(selected_paths)}",
        )
        raw = response.text.strip()

        json_items = extract_json_objects(raw)
        messages: List[dict] = []

        proj_folder = Path(req.project_path).name

        for obj, _, _ in json_items:
            action = (obj.get("action") or "").strip().lower()
            if action in ("create_project", "create project", "createproject"):
                files = obj.get("files") or []
                for file in files:
                    if not isinstance(file, dict):
                        continue
                    rel_path = _normalize_update_output_path(
                        file.get("path", ""),
                        target_directory,
                        planned_files,
                        proj_folder,
                    )
                    content = file.get("content", "")
                    if not rel_path:
                        continue
                    full_path = f"{proj_folder}/{rel_path}" if not rel_path.startswith(proj_folder) else rel_path
                    messages.append(create_file_action(full_path, content))
                continue

            rel_path = _normalize_update_output_path(
                obj.get("path", ""),
                target_directory,
                planned_files,
                proj_folder,
            )
            content = obj.get("content", "")
            if not rel_path:
                continue
            full_path = f"{proj_folder}/{rel_path}" if not rel_path.startswith(proj_folder) else rel_path
            if action in ("update_file", "update file", "updatefile",
                          "create_file", "create file", "createfile"):
                messages.append(create_file_action(full_path, content))

        if not messages:
            messages.append(response_message(raw[:2000]))

        return {"messages": messages}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /update_project: {e}")
        raise HTTPException(status_code=500, detail=str(e))



class WorkspaceRagRequest(BaseModel):
    workspace_path: str
    user_request: str
    prompt_instructions: str = "Keep the existing architecture, coding style, and folder structure."
    top_k: Optional[int] = None
    force_reindex: bool = False
    apply_changes: bool = False


class WorkspaceIndexRequest(BaseModel):
    workspace_path: str
    force_reindex: bool = False


@app.post("/rag/index")
async def rag_index_endpoint(req: WorkspaceIndexRequest):
    try:
        if SERVER_MODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hosted backend RAG indexing is not available from workspace_path alone. "
                    "Use inline project files from the extension."
                ),
            )
        engine = build_engine(workspace_path=req.workspace_path)
        return {
            "workspace_path": req.workspace_path,
            "result": engine.build_or_update_index(force=req.force_reindex),
            "workspace": engine.describe_workspace(),
        }
    except Exception as e:
        print(f"Error in /rag/index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/chat")
async def rag_chat_endpoint(req: WorkspaceRagRequest):
    try:
        if SERVER_MODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hosted backend RAG chat is not available from workspace_path alone. "
                    "Use inline project files from the extension."
                ),
            )
        engine = build_engine(workspace_path=req.workspace_path)
        engine.build_or_update_index(force=req.force_reindex)
        result = engine.chat_about_project(
            user_message=req.user_request,
            system_instruction=req.prompt_instructions,
            top_k=req.top_k,
        )
        return {
            "workspace_path": req.workspace_path,
            "result": result,
            "workspace": engine.describe_workspace(),
        }
    except Exception as e:
        print(f"Error in /rag/chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/update")
async def rag_update_endpoint(req: WorkspaceRagRequest):
    try:
        if SERVER_MODE:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Hosted backend RAG update is not available from workspace_path alone. "
                    "Use the inline project update flow from the extension."
                ),
            )
        result = get_project_update_response(
            user_request=req.user_request,
            prompt_instructions=req.prompt_instructions,
            workspace_path=req.workspace_path,
            top_k=req.top_k,
            force_reindex=req.force_reindex,
            apply_changes=req.apply_changes,
        )
        return {
            "workspace_path": req.workspace_path,
            "result": result,
        }
    except Exception as e:
        print(f"Error in /rag/update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------
# Deep Project Analysis Endpoint
# ------------------------------------------------------------
class DeepAnalyzeRequest(BaseModel):
    project_path: str
    project_name: str = ""
    max_depth: int = 5
    include_tests: bool = True


def analyze_project_structure(project_path: str, max_depth: int = 5, include_tests: bool = True) -> dict:
    structure = {
        "total_files": 0, "total_directories": 0, "file_tree": [],
        "main_entry_points": [], "config_files": [], "dependencies": [],
        "technologies": [], "code_metrics": {"total_lines": 0, "code_lines": 0,
        "comment_lines": 0, "blank_lines": 0, "average_file_size": 0, "complexity": "Low"}
    }
    entry_point_patterns = ['app.py', 'main.py', 'index.js', 'server.js', 'main.ts', 'index.ts', 'App.js', 'Main.java', 'main.go', 'main.rs', 'index.php', 'run.py', 'application.py', 'api.py', 'app.js', 'index.tsx', 'app.tsx']
    config_patterns = ['package.json', 'tsconfig.json', 'jsconfig.json', 'requirements.txt', 'setup.py', 'pyproject.toml', 'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'composer.json', 'Gemfile', '.env', '.env.example', 'docker-compose.yml', 'Dockerfile', '.gitignore', 'README.md']

    def build_tree(current_path: str, relative_path: str, depth: int) -> list:
        if depth > max_depth: return []
        nodes = []
        try:
            entries = list(Path(current_path).iterdir())
            for entry in entries:
                if entry.name.startswith('.') or entry.name in ['node_modules', '__pycache__', 'venv', '.git', 'dist', 'build', 'coverage']: continue
                if not include_tests and ('.test.' in entry.name or '.spec.' in entry.name or entry.name == '__tests__'): continue
                rel_path = str(Path(relative_path) / entry.name) if relative_path else entry.name
                if entry.is_dir():
                    structure["total_directories"] += 1
                    children = build_tree(str(entry), rel_path, depth + 1)
                    if children or entry.name not in ['node_modules', '__pycache__', 'venv']:
                        nodes.append({"name": entry.name, "path": rel_path, "type": "directory", "children": children})
                elif entry.is_file():
                    structure["total_files"] += 1
                    if entry.name.lower() in entry_point_patterns: structure["main_entry_points"].append(rel_path)
                    if entry.name.lower() in config_patterns: structure["config_files"].append(rel_path)
                    size = entry.stat().st_size
                    code_extensions = ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb', '.php', '.c', '.cpp', '.h']
                    ext = entry.suffix.lower()
                    if ext in code_extensions:
                        try:
                            with open(entry, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    structure["code_metrics"]["total_lines"] += 1
                                    stripped = line.strip()
                                    if not stripped: structure["code_metrics"]["blank_lines"] += 1
                                    elif stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or stripped.startswith('<!--'): structure["code_metrics"]["comment_lines"] += 1
                                    else: structure["code_metrics"]["code_lines"] += 1
                        except: pass
                    lang_map = {'.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.jsx': 'React', '.tsx': 'React TS', '.java': 'Java', '.go': 'Go', '.rs': 'Rust', '.rb': 'Ruby', '.php': 'PHP', '.c': 'C', '.cpp': 'C++', '.h': 'C Header', '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS', '.json': 'JSON', '.xml': 'XML', '.sql': 'SQL', '.sh': 'Shell', '.md': 'Markdown'}
                    nodes.append({"name": entry.name, "path": rel_path, "type": "file", "language": lang_map.get(ext, ext.upper().replace('.', '')), "size": size})
        except Exception as e: print(f"Error reading {current_path}: {e}")
        return sorted(nodes, key=lambda x: (x['type'] != 'directory', x['name']))

    structure["file_tree"] = build_tree(project_path, "", 0)
    if structure["total_files"] > 0: structure["code_metrics"]["average_file_size"] = structure["code_metrics"]["total_lines"] / structure["total_files"]
    avg = structure["code_metrics"]["average_file_size"]
    if avg > 1000: structure["code_metrics"]["complexity"] = "Very High"
    elif avg > 500: structure["code_metrics"]["complexity"] = "High"
    elif avg > 200: structure["code_metrics"]["complexity"] = "Medium"

    pkg_json = Path(project_path) / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, 'r') as f:
                pkg = json.load(f)
                for name, version in pkg.get('dependencies', {}).items(): structure["dependencies"].append({"name": name, "version": str(version), "type": "production"})
                for name, version in pkg.get('devDependencies', {}).items(): structure["dependencies"].append({"name": name, "version": str(version), "type": "development"})
                structure["technologies"].append({"name": "Node.js", "category": "Runtime", "usage": "Detected from package.json"})
        except: pass

    req_txt = Path(project_path) / "requirements.txt"
    if req_txt.exists():
        try:
            with open(req_txt, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'^([a-zA-Z0-9_-]+)([=<>!~]+)?(.+)?', line)
                        if match: structure["dependencies"].append({"name": match.group(1), "version": match.group(3) if match.group(3) else "latest", "type": "production"})
            structure["technologies"].append({"name": "Python", "category": "Runtime", "usage": "Detected from requirements.txt"})
        except: pass

    if (Path(project_path) / "Cargo.toml").exists(): structure["technologies"].append({"name": "Rust", "category": "Language", "usage": "Detected from Cargo.toml"})
    if (Path(project_path) / "go.mod").exists(): structure["technologies"].append({"name": "Go", "category": "Language", "usage": "Detected from go.mod"})

    return structure


@app.post("/deep_analyze")
async def deep_analyze_endpoint(req: DeepAnalyzeRequest):
    try:
        structure = analyze_project_structure(req.project_path, req.max_depth, req.include_tests)
        main_files = structure.get("main_entry_points", [])[:5]
        config_files = structure.get("config_files", [])[:10]
        deps = structure.get("dependencies", [])[:15]
        techs = structure.get("technologies", [])

        prompt = f"""Analyze this project. Project: {req.project_name or 'Unknown'} at {req.project_path}. Structure: {structure['total_files']} files, {structure['total_directories']} dirs. Entry points: {', '.join(main_files) if main_files else 'None'}. Configs: {', '.join(config_files) if config_files else 'None'}. Deps: {', '.join([d['name'] for d in deps]) if deps else 'None'}. Tech: {', '.join([t['name'] for t in techs]) if techs else 'None'}. LOC: {structure['code_metrics']['code_lines']}, Complexity: {structure['code_metrics']['complexity']}. Return JSON with projectGoal (2-3 sentences), issues (severity, description, suggestion), enhancements (category, title, description, priority, effort)."""

        response = generate_content_logged(
            label="deep_analyze",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query=req.project_name or req.project_path,
            extra=f"files={structure['total_files']}",
        )
        ai_response = response.text.strip()

        project_goal = "Unable to determine project goal"
        issues, enhancements = [], []

        try:
            json_match = re.search(r'\{[\s\S]*\}', ai_response)
            if json_match:
                ai_data = json.loads(json_match.group())
                project_goal = ai_data.get('projectGoal', project_goal)
                issues = ai_data.get('issues', [])
                enhancements = ai_data.get('enhancements', [])
        except: project_goal = ai_response[:500]

        return {"project_name": req.project_name or Path(req.project_path).name, "project_path": req.project_path, "code_structure": {"total_files": structure["total_files"], "total_directories": structure["total_directories"], "file_tree": structure["file_tree"], "main_entry_points": structure["main_entry_points"], "config_files": structure["config_files"]}, "project_goal": project_goal, "issues": issues, "enhancement_ideas": enhancements, "technologies": structure["technologies"], "dependencies": structure["dependencies"], "code_metrics": structure["code_metrics"]}
    except Exception as e:
        print(f"Error in /deep_analyze: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Smart dev Backend is running"}


@app.post("/testing/blueprint")
async def testing_blueprint_endpoint(req: TestingBlueprintRequest):
    return build_todo_blueprint(req.user_request, req.workspace_summary)


@app.post("/testing/plan")
async def testing_plan_endpoint(req: TestingPlanRequest):
    project_root = str(req.project_root or "").strip()
    if not project_root:
        raise HTTPException(status_code=400, detail="project_root is required")
    if not os.path.isdir(project_root):
        raise HTTPException(status_code=400, detail=f"Invalid project_root: {project_root}")

    plan = build_test_plan(
        project_root=project_root,
        written_files=req.written_files,
        user_request=req.user_request,
        full_project=req.full_project,
    )
    return plan


@app.post("/testing/error_summary")
async def testing_error_summary_endpoint(payload: Dict[str, Any]):
    output = str(payload.get("output", "") or "")
    return classify_terminal_error(output)

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        msgs = process_user_message(request)
        safe_msgs = []
        for m in msgs:
            try:
                import json as _json
                _json.dumps(m)
                safe_msgs.append(m)
            except Exception:
                safe_msgs.append({"type": "error", "text": str(m)})
        return ChatResponse(messages=safe_msgs)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[/chat ERROR] {e}\n{tb}")
        return ChatResponse(messages=[{"type": "error", "text": f"Server error: {str(e)}"}])

class DebugRequest(BaseModel):
    file_path: str
    content: str
    error: Optional[str] = None
    analysis: str = ""
    error_kind: str = ""
    diagnostics: str = ""
    related_files: List[str] = []
    related_file_contexts: List[dict] = []


class RuntimeErrorAnalysisRequest(BaseModel):
    error_output: str
    runtime_type: str = ""
    cwd: str = ""
    files: List[dict] = []


def _heuristic_runtime_error_kind(error_output: str) -> str:
    text = str(error_output or "")
    if re.search(
        r"Cannot use import statement outside a module|ERR_REQUIRE_ESM|Unknown file extension|node:perf_hooks|requires Node\.js|Unsupported engine",
        text,
        re.I,
    ):
        return "environment"
    if re.search(r"SyntaxError|IndentationError|Unexpected token|Unterminated", text, re.I):
        return "syntax"
    if re.search(r"ModuleNotFoundError|ImportError|Cannot find module|Can't resolve|attempted relative import", text, re.I):
        return "import"
    if re.search(r"NameError|ReferenceError", text, re.I):
        return "reference"
    if re.search(r"TypeError|AttributeError|KeyError|IndexError|ValueError", text, re.I):
        return "runtime"
    if re.search(r"Failed to compile|Build failed|TS\d+|vite:|webpack|esbuild", text, re.I):
        return "build"
    if re.search(r"command not found|externally managed environment|Unsupported engine|requires node|permission denied|EACCES", text, re.I):
        return "environment"
    return "runtime"


def _fallback_runtime_error_analysis(req: RuntimeErrorAnalysisRequest) -> Dict[str, Any]:
    kind = _heuristic_runtime_error_kind(req.error_output)
    file_paths = [
        str(item.get("path", "")).strip()
        for item in (req.files or [])
        if isinstance(item, dict) and str(item.get("path", "")).strip()
    ]
    should_edit_code = kind in {"syntax", "import", "reference", "runtime", "build"}
    root_cause_map = {
        "syntax": "The terminal output points to a syntax or parser problem in one of the project files.",
        "import": "The terminal output points to a missing, broken, or incorrect import/module reference.",
        "reference": "The code is referencing a symbol that does not exist or is misspelled.",
        "runtime": "The code reaches runtime but fails because of an invalid operation or bad assumptions.",
        "build": "The frontend/backend build tool found a compile-time problem in source code or config.",
        "environment": "The failure is primarily caused by the runtime environment, package manager, or interpreter rather than source code.",
    }
    return {
        "error_kind": kind,
        "root_cause": root_cause_map.get(kind, "The run failed and needs targeted investigation."),
        "recommended_fix": "Inspect the likely culprit files, correct the concrete issue reported in the terminal, and re-run.",
        "should_edit_code": should_edit_code,
        "environment_issue": kind == "environment",
        "likely_files": file_paths[:5],
        "confidence": "medium" if file_paths else "low",
    }


def _clean_debug_model_output(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    elif cleaned.startswith("python\n"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("javascript\n"):
        cleaned = cleaned[11:].strip()
    elif cleaned.startswith("typescript\n"):
        cleaned = cleaned[11:].strip()
    return cleaned


def _format_related_file_contexts(contexts: List[dict], max_chars: int = 6000) -> str:
    sections: List[str] = []
    for item in contexts[:4]:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("path", "")).strip() or "unknown"
        diagnostics = str(item.get("diagnostics", "")).strip() or "None"
        snippet = str(item.get("snippet", "")).strip() or "None"
        block = (
            f"File: {file_path}\n"
            f"Diagnostics:\n{diagnostics}\n"
            f"Snippet:\n{snippet}"
        )
        sections.append(block)
    return _clip_text_for_model("\n\n".join(sections), max_chars, keep="tail")


def _looks_like_placeholder_fix(code: str, filename: str, original: str) -> Optional[str]:
    text = str(code or "").strip()
    if not text:
        return "The generated fix is empty."

    placeholder_patterns = [
        r"models are defined elsewhere",
        r"models are defined in ",
        r"implement .* here",
        r"todo",
        r"placeholder",
        r"rest of the code remains",
        r"existing code",
        r"same as before",
    ]
    if any(re.search(pattern, text, re.I) for pattern in placeholder_patterns):
        return "The generated fix still contains placeholder text instead of a concrete implementation."

    ext = Path(filename or "").suffix.lower()
    stripped_lines = [line for line in text.splitlines() if line.strip()]

    if ext == ".py":
        if re.fullmatch(r"(from\s+\S+\s+import\s+.+|import\s+\S+)(\n(from\s+\S+\s+import\s+.+|import\s+\S+))*", text.strip(), re.S):
            return "The generated Python file only contains imports and no real code."
        if "APIRouter(" in text and "@router." not in text and len(stripped_lines) <= 6:
            return "The generated router file only contains a bare router declaration without endpoints."
        if (
            any(token in Path(filename).stem.lower() for token in ["model", "schema"])
            and "class " not in text
            and len(stripped_lines) <= max(8, len(original.splitlines()) // 2)
        ):
            return "The generated model/schema file does not define concrete classes."

    return None


def _validate_debug_fix_candidate(code: str, filename: str, language: str, original: str) -> Tuple[bool, Optional[str]]:
    placeholder_error = _looks_like_placeholder_fix(code, filename, original)
    if placeholder_error:
        return False, placeholder_error

    if language == "python":
        syntax_error, _ = validate_python_code(code, filename)
        if syntax_error:
            return False, syntax_error
    elif language == "json":
        try:
            pyjson.loads(code)
        except Exception as exc:
            return False, f"JSONError: {exc}"

    if "Corrected code:" in code or "```" in code:
        return False, "The generated fix still includes response wrapper text instead of raw file content."

    return True, None


@app.post("/analyze_runtime_error")
async def analyze_runtime_error_endpoint(req: RuntimeErrorAnalysisRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        file_block = "\n".join(
            (
                f"- Path: {str(item.get('path', '')).strip()}\n"
                f"  Diagnostics: {str(item.get('diagnostics', '')).strip() or 'None'}\n"
                f"  Error context: {str(item.get('error_context', '')).strip() or 'None'}"
            )
            for item in (req.files or [])[:8]
            if isinstance(item, dict)
        )

        schema = (
            "{\n"
            '  "error_kind": "syntax | import | reference | runtime | build | environment",\n'
            '  "root_cause": "short root-cause summary",\n'
            '  "recommended_fix": "short actionable fix strategy",\n'
            '  "should_edit_code": true,\n'
            '  "environment_issue": false,\n'
            '  "likely_files": ["path/to/file.py"],\n'
            '  "confidence": "high | medium | low"\n'
            "}"
        )

        prompt = (
            "You are a senior debugging analyst.\n"
            "Analyze the terminal error first, then decide whether the fix should be applied in source code files or whether the problem is mainly environment/package related.\n"
            "Focus especially on syntax issues, import issues, compile/build issues, and obvious code issues.\n"
            "Return ONLY valid JSON matching this schema:\n"
            f"{schema}\n\n"
            f"Runtime type: {req.runtime_type or 'unknown'}\n"
            f"Working directory: {req.cwd or 'unknown'}\n\n"
            f"Terminal output:\n{_clip_text_for_model(req.error_output, 5000, keep='tail')}\n\n"
            f"Candidate files and diagnostics:\n{file_block or 'None'}"
        )

        response = generate_content_logged(
            label="analyze_runtime_error",
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 1200, "temperature": 0.1},
            user_query=_compact_text_for_log(req.error_output, 220),
            extra=f"runtime={req.runtime_type or 'unknown'}; files={len(req.files or [])}",
        )
        raw = (response.text or "").strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except Exception:
                parsed = _fallback_runtime_error_analysis(req)

        fallback = _fallback_runtime_error_analysis(req)
        result = {
            "error_kind": str(parsed.get("error_kind") or fallback["error_kind"]).strip(),
            "root_cause": str(parsed.get("root_cause") or fallback["root_cause"]).strip(),
            "recommended_fix": str(parsed.get("recommended_fix") or fallback["recommended_fix"]).strip(),
            "should_edit_code": bool(parsed.get("should_edit_code", fallback["should_edit_code"])),
            "environment_issue": bool(parsed.get("environment_issue", fallback["environment_issue"])),
            "likely_files": parsed.get("likely_files") or fallback["likely_files"],
            "confidence": str(parsed.get("confidence") or fallback["confidence"]).strip(),
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze_runtime_error: {e}")
        return _fallback_runtime_error_analysis(req)


@app.post("/debug")
async def debug_endpoint(req: DebugRequest):
    try:
        language, syntax_error, _ = validate_debug_code(req.content, req.file_path)
        related_context_block = _format_related_file_contexts(req.related_file_contexts)

        prompt = f"""
You are an expert programmer fixing ONE EXISTING FILE.
Return ONLY the corrected full file content for that same file.
Do not return explanations, markdown, backticks, JSON, or extra text.
Do not create new files, folders, backend code, APIs, routes, or project structure changes unless they already exist in this exact file.
Keep the file in its current language/framework and preserve the existing functionality as much as possible.
Fix syntax, parser, compile, import, and obvious runtime issues in-place.
Use the related-file snippets only as context. Do not rewrite them unless this exact file needs matching imports/usages.

If the file is React/JSX/TSX:
- Fix only this component/entry file.
- Preserve React + Vite structure.
- Do not convert it to Python or generate backend code.
- Keep imports unless a direct fix is required for this file to compile.

File: {req.file_path}
Language: {language}
Terminal Error: {req.error if req.error else 'Not provided'}
Detected Error Kind: {req.error_kind if req.error_kind else 'unknown'}
Error Analysis: {req.analysis if req.analysis else 'Not provided'}
VS Code Diagnostics: {req.diagnostics if req.diagnostics else 'None'}
Related Files: {', '.join(req.related_files) if req.related_files else 'None'}
Related File Context:
{related_context_block if related_context_block else 'None'}
Static Validation: {syntax_error if syntax_error else 'None'}

Code:
{req.content}

Corrected code:
"""
        response = generate_content_logged(
            label="debug",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query=f"{req.file_path} | {req.error or 'debug request'}",
        )
        fixed_content = _clean_debug_model_output(response.text)
        is_valid, validation_error = _validate_debug_fix_candidate(
            fixed_content,
            req.file_path,
            language,
            req.content,
        )

        if not is_valid:
            retry_prompt = f"""
You previously attempted to fix this file but returned an invalid result.
Return ONLY the corrected full file content for the same file.
Do not return markdown, explanations, JSON, comments about what you changed, or placeholder text.
The new output must directly fix the issue and remain valid {language} code.

File: {req.file_path}
Language: {language}
Terminal Error: {req.error if req.error else 'Not provided'}
Detected Error Kind: {req.error_kind if req.error_kind else 'unknown'}
Error Analysis: {req.analysis if req.analysis else 'Not provided'}
Validation failure from previous attempt: {validation_error}
VS Code Diagnostics: {req.diagnostics if req.diagnostics else 'None'}
Related File Context:
{related_context_block if related_context_block else 'None'}
Current file content:
{req.content}

Previous invalid attempt:
{fixed_content or '[empty]'}

Corrected code:
"""
            retry_response = generate_content_logged(
                label="debug_retry",
                model=GEMINI_MODEL,
                contents=retry_prompt,
                config={"temperature": 0.05, "max_output_tokens": 12000},
                user_query=f"{req.file_path} | retry | {req.error or 'debug request'}",
            )
            fixed_content = _clean_debug_model_output(retry_response.text)
            is_valid, validation_error = _validate_debug_fix_candidate(
                fixed_content,
                req.file_path,
                language,
                req.content,
            )

        if not fixed_content:
            fixed_content = req.content

        if not is_valid:
            return {
                "fixed_content": req.content,
                "validation_error": validation_error or "Generated fix did not pass validation.",
            }

        return {"fixed_content": fixed_content}
    except Exception as e:
        import traceback
        print(f"Error in /debug: {e}\n{traceback.format_exc()}")
        return {"fixed_content": None, "error": str(e)}


# ------------------------------------------------------------
# Image Analysis Endpoint
# ------------------------------------------------------------
class ImageAnalysisRequest(BaseModel):
    image: str
    filename: str
    type: str = "screenshot"
    conversation_history: str = ""


@app.post("/analyze_image")
async def analyze_image_endpoint(req: ImageAnalysisRequest):
    try:
        prompt = f"""
You are an expert code reviewer and debugger. Analyze this image (which appears to be a {req.type}) and provide insights.

If it's a screenshot of code:
- Identify any errors or bugs visible
- Suggest fixes or improvements
- Explain what the code does

If it's a UI screenshot:
- Describe the interface
- Suggest improvements or identify issues
- Provide relevant code suggestions if applicable

If it's an error message:
- Explain the error
- Provide the solution
- Give example code to fix it

Be concise but thorough in your analysis.
"""
        import base64
        image_data = req.image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        from google.genai import types
        mime_type = "image/png"
        if req.filename.lower().endswith('.jpg') or req.filename.lower().endswith('.jpeg'):
            mime_type = "image/jpeg"
        elif req.filename.lower().endswith('.gif'):
            mime_type = "image/gif"
        elif req.filename.lower().endswith('.webp'):
            mime_type = "image/webp"
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        response = generate_content_logged(
            label="analyze_image",
            model=GEMINI_MODEL,
            contents=[prompt, image_part],
            user_query=req.filename,
        )
        analysis = response.text.strip()
        return {"analysis": analysis}
    except Exception as e:
        print(f"Error in /analyze_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# PDF Analysis Endpoint
# ------------------------------------------------------------
class PDFAnalysisRequest(BaseModel):
    pdf: str
    filename: str
    conversation_history: str = ""


@app.post("/analyze_pdf")
async def analyze_pdf_endpoint(req: PDFAnalysisRequest):
    try:
        prompt = f"""
You are an expert document analyzer. Analyze this PDF document and provide a comprehensive summary.

Please:
1. Summarize the main content and purpose
2. Extract key information, requirements, or specifications
3. Identify any code examples, error messages, or technical details
4. Suggest how this document relates to the current project context

If this is a requirements document or specification:
- List the key requirements
- Suggest implementation approaches
- Identify potential challenges

Be thorough but concise in your analysis.
"""
        pdf_data = req.pdf
        if ',' in pdf_data:
            pdf_data = pdf_data.split(',')[1]
        import base64
        pdf_bytes = base64.b64decode(pdf_data)
        from google.genai import types
        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
        response = generate_content_logged(
            label="analyze_pdf",
            model=GEMINI_MODEL,
            contents=[prompt, pdf_part],
            user_query=req.filename,
        )
        analysis = response.text.strip()
        return {"analysis": analysis}
    except Exception as e:
        print(f"Error in /analyze_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# File Analysis Endpoint
# ------------------------------------------------------------
class AnalyzeFileRequest(BaseModel):
    file_path: str
    content: str

@app.post("/analyze_file")
async def analyze_file_endpoint(req: AnalyzeFileRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        content = req.content
        was_truncated = False
        if len(content) > 6000:
            content = content[:6000]
            was_truncated = True

        truncation_note = "\n[...file truncated for analysis...]" if was_truncated else ""

        prompt = f"""Analyze this file. Respond ONLY with a JSON object, nothing else.

File: {req.file_path}
```
{content}{truncation_note}
```

JSON format (no markdown, no explanation, raw JSON only):
{{"purpose": "one sentence describing what this file does", "issues": [{{"line": null, "description": "issue description", "severity": "warning"}}]}}

If no issues found, use empty array for issues. Output raw JSON only, starting with {{"""

        response = generate_content_logged(
            label="analyze_file",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query=req.file_path,
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        brace_idx = raw.find('{')
        if brace_idx > 0:
            raw = raw[brace_idx:]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    try:
                        parsed = json.loads(_sanitize_json_string(match.group()))
                    except:
                        raise HTTPException(status_code=422, detail=f"Could not parse AI response: {raw[:200]}")
                else:
                    raise HTTPException(status_code=422, detail=f"No JSON found in AI response: {raw[:200]}")

        return {
            "purpose": str(parsed.get("purpose", "")),
            "issues": parsed.get("issues", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze_file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Project Summary Endpoint
# ------------------------------------------------------------
class AnalyzeSummaryRequest(BaseModel):
    file_list: str

@app.post("/analyze_summary")
async def analyze_summary_endpoint(req: AnalyzeSummaryRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        prompt = f"""You are a silent project analyzer. Based on these files, respond ONLY with valid JSON.

Files:
{req.file_list}

Respond with ONLY this JSON object (no markdown, no explanation, no code fences):
{{
  "summary": "2-3 sentence overview of what this project does.",
  "suggestedEnhancements": [
    "Enhancement suggestion 1",
    "Enhancement suggestion 2",
    "Enhancement suggestion 3"
  ]
}}"""

        response = generate_content_logged(
            label="analyze_summary",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query="workspace summary",
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except:
                match = re.search(r'\{[\s\S]*\}', raw)
                parsed = json.loads(_sanitize_json_string(match.group())) if match else {}

        return {
            "summary": str(parsed.get("summary", "")),
            "suggestedEnhancements": parsed.get("suggestedEnhancements", [])
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /analyze_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Project Query Endpoint
# ------------------------------------------------------------
class ProjectQueryRequest(BaseModel):
    query: str
    project_data: dict

@app.post("/project_query")
async def project_query_endpoint(req: ProjectQueryRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        data = req.project_data
        project_name = data.get("projectName", "Unknown")
        summary      = data.get("summary", "")
        tech_stack   = ", ".join(data.get("techStack", []))
        files        = data.get("files", [])
        enhancements = data.get("suggestedEnhancements", [])

        file_lines = []
        for f in files:
            issues = f.get("issues", [])
            issue_str = ""
            if issues:
                issue_str = " | Issues: " + "; ".join(
                    f"[{i.get('severity','?').upper()}] {i.get('description','')}"
                    for i in issues
                )
            deps = ", ".join(f.get("dependencies", [])[:5])
            file_lines.append(
                f"  - {f['path']} (health:{f.get('healthScore',100)}) — "
                f"{f.get('purpose','not analyzed yet')}"
                f"{(' | deps: ' + deps) if deps else ''}"
                f"{issue_str}"
            )

        context = f"""PROJECT: {project_name}
TECH STACK: {tech_stack}
SUMMARY: {summary}
OVERALL HEALTH: {data.get('totalHealthScore', 100)}/100
TOTAL FILES: {len(files)}

FILES ANALYZED:
{chr(10).join(file_lines)}

SUGGESTED ENHANCEMENTS:
{chr(10).join('  - ' + e for e in enhancements[:5]) if enhancements else '  (none yet)'}
"""

        prompt = f"""You are an expert code reviewer. A developer asked: "{req.query}"

Here is the complete analysis of their project from our analyzer:

{context}

Answer their question thoroughly using ONLY the data above.
- List specific issues by file and severity
- Suggest concrete fixes for each issue
- Mention dependencies that could be problematic
- Comment on health scores if relevant
- Be direct and actionable
- Use markdown formatting with headers and bullet points
- Do NOT say you need to scan files or use any tools — all data is already provided above
- Do NOT output any JSON action blocks"""

        response = generate_content_logged(
            label="project_query",
            model=GEMINI_MODEL,
            contents=prompt,
            user_query=req.query,
        )
        answer = response.text.strip()
        answer = re.sub(r'\{[^{}]*?"action"[^{}]*?\}', '', answer, flags=re.DOTALL).strip()

        return {"response": answer}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /project_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# Smart Edit Endpoint — FIXED (was referencing undefined variables)
# ------------------------------------------------------------
class SmartEditRequest(BaseModel):
    prompt: str

@app.post("/smart_edit")
async def smart_edit_endpoint(req: SmartEditRequest):
    """
    Receives a pre-built surgical edit prompt from the TypeScript EditPlanner.
    Returns a JSON object with patches, newImports, summary, and sideEffects.
    """
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        system_instruction = (
            "You are a surgical code patch generator. "
            "You MUST respond with ONLY a valid JSON object — no markdown, no explanation, no code fences. "
            "The JSON must have exactly these keys: "
            "\"summary\" (string), \"patches\" (array of {search, replace, description}), "
            "\"newImports\" (array of strings), \"sideEffects\" (string). "
            "The \"search\" value in each patch must be an EXACT verbatim copy from the provided file content. "
            "Never rewrite the whole file. Only patch what is necessary."
        )

        # FIX: use req.prompt directly (the original used undefined variables)
        full_prompt = f"{system_instruction}\n\n{req.prompt}"
        response = generate_content_logged(
            label="smart_edit",
            model=GEMINI_MODEL,
            contents=full_prompt,
            config={"max_output_tokens": 32000, "temperature": 0.1},
            user_query=req.prompt[:200],
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', raw)
                if match:
                    parsed = json.loads(_sanitize_json_string(match.group()))
                else:
                    raise HTTPException(status_code=422, detail="AI returned non-JSON response")

        result = {
            "summary": parsed.get("summary", ""),
            "patches": parsed.get("patches", []),
            "newImports": parsed.get("newImports", []),
            "sideEffects": parsed.get("sideEffects", "")
        }

        clean_patches = []
        for p in result["patches"]:
            if isinstance(p, dict) and "search" in p and "replace" in p:
                clean_patches.append({
                    "search": str(p["search"]),
                    "replace": str(p["replace"]),
                    "description": str(p.get("description", ""))
                })
        result["patches"] = clean_patches

        return {"result": json.dumps(result)}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /smart_edit: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GitHub OAuth ──────────────────────────────────────────────────────────────
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")

class GithubTokenRequest(BaseModel):
    code: str
    redirect_uri: str

@app.post("/github/token")
async def github_token_exchange(req: GithubTokenRequest):
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=503,
            detail="GitHub OAuth not configured. Add GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET to .env"
        )
    try:
        response = requests.post(
            "https://github.com/login/oauth/access_token",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            json={
                "client_id":     GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code":          req.code,
                "redirect_uri":  req.redirect_uri
            },
            timeout=15
        )
        if not response.ok:
            raise HTTPException(status_code=502, detail=f"GitHub OAuth request failed: {response.status_code}")
        data = response.json()
        if "error" in data:
            raise HTTPException(status_code=400, detail=f"GitHub OAuth error: {data.get('error_description', data['error'])}")
        if "access_token" not in data:
            raise HTTPException(status_code=502, detail="GitHub did not return an access token")
        return {"access_token": data["access_token"]}
    except HTTPException:
        raise
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="GitHub OAuth request timed out. Please try again.")
    except Exception as e:
        print(f"Error in /github/token: {e}")
        raise HTTPException(status_code=500, detail=str(e))






# ------------------------------------------------------------
# List Frontend Files Endpoint (FAST - no AI, just filesystem)
# Called first to show user which files will be analyzed
# Returns file list in under 100ms
# ------------------------------------------------------------
class ListFrontendRequest(BaseModel):
    project_path: str
    frontend_subdir: str = ""
    files: Dict[str, str] = Field(default_factory=dict)

FRONTEND_EXTS = {".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte"}
FRONTEND_SKIP_DIRS = {"node_modules", ".git", "dist", "build", "__pycache__", ".next", "out", ".venv", "venv"}
FRONTEND_PRIORITY = ["App.tsx", "App.jsx", "App.ts", "App.js", "main.tsx", "main.ts", "index.tsx"]

def _is_noise_frontend_file(entry: Path, base: Path) -> bool:
    rel_parts = entry.relative_to(base).parts
    if set(rel_parts) & FRONTEND_SKIP_DIRS:
        return True

    if rel_parts and rel_parts[0] in {"frontend", "backend"}:
        return True

    if not entry.is_file() or entry.suffix.lower() not in FRONTEND_EXTS:
        return True

    stem = entry.stem.strip()
    lower_stem = stem.lower()

    # Skip obvious scratch/backup copies when the canonical file exists beside them.
    numbered = re.match(r"^(.*?)(?:[\s_-]+)?(\d+)$", stem)
    if numbered:
        canonical = entry.with_name(f"{numbered.group(1).strip()}{entry.suffix}")
        if canonical.exists():
            return True

    noisy_suffixes = (" copy", "_copy", "-copy", " backup", "_backup", "-backup", " old", "_old", "-old")
    for suffix in noisy_suffixes:
        if lower_stem.endswith(suffix):
            canonical_name = stem[: -len(suffix)].rstrip(" _-")
            if canonical_name:
                canonical = entry.with_name(f"{canonical_name}{entry.suffix}")
                if canonical.exists():
                    return True

    return False

def _collect_frontend_source_files(base: Path) -> list[Path]:
    files: list[Path] = []
    for entry in sorted(base.rglob("*")):
        if _is_noise_frontend_file(entry, base):
            continue
        files.append(entry)

    files.sort(
        key=lambda entry: (
            0 if any(str(entry.relative_to(base)).endswith(name) for name in FRONTEND_PRIORITY) else 1,
            str(entry.relative_to(base))
        )
    )
    return files

def _to_snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value)
    return value.strip("_").lower()


def _collect_backend_domain_names(parsed: dict) -> list[str]:
    names: list[str] = []

    for model in parsed.get("data_models") or []:
        if not isinstance(model, dict):
            continue
        raw_name = str(model.get("name", "") or "").strip()
        snake = _to_snake_case(raw_name)
        if snake and snake not in names:
            names.append(snake)

    for endpoint in parsed.get("api_endpoints") or []:
        if not isinstance(endpoint, dict):
            continue
        raw_path = str(endpoint.get("path", "") or "")
        cleaned = re.sub(r"^https?://[^/]+", "", raw_path).strip()
        cleaned = re.sub(r"^/api", "", cleaned).strip("/")
        if not cleaned:
            continue
        for part in cleaned.split("/"):
            part = part.strip().lower()
            if not part or part.startswith("{"):
                continue
            if part in {"auth", "login", "logout", "register", "me", "health", "api", "ws"}:
                continue
            snake = _to_snake_case(part)
            if snake and snake not in names:
                names.append(snake)
                break

    goal = str(parsed.get("project_goal", "") or "")
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", goal):
        lowered = token.lower()
        if lowered in {"project", "website", "frontend", "backend", "system", "platform", "app"}:
            continue
        snake = _to_snake_case(token)
        if snake and snake not in names:
            names.append(snake)
        if len(names) >= 6:
            break

    return names[:6]


def _needs_database(parsed: dict) -> bool:
    if parsed.get("data_models"):
        return True

    methods = {
        str(endpoint.get("method", "") or "").upper()
        for endpoint in (parsed.get("api_endpoints") or [])
        if isinstance(endpoint, dict)
    }
    if methods & {"POST", "PUT", "PATCH", "DELETE"}:
        return True

    goal = str(parsed.get("project_goal", "") or "").lower()
    return any(word in goal for word in ["store", "save", "dashboard", "management", "inventory", "booking", "records"])


def _build_backend_structure_hint(parsed: dict) -> list[str]:
    structure: list[str] = ["main.py", "requirements.txt"]
    domain_names = _collect_backend_domain_names(parsed)
    auth_type = str(parsed.get("auth_type", "none") or "none").lower()
    external_services = [str(item).lower() for item in (parsed.get("external_services") or [])]
    env_vars = parsed.get("env_vars_needed") or []
    websocket_endpoints = parsed.get("websocket_endpoints") or []

    if _needs_database(parsed):
        structure.append("database.py")

    if env_vars or external_services:
        structure.append(".env.example")

    if auth_type not in {"", "none", "null"}:
        structure.append("auth.py")

    if websocket_endpoints:
        structure.append("websocket.py")

    if any(keyword in service for service in external_services for keyword in ["nlp", "openai", "huggingface", "gemini", "llm", "ai"]):
        structure.append("services/nlp_service.py")

    if domain_names:
        for name in domain_names[:5]:
            structure.append(f"{name}_routes.py")
            if _needs_database(parsed):
                structure.append(f"{name}_models.py")

    # Keep order stable and remove duplicates.
    deduped: list[str] = []
    for item in structure:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _normalize_suggested_backend_structure(parsed: dict, suggested: Any) -> list[str]:
    hint = _build_backend_structure_hint(parsed)
    auth_type = str(parsed.get("auth_type", "none") or "none").lower()
    external_services = [str(item).lower() for item in (parsed.get("external_services") or [])]
    env_vars = parsed.get("env_vars_needed") or []
    needs_db = _needs_database(parsed)

    incoming: list[str] = []
    if isinstance(suggested, list):
        for item in suggested:
            if isinstance(item, str) and item.strip():
                incoming.append(item.strip())

    has_specific_models = any(item.endswith("_models.py") for item in incoming + hint)
    has_specific_routes = any(item.endswith("_routes.py") for item in incoming + hint)
    has_specific_services = any(item.endswith("_service.py") for item in incoming + hint)

    blocked = {"crud.py", "services.py"}
    if has_specific_models:
        blocked.update({"models.py", "schemas.py"})
    if has_specific_routes:
        blocked.add("routes.py")
    if has_specific_services:
        blocked.add("service.py")
    if not needs_db:
        blocked.add("database.py")
    if auth_type in {"", "none", "null"}:
        blocked.add("auth.py")
    if not (env_vars or external_services):
        blocked.add(".env.example")

    normalized: list[str] = []
    for item in incoming + hint:
        if item in blocked:
            continue
        if item not in normalized:
            normalized.append(item)

    return normalized

@app.post("/list_frontend_files")
async def list_frontend_files_endpoint(req: ListFrontendRequest):
    """
    Fast endpoint - just lists all frontend source files without analysis.
    Returns immediately so the UI can show the file list before AI processing begins.
    """
    try:
        inline_files = _collect_frontend_request_files(req.files, req.frontend_subdir)
        if inline_files:
            files = [
                {"path": rel_path, "size": len(content)}
                for rel_path, content in sorted(
                    inline_files.items(),
                    key=lambda item: (
                        0 if any(item[0].endswith(name) for name in FRONTEND_PRIORITY) else 1,
                        item[0]
                    )
                )
            ]
            return {"files": files, "base_path": req.frontend_subdir or "."}

        base = Path(req.project_path)
        if req.frontend_subdir:
            base = base / req.frontend_subdir
        if not base.exists():
            return {"files": [], "project_path": str(base)}

        files: list = []

        for entry in _collect_frontend_source_files(base):
            rel = str(entry.relative_to(base))
            size = entry.stat().st_size
            files.append({"path": rel, "size": size})

        return {"files": files, "base_path": str(base)}

    except Exception as e:
        print(f"Error in /list_frontend_files: {e}")
        return {"files": [], "error": str(e)}

# ------------------------------------------------------------
# Frontend Code Analysis Endpoint
# Scans existing frontend code so AI can write a matching backend.
# Called when user says "write backend for worktual_chat".
# Reads all .ts/.tsx/.js/.jsx files, extracts:
#   - API endpoint calls (fetch/axios URLs)
#   - Auth patterns (JWT, session, OAuth)
#   - TypeScript interfaces / data models
#   - WebSocket usage
#   - Environment variables referenced
# Returns structured JSON the AI uses to generate backend code.
# ------------------------------------------------------------
class AnalyzeFrontendRequest(BaseModel):
    project_path: str
    frontend_subdir: str = ""
    user_request: str = ""
    files: Dict[str, str] = Field(default_factory=dict)


def _collect_frontend_request_files(
    files: Optional[Dict[str, str]],
    frontend_subdir: str = "",
) -> Dict[str, str]:
    normalized_frontend = _normalize_rel_path(frontend_subdir)
    collected: Dict[str, str] = {}

    for raw_path, raw_content in (files or {}).items():
        rel_path = _normalize_rel_path(raw_path)
        if not rel_path:
            continue

        rel_parts = Path(rel_path).parts
        if set(rel_parts) & FRONTEND_SKIP_DIRS:
            continue

        if normalized_frontend:
            prefix = normalized_frontend + "/"
            if rel_path == normalized_frontend:
                continue
            if rel_path.startswith(prefix):
                rel_path = rel_path[len(prefix):]
            else:
                continue

        if not rel_path:
            continue

        if Path(rel_path).suffix.lower() not in FRONTEND_EXTS:
            continue

        collected[rel_path] = str(raw_content or "")

    return collected
# async def analyze_frontend_endpoint(req: AnalyzeFrontendRequest):
#     try:
#         if not check_gemini_available():
#             raise HTTPException(status_code=503, detail="Gemini API unavailable")

#         base = Path(req.project_path)
#         if req.frontend_subdir:
#             base = base / req.frontend_subdir
#         if not base.exists():
#             raise HTTPException(status_code=404, detail="Path not found: " + str(base))

#         files_read: Dict[str, str] = {}
#         total_chars = 0
#         CHAR_LIMIT = 60000

#         for entry in _collect_frontend_source_files(base):
#             rel = str(entry.relative_to(base))
#             try:
#                 text = entry.read_text(encoding="utf-8", errors="ignore")
#                 if total_chars + len(text) > CHAR_LIMIT:
#                     break
#                 files_read[rel] = text
#                 total_chars += len(text)
#             except Exception:
#                 pass

#         if not files_read:
#             raise HTTPException(status_code=404, detail="No frontend source files found")

#         sorted_files = sorted(
#             files_read.items(),
#             key=lambda x: (0 if any(x[0].endswith(p) for p in FRONTEND_PRIORITY + ["index.ts"]) else 1, x[0])
#         )
#         files_list = [rel for rel, _ in sorted_files]

#         files_block = ""
#         for rel_path, text in sorted_files:
#             files_block += "\n\n===== " + rel_path + " =====\n" + text[:3000]

#         schema = (
#             "{\n"
#             '  "project_goal": "1-2 sentence description of the project",\n'
#             '  "api_endpoints": [{"method": "GET", "path": "/api/x", "description": "...", "auth_required": true}],\n'
#             '  "auth_type": "jwt or session or oauth or none",\n'
#             '  "websocket_endpoints": ["/ws/chat"],\n'
#             '  "data_models": [{"name": "User", "fields": [{"name": "id", "type": "string"}]}],\n'
#             '  "external_services": ["worktual API"],\n'
#             '  "env_vars_needed": ["WORKTUAL_API_KEY", "DATABASE_URL"],\n'
#             '  "suggested_backend_structure": ["main.py", "models.py", "auth.py", "requirements.txt"]\n'
#             "}"
#         )

#         prompt = (
#             "You are a senior backend engineer. Analyze this frontend codebase line by line.\n"
#             "Understand the project goal, all API calls made, auth patterns, data models, and env vars.\n"
#             "Then return ONLY a JSON object matching this schema (no markdown, no explanation):\n"
#             + schema
#             + "\n\nFRONTEND SOURCE FILES:"
#             + files_block
#         )

#         response = client.models.generate_content(
#             model=GEMINI_MODEL,
#             contents=prompt,
#             config={"max_output_tokens": 4000, "temperature": 0.1}
#         )
#         raw = response.text.strip()

#         if raw.startswith("```"):
#             lines = raw.split("\n")
#             if lines[0].startswith("```"):
#                 lines = lines[1:]
#             if lines and lines[-1].strip() == "```":
#                 lines = lines[:-1]
#             raw = "\n".join(lines).strip()

#         brace = raw.find("{")
#         if brace > 0:
#             raw = raw[brace:]

#         try:
#             parsed = json.loads(raw)
#         except json.JSONDecodeError:
#             try:
#                 parsed = json.loads(_sanitize_json_string(raw))
#             except Exception:
#                 m = re.search(r"\{[\s\S]*\}", raw)
#                 parsed = json.loads(_sanitize_json_string(m.group())) if m else {}

#         return {"files_analyzed": len(files_read), "files_list": files_list, "analysis": parsed}

#     except HTTPException:
#         raise
#     except Exception as e:
#         print("Error in /analyze_frontend: " + str(e))
#         raise HTTPException(status_code=500, detail=str(e))
# ─── REPLACE the /analyze_frontend endpoint in backend.py ────────────────────
# Fixes:
#   1. Truncation was 3000 chars/file — most fetch/axios calls were cut off.
#      New limit: 8000 chars/file, total 120000 chars.
#   2. A regex pre-pass extracts all actual fetch/axios/WebSocket URLs BEFORE
#      sending to Gemini, so the AI sees the real endpoints even in large files.
#   3. The prompt now includes the pre-extracted URLs as ground truth so Gemini
#      cannot hallucinate or miss endpoints.
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/analyze_frontend")
async def analyze_frontend_endpoint(req: AnalyzeFrontendRequest):
    try:
        if not check_gemini_available():
            raise HTTPException(status_code=503, detail="Gemini API unavailable")

        files_read: Dict[str, str] = {}
        total_chars = 0
        # Keep frontend analysis affordable by default; both limits are env-configurable.
        CHAR_LIMIT     = ANALYZE_FRONTEND_CONTEXT_CHAR_LIMIT
        PER_FILE_LIMIT = ANALYZE_FRONTEND_PER_FILE_CHAR_LIMIT

        inline_files = _collect_frontend_request_files(req.files, req.frontend_subdir)
        if inline_files:
            for rel, text in sorted(
                inline_files.items(),
                key=lambda item: (
                    0 if any(item[0].endswith(name) for name in FRONTEND_PRIORITY + ["index.ts"]) else 1,
                    item[0]
                )
            ):
                if total_chars + len(text) > CHAR_LIMIT:
                    break
                files_read[rel] = text
                total_chars += len(text)
            base_label = req.frontend_subdir or req.project_path or "inline_frontend_files"
        else:
            base = Path(req.project_path)
            if req.frontend_subdir:
                base = base / req.frontend_subdir
            if not base.exists():
                raise HTTPException(status_code=404, detail="Path not found: " + str(base))

            for entry in _collect_frontend_source_files(base):
                rel = str(entry.relative_to(base))
                try:
                    text = entry.read_text(encoding="utf-8", errors="ignore")
                    if total_chars + len(text) > CHAR_LIMIT:
                        break
                    files_read[rel] = text
                    total_chars += len(text)
                except Exception:
                    pass
            base_label = str(base)

        if not files_read:
            raise HTTPException(status_code=404, detail="No frontend source files found")

        sorted_files = sorted(
            files_read.items(),
            key=lambda x: (0 if any(x[0].endswith(p) for p in FRONTEND_PRIORITY + ["index.ts"]) else 1, x[0])
        )
        files_list = [rel for rel, _ in sorted_files]

        # FIX 2: regex pre-pass — extract every real fetch/axios/ws call before AI
        # These become "ground truth" URLs that are injected into the prompt so
        # Gemini cannot miss or invent endpoints.
        _URL_PATTERNS = [
            # axios.get('/api/foo'), client.post('/auth/login'), fetch('/api/x')
            re.compile(r"""(?:axios|client|api|http|fetch)\s*\.\s*(?:get|post|put|patch|delete|request)\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
            # fetch('/auth/login')
            re.compile(r"""fetch\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
            # fetch(`${API_BASE_URL}/generate`) / axios.get(`${BACKEND}/preview`)
            re.compile(r"""(?:fetch|axios\s*\.\s*(?:get|post|put|patch|delete|request)|client\s*\.\s*(?:get|post|put|patch|delete|request))\s*\(\s*`[^`]*\$\{[^}]+\}([^`]+)`""", re.IGNORECASE),
            # axios({ url: '/api/foo' })
            re.compile(r"""url\s*:\s*['"`]([/][^'"`\s]+)['"`]""", re.IGNORECASE),
            # url: `${API_BASE_URL}/project/${id}/fix-build`
            re.compile(r"""url\s*:\s*`[^`]*\$\{[^}]+\}([^`]+)`""", re.IGNORECASE),
            # baseURL: 'http://localhost:8888'  or  baseURL: '/api'
            re.compile(r"""baseURL\s*:\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
            # const API_BASE_URL = 'http://localhost:8025'
            re.compile(r"""(?:const|let|var)\s+(?:API_BASE_URL|BACKEND)\s*=\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
            # new WebSocket('ws://localhost:...')
            re.compile(r"""(?:new\s+WebSocket|io\s*\()\s*\(\s*['"`]([^'"`\s]+)['"`]""", re.IGNORECASE),
            # proxy target in vite.config  target: 'http://localhost:8888'
            re.compile(r"""target\s*:\s*['"`](http[^'"`\s]+)['"`]""", re.IGNORECASE),
        ]

        extracted_calls: list[str] = []
        for rel_path, content in sorted_files:
            for pat in _URL_PATTERNS:
                for m in pat.finditer(content):
                    url = m.group(1).strip()
                    if url and url not in extracted_calls:
                        extracted_calls.append(f"{rel_path}: {url}")

        extracted_calls_block = (
            "\n\nPRE-EXTRACTED API CALLS (regex-found, use these as ground truth):\n"
            + "\n".join(extracted_calls[:80])
        ) if extracted_calls else ""

        # FIX 3: per-file cap raised to 8000 chars
        files_block = ""
        for rel_path, text in sorted_files:
            files_block += "\n\n===== " + rel_path + " =====\n" + text[:PER_FILE_LIMIT]
            if len(text) > PER_FILE_LIMIT:
                files_block += f"\n... [{len(text) - PER_FILE_LIMIT} chars truncated] ..."

        schema = (
            "{\n"
            '  "project_goal": "1-2 sentence description of the project",\n'
            '  "api_endpoints": [{"method": "GET", "path": "/api/x", "description": "...", "auth_required": true}],\n'
            '  "auth_type": "jwt or session or oauth or none",\n'
            '  "websocket_endpoints": ["/ws/chat"],\n'
            '  "data_models": [{"name": "User", "fields": [{"name": "id", "type": "string"}]}],\n'
            '  "external_services": ["worktual API"],\n'
            '  "env_vars_needed": ["WORKTUAL_API_KEY", "DATABASE_URL"],\n'
            '  "suggested_backend_structure": ["main.py", "requirements.txt", "crop_routes.py", "services/nlp_service.py"]\n'
            "}"
        )

        prompt = (
            "You are a senior backend engineer. Analyze this frontend codebase carefully.\n"
            "PAY SPECIAL ATTENTION to the PRE-EXTRACTED API CALLS section — those are the "
            "REAL endpoints the frontend calls. Every URL listed there MUST appear in "
            "api_endpoints with the correct HTTP method.\n"
            "This frontend may be a monolithic builder/generator app in App.jsx with project/session-driven workflows, "
            "preview streaming, build repair, Figma import, and project file editing. Infer the backend from those real flows, "
            "not from generic CRUD templates.\n"
            "The suggested backend structure must be requirement-driven, not template-driven.\n"
            "List only the backend files the project actually needs.\n"
            "Avoid defaulting to the same scaffold every time.\n"
            "Do not blindly return models.py, auth.py, database.py, or .env.example unless the frontend analysis clearly requires them.\n"
            "Prefer domain-specific names such as crop_routes.py, farm_records_models.py, irrigation_service.py, or nlp_service.py when the domain is clear.\n"
            "Also scan the full file contents below for any additional fetch/axios/WebSocket calls.\n"
            "Return ONLY a JSON object matching this schema (no markdown, no explanation):\n"
            + schema
            + extracted_calls_block
            + "\n\nFRONTEND SOURCE FILES:"
            + files_block
        )

        response = generate_content_logged(
            label="analyze_frontend",
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": 6000, "temperature": 0.1},
            user_query=req.user_request or base_label,
            extra=f"files={len(files_read)}",
        )
        raw = response.text.strip()

        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        brace = raw.find("{")
        if brace > 0:
            raw = raw[brace:]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(_sanitize_json_string(raw))
            except Exception:
                m = re.search(r"\{[\s\S]*\}", raw)
                parsed = json.loads(_sanitize_json_string(m.group())) if m else {}

        parsed["suggested_backend_structure"] = _normalize_suggested_backend_structure(
            parsed,
            parsed.get("suggested_backend_structure"),
        )

        # Attach the pre-extracted raw calls so the caller can include them too
        parsed["_raw_api_calls"] = extracted_calls

        return {"files_analyzed": len(files_read), "files_list": files_list, "analysis": parsed}

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /analyze_frontend: " + str(e))
        raise HTTPException(status_code=500, detail=str(e))


class ValidateBackendIntegrationRequest(BaseModel):
    project_path: str
    backend_subdir: str = "backend"
    frontend_subdir: str = ""
    planned_files: List[str] = []
    analysis: Dict[str, Any] = {}


_BACKEND_VALIDATION_SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "__pycache__", ".venv", "venv",
    ".next", "out", "migrations", "alembic", ".pytest_cache",
}
_BACKEND_ROUTE_DECORATOR_RE = re.compile(
    r"@(?P<target>[A-Za-z_][A-Za-z0-9_]*)\.(?P<method>get|post|put|patch|delete|options|head)\s*\(\s*['\"](?P<path>[^'\"]+)['\"]",
    re.IGNORECASE,
)
_BACKEND_API_ROUTE_RE = re.compile(
    r"@(?P<target>[A-Za-z_][A-Za-z0-9_]*)\.api_route\s*\(\s*['\"](?P<path>[^'\"]+)['\"][\s\S]*?methods\s*=\s*\[(?P<methods>[^\]]+)\]",
    re.IGNORECASE,
)
_BACKEND_ROUTER_PREFIX_RE = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*APIRouter\s*\([\s\S]*?prefix\s*=\s*['\"](?P<prefix>[^'\"]+)['\"]",
    re.IGNORECASE,
)


def _collect_backend_source_files(base: Path) -> list[Path]:
    if not base.exists():
        return []

    files: list[Path] = []
    for entry in sorted(base.rglob("*")):
        try:
            rel_parts = entry.relative_to(base).parts
        except Exception:
            continue
        if set(rel_parts) & _BACKEND_VALIDATION_SKIP_DIRS:
            continue
        if not entry.is_file():
            continue
        files.append(entry)
    return files


def _normalize_http_path(raw_path: str) -> str:
    value = str(raw_path or "").strip()
    if not value:
        return ""

    value = re.sub(r"^https?://[^/]+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^wss?://[^/]+", "", value, flags=re.IGNORECASE)
    value = value.split("?")[0].split("#")[0]

    if value.startswith("`"):
        value = value.strip("`")
    if not value.startswith("/"):
        value = "/" + value.lstrip("./")

    value = re.sub(r"/{2,}", "/", value)
    if len(value) > 1 and value.endswith("/"):
        value = value[:-1]
    return value


def _route_match_candidates(raw_path: str) -> list[str]:
    normalized = _normalize_http_path(raw_path)
    if not normalized:
        return []

    candidates = [normalized]
    if normalized.startswith("/api/"):
        candidates.append(normalized[4:] or "/")
    elif normalized != "/api":
        candidates.append(f"/api{normalized}")

    deduped: list[str] = []
    for item in candidates:
        clean = _normalize_http_path(item)
        if clean and clean not in deduped:
            deduped.append(clean)
    return deduped


def _join_route_prefix(prefix: str, route_path: str) -> str:
    prefix_norm = _normalize_http_path(prefix or "")
    route_norm = _normalize_http_path(route_path or "/")

    if route_norm == "/":
        return prefix_norm or "/"
    if not prefix_norm or route_norm.startswith(prefix_norm):
        return route_norm
    return _normalize_http_path(f"{prefix_norm}/{route_norm.lstrip('/')}")


def _extract_backend_routes_from_text(text: str) -> list[Dict[str, str]]:
    routes: list[Dict[str, str]] = []
    prefixes: Dict[str, str] = {}

    for match in _BACKEND_ROUTER_PREFIX_RE.finditer(text):
        prefixes[match.group("name")] = match.group("prefix")

    for match in _BACKEND_ROUTE_DECORATOR_RE.finditer(text):
        target = match.group("target")
        method = match.group("method").upper()
        route_path = match.group("path")
        full_path = route_path if target == "app" else _join_route_prefix(prefixes.get(target, ""), route_path)
        routes.append({"method": method, "path": _normalize_http_path(full_path)})

    for match in _BACKEND_API_ROUTE_RE.finditer(text):
        target = match.group("target")
        route_path = match.group("path")
        methods_blob = match.group("methods")
        methods = re.findall(r"['\"]([A-Za-z]+)['\"]", methods_blob)
        for method in methods:
            full_path = route_path if target == "app" else _join_route_prefix(prefixes.get(target, ""), route_path)
            routes.append({"method": method.upper(), "path": _normalize_http_path(full_path)})

    deduped: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in routes:
        key = (item["method"], item["path"])
        if item["path"] and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _detect_backend_placeholder_reason(relative_path: str, content: str) -> Optional[str]:
    normalized_path = str(relative_path or "").replace("\\", "/")
    normalized_content = str(content or "").replace("\r\n", "\n").strip()
    if not normalized_content:
        return "file content was empty"

    base_name = Path(normalized_path).name.lower()
    lines = [line.strip() for line in normalized_content.split("\n") if line.strip()]
    non_comment_lines = [
        line for line in lines
        if not line.startswith("#") and not line.startswith("//")
    ]
    lower = normalized_content.lower()

    placeholder_patterns = [
        r"models?\s+are\s+defined\s+in",
        r"routes?\s+are\s+defined\s+in",
        r"schemas?\s+are\s+defined\s+in",
        r"implementation\s+goes\s+here",
        r"add\s+logic\s+here",
        r"\bplaceholder\b",
        r"\bstub\b",
        r"\btodo\b",
        r"\bcoming\s+soon\b",
    ]
    if any(re.search(pattern, lower, re.IGNORECASE) for pattern in placeholder_patterns):
        return "file contains placeholder text instead of working code"

    if base_name == "__init__.py":
        return None

    if base_name.endswith("requirements.txt"):
        deps = [line for line in lines if not line.startswith("#")]
        return None if deps else "requirements file had no dependencies"

    if base_name.startswith(".env"):
        env_lines = [line for line in lines if re.match(r"^[A-Z0-9_]+=", line)]
        return None if env_lines else "environment file had no variables"

    if not normalized_path.endswith(".py"):
        return None if non_comment_lines else "file had no meaningful content"

    has_router = "APIRouter(" in normalized_content
    has_route_decorator = bool(_BACKEND_ROUTE_DECORATOR_RE.search(normalized_content) or _BACKEND_API_ROUTE_RE.search(normalized_content))
    has_fastapi_app = "FastAPI(" in normalized_content
    has_function = bool(re.search(r"\b(?:async\s+def|def)\s+\w+\s*\(", normalized_content))
    has_class = bool(re.search(r"\bclass\s+\w+\s*[:(]", normalized_content))
    has_model_signals = bool(re.search(r"\bColumn\s*\(|\brelationship\s*\(|\bBaseModel\b|\bField\s*\(|\bSQLModel\b|\bMapped\s*\[", normalized_content))
    has_database_signals = bool(re.search(r"\bcreate_engine\s*\(|\bsessionmaker\s*\(|\bSessionLocal\b|\bdeclarative_base\s*\(", normalized_content))
    import_only = bool(re.fullmatch(r"(?:(?:from|import)\s[^\n]+\n?)+", normalized_content + "\n"))

    if import_only:
        return "file only contained imports"
    if len(non_comment_lines) <= 2 and not (has_function or has_class or has_fastapi_app or has_database_signals):
        return "file was too small to contain working logic"

    if "/routes/" in normalized_path or base_name.endswith("_routes.py") or "route" in base_name:
        if has_router and not has_route_decorator:
            return "route file declared a router but no endpoints"
        if not has_router and not has_route_decorator:
            return "route file did not include routing logic"
        return None

    if "/models/" in normalized_path or "/schemas/" in normalized_path or "model" in base_name or "schema" in base_name:
        if not has_class or not has_model_signals:
            return "model/schema file did not include real definitions"
        return None

    if base_name in {"main.py", "app.py"} and not has_fastapi_app:
        return "entry file did not define a FastAPI application"

    if any(token in normalized_path for token in ["/services/", "/utils/", "/core/", "/config/"]) or base_name in {"database.py", "auth.py"}:
        if not (has_function or has_class or has_fastapi_app or has_database_signals):
            return "backend support file did not include working implementation"

    return None


def _extract_expected_frontend_endpoints(analysis: Dict[str, Any]) -> list[Dict[str, str]]:
    expected: list[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for endpoint in analysis.get("api_endpoints") or []:
        if not isinstance(endpoint, dict):
            continue
        method = str(endpoint.get("method", "") or "GET").upper()
        raw_path = str(endpoint.get("path", "") or "")
        for candidate in _route_match_candidates(raw_path):
            key = (method, candidate)
            if key not in seen:
                seen.add(key)
                expected.append({"method": method, "path": candidate})

    for raw_entry in analysis.get("_raw_api_calls") or []:
        if not isinstance(raw_entry, str):
            continue
        raw_path = raw_entry.split(":", 1)[-1].strip()
        for candidate in _route_match_candidates(raw_path):
            key = ("ANY", candidate)
            if key not in seen:
                seen.add(key)
                expected.append({"method": "ANY", "path": candidate})

    return expected


def _find_frontend_connection_signals(frontend_base: Path) -> Dict[str, Any]:
    result = {
        "proxy_files": [],
        "api_client_files": [],
        "warnings": [],
    }

    if not frontend_base.exists():
        result["warnings"].append("Frontend directory was not found for connectivity validation.")
        return result

    vite_candidates = [frontend_base / "vite.config.ts", frontend_base / "vite.config.js"]
    for vite_file in vite_candidates:
        if not vite_file.exists():
            continue
        try:
            text = vite_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if "proxy" in text and "/api" in text:
            result["proxy_files"].append(str(vite_file.relative_to(frontend_base)))

    api_client_patterns = [
        "src/api/client.ts", "src/api/client.js",
        "src/services/api.ts", "src/services/api.js",
        "src/hooks/useIntegration.ts", "src/hooks/useIntegration.js",
    ]
    for relative in api_client_patterns:
        candidate = frontend_base / relative
        if candidate.exists():
            result["api_client_files"].append(relative)

    if not result["proxy_files"]:
        result["warnings"].append("No Vite proxy file with /api forwarding was detected.")
    if not result["api_client_files"]:
        result["warnings"].append("No dedicated frontend API client or integration hook file was detected.")

    return result


@app.post("/validate_backend_integration")
async def validate_backend_integration_endpoint(req: ValidateBackendIntegrationRequest):
    try:
        project_root = Path(req.project_path)
        backend_base = project_root / req.backend_subdir if req.backend_subdir else project_root
        frontend_base = project_root / req.frontend_subdir if req.frontend_subdir else project_root

        if not project_root.exists():
            raise HTTPException(status_code=404, detail="Project path not found")
        if not backend_base.exists():
            raise HTTPException(status_code=404, detail="Backend path not found")

        backend_files = _collect_backend_source_files(backend_base)
        backend_file_payloads: list[Dict[str, Any]] = []
        syntax_errors: list[Dict[str, str]] = []
        weak_files: list[Dict[str, str]] = []
        discovered_routes: list[Dict[str, str]] = []

        for file_path in backend_files:
            rel_backend = str(file_path.relative_to(backend_base)).replace("\\", "/")
            rel_project = str(file_path.relative_to(project_root)).replace("\\", "/")
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""

            backend_file_payloads.append({"path": rel_project, "size": len(text)})

            weak_reason = _detect_backend_placeholder_reason(rel_backend, text)
            if weak_reason:
                weak_files.append({"path": rel_project, "reason": weak_reason})

            if file_path.suffix.lower() == ".py":
                syntax_error, _ = validate_python_code(text, rel_project)
                if syntax_error:
                    syntax_errors.append({"path": rel_project, "error": syntax_error})
                for route in _extract_backend_routes_from_text(text):
                    discovered_routes.append({
                        "method": route["method"],
                        "path": route["path"],
                        "source_file": rel_project,
                    })

        expected_endpoints = _extract_expected_frontend_endpoints(req.analysis or {})

        matched_endpoints: list[Dict[str, str]] = []
        missing_endpoints: list[Dict[str, str]] = []
        for expected in expected_endpoints:
            candidates = _route_match_candidates(expected["path"])
            match = next(
                (
                    route for route in discovered_routes
                    if route["path"] in candidates
                    and (expected["method"] == "ANY" or route["method"] == expected["method"])
                ),
                None,
            )
            if match:
                matched_endpoints.append({
                    "method": expected["method"],
                    "path": expected["path"],
                    "source_file": match["source_file"],
                })
            else:
                missing_endpoints.append(expected)

        missing_planned_files: list[str] = []
        for planned in req.planned_files or []:
            clean = str(planned or "").replace("\\", "/").strip()
            if not clean:
                continue
            if not (project_root / clean).exists():
                missing_planned_files.append(clean)

        connection_signals = _find_frontend_connection_signals(frontend_base)
        route_files = [
            item["path"] for item in backend_file_payloads
            if "/routes/" in item["path"] or item["path"].endswith("_routes.py") or item["path"].endswith("/main.py")
        ]

        repair_targets: list[str] = []
        for item in missing_planned_files:
            if item not in repair_targets:
                repair_targets.append(item)
        for item in weak_files:
            if item["path"] not in repair_targets:
                repair_targets.append(item["path"])
        for item in syntax_errors:
            if item["path"] not in repair_targets:
                repair_targets.append(item["path"])
        if missing_endpoints:
            for candidate in route_files[:8] or [f"{req.backend_subdir}/main.py".strip("/")]:
                if candidate not in repair_targets:
                    repair_targets.append(candidate)

        critical_issue_count = (
            len(missing_planned_files)
            + len(weak_files)
            + len(syntax_errors)
            + len(missing_endpoints)
        )
        status = "pass" if critical_issue_count == 0 else "needs_attention"

        summary_parts = [
            f"checked {len(backend_file_payloads)} backend file(s)",
            f"matched {len(matched_endpoints)}/{len(expected_endpoints)} frontend endpoint(s)",
        ]
        if syntax_errors:
            summary_parts.append(f"{len(syntax_errors)} syntax issue(s)")
        if weak_files:
            summary_parts.append(f"{len(weak_files)} incomplete file(s)")
        if missing_planned_files:
            summary_parts.append(f"{len(missing_planned_files)} missing planned file(s)")

        return {
            "status": status,
            "summary": ", ".join(summary_parts),
            "backend_files_checked": backend_file_payloads,
            "missing_planned_files": missing_planned_files,
            "empty_or_placeholder_files": weak_files,
            "syntax_errors": syntax_errors,
            "expected_frontend_endpoints": expected_endpoints,
            "matched_endpoints": matched_endpoints,
            "missing_endpoints": missing_endpoints,
            "connection_warnings": connection_signals["warnings"],
            "frontend_proxy_files": connection_signals["proxy_files"],
            "frontend_api_client_files": connection_signals["api_client_files"],
            "repair_targets": repair_targets,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /validate_backend_integration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ─── Inline Code Completion ───────────────────────────────────────────────────

class CompletionRequest(BaseModel):
    prefix: str              # code before cursor
    suffix: str = ""         # code after cursor (next ~10 lines)
    language: str = ""
    filename: str = ""
    # FIX 3: workspace file-tree summary sent from extension.ts.
    # Gives the model context about which modules/classes exist in the project,
    # even on the very first keystroke in a brand-new project.
    project_context: str = ""
    # FIX 2: True when the TS client already detected a block-opener pattern.
    # When set we skip _wants_block() and jump straight to block-mode prompting.
    block_hint: bool = False


# ── Algorithm keywords that always trigger full-block mode ────────────────────
_ALGO_KEYWORDS = [
    "armstrong", "fibonacci", "factorial", "palindrome", "prime", "bubble",
    "selection", "insertion", "merge", "quick", "binary_search", "linear_search",
    "binary search", "linear search", "stack", "queue", "linked_list", "linked list",
    "tree", "graph", "bfs", "dfs", "dijkstra", "knapsack", "caesar", "hash",
    "anagram", "sorting", "recursion", "permutation", "combination",
]


def _wants_block(prefix: str) -> bool:
    """
    Returns True when the prefix strongly suggests the user wants a full
    function/class body generated — not just a single-line completion.

    FIX 2 additions vs. original:
      • Detects cursor-on-blank-line after a block opener (most common missed case)
      • Catches partial def/class with unclosed paren more reliably
      • Catches Python decorators (@decorator before def/class)
      • Catches JS/TS interface, enum, abstract class blocks
      • Catches partial arrow functions with no closing paren
      • Searches last 6 non-empty lines (was 5)
      • Checks last 10 lines for algo keywords (was 8)
    """
    lines = prefix.split("\n")

    # Last 6 non-empty lines (was 5)
    candidates = [l.rstrip() for l in lines if l.strip()][-6:]

    for line in candidates:
        stripped = line.strip()

        # ── Python ──────────────────────────────────────────────────────────
        # def / async def ending with ":"
        if re.match(r"^(async\s+)?def\s+\w", stripped) and stripped.endswith(":"):
            return True
        # class ending with ":"
        if re.match(r"^class\s+\w", stripped) and stripped.endswith(":"):
            return True
        # def/async def with open paren but no closing paren (mid-signature pause)
        if re.match(r"^(async\s+)?def\s+\w+\s*\(", stripped) and ")" not in stripped:
            return True
        # Decorator above a function/class — user is about to type the signature
        if re.match(r"^@\w", stripped):
            return True

        # ── JavaScript / TypeScript ──────────────────────────────────────────
        # Named function ending with {
        if re.search(r"\bfunction\s+\w+", stripped) and stripped.rstrip().endswith("{"):
            return True
        # Arrow function: ) => { or ) => (start of multiline body)
        if re.search(r"\)\s*=>\s*[\({]?\s*$", stripped):
            return True
        # interface / enum / class / abstract class ending with {
        if (re.match(r"^(export\s+)?(default\s+)?(interface|enum|class|abstract\s+class)\s+\w", stripped)
                and stripped.rstrip().endswith("{")):
            return True
        # Any line ending with { that is not trivially short
        if stripped.rstrip().endswith("{") and len(stripped) > 4:
            return True
        # Any line ending with : that looks like a signature (not a comment or ternary)
        if (stripped.endswith(":")
                and len(stripped) > 5
                and not stripped.startswith("//")
                and not stripped.startswith("#")):
            return True

    # ── FIX 2 KEY ADDITION: cursor on blank line after a block opener ─────────
    # Most common real-world case: user types "def foo():\n" then pauses.
    # The last line is blank, but the line before it is the block opener.
    last_actual = lines[-1].strip() if lines else ""
    if last_actual == "" and candidates:
        opener = candidates[-1].strip()
        is_opener = (
            (re.match(r"^(async\s+)?def\s+\w", opener) and opener.endswith(":"))
            or (re.match(r"^class\s+\w", opener) and opener.endswith(":"))
            or opener.rstrip().endswith("{")
        )
        if is_opener:
            return True

    # ── Algorithm keywords in last 10 lines (was 8) ──────────────────────────
    recent = "\n".join(lines[-10:]).lower()
    if any(kw in recent for kw in _ALGO_KEYWORDS):
        return True

    return False

# def print_1_to10(start: int, end: int):
#     if start>end:
#         return
#     print(start)
#     print_1_to10(start+1,end)
# start = int(input("Enter the starting number: "))
# end = int(input("Enter the ending number: "))
# print_1_to10(start,end)

# @app.post("/complete")
# async def complete_endpoint(req: CompletionRequest):
#     """
#     Smart Dev inline code completion — optimised for speed + multiline.

#     You must be provide suggestions based on the below usecases.

#     USECASES:
#     ==========

#        1) Based on the Files name.
#        2) Based on the project name and check the files name present inside that folder.
#        3) Based on the class names provide the suggestions.
#        4) Based on the function name provide the suggestions.
#        5) Based on the errors provide the suggestions.
#        6) Based on the incomplete codes provide the suggestion.

#     While provide the suggestion you must check the existing code is completed or not, If the code not completed mean you must complete the code with proper struture.
#     Ensure that the code is properly indented and follow the standard coding conventions.
#     If the code is completed mean you must provide the suggestion based on the existing code and also check the code is having any error or not, if the code is having any error mean you must provide the suggestion to fix the error and also provide the suggestion to improve the code quality.
#     If the code is having any incomplete code mean you must provide the suggestion to complete the code with proper structure and also provide the suggestion to improve the code quality.
#     Example:
#     =========
#     num = int(input("Enter a number: "))
#     n = num
#     power = len(str(num))
#     total = 0

#     while n > 0:
#         digit = n % 10
#         total += digit ** power
#         n //= 10

#     if total == num:
#         print("Armstrong Number")
#     else:
#         print("Not an Armstrong Number")

#     FIX 1 — SPEED:
#       Uses COMPLETION_MODEL (default: gemini-2.5-flash) instead of GEMINI_MODEL.
#       Prefix capped at last 40 lines (was 80) — primary remaining latency driver.
#       Block token budget: 600 (was 1500). Still enough for ~50 lines.
#       Short token budget: 120 (was 300). Short completions never need more.

#     FIX 2 — MULTILINE:
#       block_hint=True from TS client bypasses _wants_block() — no double work.
#       _wants_block() now catches: blank-line-after-opener, decorators,
#       partial arrow functions, interface/enum/abstract class blocks, and more.
#       Block output cap raised from 80 → 120 lines.

#     FIX 3 — NEW PROJECT CONTEXT:
#       project_context carries a compact workspace file-tree summary built by
#       buildProjectContext() in extension.ts. Both prompts include it so the
#       model knows what modules/classes exist even in brand-new projects.

#     FIX 4 — PROMPT QUALITY:
#       Short prompt now uses fill-in-the-middle framing with prefix+suffix.
#       Block prompt uses explicit output contract with a typed example.
#       Stop sequences tightened for short mode.
#       Post-processing strips leading blank lines aggressively.
# #     """
#     if not req.prefix or len(req.prefix.strip()) < 2:
#         return {"suggestion": "", "is_block": False}

#     lang_note = f"Language: {req.language}." if req.language else ""
#     file_note = f"File: {req.filename}."     if req.filename else ""

#     # Only last 40 lines as prefix context
#     prefix_lines   = req.prefix.split("\n")
#     prefix_trimmed = "\n".join(prefix_lines[-40:])
#     suffix_trimmed = "\n".join(req.suffix.split("\n")[:10]) if req.suffix else ""

#     # Optional project context block
#     ctx_section = ""
#     if req.project_context and req.project_context.strip():
#         ctx_section = (
#             f"\nPROJECT CONTEXT (workspace file tree — use for imports/references only):\n"
#             f"{req.project_context.strip()}\n"
#         )

#     # Prefer the client's block hint; fall back to our own heuristic
#     block_mode = req.block_hint or _wants_block(req.prefix)

#     if block_mode:
#         prompt = (
#             f"You are Smart Dev, an expert code completion engine.\n"
#             f"{lang_note} {file_note}{ctx_section}\n"
#             f"The developer has started writing a function, class, or algorithm.\n"
#             f"Your job: write the COMPLETE, WORKING body that goes after the cursor.\n\n"
#             f"OUTPUT CONTRACT:\n"
#             f"- Raw code ONLY. No prose, no markdown fences, no backticks.\n"
#             f"- Do NOT repeat any line already in the prefix.\n"
#             f"- Do NOT output a blank line as the very first line.\n"
#             f"- Match indentation of the prefix exactly (spaces vs tabs).\n"
#             f"- Write real logic — no stubs, no `pass`, no `# TODO`.\n"
#             f"- Finish with a proper closing token (}}, `end`, dedent, etc.) if needed.\n\n"
#             f"EXAMPLE (Python):\n"
#             f"Prefix:\n"
#             f"def add(a, b):\n"
#             f"<CURSOR>\n"
#             f"Correct output:\n"
#             f"    return a + b\n\n"
#             f"Now complete the real code below.\n\n"
#             f"Prefix:\n"
#             f"{prefix_trimmed}\n"
#             f"<CURSOR — write what comes next, starting on the very next character>\n\n"
#             f"Output (raw code only):"
#         )
#         max_tokens  = 600
#         temperature = 0.15
#         stop_seqs   = None  # Let it finish the block cleanly

#     else:
#         # Short / inline completion — fill-in-the-middle framing
#         prompt = (
#             f"You are Smart Dev, an inline code completion engine.\n"
#             f"{lang_note} {file_note}{ctx_section}\n"
#             f"Complete ONLY the missing code at <CURSOR>.\n"
#             f"The code after the cursor is already written — do not repeat it.\n\n"
#             f"OUTPUT CONTRACT:\n"
#             f"- Raw code ONLY. No prose, no markdown fences, no backticks.\n"
#             f"- Output AT MOST 5 lines. Prefer the shortest correct completion.\n"
#             f"- Do NOT repeat any code from the prefix or suffix.\n"
#             f"- Do NOT output a leading blank line.\n"
#             f"- Match existing indentation exactly.\n"
#             f"- If no useful completion exists, output an empty string.\n\n"
#             f"EXAMPLE:\n"
#             f"Prefix:  `const total = items.reduce(`\n"
#             f"Suffix:  `, 0);`\n"
#             f"Output:  `(sum, item) => sum + item.price`\n\n"
#             f"Now complete the real code below.\n\n"
#             f"Code before cursor:\n"
#             f"{prefix_trimmed}\n"
#             f"<CURSOR>\n"
#             f"Code after cursor:\n"
#             f"{suffix_trimmed}\n\n"
#             f"Output (1–5 lines, raw code only):"
#         )
#         # FIX: was incorrectly set to 600 — short completions never need more than 120
#         max_tokens  = 120
#         temperature = 0.10   # Slightly lower — short completions must be precise
#         stop_seqs   = ["\n\n", "```"]  # Stop at first blank line or accidental fence

#     try:
#         cfg = {"max_output_tokens": max_tokens, "temperature": temperature}
#         if stop_seqs:
#             cfg["stop_sequences"] = stop_seqs

#         completion_candidates = [COMPLETION_MODEL]
#         if "gemini-2.5-flash" not in completion_candidates:
#             completion_candidates.append("gemini-2.5-flash")
#         if GEMINI_MODEL not in completion_candidates:
#             completion_candidates.append(GEMINI_MODEL)

#         last_error = None
#         response = None
#         for model_name in completion_candidates:
#             try:
#                 response = generate_content_logged(
#                     label="complete",
#                     model=model_name,
#                     contents=prompt,
#                     config=cfg,
#                     user_query=f"{req.filename or 'inline completion'} | {req.language or 'unknown'}",
#                     extra=f"block={block_mode}",
#                 )
#                 break
#             except Exception as model_error:
#                 last_error = model_error
#                 if "NOT_FOUND" not in str(model_error):
#                     raise
#                 print(f"[Worktual /complete] Fallback from {model_name}: {model_error}")

#         if response is None:
#             raise last_error or RuntimeError("No completion model available")
#         raw = (response.text or "").strip()

#         # Strip accidental markdown fences
#         raw = re.sub(r"^```[\w]*\n?", "", raw)
#         raw = re.sub(r"\n?```$",        "", raw)

#         # Strip leading blank lines the model sometimes emits despite instructions
#         raw = re.sub(r"^\n+", "", raw)

#         raw = raw.strip()

#         # Enforce output line caps
#         out_lines = raw.split("\n")
#         if block_mode and len(out_lines) > 120:
#             raw = "\n".join(out_lines[:120])
#         elif not block_mode and len(out_lines) > 5:
#             raw = "\n".join(out_lines[:5])

#         print(
#             f"[Worktual /complete] {'BLOCK' if block_mode else 'SHORT'} | "
#             f"{'client-hint' if req.block_hint else 'heuristic'} | "
#             f"{req.language} | {len(raw.split(chr(10)))} lines"
#         )
#         return {"suggestion": raw, "is_block": block_mode}

#     except Exception as e:
#         print(f"[Worktual /complete] Error: {e}")
#         return {"suggestion": "", "is_block": False}

# ─── Run Project Detection Endpoints ─────────────────────────────────────────

class DetectProjectRequest(BaseModel):
    project_path: str

class DetectProjectResponse(BaseModel):
    has_backend: bool
    has_frontend: bool
    backend_dir: Optional[str] = None
    frontend_dir: Optional[str] = None
    backend_command: Optional[str] = None
    frontend_command: Optional[str] = None
    backend_port: int = 8888
    frontend_port: int = 5173
    missing_files: List[str] = []
    setup_commands: List[str] = []

@app.post("/detect_project")
async def detect_project_endpoint(req: DetectProjectRequest):
    """Analyse a project directory and return full run plan."""
    import os as _os
    project_path = req.project_path
    if not _os.path.isdir(project_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {project_path}")

    def find_dir(root, candidates):
        for c in candidates:
            d = _os.path.join(root, c)
            if _os.path.isdir(d):
                return d
        return None

    def dir_has(d, ext):
        try:
            return any(f.endswith(ext) for f in _os.listdir(d))
        except:
            return False

    backend_dir = find_dir(project_path, ["backend", "server", "api", "app"])
    if not backend_dir and dir_has(project_path, ".py"):
        backend_dir = project_path

    has_backend = backend_dir is not None
    backend_command = None
    backend_port = 8888

    if has_backend:
        main_files = ["main.py", "app.py", "run.py", "server.py"]
        main_file = next((f for f in main_files if _os.path.exists(_os.path.join(backend_dir, f))), "main.py")
        content_py = ""
        main_path = _os.path.join(backend_dir, main_file)
        if _os.path.exists(main_path):
            try:
                with open(main_path) as mf:
                    content_py = mf.read()
            except:
                pass
        port_match = re.search(r"port\s*[=:]\s*(\d{4,5})", content_py, re.IGNORECASE)
        if port_match:
            backend_port = int(port_match.group(1))
        app_var_match = re.search(r"(\w+)\s*=\s*FastAPI\s*\(", content_py)
        app_var = app_var_match.group(1) if app_var_match else "app"
        module_name = main_file.replace(".py", "")
        if "FastAPI" in content_py or "fastapi" in content_py:
            backend_command = f"uvicorn {module_name}:{app_var} --reload --port {backend_port}"
        else:
            backend_command = f"python {main_file}"

    frontend_dir = find_dir(project_path, ["frontend", "client", "web", "ui"])
    if not frontend_dir and _os.path.exists(_os.path.join(project_path, "package.json")):
        frontend_dir = project_path

    has_frontend = frontend_dir is not None
    frontend_command = None
    frontend_port = 5173

    if has_frontend:
        pkg_path = _os.path.join(frontend_dir, "package.json")
        if _os.path.exists(pkg_path):
            try:
                with open(pkg_path) as pf:
                    pkg = json.load(pf)
                scripts = pkg.get("scripts", {})
                if "dev" in scripts:
                    frontend_command = "npm run dev"
                elif "start" in scripts:
                    frontend_command = "npm start"
                    frontend_port = 3000
            except:
                frontend_command = "npm run dev"

    missing_files = []
    setup_commands = []

    if has_backend:
        req_txt = _os.path.join(backend_dir, "requirements.txt")
        if not _os.path.exists(req_txt):
            missing_files.append(_os.path.relpath(backend_dir, project_path) + "/requirements.txt")
        else:
            setup_commands.append("pip install -r " + _os.path.relpath(req_txt, project_path))

    if has_frontend and frontend_dir:
        node_modules = _os.path.join(frontend_dir, "node_modules")
        pkg_json = _os.path.join(frontend_dir, "package.json")
        if not _os.path.exists(pkg_json):
            missing_files.append(_os.path.relpath(frontend_dir, project_path) + "/package.json")
        elif not _os.path.exists(node_modules):
            setup_commands.insert(0, "cd " + _os.path.relpath(frontend_dir, project_path) + " && npm install")

    return DetectProjectResponse(
        has_backend=has_backend,
        has_frontend=has_frontend,
        backend_dir=backend_dir,
        frontend_dir=frontend_dir,
        backend_command=backend_command,
        frontend_command=frontend_command,
        backend_port=backend_port,
        frontend_port=frontend_port,
        missing_files=missing_files,
        setup_commands=setup_commands,
    )


@app.post("/fix_missing_files")
async def fix_missing_files_endpoint(req: DetectProjectRequest):
    """Auto-generate missing requirements.txt, package.json, vite.config.js."""
    project_path = req.project_path
    if not os.path.isdir(project_path):
        raise HTTPException(status_code=400, detail="Invalid project path")

    fixed = []
    STDLIB = {
        "os","sys","json","re","math","time","datetime","pathlib","typing","collections",
        "itertools","functools","contextlib","subprocess","threading","asyncio","logging",
        "warnings","inspect","hashlib","hmac","base64","uuid","enum","abc","copy",
        "dataclasses","http","urllib","email","html","xml","csv","sqlite3","struct",
        "socket","ssl","shutil","tempfile","glob","traceback","io","io"
    }
    PIP_MAP = {
        "fastapi":"fastapi","uvicorn":"uvicorn[standard]","pydantic":"pydantic",
        "sqlalchemy":"SQLAlchemy","jose":"python-jose[cryptography]","passlib":"passlib[bcrypt]",
        "dotenv":"python-dotenv","multipart":"python-multipart","aiofiles":"aiofiles",
        "starlette":"starlette","bcrypt":"bcrypt","jwt":"PyJWT","PIL":"Pillow",
        "cv2":"opencv-python","sklearn":"scikit-learn","yaml":"PyYAML",
        "requests":"requests","httpx":"httpx","aiohttp":"aiohttp","motor":"motor",
        "pymongo":"pymongo","redis":"redis","celery":"celery","alembic":"alembic",
        "stripe":"stripe","boto3":"boto3","email_validator":"email-validator",
    }
    CORE_PACKAGES = ["fastapi","uvicorn[standard]","sqlalchemy","python-dotenv","python-multipart","python-jose[cryptography]","passlib[bcrypt]"]

    for backend_name in ["backend", "server", "api"]:
        backend_dir = os.path.join(project_path, backend_name)
        if not os.path.isdir(backend_dir):
            continue
        req_txt = os.path.join(backend_dir, "requirements.txt")
        if not os.path.exists(req_txt):
            imports = set()
            for f in os.listdir(backend_dir):
                if f.endswith(".py"):
                    try:
                        fc = open(os.path.join(backend_dir, f)).read()
                        for m in re.findall(r"^(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", fc, re.MULTILINE):
                            imports.add(m)
                    except:
                        pass
            packages = sorted(set(
                PIP_MAP.get(i, i.replace("_","-"))
                for i in imports
                if i not in STDLIB and not i.startswith("_")
            ) | set(CORE_PACKAGES))
            with open(req_txt, "w") as rf:
                rf.write("\n".join(packages) + "\n")
            fixed.append(os.path.relpath(req_txt, project_path))

    for frontend_name in ["frontend", "client", "web"]:
        frontend_dir = os.path.join(project_path, frontend_name)
        if not os.path.isdir(frontend_dir):
            continue
        pkg_json = os.path.join(frontend_dir, "package.json")
        if not os.path.exists(pkg_json):
            folder_name = os.path.basename(project_path)
            pkg = {
                "name": f"{folder_name}-frontend","private": True,"version": "0.0.0","type": "module",
                "scripts": {"dev": "vite","build": "vite build","preview": "vite preview"},
                "dependencies": {"react": "^18.2.0","react-dom": "^18.2.0","react-router-dom": "^6.20.0","axios": "^1.6.0"},
                "devDependencies": {"@types/react": "^18.2.0","@types/react-dom": "^18.2.0","@vitejs/plugin-react": "^4.2.0","vite": "^5.0.0"}
            }
            with open(pkg_json, "w") as pf:
                json.dump(pkg, pf, indent=2)
            fixed.append(os.path.relpath(pkg_json, project_path))

        vite_cfg = os.path.join(frontend_dir, "vite.config.js")
        if not os.path.exists(vite_cfg):
            with open(vite_cfg, "w") as vf:
                vf.write("""import { defineConfig } from 'vite'\nimport react from '@vitejs/plugin-react'\n\nexport default defineConfig({\n  plugins: [react()],\n  server: {\n    port: 5173,\n    proxy: {\n      '/api': {\n        target: 'http://localhost:8888',\n        changeOrigin: true,\n        rewrite: (path) => path.replace(/^\\/api/, '')\n      }\n    }\n  }\n})\n""")
            fixed.append(os.path.relpath(vite_cfg, project_path))

    return {"fixed_files": fixed, "count": len(fixed)}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/health/details")
async def health_details():
    routes = sorted(
        {
            getattr(route, "path", "")
            for route in app.router.routes
            if getattr(route, "path", "")
        }
    )
    return {
        "status": "ok",
        "server_mode": SERVER_MODE,
        "route_count": len(routes),
        "routes": routes,
        "has_analyze_frontend": "/analyze_frontend" in routes,
        "has_update_project": "/update_project" in routes,
        "capabilities":{
            "inline_frontend_files":True,
            "inline_project_files":True,
            "hosted_rag_requires_local_filessystem":True,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
