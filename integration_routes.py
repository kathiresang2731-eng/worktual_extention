# ─────────────────────────────────────────────────────────────────────────────
# integration_routes.py  —  Worktual Dev Full-Stack Integration Generator
#
# POST /integrate_feature
#   Reads the user's feature request + config, asks Gemini to generate
#   a complete set of Python + React files, returns them as create_file /
#   update_file action messages (same format as /chat).
#
# Wire into backend.py:
#   from integration_routes import integration_router
#   app.include_router(integration_router)
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import re
import warnings

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from google import genai
from pydantic import BaseModel
from typing import Optional, List, Any

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", category=UserWarning, module="google")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

integration_router = APIRouter()

# ─── Models ───────────────────────────────────────────────────────────────────

class IntegrateRequest(BaseModel):
    feature_type:       str
    user_description:   str
    llm_provider:       Optional[str] = None
    llm_model:          Optional[str] = None
    llm_api_key:        Optional[str] = None
    db_type:            Optional[str] = None
    db_connection:      Optional[str] = None
    backend_port:       int = 8888
    use_websocket:      bool = False
    use_redis:          bool = False
    use_celery:         bool = False
    target_component:   Optional[str] = None
    react_src_dir:      Optional[str] = "src"
    backend_dir:        Optional[str] = "backend"
    scanned_components: List[str] = []

# ─── Feature templates ────────────────────────────────────────────────────────

FEATURE_PROMPTS = {
    "llm_chat": """
Generate a complete LLM chat integration with WebSocket streaming.

PYTHON BACKEND FILES (backend/):
1. First read the frontend files clearly and then make a plan to generate the backend.
2. After making a planner, now you have to include all the required backend files as you have read the files of frontend.
3. Now the backend file which you are generating is needs to related to the excisting frontend.
4. After genrating the backend for the excisting frontend, then connection should be intergrated between the frontend and backend.
5. If you are generating files for the excisting then you should generate the code as the files would be connected.
 
REACT FILES (src/):
7. services/api.ts — WebSocket class wrapper with connect/send/disconnect + axios for REST
8. hooks/useChat.ts — manages WS connection, messages array, sendMessage fn, loading/error state
9. {target_component} UPDATE — import useChat, render message history + input box

IMPORTANT:
- Stream tokens character by character using WebSocket text frames
- React hook auto-reconnects on disconnect
- Shared message format: {{ role: "user"|"assistant", content: string, timestamp: number }}
- CORS: allow_origins=["http://localhost:5173", "http://localhost:3000"]
""",

    "db_dashboard": """
Generate a FastAPI → Database → React dashboard data pipeline.

PYTHON BACKEND FILES (backend/):
1. main.py — FastAPI app with CORS, mount router
2. routes/dashboard.py — GET /api/dashboard/data, GET /api/dashboard/summary, POST /api/dashboard/filter
3. services/db.py — SQLAlchemy/pymongo connection, query functions, connection pooling
4. models/schemas.py — DashboardData, FilterRequest, SummaryResponse Pydantic models
5. requirements.txt — fastapi, uvicorn, sqlalchemy/pymongo, python-dotenv
6. .env.example — DATABASE_URL={db_connection}, BACKEND_PORT={port}

REACT FILES (src/):
7. services/api.ts — axios instance with base URL http://localhost:{port}, interceptors
8. hooks/useDashboardData.ts — fetches data, handles loading/error, supports filters, auto-refresh
9. {target_component} UPDATE — import hook, display data in tables/cards with loading skeleton

IMPORTANT:
- Use connection pooling in SQLAlchemy (pool_size=5)
- React hook exposes: {{ data, summary, loading, error, refetch, applyFilter }}
- Add pagination support: page, limit, total_count
""",

    "realtime_notify": """
Generate a Redis pub/sub + WebSocket real-time notification system.

PYTHON BACKEND FILES (backend/):
1. main.py — FastAPI app, WebSocket /ws/notifications, background task for Redis subscribe
2. routes/notifications.py — POST /api/notify (publish), GET /api/notifications (history)
3. services/redis_client.py — Redis connection, publish/subscribe/get_history helpers
4. models/schemas.py — Notification, NotifyRequest Pydantic models
5. requirements.txt — fastapi, uvicorn, redis, websockets, python-dotenv
6. .env.example — REDIS_URL=redis://localhost:6379, BACKEND_PORT={port}

REACT FILES (src/):
7. services/api.ts — axios base client + WebSocket wrapper
8. hooks/useNotifications.ts — WS connection, notifications array (max 50), unread count, markRead
9. {target_component} UPDATE — import hook, show notification bell with badge + dropdown list

IMPORTANT:
- Notifications shape: {{ id, title, message, type: "info"|"success"|"warning"|"error", timestamp, read }}
- Redis key: notifications:history (LPUSH, LTRIM to 100)
- WebSocket broadcasts to ALL connected clients when a new notification is published
""",

    "background_job": """
Generate a Celery background job system with progress tracking.

PYTHON BACKEND FILES (backend/):
1. main.py — FastAPI app, mount router
2. routes/jobs.py — POST /api/jobs/start, GET /api/jobs/{job_id}/status, GET /api/jobs/list
3. worker.py — Celery app, task definitions, progress updates via Redis
4. services/redis_client.py — Redis for Celery broker + result backend + progress store
5. models/schemas.py — JobRequest, JobStatus, JobResult Pydantic models
6. requirements.txt — fastapi, uvicorn, celery, redis, python-dotenv
7. .env.example — REDIS_URL=redis://localhost:6379, BACKEND_PORT={port}

REACT FILES (src/):
8. services/api.ts — axios base client
9. hooks/useBackgroundJob.ts — startJob, pollStatus (every 2s), cancel, progress %, result
10. {target_component} UPDATE — trigger job, show progress bar, display result

IMPORTANT:
- Job status: pending → running → success | failed
- Progress stored in Redis: jobs:{job_id}:progress = 0-100
- Poll via GET every 2s until done, then stop polling
- Celery result backend: redis://
""",

    "auth": """
Generate a complete JWT authentication system.

PYTHON BACKEND FILES (backend/):
1. main.py — FastAPI app with CORS, mount auth router
2. routes/auth.py — POST /api/auth/register, POST /api/auth/login, GET /api/auth/me, POST /api/auth/refresh
3. services/auth_service.py — password hashing (bcrypt), JWT create/verify, user lookup
4. services/db.py — User model + CRUD with SQLAlchemy
5. models/schemas.py — UserCreate, UserLogin, Token, UserOut Pydantic models
6. requirements.txt — fastapi, uvicorn, python-jose, passlib, bcrypt, sqlalchemy, python-dotenv
7. .env.example — JWT_SECRET=change-me-in-production, DATABASE_URL=..., BACKEND_PORT={port}

REACT FILES (src/):
8. services/api.ts — axios with Authorization header interceptor, token refresh on 401
9. hooks/useAuth.ts — login, logout, register, currentUser, isAuthenticated, loading
10. context/AuthContext.tsx — AuthProvider wrapping the app
11. {target_component} UPDATE — wrap with AuthProvider, protect routes

IMPORTANT:
- Store JWT in memory (NOT localStorage) for security — use httpOnly cookie or in-memory
- Access token: 15min expiry. Refresh token: 7 days
- Protected routes: use Depends(get_current_user) in FastAPI
""",

    "generic": """
Generate a complete frontend-backend connection for: {user_description}

FRONTEND ROOT: {frontend_root}
REACT SRC DIR: {react_src_dir}
BACKEND DIR: {backend_dir}
BACKEND PORT: {port}

PYTHON BACKEND FILES ({backend_dir}/):
1. main.py — FastAPI app with CORS and the API routes needed by the request
2. routes/api.py — all backend endpoints required by the frontend actions
3. models/schemas.py — request/response models that match the frontend payloads
4. requirements.txt — only the dependencies needed for this feature
5. .env.example — all required backend env vars

REACT FILES ({frontend_root}/ and {react_src_dir}/):
6. {frontend_root}/vite.config.js — Vite proxy for /api → http://localhost:{port}
7. {react_src_dir}/api/client.js — axios client with /api baseURL and auth header support
8. {react_src_dir}/hooks/useIntegration.js — custom hook for this feature
9. {target_component} UPDATE — import the hook/client and wire the UI actions to the backend

CRITICAL RULES:
- The frontend must call the backend through /api/* relative URLs only.
- The Vite proxy must strip /api and forward to the backend port.
- Do not return an explanation-only response. Return create_file/update_file actions for the actual files.
- If the frontend already exists, update it instead of creating a second app shell.
- Do not generate a local-only UI loop for stateful features. The visible UI actions must call the backend, and the backend must persist the relevant server-side state.
- If the project already has pages/components/store/api files, patch those existing files instead of inventing unrelated parallel components.
- If the frontend already exists, patch its current files and connection config instead of creating a second frontend shell or duplicate app scaffold.
- If the backend already exists, do not duplicate backend business logic in React. Keep persistence, auth rules, and workflow/state transitions on the server and call them through the API.
- Do not swap API-backed screens for mock data, browser-only arrays, fake repositories, or localStorage-only persistence unless the user explicitly asks for a standalone demo.
- If a feature needs live data, make the main page submit updates to the backend and make summary/list pages fetch live data from the backend.

Make it production-ready and complete.
"""
}

LLM_ENV_KEYS = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "ollama":    "OLLAMA_BASE_URL",
}

LLM_INSTALL = {
    "openai":    "openai>=1.0.0",
    "anthropic": "anthropic>=0.25.0",
    "gemini":    "google-genai>=1.0.0",
    "ollama":    "ollama>=0.2.0",
}

# ─── System prompt ────────────────────────────────────────────────────────────

INTEGRATION_SYSTEM_PROMPT = """
You are a senior full-stack engineer. Generate complete, production-ready files for a Vite+React + FastAPI integration.

CRITICAL OUTPUT RULES:
1. Return ONLY valid JSON — no prose, no markdown fences around the outer response.
2. Each file is a separate object in the "messages" array.
3. Use "create_file" for new files, "update_file" for files that already exist in the React project.
4. Every file must be COMPLETE — no "// ... rest of code" shortcuts.
5. File paths: Python files use backend/ prefix, React files use src/ prefix.
6. The .env file uses "update_file" since it already exists in the React project.

OUTPUT FORMAT (JSON only):
{
  "messages": [
    {"type": "response", "text": "Brief description of what was generated"},
    {"type": "create_file", "file_path": "backend/main.py", "content": "# complete file content"},
    {"type": "create_file", "file_path": "backend/routes/chat.py", "content": "# complete file content"},
    {"type": "update_file", "file_path": "src/App.tsx", "content": "// complete updated file"},
    {"type": "create_file", "file_path": ".env", "content": "VITE_API_URL=http://localhost:7000\\n"},
    {"type": "response", "text": "Next steps markdown"}
  ]
}

REACT CODE RULES:
- TypeScript (.ts/.tsx) only — no plain JS
- Use React hooks (useState, useEffect, useCallback, useRef)
- Export named + default from each file
- Handle loading, error, empty states
- Use proper TypeScript interfaces — no 'any' type

PYTHON CODE RULES:
- FastAPI with type hints everywhere
- Pydantic v2 models (model_config = ConfigDict(...))
- async/await for all IO operations
- Proper error handling with HTTPException
- Docstrings on all functions

ENV VARIABLE RULES:
- Same logical name for related vars: Python reads DATABASE_URL, React reads VITE_DATABASE_URL only if truly needed
- Never expose secret keys to React frontend
- All FastAPI env vars go in backend/.env.example
- React env vars go in root .env (prefixed VITE_)
"""

# ─── Route ────────────────────────────────────────────────────────────────────

@integration_router.post("/integrate_feature")
async def integrate_feature(req: IntegrateRequest):
    """
    Generate a complete full-stack integration.
    Returns messages in the same format as /chat (create_file, update_file, response).
    """
    if client is None:
        raise HTTPException(status_code=503, detail="Gemini API unavailable")

    feature_prompt_template = FEATURE_PROMPTS.get(req.feature_type, FEATURE_PROMPTS["generic"])

    env_key    = LLM_ENV_KEYS.get(req.llm_provider or "", "LLM_API_KEY")
    llm_dep    = LLM_INSTALL.get(req.llm_provider or "", "")
    react_src_dir = (req.react_src_dir or "src").rstrip("/").rstrip("\\")
    frontend_root = os.path.dirname(react_src_dir) or "."
    component  = req.target_component or f"{react_src_dir}/App.tsx"

    feature_prompt = feature_prompt_template.format(
        llm_provider=req.llm_provider or "openai",
        llm_model=req.llm_model or "gpt-4o",
        env_key=env_key,
        db_connection=req.db_connection or "postgresql://user:pass@localhost:5432/mydb",
        port=req.backend_port,
        target_component=component,
        user_description=req.user_description,
        frontend_root=frontend_root,
        react_src_dir=react_src_dir,
        backend_dir=(req.backend_dir or "backend").rstrip("/").rstrip("\\"),
    )

    user_prompt = f"""
PROJECT CONTEXT:
- Framework: Vite + React with TypeScript
- Existing React components: {', '.join(req.scanned_components[:15]) or 'App.tsx'}
- Target component to modify: {component}
- Backend directory: {req.backend_dir or 'backend'}
- Frontend root directory: {frontend_root}
- React src directory: {react_src_dir}

INTEGRATION REQUEST: {req.user_description}

FEATURE SPECIFICATION:
{feature_prompt}

CONFIGURATION:
- LLM Provider: {req.llm_provider or 'none'}
- LLM Model: {req.llm_model or 'none'}
- LLM API Key env var: {env_key}
- Database: {req.db_type or 'none'} ({req.db_connection or 'not provided'})
- Backend Port: {req.backend_port}
- Use WebSocket: {req.use_websocket}
- Use Redis: {req.use_redis}
- Use Celery: {req.use_celery}
- Extra LLM dependency: {llm_dep}

If the request is about connecting an existing frontend to an existing backend, prioritize generating:
- {frontend_root}/vite.config.js
- {react_src_dir}/api/client.js
- the existing React page/component patch that actually calls the backend
- any backend route/schema files needed for the contract

If the feature is a bug game or snake game, ensure the backend stores score/player/game-state and the frontend sends those updates through the API.
If the feature is a bug game or snake game, do not use mock leaderboard data or browser-only score persistence. The React game must import the API client and call the backend in the gameplay lifecycle.

Generate ALL files now. Be complete, production-ready, no shortcuts.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=user_prompt,
            config={
                "system_instruction": INTEGRATION_SYSTEM_PROMPT,
                "temperature": 0.2,
                "max_output_tokens": 8192,
            }
        )

        raw = response.text.strip()

        # Strip markdown fences
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",       "", raw, flags=re.MULTILINE)
        raw = raw.strip()

        data = json.loads(raw)
        messages = data.get("messages", [])

        # Inject LLM API key into .env files if provided
        if req.llm_api_key and req.llm_api_key not in ("YOUR_API_KEY_HERE", "not-needed"):
            for msg in messages:
                if msg.get("type") in ("create_file", "update_file"):
                    fp = msg.get("file_path", "")
                    if fp.endswith(".env.example") or fp == ".env":
                        content = msg.get("content", "")
                        if env_key not in content:
                            msg["content"] = content + f"\n{env_key}={req.llm_api_key}\n"

        return {"messages": messages}

    except json.JSONDecodeError:
        # AI returned non-JSON — parse files from markdown
        messages = _extract_files_from_markdown(raw if 'raw' in dir() else "", req)
        return {"messages": messages}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration generation failed: {str(e)}")


def _extract_files_from_markdown(text: str, req: IntegrateRequest) -> list:
    """Fallback: extract code blocks from a markdown response."""
    messages = []
    pattern = re.compile(r'(?:#{1,3}\s+(.+?)\n)?```(\w+)?\n([\s\S]*?)```', re.MULTILINE)

    path_hints = {
        'python': 'backend/main.py', 'typescript': 'src/hooks/useIntegration.ts',
        'tsx': 'src/App.tsx', 'bash': 'backend/requirements.txt',
    }

    for match in pattern.finditer(text):
        heading = (match.group(1) or '').strip()
        lang    = (match.group(2) or '').lower()
        code    = match.group(3).strip()

        # Try to get a filename from the heading
        file_path = None
        for word in heading.split():
            if '.' in word and '/' in word:
                file_path = word.strip('`()[]')
                break
        if not file_path:
            file_path = path_hints.get(lang, f'backend/generated_{len(messages)}.py')

        messages.append({
            "type": "create_file",
            "file_path": file_path,
            "content": code
        })

    if not messages:
        messages.append({
            "type": "response",
            "text": f"Here is the integration for **{req.feature_type}**:\n\n{text[:2000]}"
        })

    return messages
