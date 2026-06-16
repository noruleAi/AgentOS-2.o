import os, json, uuid, datetime, logging, secrets, base64, hashlib, re
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, constr
from jose import jwt, JWTError
import httpx

# ========================= CONFIGURATION =========================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SECRET_KEY = os.getenv("SECRET_KEY", "agentos-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

# Allowed origins for CORS – set via environment or default to *
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Founder emails – comma‑separated in env
FOUNDER_EMAILS = set(
    email.strip().lower()
    for email in os.getenv("FOUNDER_EMAILS", "").split(",")
    if email.strip()
)

# ========================= LOGGING =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentos")

# ========================= IN‑MEMORY STORES (WARNING: NOT PERSISTENT) =========================
# Replace with PostgreSQL / SQLite for production
users = {}      # email -> {email, password_hash, name, role, status, ...}
chats = {}      # chat_id -> {id, user_email, title, pinned, created_at, messages: []}
memory_db = {}  # user_email -> {profile: {key: value}}

# ========================= FASTAPI APP =========================
app = FastAPI(title="AgentOS AI 2.0", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================= STATIC FILES =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Frontend files are in the root directory
FRONTEND_DIR = BASE_DIR

IMAGES_DIR = os.path.join(BASE_DIR, "images")

# Serve all frontend files
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# Optional images folder
if os.path.exists(IMAGES_DIR):
    app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# ========================= SECURITY =========================
security = HTTPBearer()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    return hashlib.sha256(password.encode()).hexdigest() == hashed

def create_access_token(email: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    try:
        payload = decode_token(token)
        email = payload.get("sub")
        if email not in users:
            raise HTTPException(status_code=401, detail="User not found")
        user = users[email]
        if user.get("status") != "active":
            raise HTTPException(status_code=403, detail="Account deactivated")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def founder_required(current_user = Depends(get_current_user)):
    if current_user.get("role") != "founder":
        raise HTTPException(status_code=403, detail="Founder only")
    return current_user

# ========================= GLOBAL EXCEPTION HANDLER =========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."}
    )

# ========================= STARTUP =========================
@app.on_event("startup")
async def startup():
    logger.info("AgentOS AI 2.0 starting up")
    logger.info(f"FRONTEND_DIR: {FRONTEND_DIR} – exists: {os.path.exists(FRONTEND_DIR)}")
    logger.info(f"IMAGES_DIR: {IMAGES_DIR} – exists: {os.path.exists(IMAGES_DIR)}")
    logger.info(f"OPENROUTER_API_KEY configured: {bool(OPENROUTER_API_KEY)}")
    logger.info(f"FOUNDER_EMAILS: {FOUNDER_EMAILS}")

# ========================= ROOT & HEALTH =========================
@app.get("/")
def root():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

# ========================= AUTH SCHEMAS =========================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: constr(min_length=8)
    name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

# ========================= AUTH ENDPOINTS =========================
@app.post("/api/register")
def register(data: RegisterRequest):
    email = data.email.lower()
    if email in users:
        raise HTTPException(status_code=400, detail="Email already registered")
    # Auto‑set founder role if email matches
    role = "founder" if email in FOUNDER_EMAILS else "user"
    users[email] = {
        "email": email,
        "password_hash": hash_password(data.password),
        "name": data.name,
        "role": role,
        "status": "active",
        "is_verified": True,  # auto‑verify for demo; in production send email
        "created_at": datetime.datetime.utcnow().isoformat()
    }
    logger.info(f"User registered: {email} (role: {role})")
    return {"success": True, "message": "User registered"}

@app.post("/api/login")
def login(data: LoginRequest):
    email = data.email.lower()
    user = users.get(email)
    if not user or not verify_password(data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if user.get("status") != "active":
        raise HTTPException(status_code=403, detail="Account deactivated")
    token = create_access_token(email)
    return {
        "access_token": token,
        "user": {
            "email": user["email"],
            "name": user["name"],
            "role": user.get("role", "user")
        }
    }

@app.get("/api/me")
def get_me(current_user = Depends(get_current_user)):
    return {
        "email": current_user["email"],
        "name": current_user["name"],
        "role": current_user.get("role", "user")
    }

# ========================= CHAT SCHEMAS =========================
class ChatCreate(BaseModel):
    title: str = "New Chat"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatStreamRequest(BaseModel):
    chat_id: str
    messages: List[dict]
    model: str = "openai/gpt-3.5-turbo"

# ========================= CHAT ENDPOINTS (authenticated) =========================
@app.post("/api/chats")
def create_chat(chat: ChatCreate, current_user = Depends(get_current_user)):
    chat_id = str(uuid.uuid4())
    chats[chat_id] = {
        "id": chat_id,
        "user_email": current_user["email"],
        "title": chat.title,
        "pinned": False,
        "created_at": datetime.datetime.utcnow().isoformat(),
        "messages": []
    }
    return chats[chat_id]

@app.get("/api/chats")
def get_chats(current_user = Depends(get_current_user)):
    user_email = current_user["email"]
    user_chats = [c for c in chats.values() if c["user_email"] == user_email]
    user_chats.sort(key=lambda c: (not c["pinned"], c["created_at"]), reverse=True)
    return user_chats

@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str, current_user = Depends(get_current_user)):
    chat = chats.get(chat_id)
    if not chat or chat["user_email"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.put("/api/chats/{chat_id}")
def update_chat(chat_id: str, update: dict, current_user = Depends(get_current_user)):
    chat = chats.get(chat_id)
    if not chat or chat["user_email"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Chat not found")
    if "title" in update:
        chat["title"] = update["title"]
    if "pinned" in update:
        chat["pinned"] = update["pinned"]
    return {"success": True}

@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str, current_user = Depends(get_current_user)):
    chat = chats.get(chat_id)
    if not chat or chat["user_email"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Chat not found")
    del chats[chat_id]
    return {"success": True}

@app.post("/api/chats/{chat_id}/messages")
def add_message(chat_id: str, msg: ChatMessage, current_user = Depends(get_current_user)):
    chat = chats.get(chat_id)
    if not chat or chat["user_email"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Chat not found")
    chat["messages"].append({"role": msg.role, "content": msg.content})
    return {"success": True}

# ========================= AI STREAMING (authenticated) =========================
MODEL_MAP = {
    "GPT-4o": "openai/gpt-4o",
    "Claude 3.5": "anthropic/claude-sonnet-4-20250514",
    "Gemini 1.5 Pro": "google/gemini-1.5-pro",
    "DeepSeek": "deepseek/deepseek-chat",
    "Llama 3": "meta-llama/llama-3.3-70b-instruct",
    "Mistral Large": "mistralai/mistral-large",
    "Grok-2": "x-ai/grok-2",
    "Perplexity": "perplexity/llama-3.1-sonar-large-128k-online",
    "Cohere": "cohere/command-r-plus",
    "Copilot": "microsoft/copilot",
    "Sonar 2": "perplexity/sonar-reasoning"
}

async def stream_gemini(messages: list):
    if not GEMINI_API_KEY:
        yield f"data: {json.dumps({'content':'Gemini API key not configured'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    try:
        prompt = "\n".join(
            [f"{m.get('role','user')}: {m.get('content','')}" for m in messages]
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}",
                json={
                    "contents": [
                        {
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ]
                },
                timeout=60.0
            )

            if response.status_code != 200:
                yield f"data: {json.dumps({'content':f'Gemini Error {response.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return

            data = response.json()

            text = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

            yield f"data: {json.dumps({'content': text})}\n\n"
            yield "data: [DONE]\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'content': str(e)})}\n\n"
        yield "data: [DONE]\n\n"
async def stream_openrouter(messages: list, model_display: str, user_email: str):
    if not OPENROUTER_API_KEY:
        yield f"data: {json.dumps({'content': 'Error: OpenRouter API key not configured.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Inject memory context
    memory_context = ""
    if user_email in memory_db:
        for profile, items in memory_db[user_email].items():
            for k, v in items.items():
                memory_context += f"{k}: {v}; "
    if memory_context:
        memory_context = "[User context] " + memory_context + "\n"

    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = memory_context + messages[0]["content"]
    else:
        messages.insert(0, {"role": "system", "content": memory_context})

    model_id = MODEL_MAP.get(model_display, model_display)
    referer = os.getenv("APP_URL", "http://localhost:8000")

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": referer,
                "X-Title": "AgentOS",
            },
            json={"model": model_id, "messages": messages, "stream": True},
            timeout=60.0
        ) as response:
            if response.status_code != 200:
                yield f"data: {json.dumps({'content': f'Error: OpenRouter returned {response.status_code}'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield f"data: {json.dumps({'content': content})}\n\n"
                    except:
                        pass
            yield "data: [DONE]\n\n"

@app.post("/api/chat/stream")
async def stream_chat(req: ChatStreamRequest, current_user = Depends(get_current_user)):
    chat = chats.get(req.chat_id)
    if not chat or chat["user_email"] != current_user["email"]:
        raise HTTPException(status_code=403, detail="Not your chat")

    # Save user message
    if req.messages and req.messages[-1]["role"] == "user":
        user_msg = req.messages[-1]["content"]
        chat["messages"].append({"role": "user", "content": user_msg})

        # Auto‑title if still default
        if chat["title"] == "New Chat" and user_msg.strip():
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "openai/gpt-3.5-turbo",
                            "messages": [
                                {"role": "system", "content": "Generate a very short title (max 5 words) for a conversation that starts with: " + user_msg}
                            ],
                            "max_tokens": 20,
                            "temperature": 0.3
                        },
                        timeout=10.0
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        title = data["choices"][0]["message"]["content"].strip().strip('"')
                        if title:
                            chat["title"] = title[:60]
            except:
                pass

                async def generate():
                    accumulated_response = ""

                    if GEMINI_API_KEY:
                        response_stream =                        
            stream_gemini(req.messages)
                    else:
                        response_stream =        
            stream_openrouter(
                            req.messages,
                            req.model,
                        current_user["email"]
                    )

        async for chunk in response_stream:
            if chunk.startswith("data: [DONE]"):
                break

            yield chunk

            try:
                payload = json.loads(chunk.replace("data: ", ""))
                accumulated_response += payload.get("content", "")
            except Exception:
                continue

        if accumulated_response:
            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": accumulated_response,
                }
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )

# ========================= PUBLIC SIMPLE CHAT (no auth) =========================
class SimpleChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def simple_chat(req: SimpleChatRequest):
    """Public chat endpoint – does not require authentication."""
    if not OPENROUTER_API_KEY:
        return {"reply": "OpenRouter API key not configured."}
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-3.5-turbo",
                "messages": [{"role": "user", "content": req.message}]
            },
            timeout=30.0
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"reply": data["choices"][0]["message"]["content"]}
        else:
            return {"reply": f"Error: {resp.status_code}"}

# ========================= MEMORY (authenticated) =========================
@app.get("/api/memory")
def get_memory(current_user = Depends(get_current_user), profile: str = "default"):
    user_mem = memory_db.setdefault(current_user["email"], {})
    profile_data = user_mem.get(profile, {})
    return {"profile": profile, "items": [{"key": k, "value": v} for k, v in profile_data.items()]}

@app.post("/api/memory")
def add_memory(key: str, value: str, profile: str = "default", current_user = Depends(get_current_user)):
    user_mem = memory_db.setdefault(current_user["email"], {})
    profile_data = user_mem.setdefault(profile, {})
    profile_data[key] = value
    return {"success": True}

@app.delete("/api/memory/{key}")
def delete_memory(key: str, profile: str = "default", current_user = Depends(get_current_user)):
    user_mem = memory_db.get(current_user["email"], {})
    profile_data = user_mem.get(profile, {})
    if key in profile_data:
        del profile_data[key]
    return {"success": True}

# ========================= IMAGE ANALYSIS (authenticated) =========================
@app.post("/api/image-analyze")
async def analyze_image(file: UploadFile = File(...), query: str = "Describe this image", current_user = Depends(get_current_user)):
    # Basic validation
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=400, detail="File too large")
    base64_image = base64.b64encode(content).decode("utf-8")
    data_url = f"data:{file.content_type};base64,{base64_image}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }
    ]
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={"model": "openai/gpt-4o", "messages": messages, "max_tokens": 300},
            timeout=30.0
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"result": data["choices"][0]["message"]["content"]}
        else:
            return {"result": f"Analysis failed: HTTP {resp.status_code}"}

# ========================= IMAGE GENERATION (authenticated) =========================
class ImageGenRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024

@app.post("/api/image-generate")
def generate_image(req: ImageGenRequest, current_user = Depends(get_current_user)):
    import urllib.parse
    url = f"https://image.pollinations.ai/prompt/{urllib.parse.quote(req.prompt)}?width={req.width}&height={req.height}&seed={secrets.randbelow(10000)}&nofeed=true"
    return {"url": url}

# ========================= FILE UPLOAD (public? keeping it simple) =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    # In production, save to cloud storage. Here we just acknowledge.
    return {"filename": file.filename, "size": len(content)}

# ========================= ADMIN ENDPOINTS (founder only) =========================
@app.get("/api/admin/users", dependencies=[Depends(founder_required)])
def admin_list_users():
    return list(users.values())

@app.put("/api/admin/users/{email}/ban", dependencies=[Depends(founder_required)])
def admin_ban_user(email: str):
    user = users.get(email.lower())
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["status"] = "banned"
    return {"success": True}

@app.put("/api/admin/users/{email}/unban", dependencies=[Depends(founder_required)])
def admin_unban_user(email: str):
    user = users.get(email.lower())
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["status"] = "active"
    return {"success": True}

@app.put("/api/admin/users/{email}/suspend", dependencies=[Depends(founder_required)])
def admin_suspend_user(email: str):
    user = users.get(email.lower())
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user["status"] = "suspended"
    return {"success": True}

@app.get("/api/admin/stats", dependencies=[Depends(founder_required)])
def admin_stats():
    total_users = len(users)
    active_users = sum(1 for u in users.values() if u.get("status") == "active")
    total_chats = len(chats)
    total_messages = sum(len(c["messages"]) for c in chats.values())
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_chats": total_chats,
        "total_messages": total_messages
    }
