from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="AgentOS Production Backend API", version="1.0.0")

# Add CORS Middleware for multi-platform connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, lock this down to your React URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to AgentOS Core AI Engine",
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/health")
def read_health():
    return {"status": "ok", "service": "fastapi-backend"}
