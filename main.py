"""JARVIS backend: a thin FastAPI proxy so the AI provider key never lives on the phone.

Environment variables:
  GEMINI_API_KEY    your Gemini API key (required)
  JARVIS_APP_TOKEN  long random string; the app sends it as X-Jarvis-Token (required)
  GEMINI_MODEL      optional, default gemini-2.5-flash
"""
import hmac
import os

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
APP_TOKEN = os.environ.get("JARVIS_APP_TOKEN", "")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are JARVIS, a concise personal voice assistant. Your answers are spoken aloud, so reply in "
    "plain sentences only: no markdown, no lists, no emojis, no code blocks. Keep answers to one to "
    "three sentences unless the user asks for more detail. Address the user as 'sir' occasionally, "
    "not in every reply. If you don't know something, say so plainly."
)

app = FastAPI(title="JARVIS backend")


class Turn(BaseModel):
    role: str = Field(pattern="^(user|model)$")
    text: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=20)


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest, x_jarvis_token: str | None = Header(default=None)):
    # Fail closed: the server refuses to answer unless both secrets are configured.
    if not GEMINI_API_KEY or not APP_TOKEN:
        raise HTTPException(status_code=500, detail="Server is not configured.")
    if not hmac.compare_digest(x_jarvis_token or "", APP_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token.")

    history = list(req.history)
    while history and history[0].role != "user":
        history.pop(0)  # Gemini expects the conversation to start with a user turn

    contents = [{"role": t.role, "parts": [{"text": t.text}]} for t in history]
    contents.append({"role": "user", "parts": [{"text": req.prompt}]})
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 512},
    }

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(url, json=body, headers={"x-goog-api-key": GEMINI_API_KEY})
    except httpx.HTTPError:
        raise HTTPException(status_code=502, detail="Could not reach the AI provider.")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"AI provider error ({r.status_code}).")

    try:
        parts = r.json()["candidates"][0]["content"]["parts"]
        reply = "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, ValueError):
        reply = ""
    if not reply:
        raise HTTPException(status_code=502, detail="The AI provider gave no answer.")
    return {"reply": reply}
