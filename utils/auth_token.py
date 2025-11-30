# utils/auth_token.py
import os
from datetime import datetime, timedelta
import jwt

SECRET = os.getenv("AUTH_SECRET", "change-this-in-prod")
ALGO = "HS256"

def create_token(user_id: str, days: int = 7, extra: dict | None = None) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(days=days),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET, algorithm=ALGO)

def verify_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGO])
    except jwt.ExpiredSignatureError:
        return None
    except Exception:
        return None
