import os
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "super-secreto-kovaq")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120
if SECRET_KEY == "super-secreto-kovaq":
    print("⚠️ Define SECRET_KEY en .env: con la clave por defecto cualquiera puede fabricar sesiones.")


def crear_token_acceso(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verificar_token(token: str):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
