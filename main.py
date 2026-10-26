import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext

from ai_agent import preguntar_a_los_datos
from auth import crear_token_acceso, verificar_token
from camera import camara
from database import init_db, leer, ejecutar, obtener_estadisticas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
EXT = {"imagen": (".jpg", ".jpeg", ".png", ".bmp", ".webp"), "video": (".mp4", ".avi", ".mov", ".mkv", ".webm")}
MAX_MB = 300


def _limpiar_uploads(excepto: str = ""):
    """Los videos subidos tienen rostros: no se acumulan en el disco (solo queda el actual)."""
    for n in os.listdir("uploads"):
        ruta = os.path.join("uploads", n)
        if ruta != excepto:
            try:
                os.remove(ruta)
            except OSError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    os.makedirs("uploads", exist_ok=True)
    _limpiar_uploads()
    camara.iniciar()
    yield
    camara.detener()


app = FastAPI(title="Kovaq Production", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")


async def get_user(token: str = Depends(oauth2_scheme)):
    payload = verificar_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Token inválido")
    return payload


class ChatReq(BaseModel):
    mensaje: str = Field(..., min_length=1, max_length=500)

class UserRegister(BaseModel):
    username: str
    password: str


@app.post("/api/register")
async def register_user(user: UserRegister):
    hashed_pwd = pwd_context.hash(user.password)
    
    # Verifica si el usuario ya existe en Turso
    existe = await ejecutar("SELECT username FROM users WHERE username = ?", (user.username,), uno=True)
    if existe:
        raise HTTPException(status_code=400, detail="El usuario ya existe")
        
    # Inserta el nuevo usuario usando la estructura exacta de la tabla `users`
    await ejecutar(
        "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
        (user.username, hashed_pwd, 'admin')
    )
    return {"status": "ok", "message": f"Usuario {user.username} creado exitosamente"}


@app.get("/")
def root():
    return FileResponse("static/index.html")


@app.post("/api/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    # Corregido: consulta hacia la tabla `users` y columna `hashed_password`
    user = leer("SELECT username, hashed_password FROM users WHERE username = ?", (form.username,), uno=True)
    if user and pwd_context.verify(form.password, user[1]):
        return {"access_token": crear_token_acceso({"sub": user[0]}), "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Credenciales incorrectas")


@app.get("/api/stats")
async def stats(user: dict = Depends(get_user)):
    d = await asyncio.to_thread(obtener_estadisticas)
    d["entradas_tiempo_real"] = camara.total_entradas
    d["agarrados_tiempo_real"] = camara.total_agarrados
    d["conversion"] = round(camara.total_agarrados / camara.total_entradas * 100, 1) if camara.total_entradas else 0.0
    return d


@app.get("/api/estado")
async def estado(user: dict = Depends(get_user)):
    return camara.estado_dict()


@app.post("/api/chat")
async def chat(req: ChatReq, user: dict = Depends(get_user)):
    vivo = {"entradas": camara.total_entradas, "agarrados": camara.total_agarrados}
    return {"respuesta": await preguntar_a_los_datos(req.mensaje, vivo)}


@app.post("/api/toggle_pause")
async def toggle_pause(user: dict = Depends(get_user)):
    return {"pausado": camara.toggle_pausa()}


@app.post("/api/apagar")
async def apagar_camara(user: dict = Depends(get_user)):
    camara.apagar()
    return {"status": "apagado"}


@app.post("/api/replay")
async def replay(user: dict = Depends(get_user)):
    camara.repetir()
    return {"status": "ok"}


@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket, token: str = ""):
    if not verificar_token(token):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    ultimo = -1
    try:
        while True:
            ultimo, datos = await asyncio.to_thread(camara.get_frame_bytes, ultimo)
            if datos:
                await websocket.send_bytes(datos)
            await asyncio.sleep(0.02 if datos else 0.04)
    except (WebSocketDisconnect, RuntimeError):
        pass


@app.post("/api/upload_source")
async def upload(tipo: str, file: Optional[UploadFile] = File(None), user: dict = Depends(get_user)):
    if tipo in ("webcam", "rtsp"):
        camara.cambiar_fuente(tipo)
        return {"status": "ok"}
    if tipo not in EXT or file is None:
        raise HTTPException(400, "Fuente no válida")
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in EXT[tipo]:
        raise HTTPException(400, f"Formato no permitido para {tipo}: {', '.join(EXT[tipo])}")
    ruta = os.path.join("uploads", f"{uuid.uuid4().hex}{ext}")

    def guardar():
        with open(ruta, "wb") as f:
            shutil.copyfileobj(file.file, f, 1024 * 1024)

    await asyncio.to_thread(guardar)
    if os.path.getsize(ruta) > MAX_MB * 1024 * 1024:
        os.remove(ruta)
        raise HTTPException(413, f"El archivo supera {MAX_MB} MB")
    camara.cambiar_fuente(tipo, ruta)
    _limpiar_uploads(excepto=ruta)
    return {"status": "ok"}