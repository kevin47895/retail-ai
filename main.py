import os
import uuid
import asyncio
import threading
import jwt
from fastapi import FastAPI, Request, Form, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from database import init_db, leer, escribir, pwd_context

SECRET_KEY = os.getenv("JWT_SECRET", "cambiar_esto_en_produccion")
ALGORITHM = "HS256"
MODO = os.getenv("MODO", "local").lower()  # "local" o "nube"

# En nube NO se importa camera.py, así no se carga YOLO ni se abre la webcam
if MODO == "local":
    from camera import camara
else:
    camara = None

ROLES_GESTION = ["admin", "admin_tecnico", "dueno"]

app = FastAPI(title="Kovaq Retail Analytics")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    await init_db()
    if MODO == "local":
        camara.iniciar()
        camara.cambiar_fuente(os.getenv("FUENTE_INICIAL", "webcam"))
        from publicador_video import iniciar_publicacion
        iniciar_publicacion(camara)

@app.get("/")
async def root():
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse(request, "login.html", {"error": error})

@app.post("/login")
async def login_action(request: Request, username: str = Form(...), password: str = Form(...)):
    usuario = leer(
        "SELECT u.id, u.username, u.password_hash, u.nombre_completo, r.nombre FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.username = ?",
        (username,), uno=True
    )
    if not usuario or not pwd_context.verify(password, usuario[2]):
        return templates.TemplateResponse(request, "login.html", {
            "error": "Credenciales inválidas. Verifique su usuario y contraseña."
        })

    token = jwt.encode({"sub": usuario[1], "rol": usuario[4]}, SECRET_KEY, algorithm=ALGORITHM)
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response

async def obtener_usuario_actual(request: Request):
    token_cookie = request.cookies.get("access_token")
    if not token_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autorizado")
    try:
        token = token_cookie.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        usuario = leer(
            "SELECT u.id, u.username, u.nombre_completo, r.nombre FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.username = ?",
            (payload.get("sub"),), uno=True
        )
        if not usuario:
            raise HTTPException(status_code=401)
        return {"id": usuario[0], "username": usuario[1], "nombre": usuario[2], "rol": usuario[3]}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

MENU = [
    {"url": "/dashboard", "titulo": "Panel General", "icono": "bi bi-speedometer2", "admin_only": False},
    {"url": "/usuarios", "titulo": "Gestión de Usuarios", "icono": "bi bi-people", "admin_only": True},
]

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        user = await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request, "dashboard.html", {
        "usuario_actual": user, "menu": MENU, "modo": MODO
    })

@app.get("/usuarios", response_class=HTMLResponse)
async def listar_usuarios(request: Request):
    try:
        user = await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    if user["rol"] not in ROLES_GESTION:
        raise HTTPException(status_code=403, detail="Acceso denegado. Permisos insuficientes.")

    usuarios = leer("SELECT u.id, u.username, u.nombre_completo, r.nombre, u.estado FROM usuarios u JOIN roles r ON u.rol_id = r.id")
    roles = leer("SELECT id, nombre FROM roles")
    return templates.TemplateResponse(request, "usuarios.html", {
        "usuarios": usuarios, "roles": roles,
        "usuario_actual": user, "menu": MENU, "modo": MODO
    })

@app.post("/usuarios/crear")
async def crear_usuario(request: Request, nombre: str = Form(...), username: str = Form(...),
                        password: str = Form(...), rol_id: int = Form(...)):
    user = await obtener_usuario_actual(request)
    if user["rol"] not in ROLES_GESTION:
        raise HTTPException(status_code=403, detail="Permisos insuficientes")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres")
    try:
        escribir(
            "INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id, estado) VALUES (?, ?, ?, ?, 'activo')",
            (username.strip(), pwd_context.hash(password), nombre.strip(), rol_id)
        )
    except Exception:
        raise HTTPException(status_code=409, detail="El usuario ya existe")
    return RedirectResponse(url="/usuarios", status_code=303)

@app.post("/usuarios/eliminar/{usuario_id}")
async def eliminar_usuario(request: Request, usuario_id: int):
    user = await obtener_usuario_actual(request)
    if user["rol"] not in ROLES_GESTION:
        raise HTTPException(status_code=403, detail="Permisos insuficientes")
    if user["id"] == usuario_id:
        raise HTTPException(status_code=400, detail="No puedes eliminar tu propio usuario")
    escribir("DELETE FROM usuarios WHERE id = ?", (usuario_id,))
    return RedirectResponse(url="/usuarios", status_code=303)

@app.get("/api/video_feed")
async def video_feed(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="Video no disponible en modo nube")

    async def generar():
        ultimo = -1
        try:
            while True:
                if await request.is_disconnected():
                    break
                fid, jpg = camara.get_frame_bytes(ultimo)
                if jpg:
                    ultimo = fid
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.03)
        except (ConnectionError, asyncio.CancelledError, GeneratorExit):
            return  # el cliente cerró la conexión

    return StreamingResponse(generar(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/fuente")
async def cambiar_fuente_api(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="Cambio de fuente solo disponible en modo local")
    datos = await request.json()
    tipo = datos.get("tipo", "webcam")
    if tipo not in ("webcam", "rtsp", "video", "imagen", "apagado"):
        raise HTTPException(status_code=400, detail="Fuente no válida")
    rutas = {"video": "uploads/demo.mp4", "imagen": "uploads/demo.jpg"}
    camara.cambiar_fuente(tipo, rutas.get(tipo, ""))
    return {"ok": True}

@app.websocket("/ws/video")
async def websocket_video(websocket: WebSocket):
    token_cookie = websocket.cookies.get("access_token")
    if not token_cookie:
        await websocket.close(code=4001)
        return
    try:
        token = token_cookie.split(" ")[1]
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        await websocket.close(code=4001)
        return

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            await websocket.send_bytes(data)
    except WebSocketDisconnect:
        pass

def _obtener_stats():
    trafico = leer("SELECT COUNT(*) FROM eventos WHERE tipo = 'trafico'", uno=True)[0]
    perm = leer("SELECT COUNT(*), COALESCE(AVG(valor), 0) FROM eventos WHERE tipo = 'permanencia'", uno=True)
    inter = leer("SELECT COUNT(*) FROM eventos WHERE tipo = 'interaccion'", uno=True)[0]
    top = leer("""SELECT detalle, COUNT(*) FROM eventos WHERE tipo = 'interaccion'
                  GROUP BY detalle ORDER BY COUNT(*) DESC LIMIT 5""")
    return {
        "trafico": trafico,
        "permanencia_eventos": perm[0],
        "permanencia_promedio_seg": round(float(perm[1]), 1),
        "interacciones": inter,
        "top_productos": [{"producto": p[0], "veces": p[1]} for p in top],
    }

def _texto_resumen(s):
    partes = [f"Se registraron {s['trafico']} detecciones de personas."]
    if s["permanencia_eventos"]:
        partes.append(f"La permanencia promedio en la zona de interés fue de {s['permanencia_promedio_seg']} segundos ({s['permanencia_eventos']} casos).")
    else:
        partes.append("Aún no hay permanencias registradas en la zona de interés.")
    partes.append(f"Hubo {s['interacciones']} interacciones con productos.")
    if s["top_productos"]:
        lista = ", ".join(f"{p['producto']} ({p['veces']})" for p in s["top_productos"])
        partes.append(f"Los productos con más interacción fueron: {lista}.")
    return " ".join(partes)

@app.get("/api/stats")
async def api_stats(request: Request):
    await obtener_usuario_actual(request)
    try:
        s = _obtener_stats()
    except Exception:
        raise HTTPException(status_code=500, detail="Error consultando las estadísticas")
    s["resumen"] = _texto_resumen(s)
    return s

@app.post("/api/chat")
async def api_chat(request: Request):
    await obtener_usuario_actual(request)
    try:
        datos = await request.json()
        mensaje = (datos.get("mensaje") or "").strip()
    except Exception:
        raise HTTPException(status_code=400, detail="Solicitud inválida")
    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje está vacío")
    try:
        s = _obtener_stats()
    except Exception:
        raise HTTPException(status_code=500, detail="Error consultando los datos")

    respuesta = _texto_resumen(s)
    if any(p in mensaje.lower() for p in ("venta", "ingreso", "factur")):
        respuesta = "El sistema no registra ventas, solo tráfico, permanencia e interacciones con perchas. " + respuesta
    return {"respuesta": respuesta}

@app.post("/api/upload")
async def api_upload(request: Request, archivo: UploadFile = File(...)):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="La subida de video solo está disponible en modo local")

    ext = os.path.splitext(archivo.filename or "")[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        tipo = "video"
    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        tipo = "imagen"
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    contenido = await archivo.read()
    if len(contenido) > 200 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El archivo supera 200 MB")

    os.makedirs("uploads", exist_ok=True)
    ruta = os.path.join("uploads", f"{uuid.uuid4().hex}{ext}")
    with open(ruta, "wb") as f:
        f.write(contenido)

    camara.cambiar_fuente(tipo, ruta)
    return {"ok": True, "tipo": tipo}
@app.post("/api/control")
async def api_control(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="Control solo disponible en modo local")
    datos = await request.json()
    accion = datos.get("accion")
    if accion == "pausa":
        return {"ok": True, "pausado": camara.toggle_pausa()}
    if accion == "parar":
        camara.apagar()
        return {"ok": True, "pausado": False}
    raise HTTPException(status_code=400, detail="Acción no válida")
# ---------- VIDEO EN VIVO PARA LA WEB (solo último cuadro, nada se guarda) ----------
VIDEO_TOKEN = os.getenv("VIDEO_PUSH_TOKEN", "")
_ultimo_frame = {"jpg": None, "id": 0}
_frame_lock = threading.Lock()

@app.post("/api/frame_push")
async def frame_push(request: Request):
    if not VIDEO_TOKEN or request.headers.get("X-Token") != VIDEO_TOKEN:
        raise HTTPException(status_code=403, detail="Token inválido")
    jpg = await request.body()
    if not jpg or len(jpg) > 2_000_000:
        raise HTTPException(status_code=400, detail="Frame inválido")
    with _frame_lock:
        _ultimo_frame["jpg"] = jpg
        _ultimo_frame["id"] += 1
    return {"ok": True}

@app.get("/api/video_live")
async def video_live(request: Request):
    await obtener_usuario_actual(request)

    async def generar():
        ultimo = -1
        try:
            while True:
                if await request.is_disconnected():
                    break
                with _frame_lock:
                    jpg, fid = _ultimo_frame["jpg"], _ultimo_frame["id"]
                if jpg and fid != ultimo:
                    ultimo = fid
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                await asyncio.sleep(0.05)
        except (ConnectionError, asyncio.CancelledError, GeneratorExit):
            return

    return StreamingResponse(generar(), media_type="multipart/x-mixed-replace; boundary=frame")