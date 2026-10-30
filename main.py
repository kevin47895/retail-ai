from fastapi import FastAPI, Request, Form, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
import jwt
import os
import uuid
from fastapi import UploadFile, File
import asyncio
from fastapi.responses import StreamingResponse
from camera import camara
from database import init_db, leer, escribir, pwd_context

SECRET_KEY = os.getenv("JWT_SECRET", "cambiar_esto_en_produccion")
ALGORITHM = "HS256"
MODO = os.getenv("MODO", "local").lower()  # "local" o "nube"

app = FastAPI(title="Kovaq ")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    await init_db()
    if MODO == "local":
        camara.iniciar()
        camara.cambiar_fuente(os.getenv("FUENTE_INICIAL", "webcam"))

# 1. SOLUCIÓN AL 404: Ruta raíz que redirige automáticamente al login
@app.get("/")
async def root():
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    return templates.TemplateResponse("login.html", {"request": request, "error": error})

@app.post("/login")
async def login_action(request: Request, username: str = Form(...), password: str = Form(...)):
    # Búsqueda por username (flexible, sin requerir correo)
    usuario = leer("SELECT u.id, u.username, u.password_hash, u.nombre_completo, r.nombre FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.username = ?", (username,), uno=True)
    
    if not usuario or not pwd_context.verify(password, usuario[2]):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Credenciales inválidas. Verifique su usuario y contraseña."})
    
    # Generar Token JWT para sesión y WebSockets
    token_data = {"sub": usuario[1], "rol": usuario[4]}
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
    return response

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response

# Dependencia para obtener el usuario actual desde la cookie y validar roles
async def obtener_usuario_actual(request: Request):
    token_cookie = request.cookies.get("access_token")
    if not token_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autorizado")
    try:
        token = token_cookie.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        usuario = leer("SELECT u.id, u.username, u.nombre_completo, r.nombre FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.username = ?", (username,), uno=True)
        if not usuario:
            raise HTTPException(status_code=401)
        return {"id": usuario[0], "username": usuario[1], "nombre": usuario[2], "rol": usuario[3]}
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        user = await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "usuario_actual": user,
        "menu": [
            {"url": "/dashboard", "titulo": "Panel General", "icono": "bi bi-speedometer2", "admin_only": False},
            {"url": "/usuarios", "titulo": "Gestión de Usuarios", "icono": "bi bi-people", "admin_only": True}
        ]
    })

@app.post("/usuarios/nuevo")
async def crear_usuario_ui(request: Request, username: str = Form(...), nombre_completo: str = Form(...),
                           password: str = Form(...), rol: str = Form(...)):
    user = await obtener_usuario_actual(request)
    if user["rol"] not in ["admin", "admin_tecnico", "dueno"]:
        raise HTTPException(status_code=403, detail="Permisos insuficientes")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="La contraseña debe tener al menos 8 caracteres")
    rol_fila = leer("SELECT id FROM roles WHERE nombre = ?", (rol,), uno=True)
    if not rol_fila:
        raise HTTPException(status_code=400, detail="Rol no válido")
    try:
        escribir(
            "INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id, estado) VALUES (?, ?, ?, ?, 'activo')",
            (username, pwd_context.hash(password), nombre_completo, rol_fila[0])
        )
    except Exception:
        raise HTTPException(status_code=409, detail="El usuario ya existe")
    return RedirectResponse(url="/usuarios", status_code=303)

# Endpoint HTTP de respaldo para el feed de video (Evita fallos de conexión si el WebSocket no está activo aún)
@app.get("/api/video_feed")
async def video_feed(request: Request):
    await obtener_usuario_actual(request)

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
            return  # el cliente cerró la conexión: salimos sin romper el servidor

    return StreamingResponse(generar(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/fuente")
async def cambiar_fuente_api(request: Request):
    await obtener_usuario_actual(request)
    tipo = (await request.json()).get("tipo", "webcam")
    if tipo not in ("webcam", "rtsp", "video", "imagen", "apagado"):
        raise HTTPException(status_code=400, detail="Fuente no válida")
    rutas = {"video": "uploads/demo.mp4", "imagen": "uploads/demo.jpg"}
    camara.cambiar_fuente(tipo, rutas.get(tipo, ""))
    return {"ok": True}

# WebSocket de Video seguro validando Token JWT
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