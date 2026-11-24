import os
import uuid
import asyncio
import threading
import hmac
import logging
from typing import Optional

from fastapi import (
    FastAPI, Request, Form, HTTPException, status,
    UploadFile, File, Response
)
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (
    init_db, init_analytics, leer, escribir, pwd_context,
    obtener_usuario_por_id, contar_administradores_activos,
    actualizar_usuario
)
from ai_agent import preguntar_a_los_datos
from analytics import router as analytics_router
from security import (
    SecurityHeadersMiddleware, crear_token_acceso, decodificar_token_acceso,
    validar_password_complejidad, limiter_auth, limiter_chat, obtener_ip_cliente,
    generar_csrf_token, verificar_csrf_request, guardar_archivo_seguro,
    validar_ruta_fuente_segura, enmascarar_url_conexion, CSRF_COOKIE_NAME, TOKEN_EXPIRATION_SECONDS
)

log = logging.getLogger("kovaq.main")
MODO = os.getenv("MODO", "local").lower()  # "local" o "nube"

# En nube NO se importa camera.py, así no se carga YOLO ni se abre la webcam
if MODO == "local":
    from camera import camara
else:
    camara = None

ROLES_GESTION = ["admin", "admin_tecnico", "dueno"]

def es_rol_gestion(rol: str) -> bool:
    """Verificación segura e insensible a mayúsculas de roles administrativos."""
    return (rol or "").strip().lower() in ROLES_GESTION

app = FastAPI(title="Kovaq Retail Analytics", docs_url=None, redoc_url=None)

# 1. Registro del Middleware de Cabeceras de Seguridad (OWASP Defense-in-Depth)
app.add_middleware(SecurityHeadersMiddleware)

# 2. Inclusión del Router de Analítica
app.include_router(analytics_router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 3. Montaje de archivos estáticos y plantillas con rutas absolutas
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

DEMO_VIDEO = os.path.join(BASE_DIR, "static", "demo_supermercado.mp4")


@app.on_event("startup")
async def startup_event():
    await init_db()
    await init_analytics()
    if MODO == "local" and camara:
        camara.iniciar()
        if os.path.exists(DEMO_VIDEO):
            camara.cambiar_fuente("video", DEMO_VIDEO)
        else:
            camara.cambiar_fuente("webcam")


# ==========================================
# MENÚ LATERAL UNIFICADO
# ==========================================
MENU = [
    {"url": "/dashboard", "titulo": "1. Dashboard General", "icono": "bi bi-speedometer2", "admin_only": False},
    {"url": "/usuarios", "titulo": "Gestión Usuarios", "icono": "bi bi-people", "admin_only": True},
]


# ==========================================
# GESTIÓN DE SESIÓN & AUTENTICACIÓN (IAM)
# ==========================================
def _obtener_o_crear_csrf(request: Request, response: Optional[Response] = None) -> str:
    """Extrae el token CSRF de la cookie o genera uno nuevo y lo asocia a la respuesta."""
    token = request.cookies.get(CSRF_COOKIE_NAME)
    if not token:
        token = generar_csrf_token()
        if response:
            es_seguro = request.url.scheme == "https"
            response.set_cookie(
                key=CSRF_COOKIE_NAME,
                value=token,
                max_age=TOKEN_EXPIRATION_SECONDS,
                httponly=False,  # Debe ser legible por JS para cabeceras X-CSRF-Token
                samesite="lax",
                secure=es_seguro
            )
    return token


async def obtener_usuario_actual(request: Request) -> dict:
    """
    Verifica el token JWT en cookie, comprueba su firma y fecha de expiración,
    y valida que el usuario siga existiendo y mantenga estado 'activo'.
    """
    token_cookie = request.cookies.get("access_token")
    if not token_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No autorizado: Sesión ausente")

    try:
        token_str = token_cookie.split(" ")[1] if " " in token_cookie else token_cookie
        payload = decodificar_token_acceso(token_str)
        uid = payload.get("uid")
        sub = payload.get("sub")

        # Comprobación de existencia y estado en base de datos
        if uid:
            usuario = leer(
                "SELECT u.id, u.username, u.nombre_completo, r.nombre, u.estado "
                "FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.id = ?",
                (uid,), uno=True
            )
        else:
            usuario = leer(
                "SELECT u.id, u.username, u.nombre_completo, r.nombre, u.estado "
                "FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE u.username = ?",
                (sub,), uno=True
            )

        if not usuario:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Usuario no encontrado")

        if usuario[4] != "activo":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Esta cuenta ha sido desactivada o suspendida por el administrador."
            )

        return {
            "id": usuario[0],
            "username": usuario[1],
            "nombre": usuario[2],
            "rol": usuario[3],
            "estado": usuario[4]
        }
    except HTTPException:
        raise
    except Exception as e:
        log.warning("Fallo en verificación de sesión: %r", e)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado")


# ==========================================
# RUTAS DE AUTENTICACIÓN
# ==========================================
@app.get("/")
async def root():
    return RedirectResponse(url="/login", status_code=303)


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    # Si ya tiene sesión activa válida, redirigir al dashboard
    try:
        await obtener_usuario_actual(request)
        return RedirectResponse(url="/dashboard", status_code=303)
    except HTTPException:
        pass

    response = templates.TemplateResponse(request, "login.html", {
        "error": error,
        "csrf_token": ""
    })
    csrf_token = _obtener_o_crear_csrf(request, response)
    # Re-render con el token CSRF asociado
    response = templates.TemplateResponse(request, "login.html", {
        "error": error,
        "csrf_token": csrf_token
    })
    _obtener_o_crear_csrf(request, response)
    return response


@app.post("/login")
async def login_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    csrf_token: Optional[str] = Form(None)
):
    ip = obtener_ip_cliente(request)
    user_clean = (username or "").strip()
    clave_limiter = f"login:{ip}:{user_clean.lower()}"

    # 1. Blindaje contra Fuerza Bruta (Rate Limiting)
    permitido, espera_seg = limiter_auth.es_permitido(clave_limiter, max_intentos=5, ventana_segundos=300)
    if not permitido:
        return templates.TemplateResponse(request, "login.html", {
            "error": f"Demasiados intentos fallidos. Por seguridad, intente en {espera_seg} segundos.",
            "csrf_token": _obtener_o_crear_csrf(request)
        }, status_code=status.HTTP_429_TOO_MANY_REQUESTS)

    # 2. Validación de Token Anti-CSRF
    try:
        verificar_csrf_request(request, token_form=csrf_token)
    except HTTPException as csrf_err:
        return templates.TemplateResponse(request, "login.html", {
            "error": csrf_err.detail,
            "csrf_token": _obtener_o_crear_csrf(request)
        }, status_code=403)

    # 3. Consulta de usuario en BD
    usuario = leer(
        "SELECT u.id, u.username, u.password_hash, u.nombre_completo, r.nombre, u.estado "
        "FROM usuarios u JOIN roles r ON u.rol_id = r.id WHERE LOWER(u.username) = LOWER(?)",
        (user_clean,), uno=True
    )

    # 4. Verificación de credenciales con tiempo constante
    if not usuario or not pwd_context.verify(password, usuario[2]):
        limiter_auth.registrar(clave_limiter)
        return templates.TemplateResponse(request, "login.html", {
            "error": "Credenciales inválidas. Verifique su usuario y contraseña.",
            "csrf_token": _obtener_o_crear_csrf(request)
        }, status_code=status.HTTP_401_UNAUTHORIZED)

    # 5. Verificación de estado de cuenta
    if usuario[5] != "activo":
        return templates.TemplateResponse(request, "login.html", {
            "error": "Esta cuenta se encuentra desactivada o suspendida. Contacte al administrador.",
            "csrf_token": _obtener_o_crear_csrf(request)
        }, status_code=status.HTTP_403_FORBIDDEN)

    # Restablecer intentos fallidos tras login exitoso
    limiter_auth.reiniciar(clave_limiter)

    # 6. Emisión de JWT seguro con expiración
    token = crear_token_acceso(usuario_id=usuario[0], username=usuario[1], rol=usuario[4])
    es_seguro = request.url.scheme == "https"

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {token}",
        max_age=TOKEN_EXPIRATION_SECONDS,
        httponly=True,
        samesite="lax",
        secure=es_seguro
    )
    # Refrescar token CSRF tras inicio de sesión
    nuevo_csrf = generar_csrf_token()
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=nuevo_csrf,
        max_age=TOKEN_EXPIRATION_SECONDS,
        httponly=False,
        samesite="lax",
        secure=es_seguro
    )
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    response.delete_cookie(CSRF_COOKIE_NAME)
    return response


# ==========================================
# RUTAS DE INTERFAZ WEB (VISTAS)
# ==========================================
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    try:
        user = await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

    response = templates.TemplateResponse(request, "dashboard.html", {
        "usuario_actual": user,
        "menu": MENU,
        "modo": MODO,
        "csrf_token": ""
    })
    csrf_token = _obtener_o_crear_csrf(request, response)
    response = templates.TemplateResponse(request, "dashboard.html", {
        "usuario_actual": user,
        "menu": MENU,
        "modo": MODO,
        "csrf_token": csrf_token
    })
    _obtener_o_crear_csrf(request, response)
    return response


@app.get("/productos", response_class=HTMLResponse)
async def pagina_productos(request: Request):
    """Redirige al dashboard unificado donde se integra el catálogo de inventario."""
    try:
        await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard#tab-inventario", status_code=303)


@app.get("/usuarios", response_class=HTMLResponse)
async def listar_usuarios(request: Request, error: Optional[str] = None, mensaje: Optional[str] = None):
    try:
        user = await obtener_usuario_actual(request)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=303)

    if not es_rol_gestion(user["rol"]):
        raise HTTPException(status_code=403, detail="Acceso denegado. Permisos insuficientes.")

    usuarios = leer(
        "SELECT u.id, u.username, u.nombre_completo, r.nombre, u.estado "
        "FROM usuarios u JOIN roles r ON u.rol_id = r.id ORDER BY u.id ASC"
    )
    roles = leer("SELECT id, nombre FROM roles ORDER BY id ASC")

    response = templates.TemplateResponse(request, "usuarios.html", {
        "usuarios": usuarios,
        "roles": roles,
        "usuario_actual": user,
        "menu": MENU,
        "modo": MODO,
        "error": error,
        "mensaje": mensaje,
        "csrf_token": ""
    })
    csrf_token = _obtener_o_crear_csrf(request, response)
    response = templates.TemplateResponse(request, "usuarios.html", {
        "usuarios": usuarios,
        "roles": roles,
        "usuario_actual": user,
        "menu": MENU,
        "modo": MODO,
        "error": error,
        "mensaje": mensaje,
        "csrf_token": csrf_token
    })
    _obtener_o_crear_csrf(request, response)
    return response


# ==========================================
# GESTIÓN SEGURA DE IDENTIDADES (USUARIOS)
# ==========================================
@app.post("/usuarios/crear")
async def crear_usuario(
    request: Request,
    nombre: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    rol_id: int = Form(...),
    csrf_token: Optional[str] = Form(None)
):
    user = await obtener_usuario_actual(request)
    if not es_rol_gestion(user["rol"]):
        raise HTTPException(status_code=403, detail="Permisos insuficientes")

    # Validación Anti-CSRF
    verificar_csrf_request(request, token_form=csrf_token)

    # Validación de política de contraseñas seguras
    es_valida, msg_error = validar_password_complejidad(password)
    if not es_valida:
        return RedirectResponse(url=f"/usuarios?error={msg_error}", status_code=303)

    # Validar rol existente
    rol_existe = leer("SELECT id FROM roles WHERE id = ?", (rol_id,), uno=True)
    if not rol_existe:
        return RedirectResponse(url="/usuarios?error=El+rol+seleccionado+no+es+valido", status_code=303)

    username_clean = username.strip().lower()
    try:
        escribir(
            "INSERT INTO usuarios (username, password_hash, nombre_completo, rol_id, estado) VALUES (?, ?, ?, ?, 'activo')",
            (username_clean, pwd_context.hash(password), nombre.strip(), rol_id)
        )
    except Exception:
        return RedirectResponse(url="/usuarios?error=El+nombre+de+usuario+ya+se+encuentra+registrado", status_code=303)

    return RedirectResponse(url="/usuarios?mensaje=Usuario+registrado+exitosamente", status_code=303)


@app.post("/usuarios/editar/{usuario_id}")
async def editar_usuario(
    request: Request,
    usuario_id: int,
    nombre: str = Form(...),
    rol_id: int = Form(...),
    estado: str = Form(...),
    password: Optional[str] = Form(None),
    csrf_token: Optional[str] = Form(None)
):
    user = await obtener_usuario_actual(request)
    if not es_rol_gestion(user["rol"]):
        raise HTTPException(status_code=403, detail="Permisos insuficientes")

    # Validación Anti-CSRF
    verificar_csrf_request(request, token_form=csrf_token)

    estado_clean = "activo" if estado == "activo" else "inactivo"

    # Prevenir que un admin se desactive a sí mismo
    if user["id"] == usuario_id and estado_clean != "activo":
        return RedirectResponse(url="/usuarios?error=No+puedes+desactivar+tu+propia+cuenta+de+acceso", status_code=303)

    # Prevenir que se desactive el último administrador del sistema
    if estado_clean != "activo":
        usuario_target = obtener_usuario_por_id(usuario_id)
        if usuario_target and es_rol_gestion(usuario_target[4]):
            if contar_administradores_activos() <= 1:
                return RedirectResponse(
                    url="/usuarios?error=No+se+puede+desactivar+al+unico+administrador+activo+del+sistema",
                    status_code=303
                )

    # Validar contraseña si fue provista
    password_hash = None
    if password and password.strip():
        es_valida, msg_error = validar_password_complejidad(password.strip())
        if not es_valida:
            return RedirectResponse(url=f"/usuarios?error={msg_error}", status_code=303)
        password_hash = pwd_context.hash(password.strip())

    actualizar_usuario(
        usuario_id=usuario_id,
        nombre=nombre.strip(),
        rol_id=rol_id,
        estado=estado_clean,
        password_hash=password_hash
    )
    return RedirectResponse(url="/usuarios?mensaje=Usuario+actualizado+correctamente", status_code=303)


@app.post("/usuarios/eliminar/{usuario_id}")
async def eliminar_usuario(
    request: Request,
    usuario_id: int,
    csrf_token: Optional[str] = Form(None)
):
    user = await obtener_usuario_actual(request)
    if not es_rol_gestion(user["rol"]):
        raise HTTPException(status_code=403, detail="Permisos insuficientes")

    # Validación Anti-CSRF
    verificar_csrf_request(request, token_form=csrf_token)

    if user["id"] == usuario_id:
        return RedirectResponse(url="/usuarios?error=No+puedes+eliminar+tu+propio+usuario", status_code=303)

    usuario_target = obtener_usuario_por_id(usuario_id)
    if usuario_target and es_rol_gestion(usuario_target[4]):
        if contar_administradores_activos() <= 1:
            return RedirectResponse(
                url="/usuarios?error=No+es+posible+eliminar+al+unico+administrador+del+sistema",
                status_code=303
            )

    escribir("DELETE FROM usuarios WHERE id = ?", (usuario_id,))
    return RedirectResponse(url="/usuarios?mensaje=Usuario+eliminado+correctamente", status_code=303)


# ==========================================
# ENDPOINTS DE LA API REST (IA, STATS, VIDEO)
# ==========================================
@app.get("/api/sources")
async def listar_fuentes_camara(request: Request):
    await obtener_usuario_actual(request)
    return {
        "fuentes": [
            {"id": "webcam", "nombre": "Cámara Web Predeterminada (USB 0)"},
            {"id": "rtsp", "nombre": "Cámara IP Percha Principal (RTSP)"},
            {"id": "demo", "nombre": "Video Demostración Supermercado"}
        ]
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    user = await obtener_usuario_actual(request)
    
    # Rate Limiting preventivo para llamadas de IA
    permitido, espera_seg = limiter_chat.es_permitido(f"chat:{user['id']}", max_intentos=30, ventana_segundos=60)
    if not permitido:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Límite de mensajes por minuto alcanzado. Espere {espera_seg} segundos."
        )

    try:
        datos = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Solicitud inválida")

    mensaje = (datos.get("mensaje") or "").strip()
    modo = datos.get("modo", "datos")

    if not mensaje:
        raise HTTPException(status_code=400, detail="El mensaje está vacío")

    estado_vivo = camara.estado_dict() if camara else {}

    try:
        respuesta_ia = await preguntar_a_los_datos(mensaje, modo=modo, vivo=estado_vivo)
    except Exception as e:
        log.error(f"Fallo en motor de chat: {e}", exc_info=True)
        from ai_agent import generar_estrategia_local
        respuesta_ia = generar_estrategia_local(mensaje)

    return {"respuesta": respuesta_ia}


@app.post("/api/upload")
async def api_upload(request: Request, archivo: UploadFile = File(...)):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="La subida de video solo está disponible en modo local")

    # Verificación Anti-CSRF
    verificar_csrf_request(request)

    ext = os.path.splitext(archivo.filename or "")[1].lower()
    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        tipo = "video"
    elif ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        tipo = "imagen"
    else:
        raise HTTPException(status_code=400, detail="Formato no soportado por el motor de visión")

    os.makedirs("uploads", exist_ok=True)
    nombre_seguro = f"{uuid.uuid4().hex}{ext}"
    ruta = os.path.join("uploads", nombre_seguro)

    # Streaming seguro en bloques con validación binaria de firma (Magic Bytes)
    exito, err_msg = await guardar_archivo_seguro(archivo, ruta, max_bytes=200 * 1024 * 1024)
    if not exito:
        raise HTTPException(status_code=400, detail=err_msg)

    camara.cambiar_fuente(tipo, ruta)
    return {"ok": True, "tipo": tipo}


@app.post("/api/control")
async def api_control(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="Control solo disponible en modo local")

    verificar_csrf_request(request)
    datos = await request.json()
    accion = datos.get("accion")
    if accion == "pausa":
        return {"ok": True, "pausado": camara.toggle_pausa()}
    if accion == "parar":
        camara.apagar()
        return {"ok": True, "pausado": False}
    raise HTTPException(status_code=400, detail="Acción no válida")


@app.post("/api/fuente")
async def cambiar_fuente_api(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        raise HTTPException(status_code=503, detail="Cambio de fuente solo disponible en modo local")

    verificar_csrf_request(request)
    datos = await request.json()
    tipo = datos.get("tipo", "video")
    if tipo not in ("webcam", "rtsp", "video", "imagen", "apagado"):
        raise HTTPException(status_code=400, detail="Fuente no válida")

    ruta = datos.get("ruta")
    if not ruta:
        rutas = {
            "video": DEMO_VIDEO if os.path.exists(DEMO_VIDEO) else "static/demo_supermercado.mp4",
            "imagen": "static/demo.jpg" if os.path.exists("static/demo.jpg") else "uploads/demo.jpg"
        }
        ruta = rutas.get(tipo, "")

    # Sanitización estricta contra Path Traversal
    es_valida, ruta_sanitizada = validar_ruta_fuente_segura(tipo, ruta)
    if not es_valida:
        raise HTTPException(status_code=400, detail=ruta_sanitizada)

    camara.cambiar_fuente(tipo, ruta_sanitizada)
    # Enmascara contraseñas y credenciales sensibles para que no viajen al navegador
    ruta_enmascarada = enmascarar_url_conexion(ruta_sanitizada)
    return {"ok": True, "tipo": tipo, "ruta": ruta_enmascarada}


# ==========================================
# ESTADÍSTICAS Y TELEMETRÍA (Lógica de Negocio Centralizada en Backend)
# ==========================================
def _obtener_stats():
    trafico = leer("SELECT COUNT(*) FROM eventos WHERE tipo = 'trafico'", uno=True)[0]
    perm = leer("SELECT COUNT(*), COALESCE(AVG(valor), 0) FROM eventos WHERE tipo = 'permanencia'", uno=True)
    inter = leer("SELECT COUNT(*) FROM eventos WHERE tipo IN ('interaccion', 'producto_tomado')", uno=True)[0]
    tomados = leer("SELECT COUNT(*) FROM eventos WHERE tipo = 'producto_tomado'", uno=True)[0]
    devueltos = leer("SELECT COUNT(*) FROM eventos WHERE tipo = 'producto_devuelto'", uno=True)[0]
    top = leer("""SELECT detalle, COUNT(*) FROM eventos WHERE tipo IN ('interaccion', 'producto_tomado')
                  GROUP BY detalle ORDER BY COUNT(*) DESC LIMIT 5""")

    # 1. Tasa de conversión general (calculada en el servidor)
    tasa_conversion_num = round((inter / trafico * 100), 1) if trafico > 0 else 0.0
    tasa_conversion_formateada = f"{tasa_conversion_num:.1f}%"

    # 2. Tasa de retención/retorno de percha (calculada en el servidor)
    if tomados > 0:
        tasa_retorno_num = max(0, round(((tomados - devueltos) / tomados) * 100))
        tasa_retorno_formateada = f"{tasa_retorno_num}%"
    else:
        tasa_retorno_num = 100
        tasa_retorno_formateada = "100%"

    # 3. Lógica comercial y recomendaciones de marketing por producto (enriquecida en backend)
    top_productos_enriquecidos = []
    for p in top:
        prod_nombre = p[0]
        veces = p[1]
        trafico_prod = trafico if trafico > veces else veces + 10
        conv_prod = round((veces / trafico_prod) * 100, 1) if trafico_prod > 0 else 0.0

        if conv_prod < 10.0:
            badge_class = "bg-orange-100 text-kovaq-orange font-bold"
            recomendacion = f"Baja evacuación. Lanzar Reels geolocalizados para {prod_nombre}."
        elif conv_prod < 20.0:
            badge_class = "bg-amber-100 text-amber-900 font-bold"
            recomendacion = f"Duda de compra. Ofrecer cupón digital para {prod_nombre}."
        else:
            badge_class = "bg-emerald-100 text-emerald-800 font-bold"
            recomendacion = "Rendimiento positivo. Redirigir pauta hacia SKUs secundarios."

        top_productos_enriquecidos.append({
            "producto": prod_nombre,
            "veces": veces,
            "trafico_estimado": trafico_prod,
            "conversion_porcentaje": conv_prod,
            "badge_class": badge_class,
            "recomendacion": recomendacion
        })

    # 4. Diagnóstico de fricción y oportunidad de pauta comercial
    diagnostico_friccion = f"Evaluados {trafico} tránsitos en góndola. Tasa de conversión general calculada en {tasa_conversion_num:.1f}%."
    if tasa_conversion_num < 12.0 and trafico > 10:
        oportunidad_pauta = "Alto tráfico pero baja toma de producto. Ideal para campaña publicitaria de choque inmediata."
        oportunidad_pauta_html = '<strong class="text-kovaq-orange">Oportunidad Pauta:</strong> Alto tráfico pero baja toma de producto. Ideal para campaña publicitaria de choque inmediata.'
    else:
        oportunidad_pauta = "Rendimiento orgánico favorable en percha. Mantener rotación habitual."
        oportunidad_pauta_html = "Rendimiento orgánico favorable en percha. Mantener rotación habitual."

    return {
        "trafico": trafico,
        "permanencia_eventos": perm[0],
        "permanencia_promedio_seg": round(float(perm[1]), 1),
        "interacciones": inter,
        "tomados": tomados,
        "devueltos": devueltos,
        "tasa_conversion": tasa_conversion_num,
        "tasa_conversion_formateada": tasa_conversion_formateada,
        "tasa_retorno": tasa_retorno_num,
        "tasa_retorno_formateada": tasa_retorno_formateada,
        "diagnostico_friccion": diagnostico_friccion,
        "oportunidad_pauta": oportunidad_pauta,
        "oportunidad_pauta_html": oportunidad_pauta_html,
        "top_productos": top_productos_enriquecidos,
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
        s = await asyncio.to_thread(_obtener_stats)
    except Exception:
        raise HTTPException(status_code=500, detail="Error consultando estadísticas")

    s["resumen"] = await asyncio.to_thread(_texto_resumen, s)
    return s


@app.get("/api/live_events")
async def api_live_events(request: Request):
    await obtener_usuario_actual(request)
    if camara is None:
        return {
            "fps_cam": 0, "fps_yolo": 0, "latencia_ms": 0,
            "agarrados": 0, "devueltos": 0, "entradas": 0,
            "personas_activas": 0, "estado": "apagado", "eventos": []
        }
    return camara.obtener_telemetria()


# ==========================================
# VIDEO STREAMING (LOCAL)
# ==========================================
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
            return

    return StreamingResponse(generar(), media_type="multipart/x-mixed-replace; boundary=frame")


# ==========================================
# VIDEO EN VIVO PARA LA WEB (MODO NUBE)
# ==========================================
VIDEO_TOKEN = os.getenv("VIDEO_PUSH_TOKEN", "")
_ultimo_frame = {"jpg": None, "id": 0}
_frame_lock = threading.Lock()


@app.post("/api/frame_push")
async def frame_push(request: Request):
    token_recibido = request.headers.get("X-Token", "")
    if not VIDEO_TOKEN or not hmac.compare_digest(token_recibido, VIDEO_TOKEN):
        raise HTTPException(status_code=403, detail="Token de ingesta inválido")

    jpg = await request.body()
    if not jpg or len(jpg) > 2_000_000:
        raise HTTPException(status_code=400, detail="Frame inválido o supera tamaño permitido")

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
