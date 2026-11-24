"""
security.py — Módulo de Blindaje, Criptografía y Control de Acceso para KOVAQ.
Implementa:
1. Gestión criptográfica de JWT con expiración estricta y claims estándar (RFC 7519).
2. Rate Limiting en memoria para mitigación de Fuerza Bruta y DoS.
3. Validación y protección Anti-CSRF basada en tokens criptográficos y Origin/Referer.
4. Políticas de complejidad de contraseñas.
5. Validación binaria de archivos (Magic Bytes) y streaming seguro para prevenir OOM.
6. Sanitización contra Path Traversal en fuentes de video.
7. Middleware de inyección de Security Headers HTTP (OWASP Best Practices).
"""

import os
import re
import time
import uuid
import hmac
import secrets
import logging
from typing import Tuple, Dict, Optional
from urllib.parse import urlparse

import jwt
from fastapi import Request, HTTPException, status, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

log = logging.getLogger("kovaq.seguridad")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 1. GESTIÓN CRIPTOGRÁFICA DE JWT & SECRETOS
# ---------------------------------------------------------------------------
SECRET_KEY = (
    os.getenv("JWT_SECRET", "").strip()
    or os.getenv("SECRET_KEY", "").strip()
    or "kovaq_jwt_fallback_secret_key_2026_hardened_prod"
)
ALGORITHM = "HS256"
TOKEN_EXPIRATION_SECONDS = 12 * 3600  # 12 Horas de vigencia máxima


def crear_token_acceso(usuario_id: int, username: str, rol: str) -> str:
    """Genera un JWT blindado con timestamps iat/exp y jti único."""
    ahora = int(time.time())
    payload = {
        "sub": username.strip(),
        "uid": usuario_id,
        "rol": rol.strip(),
        "iat": ahora,
        "exp": ahora + TOKEN_EXPIRATION_SECONDS,
        "jti": uuid.uuid4().hex
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decodificar_token_acceso(token: str) -> dict:
    """Verifica y decodifica un JWT comprobando firma, expiración y claims obligatorios."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if "exp" not in payload or "sub" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token incompleto o inválido"
            )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La sesión ha expirado. Inicie sesión nuevamente."
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial de acceso inválida o adulterada."
        )


# ---------------------------------------------------------------------------
# 2. POLÍTICA DE COMPLEJIDAD DE CONTRASEÑAS
# ---------------------------------------------------------------------------
PASSWORDS_COMUNES = {
    "12345678", "password", "password123", "admin123", "admin1234",
    "123456789", "qwerty123", "supermercado", "kovaq123", "clave123"
}

def validar_password_complejidad(password: str) -> Tuple[bool, str]:
    """
    Exige una contraseña robusta:
    - Mínimo 8 caracteres
    - Al menos una mayúscula y una minúscula
    - Al menos un dígito numérico
    - Rechazo de contraseñas de diccionario comunes
    """
    if not password or len(password) < 8:
        return False, "La contraseña debe tener al menos 8 caracteres."
    if len(password) > 128:
        return False, "La contraseña no debe superar los 128 caracteres."
    if password.lower() in PASSWORDS_COMUNES:
        return False, "La contraseña ingresada es demasiado común y predecible."
    if not re.search(r"[A-Z]", password):
        return False, "La contraseña debe incluir al menos una letra mayúscula."
    if not re.search(r"[a-z]", password):
        return False, "La contraseña debe incluir al menos una letra minúscula."
    if not re.search(r"[0-9]", password):
        return False, "La contraseña debe incluir al menos un número."
    return True, ""


# ---------------------------------------------------------------------------
# 3. RATE LIMITING EN MEMORIA (Mitigación de Brute Force y DoS)
# ---------------------------------------------------------------------------
class RateLimiterMemoria:
    """Limitador de tasa en memoria con limpieza automática de registros antiguos."""
    def __init__(self):
        self._intentos: Dict[str, list] = {}
        self._ultimo_cleanup = time.time()

    def _cleanup(self):
        ahora = time.time()
        if ahora - self._ultimo_cleanup > 300:  # Cada 5 minutos
            umbral = ahora - 600
            for clave in list(self._intentos.keys()):
                self._intentos[clave] = [t for t in self._intentos[clave] if t > umbral]
                if not self._intentos[clave]:
                    del self._intentos[clave]
            self._ultimo_cleanup = ahora

    def es_permitido(self, clave: str, max_intentos: int, ventana_segundos: int) -> Tuple[bool, int]:
        self._cleanup()
        ahora = time.time()
        umbral = ahora - ventana_segundos
        historial = [t for t in self._intentos.get(clave, []) if t > umbral]
        self._intentos[clave] = historial

        if len(historial) >= max_intentos:
            segundos_restantes = int(ventana_segundos - (ahora - historial[0]))
            return False, max(1, segundos_restantes)
        return True, 0

    def registrar(self, clave: str):
        self._cleanup()
        ahora = time.time()
        if clave not in self._intentos:
            self._intentos[clave] = []
        self._intentos[clave].append(ahora)

    def reiniciar(self, clave: str):
        self._intentos.pop(clave, None)


limiter_auth = RateLimiterMemoria()   # Máx 5 intentos fallidos en 5 min por IP/usuario
limiter_chat = RateLimiterMemoria()   # Máx 25 preguntas por minuto por usuario


def obtener_ip_cliente(request: Request) -> str:
    """Extrae la IP real del cliente respetando headers de proxy inverso si existen."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


# ---------------------------------------------------------------------------
# 4. PROTECCIÓN ANTI-CSRF (Tokens de un solo uso / verificación de origen)
# ---------------------------------------------------------------------------
CSRF_COOKIE_NAME = "kovaq_csrf_token"

def generar_csrf_token() -> str:
    """Genera un token CSRF criptográficamente seguro."""
    return secrets.token_urlsafe(32)


def verificar_csrf_request(request: Request, token_form: Optional[str] = None):
    """
    Valida la legitimidad de peticiones mutantes (POST, PUT, DELETE).
    1. Verifica coincidencia estricta entre la cookie de sesión CSRF y el token recibido.
    2. Comprueba que el header Origin/Referer coincida con el Host de la solicitud.
    """
    # 1. Comprobación de Cabecera Origin / Referer
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")

    if origin:
        parsed_origin = urlparse(origin)
        if parsed_origin.netloc and parsed_origin.netloc != host:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Violación de seguridad Origin: Solicitud rechazada por política Cross-Origin."
            )
    elif referer:
        parsed_ref = urlparse(referer)
        if parsed_ref.netloc and parsed_ref.netloc != host:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Violación de seguridad Referer: Solicitud rechazada por política Cross-Origin."
            )

    # 2. Comprobación del Token Anti-CSRF
    cookie_token = request.cookies.get(CSRF_COOKIE_NAME)
    header_token = request.headers.get("X-CSRF-Token")
    token_evaluar = token_form or header_token

    if not cookie_token or not token_evaluar:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Falta el token de seguridad CSRF. Recargue la página e intente nuevamente."
        )

    if not hmac.compare_digest(cookie_token, token_evaluar):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token de validación CSRF inválido o caducado."
        )


# ---------------------------------------------------------------------------
# 5. SANITIZACIÓN DE ARCHIVOS & STREAMING SEGURO (Anti-DoS / Anti-OOM)
# ---------------------------------------------------------------------------
MAGIC_SIGNATURES = {
    ".mp4": [b"ftyp"],           # MP4 ISO
    ".jpg": [b"\xff\xd8\xff"],   # JPEG
    ".jpeg": [b"\xff\xd8\xff"],  # JPEG
    ".png": [b"\x89PNG\r\n\x1a\n"], # PNG
    ".webp": [b"RIFF"],          # WEBP
    ".bmp": [b"BM"],             # BMP
    ".avi": [b"RIFF"],           # AVI
    ".mkv": [b"\x1a\x45\xdf\xa3"]# Matroska
}

def validar_firma_magica(chunk_inicial: bytes, extension: str) -> bool:
    """Verifica que los primeros bytes del archivo correspondan realmente a su formato declarado."""
    ext = extension.lower()
    firmas = MAGIC_SIGNATURES.get(ext)
    if not firmas:
        return False
    for sig in firmas:
        if sig in chunk_inicial[:32]:
            return True
    return False


async def guardar_archivo_seguro(
    archivo: UploadFile,
    ruta_destino: str,
    max_bytes: int = 200 * 1024 * 1024
) -> Tuple[bool, str]:
    """
    Lee y escribe el archivo en bloques (chunks de 64KB) en disco sin cargarlo por
    completo en la memoria RAM, impidiendo ataques de agotamiento de memoria (OOM).
    Comprueba magic bytes en el primer bloque.
    """
    ext = os.path.splitext(archivo.filename or "")[1].lower()
    tamano_acumulado = 0
    primer_chunk = True

    try:
        with open(ruta_destino, "wb") as f_out:
            while True:
                chunk = await archivo.read(64 * 1024)
                if not chunk:
                    break
                tamano_acumulado += len(chunk)

                if tamano_acumulado > max_bytes:
                    f_out.close()
                    if os.path.exists(ruta_destino):
                        os.remove(ruta_destino)
                    return False, f"El archivo supera el tamaño máximo permitido de {max_bytes // (1024*1024)} MB."

                if primer_chunk:
                    if not validar_firma_magica(chunk, ext):
                        f_out.close()
                        if os.path.exists(ruta_destino):
                            os.remove(ruta_destino)
                        return False, "La cabecera binaria del archivo no coincide con su extensión declarada."
                    primer_chunk = False

                f_out.write(chunk)

        if tamano_acumulado == 0:
            if os.path.exists(ruta_destino):
                os.remove(ruta_destino)
            return False, "El archivo subido está completamente vacío."

        return True, ""
    except Exception as e:
        log.error("Fallo escribiendo archivo seguro: %r", e)
        if os.path.exists(ruta_destino):
            try:
                os.remove(ruta_destino)
            except Exception:
                pass
        return False, f"Error durante la escritura del archivo: {str(e)}"


# ---------------------------------------------------------------------------
# 6. SANITIZACIÓN CONTRA PATH TRAVERSAL EN FUENTES DE CÁMARA
# ---------------------------------------------------------------------------
def validar_ruta_fuente_segura(tipo: str, ruta: str) -> Tuple[bool, str]:
    """
    Asegura que el parámetro 'ruta' de /api/fuente apunte estrictamente
    a directorios permitidos (static, uploads) o a URLs RTSP/HTTP válidas.
    """
    if tipo == "webcam":
        return True, "0"

    if tipo == "rtsp":
        ruta_clean = ruta.strip()
        if re.match(r"^(rtsp|rtsps|http|https)://[a-zA-Z0-9_\-\.:@/]+", ruta_clean):
            return True, ruta_clean
        if ruta_clean in ("0", "1", "2"):
            return True, ruta_clean
        return False, "La URL RTSP proporcionada tiene un formato no válido."

    if tipo in ("video", "imagen"):
        if not ruta:
            return True, ""
        ruta_normalizada = os.path.normpath(ruta)
        ruta_abs = os.path.abspath(ruta_normalizada)
        permitido_static = os.path.abspath(os.path.join(BASE_DIR, "static"))
        permitido_uploads = os.path.abspath(os.path.join(BASE_DIR, "uploads"))

        if not (ruta_abs.startswith(permitido_static) or ruta_abs.startswith(permitido_uploads)):
            return False, "Acceso denegado: La ruta no pertenece a los directorios autorizados del sistema."

        if not os.path.exists(ruta_abs):
            return False, "El archivo multimedia especificado no existe en el almacenamiento."

        return True, ruta_abs

    if tipo == "apagado":
        return True, ""

    return False, "Tipo de fuente desconocido."


def enmascarar_url_conexion(url: str) -> str:
    """
    Oculta contraseñas y credenciales sensibles en URLs RTSP, HTTP o cadenas de conexión.
    Ejemplo: 'rtsp://admin:pass123@192.168.1.50:554/live' -> 'rtsp://admin:***@192.168.1.50:554/live'
    """
    if not url:
        return ""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


# ---------------------------------------------------------------------------
# 7. MIDDLEWARE DE SECURITY HEADERS HTTP (Defensa en Profundidad)
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inyecta cabeceras de seguridad estrictas en todas las respuestas HTTP."""
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)

        # 1. Anti-Clickjacking: Prohíbe incrustación en <iframe> ajenos
        response.headers["X-Frame-Options"] = "SAMEORIGIN"

        # 2. Anti-MIME-Sniffing: Obliga al navegador a respetar el Content-Type real
        response.headers["X-Content-Type-Options"] = "nosniff"

        # 3. Referrer Policy: Protege privacidad en enlaces salientes
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # 4. Permissions Policy: Deshabilita hardware no utilizado
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"

        # 5. X-XSS-Protection (Compatibilidad con navegadores antiguos)
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # 6. Content Security Policy (Permite CDNs del diseño existente sin alterarlo)
        csp_policy = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net; "
            "font-src 'self' https://fonts.gstatic.com https://cdn.jsdelivr.net data:; "
            "img-src 'self' data: blob: https:; "
            "media-src 'self' data: blob:; "
            "connect-src 'self' ws: wss:; "
            "frame-ancestors 'self';"
        )
        response.headers["Content-Security-Policy"] = csp_policy

        return response
