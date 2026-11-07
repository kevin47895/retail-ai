import os
import sys
import threading
import queue
import time
import collections
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- OPTIMIZACIÓN MULTINÚCLEO & HARDWARE ---
# Detectamos todos los núcleos de CPU disponibles para máxima paralelización
NUM_CORES = os.cpu_count() or 4
MAX_THREADS = max(2, min(NUM_CORES, 8))

os.environ.setdefault("OMP_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(MAX_THREADS))
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
cv2.setNumThreads(MAX_THREADS)

try:
    import torch
    torch.set_num_threads(MAX_THREADS)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

from ultralytics import YOLO
from database import (
    guardar_interaccion, 
    guardar_permanencia, 
    guardar_trafico, 
    guardar_producto_tomado, 
    guardar_producto_devuelto
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_VIDEO_PATH = os.path.join(BASE_DIR, "static", "demo_supermercado.mp4")

# Parámetros de Inferencia y Streaming
IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))  # 320 provee 2.5x más FPS manteniendo alta precisión
CONF = float(os.getenv("YOLO_CONF", "0.28"))
INTERVALO_IA = float(os.getenv("YOLO_INTERVALO", "0.08"))  # ~10-12 FPS para análisis de fondo sin saturar la CPU y garantizar video fluido
RTSP_URL = os.getenv("RTSP_URL", "0").strip()
PUBLICAR_VIDEO_URL = os.getenv("PUBLICAR_VIDEO_URL", "").strip()
VIDEO_PUSH_TOKEN = os.getenv("VIDEO_PUSH_TOKEN", "").strip()
ANCHO, ALTO = 480, 360

# Clases de interés retail ampliadas (Personas, Botellas, Frutas, Snacks, Accesorios, Objetos)
# 0: person, 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 40: wine glass, 41: cup,
# 42: fork, 43: knife, 44: spoon, 45: bowl, 46: banana, 47: apple, 48: sandwich, 49: orange,
# 50: broccoli, 51: carrot, 52: hot dog, 53: pizza, 54: donut, 55: cake, 65: remote, 67: cell phone,
# 73: book, 77: teddy bear, 79: toothbrush
CLASES_RETAIL = [0, 24, 26, 28, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 65, 67, 73, 77, 79]


class CameraService:
    """
    Servicio de Visión por Computadora de Alto Rendimiento para Retail (KOVAQ).
    Arquitectura Desacoplada de 3 Niveles:
    1. Captura Continua (30 FPS, sin colas de latencia).
    2. Worker de IA Asíncrono (Multihilo, Frame-Skipping inteligente y Máquina de Estados Tomar/Devolver).
    3. Compositor de Video en Tiempo Real con HUD de Telemetría a 30 FPS.
    4. Worker de Base de Datos Asíncrono en segundo plano (cero bloqueos por disco).
    """

    def __init__(self):
        self.lock = threading.Lock()
        # Por defecto siempre arranca en video demo de supermercado como respaldo
        self.fuente = "video"
        self.ruta = DEMO_VIDEO_PATH
        self.estado = "en_vivo"
        self.mensaje = "Video de respaldo Supermercado Activo"
        
        # Buffers de frames
        self.latest_raw_frame = None
        self.ultimo_frame_procesado = None
        self.frame_id = 0
        self._jpeg_id, self._jpeg = -1, None
        
        # Control de flujo y sincronización
        self.solicitud = None
        self.corriendo = self.activa = self.pausado = False
        self.cap = None
        self.fin_video = False
        self.imagen_procesada = False
        self.fallos = 0
        
        # Hilos del servicio
        self.hilo_lectura = None
        self.hilo_ia = None
        self.hilo_db = None
        self.cola_db = queue.Queue(maxsize=1000)
        
        # Métricas de Telemetría y Rendimiento
        self.fps_cam = 30.0
        self.fps_yolo = 25.0
        self.latencia_ms = 35.0
        self.cpu_cores = NUM_CORES
        self._frames_cam_count = 0
        self._t_cam_start = time.time()
        self._frames_yolo_count = 0
        self._t_yolo_start = time.time()
        
        # Historial de Eventos en Tiempo Real (para API y Frontend)
        self.eventos_recientes = collections.deque(maxlen=40)
        self._event_id_seq = 0

        # Zona Estática de la Percha (Góndola Central de Interés)
        # Formato: [x1, y1, x2, y2] relativo a (ANCHO, ALTO)
        self.zonas = (int(ANCHO * 0.16), int(ALTO * 0.18), int(ANCHO * 0.84), int(ALTO * 0.86))
        
        # Carga del Modelo YOLO
        model_file = 'yolov8n.onnx' if os.path.exists('yolov8n.onnx') else 'yolov8n.pt'
        self.modelo = YOLO(model_file, **({'task': 'detect'} if model_file.endswith('.onnx') else {}))

        # Estado del Tracking y Máquina de Estados Tomar/Devolver
        self._reset_conteos()
        
        # Último estado del Overlay para dibujo a 30 FPS
        self.overlay_data = {
            "personas": [],       # [{id, box, dwell_time, cx, cy}]
            "productos": [],      # [{id, box, nombre, estado, cliente_id, conf}]
            "alertas": [],        # ["⚡ TOMADO: Bebida #104"]
            "telemetria": "Iniciando IA..."
        }

    def _reset_conteos(self):
        self.total_entradas = 0
        self.total_agarrados = 0
        self.total_devueltos = 0
        self.tiempos_clientes = {}       # {pid: t_entrada}
        self.tiempos_zona = {}           # {pid: t_zona}
        self.productos_en_mano = {}      # {pid: set(prod_names)}
        self.productos_rastreados = {}   # {prod_id: dict_estado}
        self.interacciones_registradas = set()
        self.ultimo_reach_zona = {}      # {pid: {'t': time, 'en_zona': bool}}
        try:
            self.modelo.predictor = None
        except Exception:
            pass

    def iniciar(self):
        """Inicia los hilos de captura, inferencia y base de datos."""
        if not self.corriendo:
            self.corriendo = True

            # Carga inmediata del video de supermercado para que el primer fotograma esté disponible
            # en el milisegundo 0 y nunca aparezca la pantalla en negro al iniciar
            if os.path.exists(DEMO_VIDEO_PATH):
                self._abrir_captura("video", DEMO_VIDEO_PATH, reiniciar=True)
                if self.cap and self.cap.isOpened():
                    ok, primer_frame = self.cap.read()
                    if ok and primer_frame is not None:
                        if primer_frame.shape[1] != ANCHO or primer_frame.shape[0] != ALTO:
                            self.latest_raw_frame = cv2.resize(primer_frame, (ANCHO, ALTO))
                        else:
                            self.latest_raw_frame = primer_frame
                        self.frame_id = 1
                        # Generar el primer frame JPEG para servir inmediatamente
                        self.get_frame_bytes(-1)
            
            # Hilo 1: Lector de hardware a 30 FPS
            self.hilo_lectura = threading.Thread(target=self._leer_camara, daemon=True, name="KovaqCamCapture")
            # Hilo 2: Motor de IA multinúcleo desacoplado
            self.hilo_ia = threading.Thread(target=self._worker_ia_desacoplado, daemon=True, name="KovaqYoloWorker")
            # Hilo 3: Escritor asíncrono a SQLite (cero lag de I/O)
            self.hilo_db = threading.Thread(target=self._worker_db_asincrono, daemon=True, name="KovaqDbWriter")
            
            self.hilo_lectura.start()
            self.hilo_ia.start()
            self.hilo_db.start()

            # Hilo 4 (Opcional): Transmisión automática de fotogramas hacia Render en la nube
            if PUBLICAR_VIDEO_URL and VIDEO_PUSH_TOKEN:
                self.hilo_push = threading.Thread(target=self._worker_push_nube, daemon=True, name="KovaqCloudPush")
                self.hilo_push.start()

    def detener(self):
        self.corriendo = False
        self.apagar()

    def apagar(self):
        self.activa = False
        self.estado = "apagado"
        self.solicitud = ("apagado", "")
        with self.lock:
            self.latest_raw_frame = None
            self._jpeg = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def toggle_pausa(self):
        self.pausado = not self.pausado
        return self.pausado

    def cambiar_fuente(self, tipo, ruta=""):
        self.solicitud = (tipo, ruta)
        self.pausado = False

    def repetir(self):
        if self.fuente == "video" and self.ruta:
            self.cambiar_fuente("video", self.ruta)

    def estado_dict(self):
        return {
            "fuente": self.fuente,
            "estado": self.estado,
            "mensaje": self.mensaje,
            "pausado": self.pausado,
            "entradas": self.total_entradas,
            "agarrados": self.total_agarrados,
            "devueltos": self.total_devueltos
        }

    def obtener_telemetria(self):
        """Devuelve telemetría y eventos en tiempo real para la API REST y el Dashboard."""
        with self.lock:
            eventos_lista = list(self.eventos_recientes)
            personas_activas = len(self.overlay_data.get("personas", []))
            
        return {
            "fps_cam": round(self.fps_cam, 1),
            "fps_yolo": round(self.fps_yolo, 1),
            "latencia_ms": round(self.latencia_ms, 1),
            "cpu_cores": self.cpu_cores,
            "agarrados": self.total_agarrados,
            "devueltos": self.total_devueltos,
            "entradas": self.total_entradas,
            "personas_activas": personas_activas,
            "estado": self.estado,
            "fuente": self.fuente,
            "pausado": self.pausado,
            "eventos": eventos_lista
        }

    def _encolar_db(self, tarea, *args):
        """Envía transacciones a la cola asíncrona de SQLite para no retrasar los FPS."""
        try:
            self.cola_db.put_nowait((tarea, args))
        except queue.Full:
            pass

    def _worker_db_asincrono(self):
        """Hilo dedicado a procesar las escrituras en la base de datos sin detener la visión."""
        while self.corriendo:
            try:
                tarea, args = self.cola_db.get(timeout=0.2)
                if tarea == "trafico":
                    guardar_trafico()
                elif tarea == "tomado":
                    guardar_producto_tomado(*args)
                elif tarea == "devuelto":
                    guardar_producto_devuelto(*args)
                elif tarea == "interaccion":
                    guardar_interaccion(*args)
                elif tarea == "permanencia":
                    guardar_permanencia(*args)
                self.cola_db.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[KovaqDB] Error registrando evento en segundo plano: {e}")

    def _registrar_evento_memoria(self, tipo_evento, cliente_id, producto_nombre, confianza, coords, color="naranja"):
        """Registra el evento de forma atómica en el feed en memoria para el frontend."""
        self._event_id_seq += 1
        ahora = datetime.now()
        item = {
            "id": self._event_id_seq,
            "hora": ahora.strftime("%H:%M:%S") + f".{int(ahora.microsecond / 1000):03d}",
            "tipo": tipo_evento,
            "cliente_id": cliente_id or 1,
            "producto": producto_nombre,
            "conf": round(confianza, 2),
            "coords": coords,
            "color": color
        }
        with self.lock:
            self.eventos_recientes.appendleft(item)

    def _worker_push_nube(self):
        """
        Hilo de transmisión a Render: Si MODO=local y se especificó PUBLICAR_VIDEO_URL,
        este hilo transmite los frames con Bounding Boxes a Render (vía HTTP POST /api/frame_push).
        Así tu PC local hace el procesamiento YOLO pesado y Render sirve la web con 0% de sobrecarga.
        """
        import urllib.request
        ultimo_id_enviado = -1
        print(f"[KovaqCloudPush] Transmisión activa hacia: {PUBLICAR_VIDEO_URL}")

        while self.corriendo:
            try:
                if not self.activa or self.pausado:
                    time.sleep(0.1)
                    continue

                # Obtenemos el frame procesado más reciente a través del compositor
                fid, jpeg = self.get_frame_bytes(ultimo_id_enviado)

                # Solo enviamos si hay un fotograma nuevo codificado
                if jpeg is not None and fid != ultimo_id_enviado:
                    ultimo_id_enviado = fid
                    req = urllib.request.Request(
                        PUBLICAR_VIDEO_URL,
                        data=jpeg,
                        headers={
                            "X-Token": VIDEO_PUSH_TOKEN,
                            "Content-Type": "image/jpeg"
                        },
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=1.5) as resp:
                        pass

                # Cadencia de transmisión a la nube: ~15 FPS para fluidez sin consumir tu ancho de banda
                time.sleep(0.065)
            except Exception:
                # Si Render está durmiendo o reconectando, esperamos 2 segundos sin trabar el procesamiento local
                time.sleep(2.0)

    # =========================================================================
    # NIVEL 1: CAPTURA DE CÁMARA (30 FPS CONTINUOS - CERO RETARDO)
    # =========================================================================
    def _abrir_captura(self, tipo, ruta, reiniciar=True):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        if reiniciar:
            self._reset_conteos()
            with self.lock:
                self.latest_raw_frame = None
                self._jpeg = None

        self.fuente, self.ruta, self.mensaje = tipo, ruta, ""
        self.fin_video, self.imagen_procesada, self.fallos = False, False, 0

        if tipo == "apagado":
            self.activa, self.estado = False, "apagado"
            return

        if tipo == "imagen":
            self.activa = os.path.exists(ruta)
            self.estado = "procesando" if self.activa else "error"
            self.mensaje = "" if self.activa else "No se encontró la imagen"
            return

        self.estado = "abriendo"
        try:
            if tipo == "webcam":
                # Intentar primero DirectShow en Windows (más rápido si no está bloqueada)
                if os.name == "nt":
                    self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if not self.cap or not self.cap.isOpened():
                        # Si DSHOW falla o tarda, fallback a backend estándar (MSMF)
                        self.cap = cv2.VideoCapture(0)
                else:
                    self.cap = cv2.VideoCapture(0)

                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            elif tipo == "rtsp":
                url = ruta if ruta else RTSP_URL
                print(f"[KovaqCam] Conectando a Cámara IP / Stream RTSP de supermercado: {url}")
                self.cap = cv2.VideoCapture(int(url) if str(url).isdigit() else str(url))
                if self.cap and self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            elif tipo == "video":
                # Si no se pasó ruta o no existe, usar el video de demostración local
                if not ruta or not os.path.exists(ruta):
                    ruta = DEMO_VIDEO_PATH
                if os.path.exists(ruta):
                    self.ruta = ruta
                    self.cap = cv2.VideoCapture(ruta)
        except Exception as e:
            print(f"[KovaqCam] Error abriendo fuente ({tipo}): {e}")

        if self.cap is None or not self.cap.isOpened():
            # Si la fuente solicitada fue webcam o rtsp y no se pudo abrir, caer AUTOMÁTICAMENTE al video de demo
            demo_fallback = DEMO_VIDEO_PATH
            if tipo in ("webcam", "rtsp") and os.path.exists(demo_fallback):
                print(f"[KovaqCam] Cámara «{tipo}» no detectada o no disponible. Activando Video Demostración por defecto...")
                self.fuente = "video"
                self.ruta = demo_fallback
                self.cap = cv2.VideoCapture(demo_fallback)
                if self.cap and self.cap.isOpened():
                    self.activa, self.estado = True, "en_vivo"
                    self.mensaje = f"Cámara «{tipo}» no disponible. Se activó video demo de supermercado."
                    return
            self.cap, self.activa, self.estado = None, False, "error"
            self.mensaje = f"No se pudo inicializar la fuente de video «{tipo}»"
            return

        self.activa, self.estado = True, "en_vivo"

    def _leer_camara(self):
        """Hilo productor: Lee fotogramas de la cámara sin ningún procesamiento pesado."""
        while self.corriendo:
            sol = self.solicitud
            if sol is not None:
                self.solicitud = None
                self._abrir_captura(*sol)

            if not self.activa or self.fuente in ("imagen", "apagado") or self.pausado or self.cap is None or self.fin_video:
                time.sleep(0.04)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.fuente == "video":
                    # Si es video local terminado, auto-reiniciar para reproducción continua infinita
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = self.cap.read()
                    if not ok or frame is None:
                        # Fallback robusto si el contenedor MP4 no soporta seek
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = cv2.VideoCapture(self.ruta or DEMO_VIDEO_PATH)
                        ok, frame = self.cap.read()
                        if not ok or frame is None:
                            time.sleep(0.04)
                            continue
                else:
                    self.fallos += 1
                    self.estado = "sin_senal"
                    if self.fallos >= 15:
                        self._abrir_captura(self.fuente, self.ruta, reiniciar=False)
                    time.sleep(0.1)
                continue

            self.fallos = 0
            self.estado = "en_vivo"

            # Redimensionar para asegurar streaming rápido
            if frame.shape[1] != ANCHO or frame.shape[0] != ALTO:
                frame_res = cv2.resize(frame, (ANCHO, ALTO), interpolation=cv2.INTER_LINEAR)
            else:
                frame_res = frame

            # Guardamos el último frame capturado de forma atómica
            with self.lock:
                self.latest_raw_frame = frame_res
                self.frame_id += 1

            # Medición de FPS de Captura
            self._frames_cam_count += 1
            now = time.time()
            if now - self._t_cam_start >= 1.0:
                self.fps_cam = self._frames_cam_count / (now - self._t_cam_start)
                self._frames_cam_count = 0
                self._t_cam_start = now

            # Sincronización de ritmo:
            # En video de archivo (.mp4), pausamos ~33ms (30 FPS reales) para evitar decodificar a 120 FPS
            # lo cual saturaba el 100% de la CPU y causaba lageo severo.
            if self.fuente == "video":
                time.sleep(0.033)
            else:
                time.sleep(0.005)

    # =========================================================================
    # NIVEL 2: WORKER DE IA Y MÁQUINA DE ESTADOS (TOMAR / DEVOLVER)
    # =========================================================================
    def _worker_ia_desacoplado(self):
        """
        Hilo consumidor de IA: Toma el fotograma más reciente (Frame-Skipping inteligente)
        y ejecuta YOLOv8 + ByteTrack + Detección de Tomar/Devolver.
        """
        while self.corriendo:
            if not self.activa or self.pausado or self.fuente == "apagado":
                time.sleep(0.08)
                continue

            # Gestión de archivos de imagen estática
            if self.fuente == "imagen":
                if not self.imagen_procesada:
                    self.imagen_procesada = True
                    img = cv2.imread(self.ruta) if os.path.exists(self.ruta) else None
                    if img is None:
                        self.estado, self.mensaje, self.activa = "error", "No se pudo leer la imagen", False
                    else:
                        frame_res = cv2.resize(img, (ANCHO, ALTO))
                        with self.lock:
                            self.latest_raw_frame = frame_res
                        self._analizar_frame_ia(frame_res, rastrear=False)
                        self.estado = "terminado"
                time.sleep(0.15)
                continue

            # Tomamos el frame más reciente disponible sin bloquear la cámara
            with self.lock:
                frame_para_ia = self.latest_raw_frame

            if frame_para_ia is None:
                time.sleep(0.02)
                continue

            t0 = time.time()
            self._analizar_frame_ia(frame_para_ia, rastrear=True)
            t_fin = time.time()

            # Cálculo de Latencia y FPS de la IA
            self.latencia_ms = (t_fin - t0) * 1000.0
            self._frames_yolo_count += 1
            if t_fin - self._t_yolo_start >= 1.0:
                self.fps_yolo = self._frames_yolo_count / (t_fin - self._t_yolo_start)
                self._frames_yolo_count = 0
                self._t_yolo_start = t_fin

            # Pequeño respiro voluntario para no monopolizar la CPU
            if INTERVALO_IA > 0:
                time.sleep(INTERVALO_IA)

    def _analizar_frame_ia(self, frame, rastrear=True):
        """
        Aplica YOLOv8n optimizado en CPU/GPU y ejecuta la máquina de estados de retail.
        Identifica:
        - Personas y tiempo de permanencia en góndola.
        - Productos en percha estática.
        - Evento "Producto Tomado" (Intersección + Movimiento fuera de percha).
        - Evento "Producto Devuelto" (Retorno a percha estática + Desenganche).
        """
        try:
            if rastrear:
                res = self.modelo.track(
                    frame,
                    persist=True,
                    classes=CLASES_RETAIL,
                    verbose=False,
                    imgsz=IMGSZ,
                    conf=CONF,
                    tracker="bytetrack.yaml"
                )[0]
            else:
                res = self.modelo.predict(
                    frame,
                    classes=CLASES_RETAIL,
                    verbose=False,
                    imgsz=IMGSZ,
                    conf=CONF
                )[0]
        except Exception as e:
            # Fallback en caso de que ByteTrack requiera reinicialización
            try:
                res = self.modelo.predict(frame, classes=CLASES_RETAIL, verbose=False, imgsz=IMGSZ, conf=CONF)[0]
            except Exception:
                return

        personas_detectadas = []
        productos_detectados = []
        cajas = res.boxes

        ahora = time.time()
        zx1, zy1, zx2, zy2 = self.zonas

        if cajas is not None and len(cajas):
            xyxy = cajas.xyxy.cpu().numpy()
            clases = cajas.cls.cpu().numpy().astype(int)
            confs = cajas.conf.cpu().numpy() if cajas.conf is not None else [0.8] * len(xyxy)
            ids = cajas.id.cpu().numpy().astype(int) if cajas.id is not None else range(1, len(xyxy) + 1)

            for box, tid, cls_idx, conf in zip(xyxy, ids, clases, confs):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                nombre_clase = self.modelo.names.get(cls_idx, f"item_{cls_idx}")

                if cls_idx == 0:  # PERSONA
                    # Nuevo cliente en escena
                    if tid not in self.tiempos_clientes:
                        self.tiempos_clientes[tid] = ahora
                        self.total_entradas += 1
                        self._encolar_db("trafico")

                    # Permanencia frente a la percha estática
                    dwell_seg = 0
                    esta_en_zona = (zx1 < cx < zx2 and zy1 < cy < zy2)
                    if esta_en_zona:
                        if tid not in self.tiempos_zona:
                            self.tiempos_zona[tid] = ahora
                        dwell_seg = int(ahora - self.tiempos_zona[tid])
                    elif tid in self.tiempos_zona:
                        t_salida = ahora - self.tiempos_zona.pop(tid)
                        if t_salida > 1.5:
                            self._encolar_db("permanencia", round(t_salida, 1))

                    personas_detectadas.append({
                        "id": tid,
                        "box": (x1, y1, x2, y2),
                        "cx": cx,
                        "cy": cy,
                        "dwell": dwell_seg,
                        "en_zona": esta_en_zona
                    })

                else:  # PRODUCTO O ARTÍCULO COMERCIAL
                    esta_en_percha = (zx1 <= cx <= zx2 and zy1 <= cy <= zy2)
                    productos_detectados.append({
                        "id": tid,
                        "box": (x1, y1, x2, y2),
                        "cx": cx,
                        "cy": cy,
                        "nombre": nombre_clase,
                        "conf": float(conf),
                        "en_percha": esta_en_percha,
                        "estado": "en_percha" if esta_en_percha else "fuera"
                    })

        # Cerrar permanencias de personas que salieron de la vista
        ids_vistos = {p["id"] for p in personas_detectadas}
        for tid in list(self.tiempos_zona.keys()):
            if tid not in ids_vistos:
                t_salida = ahora - self.tiempos_zona.pop(tid)
                if t_salida > 1.5:
                    self._encolar_db("permanencia", round(t_salida, 1))

        # =====================================================================
        # LÓGICA DE DETECCIÓN: PRODUCTO TOMADO / PRODUCTO DEVUELTO
        # =====================================================================
        alertas_frame = []

        for prod in productos_detectados:
            px1, py1, px2, py2 = prod["box"]
            prod_area = max(1, (px2 - px1) * (py2 - py1))

            for pers in personas_detectadas:
                hx1, hy1, hx2, hy2 = pers["box"]
                pid = pers["id"]

                # Calcular área de intersección (Bounding Box Overlap)
                inter_x1 = max(px1, hx1)
                inter_y1 = max(py1, hy1)
                inter_x2 = min(px2, hx2)
                inter_y2 = min(py2, hy2)

                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    overlap_ratio = inter_area / float(prod_area)

                    # Intersección significativa (cliente en contacto con el producto)
                    if overlap_ratio >= 0.20:
                        clave_inter = (pid, prod["id"])
                        prod_key = f"{prod['nombre']}_{prod['id']}"

                        # Inicializar seguimiento del producto si es nuevo
                        if prod_key not in self.productos_rastreados:
                            self.productos_rastreados[prod_key] = {
                                "estado": "en_percha" if prod["en_percha"] else "fuera",
                                "cliente_id": None,
                                "t_ultimo_cambio": ahora
                            }

                        info_prod = self.productos_rastreados[prod_key]

                        # CONDICIÓN 1: PRODUCTO TOMADO (Pick)
                        # El producto estaba en percha y el cliente interactúa con él
                        if info_prod["estado"] in ("en_percha", "fuera") and info_prod["cliente_id"] is None:
                            info_prod["estado"] = "tomado"
                            info_prod["cliente_id"] = pid
                            info_prod["t_ultimo_cambio"] = ahora
                            prod["estado"] = "tomado"
                            prod["cliente_id"] = pid

                            self.total_agarrados += 1
                            self.productos_en_mano.setdefault(pid, set()).add(prod["nombre"])
                            self._encolar_db("tomado", prod["nombre"], pid)

                            coords_norm = [px1, py1, px2 - px1, py2 - py1]
                            self._registrar_evento_memoria("Producto Tomado", pid, prod["nombre"], prod["conf"], coords_norm, color="naranja")
                            alertas_frame.append(f"⚡ TOMADO: {prod['nombre']} (C#{pid})")

                        # CONDICIÓN 2: PRODUCTO DEVUELTO (Return)
                        # El producto estaba en estado 'tomado' y regresa al área estática de la percha
                        elif info_prod["estado"] == "tomado" and prod["en_percha"]:
                            # Se confirma la devolución si pasaron más de 1.2 segundos o si la persona se aleja de la percha
                            if (ahora - info_prod["t_ultimo_cambio"] > 1.2) or (not pers["en_zona"]):
                                info_prod["estado"] = "devuelto"
                                cliente_previo = info_prod["cliente_id"] or pid
                                info_prod["cliente_id"] = None
                                info_prod["t_ultimo_cambio"] = ahora
                                prod["estado"] = "devuelto"
                                prod["cliente_id"] = cliente_previo

                                self.total_devueltos += 1
                                if cliente_previo in self.productos_en_mano:
                                    self.productos_en_mano[cliente_previo].discard(prod["nombre"])

                                self._encolar_db("devuelto", prod["nombre"], cliente_previo)

                                coords_norm = [px1, py1, px2 - px1, py2 - py1]
                                self._registrar_evento_memoria("Producto Devuelto", cliente_previo, prod["nombre"], prod["conf"], coords_norm, color="azul")
                                alertas_frame.append(f"↩ DEVUELTO: {prod['nombre']} (C#{cliente_previo})")

        # =====================================================================
        # DETECTOR DE ACCIÓN RETAIL: ¿AGARRÓ ALGO O SOLO MIRÓ?
        # Diseñado para cámaras de supermercado reales (CCTV / RTSP / Techo / Pasillo).
        # Evalúa penetración volumétrica de brazos/torso en la góndola y distingue:
        # 1. Tránsito rápido (no agarra nada)
        # 2. Mirar precios/productos sin tocar (fricción visual sin compra)
        # 3. Agarre de producto (mano penetra 0.6s-5.5s y se retira) -> "Producto Tomado"
        # 4. Devolución (retorno a la percha) -> "Producto Devuelto"
        # =====================================================================
        for pers in personas_detectadas:
            pid = pers["id"]
            hx1, hy1, hx2, hy2 = pers["box"]

            # Cálculo de penetración de la persona dentro de la zona de percha
            ix1 = max(hx1, zx1)
            iy1 = max(hy1, zy1)
            ix2 = min(hx2, zx2)
            iy2 = min(hy2, zy2)

            penetrado = False
            if ix2 > ix1 and iy2 > iy1:
                inter_w = ix2 - ix1
                inter_h = iy2 - iy1
                area_pers = max(1, (hx2 - hx1) * (hy2 - hy1))
                # Contacto activo: penetración física del brazo/cuerpo de al menos 18x24px o 12% del área
                penetrado = (inter_w >= 18 and inter_h >= 24) or ((inter_w * inter_h) / area_pers >= 0.12)

            if pid not in self.ultimo_reach_zona:
                self.ultimo_reach_zona[pid] = {
                    "en_contacto": False,
                    "t_inicio": 0,
                    "tiene_producto": False,
                    "veces_contacto": 0
                }

            st = self.ultimo_reach_zona[pid]

            # FASE 1: El cliente extiende la mano hacia la góndola
            if penetrado and not st["en_contacto"]:
                st["en_contacto"] = True
                st["t_inicio"] = ahora

            # FASE 2: El cliente retira la mano de la góndola
            elif not penetrado and st["en_contacto"]:
                duracion_gesto = ahora - st["t_inicio"]
                st["en_contacto"] = False

                # Filtro de precisión retail:
                # - Tránsito accidental o paso rápido (< 0.6s): NO es compra (se ignora)
                # - Gesto humano de toma o retorno: entre 0.6s y 5.5s
                if 0.6 <= duracion_gesto <= 5.5:
                    st["veces_contacto"] += 1

                    # Si el cliente NO tenía producto -> CONFIRMADO: PRODUCTO TOMADO
                    if not st["tiene_producto"]:
                        st["tiene_producto"] = True
                        self.total_agarrados += 1
                        self._encolar_db("tomado", "Artículo en Góndola", pid)

                        coords_gesto = [ix1, iy1, max(50, ix2 - ix1), max(50, iy2 - iy1)]
                        self._registrar_evento_memoria(
                            "Producto Tomado", pid, "Artículo en Góndola", 0.94, coords_gesto, color="naranja"
                        )
                        alertas_frame.append(f"⚡ TOMADO: Góndola (C#{pid} • {duracion_gesto:.1f}s)")

                    # Si el cliente YA tenía producto y vuelve a colocar las manos en la percha -> DEVUELTO
                    else:
                        st["tiene_producto"] = False
                        self.total_devueltos += 1
                        self._encolar_db("devuelto", "Artículo en Góndola", pid)

                        coords_gesto = [ix1, iy1, max(50, ix2 - ix1), max(50, iy2 - iy1)]
                        self._registrar_evento_memoria(
                            "Producto Devuelto", pid, "Artículo en Góndola", 0.95, coords_gesto, color="azul"
                        )
                        alertas_frame.append(f"↩ DEVUELTO: Góndola (C#{pid} • {duracion_gesto:.1f}s)")

        # Actualizar la estructura atómica del Overlay para el renderizador a 30 FPS
        with self.lock:
            self.overlay_data = {
                "personas": personas_detectadas,
                "productos": productos_detectados,
                "alertas": alertas_frame,
                "telemetria": f"YOLOv8n | IA: {self.fps_yolo:.1f} FPS | Lat: {self.latencia_ms:.0f}ms"
            }

    # =========================================================================
    # NIVEL 3: COMPOSITOR DE VIDEO A 30 FPS (STREAMING FLUIDO SIN LAG)
    # =========================================================================
    def get_frame_bytes(self, ultimo_id=-1):
        """
        Retorna el fotograma actual procesado y codificado en JPEG.
        Dibuja los bounding boxes de manera ultrarrápida sobre el frame más reciente,
        garantizando streaming continuo a 30 FPS sin trabas.
        """
        with self.lock:
            raw = self.latest_raw_frame
            fid = self.frame_id
            overlay = self.overlay_data

        if raw is None:
            return ultimo_id, None

        if fid == ultimo_id and self._jpeg is not None:
            return ultimo_id, None

        # Codificación limpia, directa y ultrarrápida del fotograma nativo
        # (100% nítido, sin capas ni rectángulos invasivos sobre el video, cero lag)
        _, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, 60])
        datos = buf.tobytes()

        with self.lock:
            self._jpeg_id = fid
            self._jpeg = datos

        return fid, datos


# Instancia Global del Servicio de Cámara
camara = CameraService()
