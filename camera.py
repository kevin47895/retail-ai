import os

# --- Rendimiento: esto debe ir ANTES de importar cv2/ultralytics (antes no tenía ningún efecto) ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # RTSP más estable

import threading
import time

import cv2
from dotenv import load_dotenv
from ultralytics import YOLO

from database import guardar_interaccion, guardar_permanencia, guardar_trafico

load_dotenv()
cv2.setNumThreads(2)
try:
    import torch
    torch.set_num_threads(2)
except Exception:
    pass

IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))        # más alto = detecta personas lejanas, pero más lento
CONF = float(os.getenv("YOLO_CONF", "0.4"))
INTERVALO = float(os.getenv("YOLO_INTERVALO", "0.04"))  # pausa mínima entre análisis (CPU más libre)
RTSP_URL = os.getenv("RTSP_URL", "0").strip()       # "0" = webcam; o rtsp://usuario:clave@ip/stream
CLASES = [0, 39, 41, 46, 47, 67]                    # persona, botella, taza, banana, manzana, celular
ANCHO, ALTO = 480, 360


class CameraService:
    def __init__(self):
        self.lock = threading.Lock()
        self.fuente, self.ruta, self.estado, self.mensaje = "apagado", "", "apagado", ""
        self.frame = None
        self.frame_id = 0
        self.latest_raw_frame = None
        self._jpeg_id, self._jpeg = -1, None
        self.solicitud = None            # (tipo, ruta) pedido por la API; lo atiende el hilo lector
        self.corriendo = self.activa = self.pausado = False
        self.cap = None
        self.hilo_lectura = self.hilo_procesamiento = None
        self.frame_anterior = None
        self.ultimo_movimiento = 0.0
        self.movimiento_detectado = False
        self.fin_video = False
        self.imagen_procesada = False
        self.fallos = 0

        model_file = 'yolov8n.onnx' if os.path.exists('yolov8n.onnx') else 'yolov8n.pt'
        self.modelo = YOLO(model_file, **({'task': 'detect'} if model_file.endswith('.onnx') else {}))
        self.zonas = (120, 80, 360, 280)
        self._reset_conteos()

    # ---------- control ----------
    def _reset_conteos(self):
        """Cada fuente nueva empieza de cero (antes los contadores crecían sin parar en el bucle)."""
        self.total_entradas = self.total_agarrados = 0
        self.tiempos, self.tiempos_zona, self.interacciones = {}, {}, set()
        try:
            self.modelo.predictor = None  # reinicia el rastreador (los ID vuelven a empezar)
        except Exception:
            pass

    def iniciar(self):
        if not self.corriendo:
            self.corriendo = True
            self.hilo_lectura = threading.Thread(target=self._leer_camara, daemon=True)
            self.hilo_procesamiento = threading.Thread(target=self._procesar_frames, daemon=True)
            self.hilo_lectura.start()
            self.hilo_procesamiento.start()

    def detener(self):
        self.corriendo = False
        self.apagar()

    def apagar(self):
        """Botón «Detener»: lo pide la API y lo ejecuta el hilo lector (que es dueño de la captura)."""
        self.activa = False
        self.estado = "apagado"
        self.solicitud = ("apagado", "")
        with self.lock:
            self.frame = None
            self.latest_raw_frame = None

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
        return {"fuente": self.fuente, "estado": self.estado, "mensaje": self.mensaje, "pausado": self.pausado,
                "entradas": self.total_entradas, "agarrados": self.total_agarrados}

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
                self.frame = None
                self.latest_raw_frame = None
        self.fuente, self.ruta, self.mensaje = tipo, ruta, ""
        self.frame_anterior, self.fin_video, self.imagen_procesada, self.fallos = None, False, False, 0

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
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # más cuadros por segundo
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            elif tipo == "rtsp":
                self.cap = cv2.VideoCapture(int(RTSP_URL) if RTSP_URL.isdigit() else RTSP_URL)
            elif tipo == "video" and os.path.exists(ruta):
                self.cap = cv2.VideoCapture(ruta)
        except Exception as e:
            print(f"Error abriendo captura ({tipo}): {e}")
        if self.cap is None or not self.cap.isOpened():
            self.cap, self.activa, self.estado = None, False, "error"
            self.mensaje = f"No se pudo abrir la fuente «{tipo}»"
            return
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # sin cola de cuadros viejos: menos retraso
        except Exception:
            pass
        self.activa, self.estado = True, "en_vivo"

    # ---------- hilos ----------
    def _detectar_movimiento_ligero(self, gris):
        if self.frame_anterior is None:
            self.frame_anterior = gris
            return True
        _, umbral = cv2.threshold(cv2.absdiff(self.frame_anterior, gris), 25, 255, cv2.THRESH_BINARY)
        self.frame_anterior = gris
        return cv2.countNonZero(umbral) > 1200

    def _leer_camara(self):
        while self.corriendo:
            sol = self.solicitud
            if sol is not None:
                self.solicitud = None
                self._abrir_captura(*sol)
            if (not self.activa or self.fuente in ("imagen", "apagado") or self.pausado or self.cap is None
                    or self.fin_video):
                time.sleep(0.05)
                continue
            en_vivo = self.fuente in ("webcam", "rtsp")
            if not en_vivo and self.latest_raw_frame is not None:
                time.sleep(0.005)  # video archivo: espera a la IA, no salta cuadros (conteo más fiel)
                continue
            ok, frame = self.cap.read()
            if not ok:
                if not en_vivo:
                    self.fin_video = True  # termina UNA vez: ya no se repite en bucle
                else:
                    self.fallos += 1
                    self.estado = "sin_senal"
                    if self.fallos >= 30:  # RNF14: reintenta abrir la cámara
                        self._abrir_captura(self.fuente, self.ruta, reiniciar=False)
                    time.sleep(0.1)
                continue
            self.fallos = 0
            if en_vivo:
                self.estado = "en_vivo"
            pequeno = cv2.resize(frame, (ANCHO, ALTO))
            if en_vivo:
                gris = cv2.GaussianBlur(cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                if self._detectar_movimiento_ligero(gris):
                    self.ultimo_movimiento, self.movimiento_detectado = time.time(), True
                elif time.time() - self.ultimo_movimiento > 5.0:
                    self.movimiento_detectado = False
                time.sleep(0.01)
            with self.lock:
                self.latest_raw_frame = pequeno

    def _procesar_frames(self):
        while self.corriendo:
            if not self.activa or self.pausado or self.fuente == "apagado":
                time.sleep(0.1)
                continue
            if self.fuente == "imagen":
                if not self.imagen_procesada:  # antes se repetía la IA sobre la misma foto todo el tiempo
                    self.imagen_procesada = True
                    img = cv2.imread(self.ruta) if os.path.exists(self.ruta) else None
                    if img is None:
                        self.estado, self.mensaje, self.activa = "error", "No se pudo leer la imagen", False
                    else:
                        salida = self._aplicar_yolo(cv2.resize(img, (ANCHO, ALTO)), rastrear=False)
                        with self.lock:
                            self.frame = salida
                            self.frame_id += 1
                        self.estado = "terminado"
                time.sleep(0.2)
                continue
            t0 = time.time()
            with self.lock:
                f, self.latest_raw_frame = self.latest_raw_frame, None
            if f is None:
                if self.fin_video and self.estado != "terminado":
                    self.estado = "terminado"
                time.sleep(0.01)
                continue
            if self.fuente in ("webcam", "rtsp") and not self.movimiento_detectado:
                cv2.putText(f, "MODO AHORRO (ESPERANDO MOVIMIENTO)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                z = self.zonas
                cv2.rectangle(f, (z[0], z[1]), (z[2], z[3]), (255, 0, 0), 2)
            else:
                f = self._aplicar_yolo(f)
            with self.lock:
                self.frame = f
                self.frame_id += 1
            time.sleep(max(0.0, INTERVALO - (time.time() - t0)))

    # ---------- IA ----------
    def _aplicar_yolo(self, frame, rastrear=True):
        if rastrear:
            res = self.modelo.track(frame, persist=True, classes=CLASES, verbose=False, imgsz=IMGSZ, conf=CONF,
                                    tracker="bytetrack.yaml")[0]
        else:
            res = self.modelo.predict(frame, classes=CLASES, verbose=False, imgsz=IMGSZ, conf=CONF)[0]
        personas, productos = {}, {}
        cajas = res.boxes
        if cajas is not None and len(cajas):
            xyxy = cajas.xyxy.cpu().numpy()
            clases = cajas.cls.cpu().numpy().astype(int)
            if cajas.id is not None:
                ids = cajas.id.cpu().numpy().astype(int)
            else:
                ids = range(1, len(xyxy) + 1) if not rastrear else []  # en foto suelta no hay rastreo
            for box, tid, cls in zip(xyxy, ids, clases):
                x1, y1, x2, y2 = map(int, box)
                if cls == 0:
                    personas[tid] = (x1, y1, x2, y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if tid not in self.tiempos:
                        self.tiempos[tid] = time.time()
                        self.total_entradas += 1
                        guardar_trafico()
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    z = self.zonas
                    if z[0] < cx < z[2] and z[1] < cy < z[3]:
                        self.tiempos_zona.setdefault(tid, time.time())
                        cv2.putText(frame, f"{int(time.time() - self.tiempos_zona[tid])}s", (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    elif tid in self.tiempos_zona:
                        self._cerrar_zona(tid)
                else:
                    productos[tid] = (x1, y1, x2, y2, self.modelo.names[cls])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, self.modelo.names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for tid in [t for t in self.tiempos_zona if t not in personas]:
            self._cerrar_zona(tid)  # la persona salió de cámara dentro de la zona: también cuenta su permanencia
        for pid, pb in personas.items():
            for prod in productos.values():
                if max(pb[0], prod[0]) < min(pb[2], prod[2]) and max(pb[1], prod[1]) < min(pb[3], prod[3]):
                    if (pid, prod[4]) not in self.interacciones:  # una vez por persona y tipo de objeto
                        self.interacciones.add((pid, prod[4]))
                        self.total_agarrados += 1
                        guardar_interaccion(prod[4])
        z = self.zonas
        cv2.rectangle(frame, (z[0], z[1]), (z[2], z[3]), (255, 0, 0), 2)
        return frame

    def _cerrar_zona(self, tid):
        t = time.time() - self.tiempos_zona.pop(tid)
        if t > 2:
            guardar_permanencia(t)

    # ---------- salida ----------
    def get_frame_bytes(self, ultimo_id=-1):
        """Devuelve (id, jpeg|None). Cada navegador lleva su propio `ultimo_id`; el JPEG se codifica una vez."""
        with self.lock:
            f, fid = self.frame, self.frame_id
            if f is None or fid == ultimo_id:
                return ultimo_id, None
            if self._jpeg_id == fid:
                return fid, self._jpeg
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 55])
        datos = buf.tobytes()
        with self.lock:
            self._jpeg_id, self._jpeg = fid, datos
        return fid, datos


camara = CameraService()
