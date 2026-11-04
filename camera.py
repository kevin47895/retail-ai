import os

# --- Rendimiento: esto debe ir ANTES de importar cv2/ultralytics ---
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

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

IMGSZ = int(os.getenv("YOLO_IMGSZ", "320"))
CONF = float(os.getenv("YOLO_CONF", "0.3"))
INTERVALO = float(os.getenv("YOLO_INTERVALO", "0.04"))
RTSP_URL = os.getenv("RTSP_URL", "0").strip()
CLASES = [0, 39, 41, 46, 47, 67]
ANCHO, ALTO = 480, 360


class CameraService:
    def __init__(self):
        self.lock = threading.Lock()
        self.fuente, self.ruta, self.estado, self.mensaje = "apagado", "", "apagado", ""
        self.frame = None
        self.frame_id = 0
        self.latest_raw_frame = None
        self._jpeg_id, self._jpeg = -1, None
        self.solicitud = None
        self.corriendo = self.activa = self.pausado = False
        self.cap = None
        self.hilo_lectura = self.hilo_procesamiento = None
        self.frame_anterior = None
        self.ultimo_movimiento = 0.0
        self.movimiento_detectado = True  # Arrancar en True para evitar pantalla negra inicial
        self.fin_video = False
        self.imagen_procesada = False
        self.fallos = 0

        model_file = 'yolov8n.onnx' if os.path.exists('yolov8n.onnx') else 'yolov8n.pt'
        self.modelo = YOLO(model_file, **({'task': 'detect'} if model_file.endswith('.onnx') else {}))
        self.zonas = (120, 80, 360, 280)
        self._reset_conteos()

    def _reset_conteos(self):
        self.total_entradas = self.total_agarrados = 0
        self.tiempos, self.tiempos_zona, self.interacciones = {}, {}, set()
        try:
            self.modelo.predictor = None
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
        self.activa = False
        self.estado = "apagado"
        self.solicitud = ("apagado", "")
        with self.lock:
            self.frame = None
            self.latest_raw_frame = None
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
                # Forzamos un inicio limpio con un pequeño retraso de estabilización
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
                time.sleep(0.5)  # Da tiempo al hardware USB a liberar la señal
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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.activa, self.estado = True, "en_vivo"

        self.estado = "abriendo"
        try:
            if tipo == "webcam":
                # Abrir webcam con índice 0 por defecto de forma limpia
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.activa, self.estado = True, "en_vivo"

    def _leer_camara(self):
        """Hilo dedicado exclusivamente a capturar de la cámara sin bloquear la IA"""
        while self.corriendo:
            sol = self.solicitud
            if sol is not None:
                self.solicitud = None
                self._abrir_captura(*sol)

            if (not self.activa or self.fuente in ("imagen", "apagado") or self.pausado or self.cap is None or self.fin_video):
                time.sleep(0.05)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.fuente == "video":
                    self.fin_video = True
                else:
                    self.fallos += 1
                    self.estado = "sin_senal"
                    if self.fallos >= 15:
                        self._abrir_captura(self.fuente, self.ruta, reiniciar=False)
                    time.sleep(0.1)
                continue

            self.fallos = 0
            self.estado = "en_vivo"
            pequeno = cv2.resize(frame, (ANCHO, ALTO))
            
            with self.lock:
                self.latest_raw_frame = pequeno
            
            time.sleep(0.01)

    def _procesar_frames(self):
        """Hilo dedicado a aplicar YOLO y actualizar el frame web de forma independiente"""
        while self.corriendo:
            if not self.activa or self.pausado or self.fuente == "apagado":
                time.sleep(0.1)
                continue

            if self.fuente == "imagen":
                if not self.imagen_procesada:
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
                time.sleep(0.02)
                continue

            # Procesamiento con YOLO directo para evitar retrasos en la web
            f = self._aplicar_yolo(f)
            
            with self.lock:
                self.frame = f
                self.frame_id += 1

            elapsed = time.time() - t0
            if elapsed < INTERVALO:
                time.sleep(INTERVALO - elapsed)

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
            ids = cajas.id.cpu().numpy().astype(int) if cajas.id is not None else range(1, len(xyxy) + 1)
            
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
            self._cerrar_zona(tid)
            
        for pid, pb in personas.items():
            for prod in productos.values():
                if max(pb[0], prod[0]) < min(pb[2], prod[2]) and max(pb[1], prod[1]) < min(pb[3], prod[3]):
                    if (pid, prod[4]) not in self.interacciones:
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

    def get_frame_bytes(self, ultimo_id=-1):
        with self.lock:
            f, fid = self.frame, self.frame_id
            if f is None or fid == ultimo_id:
                return ultimo_id, None
            if self._jpeg_id == fid:
                return fid, self._jpeg
        
        # Compresión ligera (calidad 55) para que viaje fluido a la web
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 55])
        datos = buf.tobytes()
        with self.lock:
            self._jpeg_id, self._jpeg = fid, datos
        return fid, datos


camara = CameraService()