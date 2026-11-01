import os
import time
import threading
import urllib.request


def iniciar_publicacion(camara):
    url = os.getenv("PUBLICAR_VIDEO_URL", "").strip()
    token = os.getenv("VIDEO_PUSH_TOKEN", "").strip()
    if not url or not token:
        print("Publicación de video desactivada: falta PUBLICAR_VIDEO_URL o VIDEO_PUSH_TOKEN")
        return

    def bucle():
        ultimo = -1
        while True:
            try:
                fid, jpg = camara.get_frame_bytes(ultimo)
                if jpg:
                    ultimo = fid
                    req = urllib.request.Request(
                        url, data=jpg, method="POST",
                        headers={"X-Token": token, "Content-Type": "image/jpeg"}
                    )
                    urllib.request.urlopen(req, timeout=10).read()
            except Exception as e:
                print("Error publicando video:", e)
                time.sleep(2)
            time.sleep(0.2)

    threading.Thread(target=bucle, daemon=True).start()