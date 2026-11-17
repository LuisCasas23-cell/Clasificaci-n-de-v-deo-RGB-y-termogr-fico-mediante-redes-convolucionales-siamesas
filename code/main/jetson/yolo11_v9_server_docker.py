#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import socket
import struct
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# YOLO11n DATASET SERVER - DOCKER/CUDA - JETSON NANO
#
# Recibe UN frame por solicitud:
#   camera_id 0 = RGB
#   camera_id 1 = NoIR
#
# Detecta TODAS las clases COCO.
# Se conserva deteccion debil desde 0.10 para investigacion.
# La aplicacion cliente decide que >=0.25 es deteccion valida.
# ============================================================

HOST = "127.0.0.1"
PORT = 5070

MODEL_PATH = "yolo11n.pt"

# Mantener carga moderada mientras no hay ventilador.
IMGSZ = 320
CONFIDENCE_MIN = 0.10
IOU = 0.70
MAX_DET = 40

DEVICE = 0
USE_FP16 = True

MAX_FRAME_BYTES = 8 * 1024 * 1024


def recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(count - len(data))
        if not chunk:
            raise ConnectionError("Conexion cerrada")
        data.extend(chunk)
    return bytes(data)


def send_json(sock, obj):
    payload = json.dumps(
        obj,
        separators=(",", ":")
    ).encode("utf-8")

    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


print("")
print("====================================================")
print(" YOLO11n DATASET SERVER - RGB + NoIR - CUDA")
print("====================================================")
print("Modelo:", MODEL_PATH)
print("imgsz:", IMGSZ)
print("conf minima guardada:", CONFIDENCE_MIN)
print("iou:", IOU)
print("FP16:", USE_FP16)
print("Puerto:", PORT)
print("====================================================")
print("")

print("Cargando YOLO11n...")
model = YOLO(MODEL_PATH)

names = {
    int(k): str(v)
    for k, v in model.names.items()
}

print("Calentando CUDA...")
dummy = np.zeros(
    (IMGSZ, IMGSZ, 3),
    dtype=np.uint8
)

_ = model.predict(
    source=dummy,
    imgsz=IMGSZ,
    conf=CONFIDENCE_MIN,
    iou=IOU,
    device=DEVICE,
    half=USE_FP16,
    max_det=MAX_DET,
    verbose=False,
)

print(">>> YOLO11/CUDA LISTO <<<")

server = socket.socket(
    socket.AF_INET,
    socket.SOCK_STREAM
)
server.setsockopt(
    socket.SOL_SOCKET,
    socket.SO_REUSEADDR,
    1
)
server.bind((HOST, PORT))
server.listen(1)

try:
    while True:
        print(
            "Esperando cliente en 127.0.0.1:{}..."
            .format(PORT)
        )

        conn, _addr = server.accept()
        print(">>> CLIENTE CONECTADO <<<")

        try:
            while True:
                magic = recv_exact(conn, 4)

                if magic != b"FRM2":
                    raise RuntimeError(
                        "Cabecera invalida: {!r}".format(
                            magic
                        )
                    )

                camera_id = recv_exact(conn, 1)[0]

                frame_len = struct.unpack(
                    "!I",
                    recv_exact(conn, 4)
                )[0]

                if (
                    frame_len <= 0
                    or frame_len > MAX_FRAME_BYTES
                ):
                    raise RuntimeError(
                        "Tamano de frame invalido: {}"
                        .format(frame_len)
                    )

                jpeg_data = recv_exact(
                    conn,
                    frame_len
                )

                frame = cv2.imdecode(
                    np.frombuffer(
                        jpeg_data,
                        dtype=np.uint8
                    ),
                    cv2.IMREAD_COLOR
                )

                if frame is None:
                    send_json(
                        conn,
                        {
                            "ok": False,
                            "camera_id": camera_id,
                            "detections": [],
                            "error": "JPEG invalido",
                        }
                    )
                    continue

                t0 = time.perf_counter()

                results = model.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONFIDENCE_MIN,
                    iou=IOU,
                    device=DEVICE,
                    half=USE_FP16,
                    max_det=MAX_DET,
                    verbose=False,
                )

                elapsed_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                detections = []

                if (
                    results
                    and results[0].boxes is not None
                ):
                    for box in results[0].boxes:
                        cls_id = int(
                            box.cls[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        conf = float(
                            box.conf[0]
                            .detach()
                            .cpu()
                            .item()
                        )

                        xyxy = (
                            box.xyxy[0]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        )

                        detections.append(
                            {
                                "class_id": cls_id,
                                "class_name": names.get(
                                    cls_id,
                                    str(cls_id)
                                ),
                                "confidence": round(
                                    conf,
                                    5
                                ),
                                "box": [
                                    int(xyxy[0]),
                                    int(xyxy[1]),
                                    int(xyxy[2]),
                                    int(xyxy[3]),
                                ],
                            }
                        )

                send_json(
                    conn,
                    {
                        "ok": True,
                        "camera_id": int(camera_id),
                        "detections": detections,
                        "inference_ms": round(
                            elapsed_ms,
                            3
                        ),
                        "imgsz": IMGSZ,
                    }
                )

        except Exception as e:
            print("Cliente desconectado:", e)

        finally:
            try:
                conn.close()
            except Exception:
                pass

finally:
    server.close()
