#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import socket
import struct
import json
import io
import zipfile
import threading
import queue
import time
import os
import sys
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

try:
    import Jetson.GPIO as GPIO
except ImportError:
    GPIO = None

# ============================================================
# JETSON STREAM CLIENT - RGB + NOIR + TFMini + YOLO + IR AUTO
# IMX219 RGB + OV5647 NoIR
#
# CAMBIO PRINCIPAL:
# - Jetson NO genera los 6 filtros al capturar.
# - Jetson NO hace landmarks.
# - Jetson NO bloquea la interfaz durante el envio.
# - Envia solamente 1 RGB + 1 NoIR originales a la laptop.
# - La laptop genera los 6 resultados NoIR con la logica V8.1.
# - TFmini V1.81 controla una zona optima de distancia.
# - NO se permite capturar fuera de la zona optima.
#
# Resultado final en laptop:
#   1 RGB + 6 NoIR = 7 imagenes por evento.
# ============================================================

RGB_SENSOR_ID = 0
NOIR_SENSOR_ID = 1

EXPOSURE_COMPENSATION = 1.2
GAIN_MAX_PRIMARY = 9
GAIN_MAX_FALLBACK = 8
EXPOSURE_MIN_NS = 20000000
EXPOSURE_MAX_NS = 33333333

LAPTOP_PORT = 5050
PREVIEW_PORT = 5051
ORIGINAL_STREAM_PORT = 5052

PREVIEW_EACH_W = 480
PREVIEW_EACH_H = 270
PREVIEW_JPEG_QUALITY = 50
PREVIEW_SEND_INTERVAL = 1.0 / 30.0   # tracking/microexpresiones en vivo

# Stream secundario continuo, sin boton CAPT: conserva el formato original
# que entrega OpenCV (960x540 por camara) para GUARDAR LOTE.
ORIGINAL_JPEG_QUALITY = 90
ORIGINAL_SEND_INTERVAL = 1.0 / 5.0

PNG_COMPRESSION = 1

# ============================================================
# YOLO11n LOCAL (DOCKER/CUDA) - SIN CAMBIOS PARA MICROEXPRESIONES
# ============================================================

YOLO_HOST = "127.0.0.1"
YOLO_PORT = 5070

YOLO_LIVE_TOTAL_FPS = 5.0
YOLO_LIVE_INTERVAL = 1.0 / YOLO_LIVE_TOTAL_FPS
YOLO_VALID_CONF = 0.25
YOLO_WEAK_CONF = 0.10

yolo_queue = queue.Queue(maxsize=4)
yolo_state_lock = threading.Lock()
latest_yolo = {
    0: {"detections": [], "inference_ms": 0.0, "updated": 0.0},
    1: {"detections": [], "inference_ms": 0.0, "updated": 0.0},
}


# ============================================================
# TFmini V1.81 - CONTROL DE DISTANCIA
# ============================================================

TFMINI_PORT = "/dev/ttyTHS1"
TFMINI_BAUD = 115200

# Zona optima inicial para dataset facial.
# Puedes cambiar estos dos valores despues si el protocolo
# experimental requiere otra distancia.
DISTANCE_MIN_CM = 30
DISTANCE_MAX_CM = 60
TARGET_DISTANCE_CM = 45

# ============================================================
# ILUMINADOR IR - CONTROL AUTOMATICO POR TFMini
# ============================================================
# IMPORTANTE: el GPIO solo debe controlar la entrada EN/GATE de un
# MOSFET, transistor o relay. NO alimentes el modulo IR directamente
# desde un pin GPIO.
#
# Configuracion recomendada sin editar el archivo:
#   IR_GPIO_PIN=12 IR_GPIO_MODE=BOARD python3 jetson_v11_ir_auto_tfmini.py
#
# Si IR_GPIO_PIN no se define, el programa funciona en modo monitor:
# NO puede apagar hardware. La laptop V16.3 bloqueara la captura hasta que exista control fisico.
_ir_pin_env = ""
IR_GPIO_PIN = None
IR_GPIO_MODE = "BOARD"
IR_ACTIVE_HIGH = True
IR_VALID_STABLE_READS = 1
IR_INVALID_STABLE_READS = 5

ir_lock = threading.Lock()
ir_state = {
    "control_enabled": False,
    # None = no existe control fisico configurado; no fingir OFF.
    "active": None,
    "reason": "SIN CONTROL FISICO",
    "gpio_pin": IR_GPIO_PIN,
    "gpio_mode": IR_GPIO_MODE,
    "gpio_level": None,
    "gpio_readback": None,
    "last_change_iso": None,
}

# Las dos camaras estan montadas fisicamente invertidas.
# Rotar aqui corrige tanto el preview como las capturas enviadas.
ROTATE_RGB_180 = True
ROTATE_NOIR_180 = False

# Una lectura mas vieja que esto se considera invalida.
TFMINI_MAX_AGE_SEC = 1.0

tfmini_lock = threading.Lock()
tfmini_state = {
    "distance_cm": None,
    "strength": None,
    "last_update_monotonic": 0.0,
    "last_update_iso": None,
    "connected": False,
    "error": "SIN INICIAR",
}

CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "laptop_transfer_config.json"
)

LEGACY_CONFIG_FILES = [
    os.path.expanduser("~/noir_filters/laptop_transfer_config.json"),
    os.path.expanduser("~/yolo_rgb/laptop_transfer_config.json"),
    os.path.expanduser("~/rgb_noir_tfmini_face_scanner_v8/laptop_transfer_config.json"),
]

WINDOW_NAME = "RGB + NOIR + YOLO11n + TFmini + STREAM LIMPIO v11.5"
DISPLAY_W = 960
DISPLAY_H = 400

BUTTON_X1 = 250
BUTTON_Y1 = 325
BUTTON_X2 = 710
BUTTON_Y2 = 382

last_status = ""
last_status_time = 0.0

preview_queue = queue.Queue(maxsize=1)
original_queue = queue.Queue(maxsize=1)
stop_event = threading.Event()

LAPTOP_IP = None


# ============================================================
# CONFIG IP
# ============================================================

def save_laptop_ip(ip):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"laptop_ip": ip}, f, indent=2)
    except Exception:
        pass


def resolve_laptop_ip():
    global LAPTOP_IP

    if len(sys.argv) >= 2:
        LAPTOP_IP = sys.argv[1].strip()
        if LAPTOP_IP:
            save_laptop_ip(LAPTOP_IP)
            return LAPTOP_IP

    candidates = [CONFIG_FILE] + LEGACY_CONFIG_FILES

    for cfg in candidates:
        if not os.path.exists(cfg):
            continue
        try:
            with open(cfg, "r") as f:
                data = json.load(f)
            ip = str(data.get("laptop_ip", "")).strip()
            if ip:
                LAPTOP_IP = ip
                save_laptop_ip(ip)
                print("IP laptop recuperada de:", cfg)
                return LAPTOP_IP
        except Exception:
            pass

    print("")
    print("ERROR: no encontre la configuracion de IP de la V8.")
    print("Copia ~/noir_filters/laptop_transfer_config.json a ~/yolo_rgb/")
    print("o ejecuta una sola vez este archivo con la IP como argumento.")
    print("")
    LAPTOP_IP = None
    return None


def ping_laptop():
    if not LAPTOP_IP:
        return False

    try:
        with socket.create_connection(
            (LAPTOP_IP, LAPTOP_PORT),
            timeout=1.5,
        ) as sock:
            sock.sendall(b"PING")
            sock.settimeout(1.5)
            return sock.recv(4) == b"PONG"
    except Exception:
        return False



# ============================================================
# TFmini UART
# ============================================================

def read_tfmini_frame(ser):
    """Lee una trama estandar TFmini de 9 bytes (0x59 0x59...)."""
    while not stop_event.is_set():
        b1 = ser.read(1)
        if not b1:
            return None

        if b1[0] != 0x59:
            continue

        b2 = ser.read(1)
        if not b2 or b2[0] != 0x59:
            continue

        rest = ser.read(7)
        if len(rest) != 7:
            return None

        frame = bytes([0x59, 0x59]) + rest

        checksum = sum(frame[:8]) & 0xFF
        if checksum != frame[8]:
            continue

        distance_cm = frame[2] | (frame[3] << 8)
        strength = frame[4] | (frame[5] << 8)

        # Valores imposibles/no utiles se ignoran.
        if distance_cm <= 0 or distance_cm >= 65535:
            continue

        return distance_cm, strength


def tfmini_reader_worker():
    """Mantiene la ultima distancia del TFmini actualizada en segundo plano."""
    global tfmini_state

    if serial is None:
        with tfmini_lock:
            tfmini_state["connected"] = False
            tfmini_state["error"] = "FALTA python3-serial"
        print("TFmini: falta python3-serial.")
        print("Instala: sudo apt install -y python3-serial")
        return

    while not stop_event.is_set():
        ser = None
        try:
            ser = serial.Serial(
                port=TFMINI_PORT,
                baudrate=TFMINI_BAUD,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.25,
            )

            ser.reset_input_buffer()

            with tfmini_lock:
                tfmini_state["connected"] = True
                tfmini_state["error"] = ""

            print("TFmini conectado:", TFMINI_PORT, "@", TFMINI_BAUD)

            while not stop_event.is_set():
                result = read_tfmini_frame(ser)

                if result is None:
                    continue

                distance_cm, strength = result

                with tfmini_lock:
                    tfmini_state["distance_cm"] = int(distance_cm)
                    tfmini_state["strength"] = int(strength)
                    tfmini_state["last_update_monotonic"] = time.monotonic()
                    tfmini_state["last_update_iso"] = datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                    tfmini_state["connected"] = True
                    tfmini_state["error"] = ""

        except Exception as e:
            with tfmini_lock:
                tfmini_state["connected"] = False
                tfmini_state["error"] = str(e)[:120]

            print("TFmini UART:", str(e))
            time.sleep(1.0)

        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass


def get_tfmini_snapshot():
    """Devuelve telemetria y determina si la persona esta en zona optima."""
    with tfmini_lock:
        state = dict(tfmini_state)

    distance = state.get("distance_cm")
    last_mono = float(state.get("last_update_monotonic") or 0.0)
    age = None

    if last_mono > 0:
        age = max(0.0, time.monotonic() - last_mono)

    fresh = (
        distance is not None
        and age is not None
        and age <= TFMINI_MAX_AGE_SEC
    )

    valid = (
        fresh
        and DISTANCE_MIN_CM <= distance <= DISTANCE_MAX_CM
    )

    if not fresh:
        status = "SIN DATOS"
    elif distance < DISTANCE_MIN_CM:
        status = "MUY CERCA - ALEJATE"
    elif distance > DISTANCE_MAX_CM:
        status = "MUY LEJOS - ACERCATE"
    else:
        status = "DISTANCIA OPTIMA"

    return {
        "distance_cm": distance,
        "distance_m": (
            round(float(distance) / 100.0, 3)
            if distance is not None
            else None
        ),
        "strength": state.get("strength"),
        "fresh": bool(fresh),
        "valid": bool(valid),
        "status": status,
        "zone_min_cm": DISTANCE_MIN_CM,
        "zone_max_cm": DISTANCE_MAX_CM,
        "age_sec": round(age, 3) if age is not None else None,
        "last_update_iso": state.get("last_update_iso"),
        "connected": bool(state.get("connected")),
        "error": state.get("error", ""),
        "ir_active": get_ir_snapshot().get("active"),
        "ir_control_enabled": bool(
            get_ir_snapshot().get(
                "control_enabled",
                False
            )
        ),
        "ir_reason": get_ir_snapshot().get(
            "reason",
            ""
        ),
        "ir_gpio_pin": get_ir_snapshot().get(
            "gpio_pin"
        ),
        "ir_gpio_mode": get_ir_snapshot().get(
            "gpio_mode"
        ),
        "ir_gpio_level": get_ir_snapshot().get(
            "gpio_level"
        ),
        "ir_gpio_readback": get_ir_snapshot().get(
            "gpio_readback"
        ),
        "ir_active_high": bool(
            IR_ACTIVE_HIGH
        ),
    }


def distance_capture_allowed():
    return bool(get_tfmini_snapshot().get("valid"))


def setup_ir_gpio():
    if GPIO is None or IR_GPIO_PIN is None:
        with ir_lock:
            ir_state["control_enabled"] = False
            ir_state["active"] = None
            ir_state["gpio_level"] = None
            ir_state["gpio_readback"] = None
            ir_state["reason"] = (
                "SIN Jetson.GPIO"
                if GPIO is None
                else "IR_GPIO_PIN NO CONFIGURADO"
            )
        print("")
        print("IR AUTO: SIN CONTROL FISICO.")
        print("TFmini SI mide distancia, pero NO existe un GPIO que corte el iluminador.")
        print("V16.3 NO guardara capturas mientras ir_control_enabled=0.")
        print("Conecta EN/GATE/MOSFET/relay a un GPIO y define IR_GPIO_PIN.")
        print("")
        return False
    try:
        if IR_GPIO_MODE == "BCM":
            GPIO.setmode(GPIO.BCM)
        else:
            GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        GPIO.setup(IR_GPIO_PIN, GPIO.OUT, initial=(GPIO.HIGH if (False == IR_ACTIVE_HIGH) else GPIO.LOW))
        with ir_lock:
            ir_state["control_enabled"] = True
            ir_state["reason"] = "INICIADO"
        set_ir_active(True, "ARRANQUE/FUERA DE ZONA")
        print("IR AUTO: GPIO", IR_GPIO_PIN, "modo", IR_GPIO_MODE, "LISTO")
        return True
    except Exception as e:
        with ir_lock:
            ir_state["control_enabled"] = False
            ir_state["active"] = False
            ir_state["reason"] = "GPIO ERROR: {}".format(str(e)[:80])
        print("IR AUTO GPIO:", e)
        return False


def set_ir_active(active, reason=""):
    active = bool(active)

    with ir_lock:
        enabled = bool(
            ir_state.get(
                "control_enabled"
            )
        )
        old_value = ir_state.get(
            "active"
        )

    if not enabled or GPIO is None or IR_GPIO_PIN is None:
        with ir_lock:
            ir_state["active"] = None
            ir_state["gpio_level"] = None
            ir_state["gpio_readback"] = None
            ir_state["reason"] = (
                "SIN CONTROL FISICO | "
                + str(reason)
            )
        return False

    level = (
        GPIO.HIGH
        if (
            active
            == IR_ACTIVE_HIGH
        )
        else GPIO.LOW
    )

    try:
        GPIO.output(
            IR_GPIO_PIN,
            level
        )

        # Verifica al menos el nivel electrico del pin.
        readback = GPIO.input(
            IR_GPIO_PIN
        )

    except Exception as e:
        with ir_lock:
            ir_state["reason"] = (
                "GPIO WRITE ERROR: {}"
                .format(
                    str(e)[:80]
                )
            )
        return False

    with ir_lock:
        ir_state["active"] = active
        ir_state["gpio_level"] = int(
            level
        )
        ir_state["gpio_readback"] = int(
            readback
        )
        ir_state["reason"] = str(
            reason
        )

        if old_value != active:
            ir_state["last_change_iso"] = (
                datetime.now().isoformat(
                    timespec="milliseconds"
                )
            )

    if old_value != active:
        print(
            "IR AUTO:",
            "ON"
            if active
            else "OFF",
            "| GPIO",
            IR_GPIO_PIN,
            "| nivel",
            int(level),
            "| readback",
            int(readback),
            "|",
            reason
        )

    return True


def get_ir_snapshot():
    with ir_lock:
        return dict(ir_state)


def ir_distance_control_worker():
    """
    Politica:
      - 1 lectura valida fresca en 30-60 cm -> IR OFF inmediato.
      - 5 lecturas fuera de zona/sin datos -> IR ON (histeresis).
    La estabilidad evita parpadeo cerca de los limites.
    """
    valid_count = 0
    invalid_count = 0
    while not stop_event.is_set():
        snap = get_tfmini_snapshot()
        if snap.get("valid"):
            valid_count += 1
            invalid_count = 0
            if valid_count >= IR_VALID_STABLE_READS:
                set_ir_active(False, "DISTANCIA OPTIMA {} cm".format(snap.get("distance_cm")))
        else:
            invalid_count += 1
            valid_count = 0
            if invalid_count >= IR_INVALID_STABLE_READS:
                set_ir_active(True, "FUERA DE ZONA / SIN DATOS")
        time.sleep(0.05)


# ============================================================
# GSTREAMER
# ============================================================

def make_rgb_pipeline():
    return (
        'nvarguscamerasrc sensor-id={} ! '
        'video/x-raw(memory:NVMM), '
        'width=1920, height=1080, '
        'framerate=30/1, format=NV12 ! '
        'nvvidconv ! '
        'video/x-raw, width=960, height=540, format=BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    ).format(RGB_SENSOR_ID)


def make_noir_pipeline(gain_max):
    return (
        'nvarguscamerasrc sensor-id={} '
        'exposurecompensation={} '
        'gainrange="1 {}" '
        'ispdigitalgainrange="1 1" '
        'exposuretimerange="{} {}" ! '
        'video/x-raw(memory:NVMM), '
        'width=1920, height=1080, '
        'framerate=30/1, format=NV12 ! '
        'nvvidconv ! '
        'video/x-raw, width=960, height=540, format=BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=BGR ! '
        'appsink drop=true max-buffers=1 sync=false'
    ).format(
        NOIR_SENSOR_ID,
        EXPOSURE_COMPENSATION,
        gain_max,
        EXPOSURE_MIN_NS,
        EXPOSURE_MAX_NS,
    )


def open_cameras():
    print("Abriendo IMX219 RGB...")
    cap_rgb = cv2.VideoCapture(
        make_rgb_pipeline(),
        cv2.CAP_GSTREAMER
    )

    if not cap_rgb.isOpened():
        print("ERROR: no se pudo abrir IMX219 RGB.")
        return None, None

    print("Abriendo OV5647 NoIR gain 1-9...")
    cap_noir = cv2.VideoCapture(
        make_noir_pipeline(GAIN_MAX_PRIMARY),
        cv2.CAP_GSTREAMER
    )

    if not cap_noir.isOpened():
        cap_noir.release()
        print("Gain 1-9 no aceptado. Probando 1-8...")
        cap_noir = cv2.VideoCapture(
            make_noir_pipeline(GAIN_MAX_FALLBACK),
            cv2.CAP_GSTREAMER
        )

    if not cap_noir.isOpened():
        print("ERROR: no se pudo abrir OV5647.")
        cap_rgb.release()
        return None, None

    return cap_rgb, cap_noir


# ============================================================
# RED: PREVIEW ASINCRONO
# ============================================================

def put_latest(q, item):
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass



# ============================================================
# YOLO11n LOCAL - CLIENTE UNICO HACIA DOCKER
# ============================================================

def recv_exact_yolo(sock, count):
    data = bytearray()

    while len(data) < count:
        chunk = sock.recv(count - len(data))

        if not chunk:
            raise ConnectionError("Conexion YOLO cerrada")

        data.extend(chunk)

    return bytes(data)


def recv_json_yolo(sock):
    size = struct.unpack(
        "!I",
        recv_exact_yolo(sock, 4)
    )[0]

    payload = recv_exact_yolo(sock, size)

    return json.loads(
        payload.decode("utf-8")
    )


def yolo_send_frame(sock, frame, camera_id):
    ok, enc = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 72]
    )

    if not ok:
        raise RuntimeError("No se pudo codificar frame para YOLO")

    payload = enc.tobytes()

    sock.sendall(b"FRM2")
    sock.sendall(bytes([int(camera_id)]))
    sock.sendall(
        struct.pack("!I", len(payload))
    )
    sock.sendall(payload)

    return recv_json_yolo(sock)


def yolo_worker():
    sock = None

    while not stop_event.is_set():
        try:
            job = yolo_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        if job is None:
            break

        camera_id = int(job["camera_id"])
        frame = job["frame"]
        result_holder = job.get("result_holder")
        event = job.get("event")

        try:
            if sock is None:
                print(
                    "Conectando YOLO11n local {}:{}..."
                    .format(
                        YOLO_HOST,
                        YOLO_PORT
                    )
                )

                sock = socket.create_connection(
                    (YOLO_HOST, YOLO_PORT),
                    timeout=5.0
                )
                sock.settimeout(30.0)

                print(">>> YOLO11n LOCAL CONECTADO <<<")

            result = yolo_send_frame(
                sock,
                frame,
                camera_id
            )

            with yolo_state_lock:
                latest_yolo[camera_id] = {
                    "detections": result.get(
                        "detections",
                        []
                    ),
                    "inference_ms": float(
                        result.get(
                            "inference_ms",
                            0.0
                        )
                    ),
                    "updated": time.time(),
                }

            if result_holder is not None:
                result_holder["result"] = result
                result_holder["error"] = None

        except Exception as e:
            if result_holder is not None:
                result_holder["result"] = None
                result_holder["error"] = str(e)

            print(
                "YOLO local:",
                str(e)[:120]
            )

            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

            sock = None
            time.sleep(0.3)

        finally:
            if event is not None:
                event.set()

            yolo_queue.task_done()

    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass


def submit_yolo_live(frame, camera_id):
    job = {
        "camera_id": int(camera_id),
        "frame": frame.copy(),
        "result_holder": None,
        "event": None,
    }

    try:
        yolo_queue.put_nowait(job)
        return True
    except queue.Full:
        return False


def yolo_exact(frame, camera_id, timeout=30.0):
    holder = {}
    event = threading.Event()

    job = {
        "camera_id": int(camera_id),
        "frame": frame.copy(),
        "result_holder": holder,
        "event": event,
    }

    try:
        yolo_queue.put(job, timeout=3.0)
    except queue.Full:
        return {
            "ok": False,
            "camera_id": int(camera_id),
            "detections": [],
            "inference_ms": None,
            "error": "cola YOLO llena",
        }

    if not event.wait(timeout):
        return {
            "ok": False,
            "camera_id": int(camera_id),
            "detections": [],
            "inference_ms": None,
            "error": "timeout YOLO exacto",
        }

    if holder.get("result") is not None:
        return holder["result"]

    return {
        "ok": False,
        "camera_id": int(camera_id),
        "detections": [],
        "inference_ms": None,
        "error": holder.get(
            "error",
            "YOLO sin resultado"
        ),
    }


def yolo_telemetry_snapshot():
    with yolo_state_lock:
        rgb = dict(latest_yolo[0])
        noir = dict(latest_yolo[1])

    return {
        "rgb": rgb,
        "noir": noir,
        "valid_conf": YOLO_VALID_CONF,
        "weak_conf": YOLO_WEAK_CONF,
    }


def draw_yolo_boxes(frame, detections):
    out = frame.copy()

    for det in detections:
        box = det.get("box", [])
        if len(box) != 4:
            continue

        conf = float(
            det.get(
                "confidence",
                0.0
            )
        )

        if conf < YOLO_WEAK_CONF:
            continue

        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]

        name = str(
            det.get(
                "class_name",
                "objeto"
            )
        )

        if conf >= YOLO_VALID_CONF:
            color = (
                (0, 255, 0)
                if name.lower() == "person"
                else (0, 255, 255)
            )
            prefix = ""
        else:
            color = (140, 140, 140)
            prefix = "?"

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            out,
            "{}{} {:.0f}%".format(
                prefix,
                name.upper(),
                conf * 100.0
            ),
            (
                x1,
                max(18, y1 - 5)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA
        )

    return out


def preview_sender_worker():
    """
    SOLO STREAM:
    envia RGB + NoIR juntos en un JPEG horizontal ligero (480x270 por camara).
    No recibe CAPT ni ninguna orden de captura.
    GUARDAR LOTE y el dataset pertenecen completamente a la laptop.
    """
    sock = None

    while not stop_event.is_set():
        try:
            item = preview_queue.get(
                timeout=0.25
            )
        except queue.Empty:
            continue

        if item is None:
            break

        try:
            rgb_frame, noir_frame, telemetry = item

            rgb_small = cv2.resize(
                rgb_frame,
                (
                    PREVIEW_EACH_W,
                    PREVIEW_EACH_H
                ),
                interpolation=cv2.INTER_AREA
            )

            noir_small = cv2.resize(
                noir_frame,
                (
                    PREVIEW_EACH_W,
                    PREVIEW_EACH_H
                ),
                interpolation=cv2.INTER_AREA
            )

            combined = np.hstack(
                [
                    rgb_small,
                    noir_small
                ]
            )

            ok, enc = cv2.imencode(
                ".jpg",
                combined,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    PREVIEW_JPEG_QUALITY
                ]
            )

            if not ok:
                continue

            payload = enc.tobytes()

            if sock is None:
                sock = socket.create_connection(
                    (
                        LAPTOP_IP,
                        PREVIEW_PORT
                    ),
                    timeout=0.8
                )

                sock.settimeout(
                    0.8
                )

            telemetry_bytes = json.dumps(
                telemetry,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":"
                )
            ).encode(
                "utf-8"
            )

            sock.sendall(
                b"FPR8"
            )

            sock.sendall(
                struct.pack(
                    "!I",
                    len(
                        telemetry_bytes
                    )
                )
            )

            sock.sendall(
                telemetry_bytes
            )

            sock.sendall(
                struct.pack(
                    "!I",
                    len(
                        payload
                    )
                )
            )

            sock.sendall(
                payload
            )

        except Exception:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

            sock = None
            time.sleep(
                0.15
            )

    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass



def original_sender_worker():
    """Stream continuo 5052: RGB + NoIR 960x540 por camara, sin CAPT."""
    sock = None
    while not stop_event.is_set():
        try:
            item = original_queue.get(timeout=0.25)
        except queue.Empty:
            continue
        if item is None:
            break
        try:
            rgb_frame, noir_frame, telemetry = item
            combined = np.hstack([rgb_frame, noir_frame])
            ok, enc = cv2.imencode(
                ".jpg", combined,
                [cv2.IMWRITE_JPEG_QUALITY, ORIGINAL_JPEG_QUALITY]
            )
            if not ok:
                continue
            payload = enc.tobytes()
            if sock is None:
                sock = socket.create_connection(
                    (LAPTOP_IP, ORIGINAL_STREAM_PORT), timeout=0.8
                )
                sock.settimeout(0.8)
            meta = {
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "width_each": int(rgb_frame.shape[1]),
                "height_each": int(rgb_frame.shape[0]),
                "ir_active": telemetry.get("ir_active"),
                "ir_control_enabled": telemetry.get("ir_control_enabled"),
                "distance_cm": telemetry.get("distance_cm"),
            }
            meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
            sock.sendall(b"FORG")
            sock.sendall(struct.pack("!I", len(meta_bytes)))
            sock.sendall(meta_bytes)
            sock.sendall(struct.pack("!I", len(payload)))
            sock.sendall(payload)
        except Exception:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
            sock = None
            time.sleep(0.15)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass


# ============================================================
# UI
# ============================================================

def mouse_callback(event, x, y, flags, param):
    # V11.1: captura manual eliminada.
    return


def make_preview(rgb_frame, noir_frame, fps):
    with yolo_state_lock:
        rgb_dets = list(latest_yolo[0].get("detections", []))
        noir_dets = list(latest_yolo[1].get("detections", []))

    rgb_yolo_view = draw_yolo_boxes(rgb_frame, rgb_dets)
    noir_yolo_view = draw_yolo_boxes(noir_frame, noir_dets)

    rgb = cv2.resize(rgb_yolo_view, (480, 270), interpolation=cv2.INTER_AREA)
    noir = cv2.resize(noir_yolo_view, (480, 270), interpolation=cv2.INTER_AREA)

    cv2.putText(rgb, "RGB - IMX219", (12,27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(noir, "NoIR - OV5647", (12,27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    top = np.hstack([rgb, noir])
    canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    canvas[0:270, 0:960] = top

    telemetry = get_tfmini_snapshot()
    distance = telemetry.get("distance_cm")
    valid = bool(telemetry.get("valid"))
    status = telemetry.get("status", "SIN DATOS")
    strength = telemetry.get("strength")
    ir_active = bool(telemetry.get("ir_active", False))
    ir_enabled = bool(telemetry.get("ir_control_enabled", False))

    distance_color = (0,255,0) if valid else (0,0,255)
    dist_text = "{} cm".format(distance) if distance is not None else "--- cm"

    cv2.putText(canvas, "TFmini: {} | {} | ZONA: {}-{} cm".format(dist_text, status, DISTANCE_MIN_CM, DISTANCE_MAX_CM), (15,298), cv2.FONT_HERSHEY_SIMPLEX, 0.52, distance_color, 2, cv2.LINE_AA)
    ir_label = (
        ("ON" if bool(ir_active) else "OFF")
        if ir_enabled
        else "SIN CONTROL FISICO"
    )
    cv2.putText(
        canvas,
        "IR: {} | control {} | Fuerza: {} | FPS {:.1f}".format(
            ir_label,
            "GPIO" if ir_enabled else "NO",
            strength if strength is not None else "---",
            fps
        ),
        (15,326),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0,255,255) if bool(ir_active) else (0,255,0),
        1,
        cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "SOLO STREAM -> dataset/captura viven en LAPTOP",
        (15,358),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255,255,255),
        1,
        cv2.LINE_AA
    )
    cv2.putText(canvas, "Stream limpio 5051 + original 5052 | TFmini fuera del rostro | Q=salir", (15,386), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)

    if last_status and time.time() - last_status_time < 5.0:
        cv2.putText(canvas, last_status, (15,414), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,255,255), 1, cv2.LINE_AA)

    return canvas


# ============================================================
# MAIN
# ============================================================

resolve_laptop_ip()

if not LAPTOP_IP:
    print("ERROR: falta IP de laptop.")
    raise SystemExit(1)

print("")
print("==============================================")
print(" JETSON v11.5 RGB + NOIR + YOLO11n + TFmini + IR AUTO | SOLO STREAM")
print("==============================================")
print("Laptop:", LAPTOP_IP)
print("Preview puerto:", PREVIEW_PORT)
print("Jetson SOLO envia stream RGB + NoIR + telemetria. No captura dataset.")
print("GUARDAR LOTE y TODO el dataset se ejecutan en la laptop.")
print("YOLO11n SI corre en Jetson/CUDA. 68 puntos corren en laptop.")
print("TFmini:", TFMINI_PORT, "@", TFMINI_BAUD)
print("IR auto GPIO:", IR_GPIO_PIN, "| modo:", IR_GPIO_MODE, "| active_high:", IR_ACTIVE_HIGH, "| control:", "ACTIVO" if IR_GPIO_PIN is not None and GPIO is not None else "SIN CONTROL FISICO")
print("Distancia objetivo:", TARGET_DISTANCE_CM, "cm | tolerancia:", DISTANCE_MIN_CM, "-", DISTANCE_MAX_CM, "cm")
print("Receiver:", "CONECTADO" if ping_laptop() else "NO RESPONDE")
print("==============================================")
print("")

cap_rgb, cap_noir = open_cameras()

if cap_rgb is None or cap_noir is None:
    raise SystemExit(1)

tfmini_thread = threading.Thread(
    target=tfmini_reader_worker,
    daemon=True
)
tfmini_thread.start()

setup_ir_gpio()
ir_thread = threading.Thread(
    target=ir_distance_control_worker,
    daemon=True
)
ir_thread.start()

yolo_thread = threading.Thread(
    target=yolo_worker,
    daemon=True
)
yolo_thread.start()

preview_thread = threading.Thread(
    target=preview_sender_worker,
    daemon=True
)
preview_thread.start()

original_thread = threading.Thread(
    target=original_sender_worker,
    daemon=True
)
original_thread.start()

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    DISPLAY_W,
    DISPLAY_H
)

fps = 0.0
frames = 0
t0 = time.time()

last_preview_send = 0.0
last_original_send = 0.0
last_rgb = None
last_noir = None
last_yolo_live = 0.0
next_yolo_camera = 0

try:
    while True:
        ok_rgb = cap_rgb.grab()
        ok_noir = cap_noir.grab()

        if not ok_rgb or not ok_noir:
            print("ERROR en grab de camaras.")
            break

        ret_rgb, rgb_frame = cap_rgb.retrieve()
        ret_noir, noir_frame = cap_noir.retrieve()

        if (
            not ret_rgb
            or rgb_frame is None
        ):
            print("ERROR al recibir IMX219.")
            break

        if (
            not ret_noir
            or noir_frame is None
        ):
            print("ERROR al recibir OV5647.")
            break

        # Corregir orientacion ANTES de preview/captura, para no guardar imagenes de cabeza.
        if ROTATE_RGB_180:
            rgb_frame = cv2.rotate(rgb_frame, cv2.ROTATE_180)
        if ROTATE_NOIR_180:
            noir_frame = cv2.rotate(noir_frame, cv2.ROTATE_180)

        last_rgb = rgb_frame
        last_noir = noir_frame

        frames += 1
        now = time.time()

        if now - last_yolo_live >= YOLO_LIVE_INTERVAL:
            if next_yolo_camera == 0:
                submitted = submit_yolo_live(
                    rgb_frame,
                    0
                )
            else:
                submitted = submit_yolo_live(
                    noir_frame,
                    1
                )

            if submitted:
                next_yolo_camera = (
                    1
                    if next_yolo_camera == 0
                    else 0
                )
                last_yolo_live = now

        if now - t0 >= 1.0:
            fps = frames / (now - t0)
            frames = 0
            t0 = now

        # Preview de laptop:
        # solo copiamos a una cola; JAMAS esperamos red aqui.
        if now - last_preview_send >= PREVIEW_SEND_INTERVAL:
            preview_meta = get_tfmini_snapshot()
            preview_meta["yolo11n"] = yolo_telemetry_snapshot()

            put_latest(
                preview_queue,
                (
                    rgb_frame.copy(),
                    noir_frame.copy(),
                    preview_meta
                )
            )
            last_preview_send = now

        # Stream original independiente: no bloquea el tracking de 30 FPS.
        if now - last_original_send >= ORIGINAL_SEND_INTERVAL:
            original_meta = get_tfmini_snapshot()
            put_latest(
                original_queue,
                (rgb_frame.copy(), noir_frame.copy(), original_meta)
            )
            last_original_send = now

        cv2.imshow(
            WINDOW_NAME,
            make_preview(
                rgb_frame,
                noir_frame,
                fps
            )
        )

        key = cv2.waitKey(1) & 0xFF

        if key in (
            ord("q"),
            ord("Q")
        ):
            break

finally:
    stop_event.set()
    try:
        set_ir_active(False, "SALIDA DEL PROGRAMA")
        if GPIO is not None and IR_GPIO_PIN is not None:
            GPIO.cleanup(IR_GPIO_PIN)
    except Exception:
        pass

    try:
        put_latest(
            preview_queue,
            None
        )
    except Exception:
        pass

    try:
        put_latest(
            original_queue,
            None
        )
    except Exception:
        pass

    try:
        yolo_queue.put_nowait(None)
    except Exception:
        pass

    cap_rgb.release()
    cap_noir.release()
    cv2.destroyAllWindows()

    print("")
    print("Programa cerrado.")
    print("La Jetson no guarda el dataset localmente.")