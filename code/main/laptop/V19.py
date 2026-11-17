#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import argparse
import shutil
import csv
import queue
import uuid
from collections import deque
import io
import json
import math
import os
import socket
import struct
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("")
    print("FALTA MEDIAPIPE.")
    print("Ejecuta:")
    print("py -3.13 -m pip install --upgrade mediapipe")
    print("")
    raise

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.drawing.spreadsheet_drawing import AnchorMarker, OneCellAnchor
    from openpyxl.drawing.xdr import XDRPositiveSize2D
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.utils.cell import coordinate_to_tuple
    from openpyxl.utils.units import pixels_to_EMU, points_to_pixels
except ImportError:
    print("")
    print("FALTA openpyxl.")
    print("Ejecuta:")
    print("py -3.13 -m pip install openpyxl pillow")
    print("")
    raise

# ============================================================
# LAPTOP v9 SPLIT
#
# JETSON:
#   - RGB + NoIR
#   - TFmini
#   - YOLO11n / CUDA
#
# LAPTOP:
#   - 68 landmarks RGB
#   - 68 landmarks NoIR
#   - rasgos geometricos por region
#   - Excel / dataset
#
# Puertos conservados de V8:
#   5050 = capturas
#   5051 = preview
# ============================================================

HOST = "0.0.0.0"
DATASET_PORT = 5050
PREVIEW_PORT = 5051
ORIGINAL_STREAM_PORT = 5052

BASE_DIR = Path(__file__).resolve().parent

LEGACY_DATASET_DIR = (
    BASE_DIR
    / "dataset_rgb_noir_yolo_landmarks_v9"
)

LEGACY_DATASET_DIR_V13 = (
    BASE_DIR
    / "dataset_rgb_noir_micro_v13"
)

LEGACY_DATASET_DIR_V14 = (
    BASE_DIR
    / "dataset_rgb_noir_micro_v14"
)

DATASET_DIR = (
    BASE_DIR
    / "dataset_rgb_noir_micro_v16_19_calidad"
)

CAPTURES_DIR = (
    DATASET_DIR
    / "capturas"
)

EXCEL_FILE = (
    DATASET_DIR
    / "registro_dataset_rgb_noir_micro_v16_19_calidad.xlsx"
)

# V16.11 - escritura Excel robusta
EXCEL_WRITE_LOCK = threading.Lock()
ACTIVE_EXCEL_FILE = EXCEL_FILE
LAST_EXCEL_PATH_FILE = DATASET_DIR / "ULTIMO_EXCEL_GUARDADO.txt"
EXCEL_WRITE_LOG = DATASET_DIR / "excel_write_log.txt"

# V16.19 - control de calidad historico de landmarks.
QC_HISTORY_FILE = DATASET_DIR / "landmark_qc_history_v16_19.json"
QC_HISTORY_MIN_SAMPLES = 8
QC_HISTORY_MAX_SAMPLES = 200
QC_IQR_MULTIPLIER = 1.5



def _excel_log(message):
    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().isoformat(timespec="milliseconds")
        with EXCEL_WRITE_LOG.open("a", encoding="utf-8") as f:
            f.write("{} | {}\n".format(stamp, message))
    except Exception:
        pass


def _active_excel_path():
    return Path(ACTIVE_EXCEL_FILE)


def _set_active_excel_path(path):
    global ACTIVE_EXCEL_FILE
    ACTIVE_EXCEL_FILE = Path(path)
    try:
        LAST_EXCEL_PATH_FILE.write_text(str(ACTIVE_EXCEL_FILE), encoding="utf-8")
    except Exception:
        pass


def _save_workbook_robust(wb, target_path, capture_id):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error = None

    # Primero intenta guardar de forma normal.
    for attempt in range(1, 4):
        tmp = target_path.with_name(target_path.stem + ".tmp.xlsx")
        try:
            if tmp.exists():
                tmp.unlink()
            wb.save(str(tmp))
            os.replace(str(tmp), str(target_path))
            _set_active_excel_path(target_path)
            _excel_log("OK {} {} intento={}".format(capture_id, target_path, attempt))
            return target_path
        except Exception as exc:
            last_error = exc
            _excel_log("FALLO {} {} intento={} {!r}".format(capture_id, target_path, attempt, exc))

            # Si el temporal ya es un XLSX valido, wb.save SI funciono y
            # solamente Windows bloqueo el replace del archivo abierto.
            try:
                if tmp.exists() and tmp.stat().st_size > 0 and zipfile.is_zipfile(str(tmp)):
                    recovery = target_path.with_name(
                        target_path.stem
                        + "_RECUPERACION_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S")
                        + ".xlsx"
                    )
                    os.replace(str(tmp), str(recovery))
                    _set_active_excel_path(recovery)
                    _excel_log(
                        "RECUPERACION DESDE TEMP {} {} error={!r}".format(
                            capture_id, recovery, exc
                        )
                    )
                    print(
                        "[EXCEL AVISO] El Excel principal esta abierto/bloqueado. "
                        "Los datos SI se guardaron en:",
                        recovery
                    )
                    return recovery
            except Exception as recovery_exc:
                _excel_log("FALLO recovery temporal {!r}".format(recovery_exc))

            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            time.sleep(0.25 * attempt)

    # Si el guardado temporal tambien fallo, ultimo respaldo directo.
    recovery = target_path.with_name(
        target_path.stem
        + "_RECUPERACION_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".xlsx"
    )
    wb.save(str(recovery))
    _set_active_excel_path(recovery)
    _excel_log("RECUPERACION {} {} error={!r}".format(capture_id, recovery, last_error))
    print("[EXCEL AVISO] Archivo principal bloqueado. Se seguira usando:", recovery)
    return recovery


def _safe_excel_image(ws, cell_name, image_path):
    try:
        if image_path and Path(image_path).exists():
            add_excel_image(ws, cell_name, image_path)
            return True
    except Exception as exc:
        _excel_log("FOTO OMITIDA cell={} path={} error={!r}".format(cell_name, image_path, exc))
        print("[EXCEL] Foto omitida, pero las filas SI se guardaran:", exc)
    return False

DATASET_SCHEMA_VERSION = "16.19"

SESSION_ID = (
    "SESSION_"
    + datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    + "_"
    + uuid.uuid4().hex[:8]
)

BLENDSHAPE_SERIES_CSV = (
    DATASET_DIR
    / "blendshapes_serie_temporal.csv"
)

DATASET_MANIFEST = (
    DATASET_DIR
    / "dataset_manifest_v16_11.json"
)


def prepare_dataset(
    reset=False
):
    """
    Modo seguro:
    - NUNCA borra datasets anteriores.
    - Siempre trabaja dentro del dataset V16.3 nuevo.
    - El argumento reset se ignora por seguridad.
    """
    if reset:
        print(
            "AVISO: --reset-dataset fue ignorado. "
            "Esta version NO borra ningun dataset."
        )

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CAPTURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "session_id": SESSION_ID,
        "created_at": datetime.now().isoformat(
            timespec="milliseconds"
        ),
        "distance_protocol_cm": {
            "min": 30,
            "target": 45,
            "max": 60,
        },
        "missing_value_policy": (
            "None/blank means unavailable. Numeric zero is used only "
            "when the measured quantity is truly zero."
        ),
        "manual_condition_labels": {
            "lentes": "manual boolean selected before capture",
            "barba": "manual boolean selected before capture",
            "cabello_frente": (
                "manual boolean: hair visibly covers part of forehead"
            ),
        },
        "sources": [
            "RGB",
            "NOIR_ORIGINAL",
            "NOIR_RED",
            "NOIR_IRCUT",
            "NOIR_BLUE",
            "NOIR_IR",
            "NOIR_UV",
        ],
        "virtual_filter_warning": (
            "RED/IR-CUT/BLUE/IR/UV are digital virtual transforms of "
            "the NoIR image, not physical spectral bandpass filters."
        ),
        "microexpression_policy": (
            "All temporal facial events are stored. is_micro_candidate=1 "
            "only when duration/activity satisfy the micro-expression rule. "
            "FACIAL_CHANGE rows are retained as non-micro temporal events."
        ),
        "multiface_policy": {
            "training_subjects": (
                "All MediaPipe faces up to num_faces=6 are kept independently. "
                "Each face receives a Pxx ID, crop, blendshape series and "
                "micro-expression tracking."
            ),
            "background_people": (
                "YOLO people without a MediaPipe face remain only in OBJETOS. "
                "People with a valid visible face may enter the emotional dataset."
            ),
            "image_policy": (
                "Full frames are archived for traceability. Excel CAPTURAS uses "
                "a face strip without background; PERSONAS/FILTROS_NOIR use "
                "individual 256x256 face crops."
            ),
            "manual_labels": (
                "LENTES/BARBA/CABELLO_FRENTE are selected independently for "
                "each Pxx card in the interface."
            ),
        },
        "person_id_policy": {
            "PERSONAS_sheet": (
                "Pxx IDs correspond to tracked visible faces. RGB tracking "
                "provides the canonical IDs and NoIR/filter faces are matched "
                "spatially to those IDs."
            ),
            "note": (
                "Pxx is a tracking/session identifier, not a real name. "
                "It is used to separate samples and must not be interpreted "
                "as an emotion label."
            ),
        },
        "yolo_thresholds": {
            "general_valid_conf": 0.25,
            "person_valid_conf": 0.45,
            "note": (
                "Person uses a stricter threshold to reduce false people "
                "without changing other COCO object classes."
            ),
        },
        "training_note": (
            "Use session/person-level splits, not random frame splits, "
            "to avoid temporal leakage."
        ),
    }

    DATASET_MANIFEST.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

FACE_LANDMARKER_MODEL = (
    BASE_DIR
    / "face_landmarker.task"
)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/"
    "face_landmarker.task"
)

# 68 puntos equivalentes construidos desde la malla MediaPipe.
# Orden conservado:
# 0-16 mandibula
# 17-21 ceja izquierda
# 22-26 ceja derecha
# 27-35 nariz
# 36-41 ojo izquierdo
# 42-47 ojo derecho
# 48-59 boca exterior
# 60-67 boca interior
MP_TO_68 = [
    # mandibula 17
    234, 93, 132, 58, 172, 136, 150, 149, 152,
    378, 379, 365, 397, 288, 361, 323, 454,

    # ceja izquierda 5
    70, 63, 105, 66, 107,

    # ceja derecha 5
    336, 296, 334, 293, 300,

    # nariz 9
    168, 6, 197, 195, 5, 4, 45, 275, 1,

    # ojo izquierdo 6
    33, 160, 158, 133, 153, 144,

    # ojo derecho 6
    362, 385, 387, 263, 373, 380,

    # boca exterior 12
    61, 185, 40, 39, 37, 0,
    267, 269, 270, 409, 291, 17,

    # boca interior 8
    78, 191, 80, 81, 13, 311, 310, 415,
]


# ============================================================
# NOIR FILTER LAB v8.1 - FILTROS CANONICOS
# ============================================================

RGB_COLOR_BOOST = 1.20

FILTERS_NOIR = [
    "ORIGINAL NOIR",
    "RED BANDPASS VIRTUAL",
    "IR-CUT / RGB VIRTUAL",
    "BLUE BANDPASS VIRTUAL",
    "IR BANDPASS VIRTUAL",
    "UV BANDPASS VIRTUAL",
]

NOIR_FILTER_FILENAMES = [
    "NOIR_original.png",
    "NOIR_red_bandpass_virtual.png",
    "NOIR_ir_cut_rgb_virtual.png",
    "NOIR_blue_bandpass_virtual.png",
    "NOIR_ir_bandpass_virtual.png",
    "NOIR_uv_bandpass_virtual.png",
]


NOIR_FILTER_IDS = [
    "NOIR_ORIGINAL",
    "NOIR_RED",
    "NOIR_IRCUT",
    "NOIR_BLUE",
    "NOIR_IR",
    "NOIR_UV",
]

NOIR_FILTER_SHORT = [
    "ORIG",
    "RED",
    "IRCUT",
    "BLUE",
    "IR",
    "UV",
]

SHADOW_X = np.array(
    [0, 16, 32, 64, 96, 128, 160, 192, 224, 255],
    dtype=np.float32
)

SHADOW_Y = np.array(
    [0, 22, 42, 76, 110, 143, 174, 204, 232, 255],
    dtype=np.float32
)

SHADOW_LUT = np.interp(
    np.arange(256),
    SHADOW_X,
    SHADOW_Y
).astype(np.uint8)

CONTRAST_LUT = np.array(
    [
        np.clip(
            (i - 128.0) * 1.035 + 128.0,
            0,
            255
        )
        for i in range(256)
    ],
    dtype=np.uint8
)


def light_chroma_denoise(frame):
    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(
        ycrcb
    )

    cr = cv2.medianBlur(
        cr,
        3
    )

    cb = cv2.medianBlur(
        cb,
        3
    )

    return cv2.cvtColor(
        cv2.merge(
            [y, cr, cb]
        ),
        cv2.COLOR_YCrCb2BGR
    )


def mild_shadow_lift(frame):
    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(
        ycrcb
    )

    y = cv2.LUT(
        y,
        SHADOW_LUT
    )

    y = cv2.LUT(
        y,
        CONTRAST_LUT
    )

    return cv2.cvtColor(
        cv2.merge(
            [y, cr, cb]
        ),
        cv2.COLOR_YCrCb2BGR
    )


def sharpen_light(frame):
    blur = cv2.GaussianBlur(
        frame,
        (0, 0),
        0.65
    )

    return cv2.addWeighted(
        frame,
        1.10,
        blur,
        -0.10,
        0
    )


def prepare_base(frame):
    out = light_chroma_denoise(
        frame
    )

    out = mild_shadow_lift(
        out
    )

    out = sharpen_light(
        out
    )

    return out


def clear_gray(frame):
    base = prepare_base(
        frame
    )

    b, g, r = cv2.split(
        base
    )

    gray = (
        0.22 * b.astype(np.float32)
        + 0.33 * g.astype(np.float32)
        + 0.45 * r.astype(np.float32)
    )

    gray = np.clip(
        gray,
        0,
        255
    ).astype(np.uint8)

    gray = cv2.medianBlur(
        gray,
        3
    )

    return gray


def tint_gray(
    gray,
    b_gain,
    g_gain,
    r_gain
):
    f = gray.astype(
        np.float32
    )

    b = np.clip(
        f * b_gain,
        0,
        255
    ).astype(np.uint8)

    g = np.clip(
        f * g_gain,
        0,
        255
    ).astype(np.uint8)

    r = np.clip(
        f * r_gain,
        0,
        255
    ).astype(np.uint8)

    return cv2.merge(
        [b, g, r]
    )


def gray_world_limited(frame):
    work = frame.astype(
        np.float32
    )

    means = (
        work
        .reshape(-1, 3)
        .mean(axis=0)
        + 1e-6
    )

    target = float(
        np.mean(means)
    )

    desired = (
        target / means
    )

    strength = 0.52

    gains = (
        1.0
        + strength
        * (desired - 1.0)
    )

    gains = np.clip(
        gains,
        [0.86, 0.90, 0.82],
        [1.14, 1.26, 1.12]
    )

    work[:, :, 0] *= gains[0]
    work[:, :, 1] *= gains[1]
    work[:, :, 2] *= gains[2]

    return np.clip(
        work,
        0,
        255
    ).astype(np.uint8)


def rgb_virtual(frame):
    out = prepare_base(
        frame
    )

    out = gray_world_limited(
        out
    )

    work = out.astype(
        np.float32
    )

    b = work[:, :, 0]
    g = work[:, :, 1]
    r = work[:, :, 2]

    b2 = (
        0.90 * b
        + 0.10 * g
    )

    g2 = (
        0.10 * b
        + 1.14 * g
        + 0.08 * r
    )

    r2 = (
        0.05 * g
        + 0.80 * r
    )

    out = cv2.merge(
        [
            np.clip(
                b2,
                0,
                255
            ).astype(np.uint8),

            np.clip(
                g2,
                0,
                255
            ).astype(np.uint8),

            np.clip(
                r2,
                0,
                255
            ).astype(np.uint8),
        ]
    )

    hsv = cv2.cvtColor(
        out,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(
        hsv
    )

    s = np.clip(
        s.astype(np.float32)
        * RGB_COLOR_BOOST,
        0,
        255
    ).astype(np.uint8)

    out = cv2.cvtColor(
        cv2.merge(
            [h, s, v]
        ),
        cv2.COLOR_HSV2BGR
    )

    return sharpen_light(
        out
    )


def filter_original(frame):
    return frame.copy()


def filter_red(frame):
    gray = clear_gray(
        frame
    )

    out = tint_gray(
        gray,
        0.10,
        0.20,
        1.00
    )

    return sharpen_light(
        out
    )


def filter_blue(frame):
    gray = clear_gray(
        frame
    )

    out = tint_gray(
        gray,
        1.00,
        0.24,
        0.10
    )

    return sharpen_light(
        out
    )


def filter_ir(frame):
    gray = clear_gray(
        frame
    )

    out = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    return sharpen_light(
        out
    )


def filter_uv(frame):
    gray = clear_gray(
        frame
    )

    out = tint_gray(
        gray,
        0.82,
        0.12,
        1.00
    )

    return sharpen_light(
        out
    )


def apply_noir_filter(
    frame,
    index
):
    if index == 0:
        return filter_original(
            frame
        )

    if index == 1:
        return filter_red(
            frame
        )

    if index == 2:
        return rgb_virtual(
            frame
        )

    if index == 3:
        return filter_blue(
            frame
        )

    if index == 4:
        return filter_ir(
            frame
        )

    if index == 5:
        return filter_uv(
            frame
        )

    return frame.copy()


VALID_CONF_DEFAULT = 0.25
PERSON_VALID_CONF = 0.45
WEAK_CONF_DEFAULT = 0.10

# ============================================================
# PREVIEW / UI
# ============================================================

WINDOW_NAME = (
    "LAPTOP v16.18 | PERSONAS + FILTROS NOIR | SIN ETIQUETAS"
)

DISPLAY_EACH_W = 480
DISPLAY_EACH_H = 270
BOTTOM_H = 390
DISPLAY_W = 960
DISPLAY_H = DISPLAY_EACH_H + BOTTOM_H

latest_frame_lock = threading.Lock()
latest_frame = None

latest_telemetry_lock = threading.Lock()
latest_telemetry = {
    "distance_cm": None,
    "valid": False,
    "status": "SIN DATOS",
    "zone_min_cm": 30,
    "zone_max_cm": 60,
    "yolo11n": {},
}


# ============================================================
# V16.3 - CAPTURA LOCAL DESDE EL STREAM + VENTANA 6 FILTROS
# ============================================================
# GUARDAR LOTE ya NO ordena una captura a la Jetson.
# La laptop congela el ultimo RGB + NoIR recibidos por preview,
# construye el paquete en memoria y llama directamente al mismo
# pipeline process_dataset_zip() que genera Excel, filtros y metadata.
FILTER_WINDOW_NAME = "V16.18 | PERSONAS + FILTROS NOIR"
FILTER_GRID_W = 960
FILTER_GRID_H = 600

local_capture_lock = threading.Lock()
local_capture_state = {
    "busy": False,
    "status": "SIN CAPTURA",
    "last_capture_id": None,
    "last_path": None,
    "last_error": None,
}

# Las detecciones YOLO que llegan en telemetria fueron calculadas en Jetson
# sobre 960x540 por camara. Al guardar desde preview se reescalan.
YOLO_SOURCE_W = 960
YOLO_SOURCE_H = 540


def local_capture_status_text():
    with local_capture_lock:
        return str(local_capture_state.get("status", "SIN CAPTURA"))


def _encode_png_bytes(frame):
    ok, enc = cv2.imencode(
        ".png",
        frame,
        [int(cv2.IMWRITE_PNG_COMPRESSION), 2]
    )
    if not ok:
        raise RuntimeError("No se pudo codificar PNG local")
    return enc.tobytes()


def _scaled_yolo_for_local_frame(yolo, rgb_shape, noir_shape):
    yolo = copy.deepcopy(yolo or {})

    valid_conf = float(
        yolo.get(
            "valid_conf",
            VALID_CONF_DEFAULT
        )
    )

    weak_conf = float(
        yolo.get(
            "weak_conf",
            WEAK_CONF_DEFAULT
        )
    )

    rgb_info = copy.deepcopy(
        yolo.get(
            "rgb",
            {}
        )
    )

    noir_info = copy.deepcopy(
        yolo.get(
            "noir",
            {}
        )
    )

    rgb_info["detections"] = scale_detections(
        rgb_info.get(
            "detections",
            []
        ),
        YOLO_SOURCE_W,
        YOLO_SOURCE_H,
        int(rgb_shape[1]),
        int(rgb_shape[0])
    )

    noir_info["detections"] = scale_detections(
        noir_info.get(
            "detections",
            []
        ),
        YOLO_SOURCE_W,
        YOLO_SOURCE_H,
        int(noir_shape[1]),
        int(noir_shape[0])
    )

    return {
        "rgb": rgb_info,
        "noir": noir_info,
        "valid_conf": valid_conf,
        "weak_conf": weak_conf,
    }



def _json_safe_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    return value


def _face_to_json(face):
    if not isinstance(face, dict):
        return {}
    keep = {}
    for key in (
        "person_id", "track_id", "bbox", "landmarks_detected", "landmarks",
        "metrics", "blendshapes", "quality_gate", "quality_status",
        "training_valid"
    ):
        if key in face:
            keep[key] = _json_safe_value(face.get(key))
    return keep


def _faces_to_json(faces):
    return [_face_to_json(f) for f in (faces or [])]


def _ids_with_landmarks(faces):
    out = set()
    for face in (faces or []):
        pid = face.get("person_id") if isinstance(face, dict) else None
        if not pid:
            continue
        if bool(face.get("landmarks_detected", False)) and face.get("landmarks") is not None:
            out.add(str(pid))
    return out


def _set_people_back_to_pending(person_ids):
    with registration_lock:
        for pid in (person_ids or []):
            item = person_registration.get(str(pid))
            if item is not None:
                item["status"] = "PENDING"


def _wait_for_complete_live_snapshot(selected_person_ids, timeout_s=8.0):
    """Obtener RGB+NoIR limpios; NUNCA exigir 7 detecciones para guardar.

    Se prefieren resultados RGB/NoIR del mismo seq. Si el motor RGB tarda,
    se permite un frame RGB reciente y se registra la diferencia de seq.
    Las mediciones ausentes se mantienen ausentes, sin inventar landmarks.
    """
    with latest_noir_cycle_lock:
        start_seq = int(latest_noir_cycle_snapshot.get("seq", -1))
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        with latest_noir_cycle_lock:
            ns = latest_noir_cycle_snapshot
            seq = int(ns.get("seq", -1))
            noir = ns.get("frame")
            noir = None if noir is None else noir.copy()
            filters = copy.deepcopy(ns.get("filter_faces", {}))
            micro = copy.deepcopy(ns.get("filter_micro", {}))
            ts = int(ns.get("timestamp_ms", 0))
        if noir is None or seq <= start_seq:
            time.sleep(0.02)
            continue
        with live_capture_cache_lock:
            rgb_snap = live_rgb_capture_cache.get(seq)
            if rgb_snap is None and live_rgb_capture_cache:
                closest_seq = min(live_rgb_capture_cache,
                                  key=lambda k: abs(int(k) - seq))
                rgb_snap = live_rgb_capture_cache[closest_seq]
            else:
                closest_seq = seq
            if rgb_snap is not None:
                rgb = rgb_snap.get("frame")
                rgb = None if rgb is None else rgb.copy()
                faces = copy.deepcopy(rgb_snap.get("faces", []))
                rgb_ts = int(rgb_snap.get("timestamp_ms", 0))
            else:
                rgb = None
                faces = []
                rgb_ts = 0
        if rgb is None:
            time.sleep(0.02)
            continue
        with latest_dataset_input_lock:
            raw_rgb = latest_dataset_input.get(0)
            raw_noir = latest_dataset_input.get(1)
            raw_rgb = None if raw_rgb is None else raw_rgb.copy()
            raw_noir = None if raw_noir is None else raw_noir.copy()
        return {
            "seq": seq, "rgb_seq": int(closest_seq),
            "timestamp_ms": max(ts, rgb_ts),
            "rgb_frame": rgb, "noir_frame": noir,
            "rgb_faces": faces, "filter_faces": filters,
            "filter_micro": micro,
            "raw_rgb": raw_rgb, "raw_noir": raw_noir,
        }
    raise RuntimeError("No llegaron frames limpios RGB y NoIR en {:.1f}s".format(timeout_s))


def build_local_dataset_payload(selected_person_ids=None):
    """
    V16.11: toma un frame que YA fue validado por los motores LIVE.
    RGB y los seis filtros NoIR corresponden al mismo seq del preview limpio.
    """
    selected_ids = [str(x) for x in (selected_person_ids or []) if x]
    snapshot = _wait_for_complete_live_snapshot(selected_ids, timeout_s=8.0)

    rgb = snapshot["rgb_frame"]
    noir = snapshot["noir_frame"]
    raw_rgb = snapshot.get("raw_rgb")
    raw_noir = snapshot.get("raw_noir")

    with latest_telemetry_lock:
        telemetry = copy.deepcopy(latest_telemetry)

    now = datetime.now()
    capture_id = "CAP_LOCAL_" + now.strftime("%Y%m%d_%H%M%S_%f")[:-3]

    yolo_scaled = _scaled_yolo_for_local_frame(
        telemetry.get("yolo11n", {}), rgb.shape, noir.shape
    )
    tfmini = copy.deepcopy(telemetry)
    tfmini.pop("yolo11n", None)

    analysis_faces = {
        "RGB": _faces_to_json(snapshot.get("rgb_faces", [])),
        "filters": {},
        "filter_micro": _json_safe_value(snapshot.get("filter_micro", {})),
    }
    for idx, fid in enumerate(NOIR_FILTER_IDS):
        faces = snapshot.get("filter_faces", {}).get(
            idx,
            snapshot.get("filter_faces", {}).get(str(idx), [])
        )
        analysis_faces["filters"][fid] = _faces_to_json(faces)

    metadata = {
        "protocol_version": 16.12,
        "capture_id": capture_id,
        "created_at": now.isoformat(timespec="milliseconds"),
        "source": "LIVE_RGB_NOIR_CAPTURE_5051",
        "filter_base": "NOIR FILTER LAB v8.1",
        "event": "GUARDAR_LOTE_REGISTRO",
        "selected_person_ids": selected_ids,
        "analysis_seq": int(snapshot.get("seq", -1)),
        "rgb_analysis_seq": int(snapshot.get("rgb_seq", -1)),
        "analysis_timestamp_ms": int(snapshot.get("timestamp_ms", 0)),
        "rgb_filename": "01_RGB_validated.png",
        "noir_filename": "02_NOIR_validated.png",
        "raw_rgb_filename": "03_RGB_highres_archive.png" if raw_rgb is not None else None,
        "raw_noir_filename": "04_NOIR_highres_archive.png" if raw_noir is not None else None,
        "tfmini": tfmini,
        "ir_active": telemetry.get("ir_active"),
        "ir_control_enabled": telemetry.get("ir_control_enabled", False),
        "ir_reason": telemetry.get("ir_reason", ""),
        "capture_resolution": {
            "rgb": [int(rgb.shape[1]), int(rgb.shape[0])],
            "noir": [int(noir.shape[1]), int(noir.shape[0])],
        },
        "analysis_faces": analysis_faces,
        "yolo11n": yolo_scaled,
    }

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("images/01_RGB_validated.png", _encode_png_bytes(rgb))
        zf.writestr("images/02_NOIR_validated.png", _encode_png_bytes(noir))
        if raw_rgb is not None:
            zf.writestr("images/03_RGB_highres_archive.png", _encode_png_bytes(raw_rgb))
        if raw_noir is not None:
            zf.writestr("images/04_NOIR_highres_archive.png", _encode_png_bytes(raw_noir))
        zf.writestr(
            "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
        )
    return mem.getvalue()



def capture_current_batch_locally(registered_ids):
    ids = [str(x) for x in (registered_ids or []) if x]

    with local_capture_lock:
        if local_capture_state.get("busy"):
            print("[V16.11] Ya hay una captura local en proceso.")
            return
        local_capture_state.update({
            "busy": True,
            "status": "CAPTURANDO RGB + NOIR | " + ",".join(ids),
            "last_error": None,
        })

    try:
        payload = build_local_dataset_payload(selected_person_ids=ids)
        capture_id, event_dir = process_dataset_zip(payload)
        with local_capture_lock:
            local_capture_state.update({
                "status": "CAPTURA REGISTRADA {}".format(capture_id),
                "last_capture_id": capture_id,
                "last_path": str(event_dir),
                "last_error": None,
            })
        print("[DATASET V16.11 COMPLETO]", capture_id, "|", event_dir)
        return

    except Exception as exc:
        _set_people_back_to_pending(ids)
        with local_capture_lock:
            local_capture_state.update({
                "status": "NO GUARDADO - VUELVE A PENDING | " + str(exc)[:75],
                "last_error": str(exc),
            })
        print("[DATASET V16.11 NO GUARDADO]", repr(exc))
        print("[REGISTRO V16.11] Regresan a PENDING:", ", ".join(ids))

    finally:
        with local_capture_lock:
            local_capture_state["busy"] = False



def request_local_dataset_capture(
    registered_ids
):
    thread = threading.Thread(
        target=capture_current_batch_locally,
        args=(
            list(
                registered_ids
                or []
            ),
        ),
        daemon=True
    )

    thread.start()
    return True


# ============================================================
# ETIQUETAS MANUALES POR ROSTRO / PERSONA
# ============================================================
# Cada rostro detectado en RGB recibe un identificador P01, P02...
# y conserva sus propias etiquetas manuales:
#   - lentes
#   - barba
#   - cabello_frente
#
# Estas etiquetas se asignan POR PERSONA, no globalmente.
person_conditions_lock = threading.Lock()
person_conditions = {}

person_button_rects_lock = threading.Lock()
person_button_rects = []

rgb_track_person_lock = threading.Lock()
rgb_track_to_person_id = {}
next_rgb_person_number = 1

source_track_person_lock = threading.Lock()
source_track_to_person_id = {}

latest_rgb_person_refs_lock = threading.Lock()
latest_rgb_person_refs = []

last_event_person_lock = threading.Lock()
last_event_by_source_person = {}

# ============================================================
# V16 - REGISTRO NATURAL EN LOTES DE 3 + QUALITY GATE
# ============================================================
REGISTRATION_BATCH_SIZE = 3
REGISTRATION_DIR = DATASET_DIR / "personas_registradas"
REGISTRATION_DIR.mkdir(parents=True, exist_ok=True)

registration_lock = threading.Lock()
person_registration = {}
registration_save_rect = None

# ============================================================
# V16.15 - NORMALIZACION GEOMETRICA AUTOMATICA
# ============================================================
# No se usan etiquetas manuales de emociones ni estado NEUTRAL.
# La normalizacion por ojos es puramente geometrica y se calcula
# automaticamente a partir de los 68 landmarks detectados.

def _eye_normalization(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 48:
        return None, None
    left = np.mean(pts[36:42, :2], axis=0)
    right = np.mean(pts[42:48, :2], axis=0)
    center = (left + right) * 0.5
    scale = float(np.linalg.norm(right - left))
    if scale <= 1e-6:
        return None, None
    return center, scale


def _v1615_enrich_normalized_landmarks(records):
    """Agrega x_norm_eyes/y_norm_eyes sin usar ninguna etiqueta manual."""
    groups = {}
    for row in records or []:
        pid = str(row.get("person_id") or "")
        source = str(row.get("camera") or "")
        if not pid or not source:
            continue
        groups.setdefault((pid, source), []).append(row)

    for (_pid, _source), rows in groups.items():
        rows.sort(key=lambda r: int(r.get("point_index", 0)))
        if len(rows) < 68:
            continue
        try:
            points = np.asarray(
                [[float(r.get("x_px")), float(r.get("y_px"))] for r in rows[:68]],
                dtype=np.float32,
            )
        except Exception:
            continue
        center, scale = _eye_normalization(points)
        if center is None or scale is None:
            continue
        for idx, row in enumerate(rows[:68]):
            row["x_norm_eyes"] = round(
                (float(points[idx, 0]) - float(center[0])) / scale, 6
            )
            row["y_norm_eyes"] = round(
                (float(points[idx, 1]) - float(center[1])) / scale, 6
            )
    return records


def _v1614_yolo_map_for_ids(detections, valid_conf, selected_ids, frame_shape):
    """Asigna cajas YOLO a Pxx aunque MediaPipe no vea el rostro en ese frame."""
    persons = person_detections(detections, valid_conf)
    if not persons or not selected_ids:
        return {}
    persons = sorted(persons, key=lambda p: float((p.get("box") or [0, 0, 0, 0])[0]))
    h, w = frame_shape[:2]
    with latest_rgb_person_refs_lock:
        refs = copy.deepcopy(latest_rgb_person_refs)
    ref_by_id = {str(r.get("person_id")): r for r in refs if r.get("person_id")}
    available = set(range(len(persons)))
    result = {}

    for pid in selected_ids:
        ref = ref_by_id.get(str(pid))
        best = None
        if ref and ref.get("center_norm") and len(ref.get("center_norm")) >= 2:
            rx = float(ref["center_norm"][0]) * float(w)
            ry = float(ref["center_norm"][1]) * float(h)
            for idx in available:
                box = persons[idx].get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(v) for v in box]
                cx = (x1 + x2) * 0.5
                cy = (y1 + y2) * 0.5
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                inside = (x1 - 0.08*bw <= rx <= x2 + 0.08*bw and y1 - 0.10*bh <= ry <= y2 + 0.10*bh)
                dist = ((cx-rx)/max(1.0,w))**2 + ((cy-ry)/max(1.0,h))**2
                score = (0 if inside else 1, dist)
                if best is None or score < best[0]:
                    best = (score, idx)
        if best is None and available:
            # Fallback estable izquierda->derecha.
            best = ((2, 0.0), min(available))
        if best is not None:
            idx = best[1]
            available.discard(idx)
            p = persons[idx]
            result[str(pid)] = {
                "box": [int(v) for v in (p.get("box") or [0, 0, 0, 0])],
                "confidence": float(p.get("confidence", 0.0)),
            }
    return result


def _v1614_enrich_person_rows_with_yolo(records, detections, valid_conf, selected_ids, frame_shape):
    mapped = _v1614_yolo_map_for_ids(detections, valid_conf, selected_ids, frame_shape)
    by_id = {str(r.get("person_id")): r for r in records if r.get("person_id")}
    for pid in selected_ids or []:
        row = by_id.get(str(pid))
        info = mapped.get(str(pid))
        if row is None or info is None:
            continue
        box = info.get("box") or [None, None, None, None]
        if row.get("person_x1") is None:
            row["person_x1"], row["person_y1"], row["person_x2"], row["person_y2"] = box
        if row.get("yolo_person_confidence") is None:
            row["yolo_person_confidence"] = round(float(info.get("confidence", 0.0)), 5)
        if not bool(row.get("face_detected")):
            row["source"] = "YOLO_PERSON_NO_FACE"
    return records

pending_blendshape_lock = threading.Lock()
pending_blendshape_rows = {}
PENDING_BLENDSHAPE_MAX = 300  # ~10 s a 30 FPS por persona

pending_micro_lock = threading.Lock()
pending_micro_events = {}
PENDING_MICRO_MAX = 100

# Quality gate inicial. Ajustable despues con datos reales de RGB/NoIR.
QUALITY_MIN_FACE_W = 70
QUALITY_MIN_FACE_H = 70
QUALITY_MIN_AREA_RATIO = 0.018
QUALITY_MIN_SHARPNESS = 28.0
QUALITY_MIN_FRONTAL_SCORE = 0.55
QUALITY_MAX_ROLL_DEG = 22.0
QUALITY_HOLD_MS = 220

# Muestreo automatico de IMAGEN LIMPIA solo para RGB + NoIR original.
# Los 6 filtros virtuales se pueden regenerar desde NoIR original y sus
# blendshapes/eventos si se guardan continuamente.
AUTO_FACE_SAMPLE_FPS = 5.0
AUTO_FACE_SAMPLE_ENABLED = False
AUTO_FACE_SAMPLE_INTERVAL_MS = int(1000.0 / AUTO_FACE_SAMPLE_FPS)
AUTO_FACE_SAMPLES_DIR = DATASET_DIR / "auto_face_samples"
AUTO_FACE_SAMPLES_CSV = DATASET_DIR / "auto_face_samples.csv"
auto_sample_lock = threading.Lock()
auto_sample_last_ms = {}


def maybe_save_auto_face_sample(source, frame, face, timestamp_ms):
    # V16.3: no guardar imagenes automaticas 256x256.
    # La serie temporal de blendshapes/microeventos sigue activa; las imagenes
    # definitivas se guardan en GUARDAR LOTE usando el formato original.
    return None

    person_id = face.get("person_id")
    if not person_id or not person_is_registered(person_id):
        return None
    q = face.get("quality_gate", {})
    if not bool(q.get("training_valid", 0)):
        return None
    key = (str(source), str(person_id))
    with auto_sample_lock:
        last = int(auto_sample_last_ms.get(key, 0))
        if int(timestamp_ms) - last < AUTO_FACE_SAMPLE_INTERVAL_MS:
            return None
        auto_sample_last_ms[key] = int(timestamp_ms)
    crop = crop_face_for_dataset(frame, face.get("bbox"), output_size=None)
    if crop is None:
        return None
    person_dir = AUTO_FACE_SAMPLES_DIR / person_id / str(source)
    person_dir.mkdir(parents=True, exist_ok=True)
    filename = "{}_{}_{}.jpg".format(person_id, source, int(timestamp_ms))
    path = person_dir / filename
    cv2.imwrite(str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    with latest_telemetry_lock:
        telem = dict(latest_telemetry)
    row = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "session_id": SESSION_ID,
        "timestamp_ms": int(timestamp_ms),
        "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
        "person_id": person_id,
        "source": str(source),
        "distance_cm": telem.get("distance_cm"),
        "distance_valid": int(bool(telem.get("valid", False))),
        "ir_active": int(bool(telem.get("ir_active", False))),
        "quality_status": q.get("status", "UNKNOWN"),
        "sharpness": q.get("sharpness"),
        "frontal_score": q.get("frontal_score"),
        "area_ratio": q.get("area_ratio"),
        "path": str(path),
    }
    headers=list(row.keys())
    with auto_sample_lock:
        exists=AUTO_FACE_SAMPLES_CSV.exists() and AUTO_FACE_SAMPLES_CSV.stat().st_size>0
        with AUTO_FACE_SAMPLES_CSV.open("a",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=headers)
            if not exists: w.writeheader()
            w.writerow(row)
    return str(path)


def _registration_default(person_id):
    now_ms = int(time.monotonic_ns() // 1_000_000)
    return {
        "person_id": str(person_id),
        "status": "PENDING",
        "first_seen_ms": now_ms,
        "last_seen_ms": now_ms,
        "best_rgb_crop": None,
        "best_rgb_score": -1.0,
        "best_rgb_quality": "SIN_DATOS",
        "best_noir_crop": None,
        "best_noir_score": -1.0,
        "best_noir_quality": "SIN_DATOS",
        "rgb_reference_path": None,
        "noir_reference_path": None,
    }


def ensure_person_registration(person_id):
    if not person_id:
        return None
    with registration_lock:
        if person_id not in person_registration:
            person_registration[person_id] = _registration_default(person_id)
        person_registration[person_id]["last_seen_ms"] = int(time.monotonic_ns() // 1_000_000)
        return dict(person_registration[person_id])


def get_registration_status(person_id):
    if not person_id:
        return "UNASSIGNED"
    with registration_lock:
        item = person_registration.get(person_id)
        return str(item.get("status", "PENDING")) if item else "PENDING"


def person_is_registered(person_id):
    return get_registration_status(person_id) == "REGISTERED"


def registration_batch_ids():
    with registration_lock:
        pending = [
            (int(v.get("first_seen_ms", 0)), pid)
            for pid, v in person_registration.items()
            if v.get("status") == "PENDING"
        ]
    pending.sort(key=lambda x: x[0])
    return [pid for _, pid in pending[:REGISTRATION_BATCH_SIZE]]


def registration_counts():
    with registration_lock:
        vals = list(person_registration.values())
    return {
        "pending": sum(1 for v in vals if v.get("status") == "PENDING"),
        "registered": sum(1 for v in vals if v.get("status") == "REGISTERED"),
        "ignored": sum(1 for v in vals if v.get("status") == "IGNORED"),
    }


def _frontal_score_from_68(face):
    pts = face.get("landmarks")
    if pts is None or len(pts) < 55:
        return 0.0
    pts = np.asarray(pts, dtype=np.float32)
    left_eye = np.mean(pts[36:42], axis=0)
    right_eye = np.mean(pts[42:48], axis=0)
    nose = pts[30]
    dl = float(np.linalg.norm(nose - left_eye))
    dr = float(np.linalg.norm(nose - right_eye))
    if max(dl, dr) <= 1e-6:
        return 0.0
    return float(min(dl, dr) / max(dl, dr))


def evaluate_face_quality(frame, face):
    if frame is None or face is None:
        return {"status":"RED", "training_valid":0, "score":0.0, "sharpness":None, "frontal_score":0.0}
    x,y,w,h = [int(v) for v in face.get("bbox", [0,0,0,0])]
    fh, fw = frame.shape[:2]
    if w <= 1 or h <= 1:
        return {"status":"RED", "training_valid":0, "score":0.0, "sharpness":None, "frontal_score":0.0}
    q = image_quality(frame, [x,y,w,h])
    sharp = float(q.get("sharpness") or 0.0)
    area_ratio = float(w*h) / float(max(1, fw*fh))
    frontal = _frontal_score_from_68(face)
    roll = abs(float(face.get("metrics", {}).get("head_roll_deg") or 0.0))
    edge_ok = x > 2 and y > 2 and (x+w) < (fw-2) and (y+h) < (fh-2)
    size_ok = w >= QUALITY_MIN_FACE_W and h >= QUALITY_MIN_FACE_H and area_ratio >= QUALITY_MIN_AREA_RATIO
    sharp_ok = sharp >= QUALITY_MIN_SHARPNESS
    frontal_ok = frontal >= QUALITY_MIN_FRONTAL_SCORE and roll <= QUALITY_MAX_ROLL_DEG
    if size_ok and sharp_ok and frontal_ok and edge_ok:
        status = "GREEN"; valid = 1
    elif w >= 45 and h >= 45 and frontal >= 0.42 and sharp >= 14.0:
        status = "YELLOW"; valid = 0
    else:
        status = "RED"; valid = 0
    score = (min(1.0, area_ratio/0.08)*0.25 + min(1.0, sharp/120.0)*0.35 + min(1.0, frontal)*0.30 + (0.10 if edge_ok else 0.0))
    return {
        "status": status, "training_valid": int(valid), "score": round(float(score),5),
        "sharpness": round(sharp,3), "frontal_score": round(frontal,4),
        "area_ratio": round(area_ratio,5), "roll_deg": round(roll,3),
    }


def apply_quality_gate(frame, faces):
    out=[]
    for face in faces:
        f=copy.deepcopy(face)
        q=evaluate_face_quality(frame, f)
        f["quality_gate"] = q
        f["quality_status"] = q["status"]
        f["training_valid"] = q["training_valid"]
        out.append(f)
    return out


def update_best_reference(person_id, frame, face, source="RGB"):
    if not person_id or frame is None:
        return
    ensure_person_registration(person_id)
    q = face.get("quality_gate") or evaluate_face_quality(frame, face)
    crop = crop_face_for_dataset(frame, face.get("bbox"), output_size=None)
    if crop is None:
        return
    score = float(q.get("score", 0.0)) + (0.15 if q.get("status") == "GREEN" else 0.0)
    key_crop = "best_rgb_crop" if source == "RGB" else "best_noir_crop"
    key_score = "best_rgb_score" if source == "RGB" else "best_noir_score"
    key_quality = "best_rgb_quality" if source == "RGB" else "best_noir_quality"
    with registration_lock:
        item = person_registration.setdefault(person_id, _registration_default(person_id))
        item["last_seen_ms"] = int(time.monotonic_ns() // 1_000_000)
        if score > float(item.get(key_score, -1.0)):
            item[key_crop] = crop.copy()
            item[key_score] = score
            item[key_quality] = q.get("status", "SIN_DATOS")


def registration_thumbnail(person_id):
    with registration_lock:
        item = person_registration.get(person_id, {})
        crop = item.get("best_rgb_crop")
        quality = item.get("best_rgb_quality", "SIN_DATOS")
    return (None if crop is None else crop.copy(), quality)


def _refresh_manual_labels(row, person_id):
    state = get_person_conditions(person_id)
    row["lentes"] = int(bool(state.get("lentes")))
    row["barba"] = int(bool(state.get("barba")))
    row["cabello_frente"] = int(bool(state.get("cabello_frente")))
    row["condiciones_manual"] = person_conditions_summary(state)
    row["registration_status"] = "REGISTERED"
    if "training_valid" in row:
        row["training_valid"] = int(
            str(row.get("quality_status", "")) == "GREEN"
        )
    return row


def flush_pending_person(person_id):
    with pending_blendshape_lock:
        rows = list(pending_blendshape_rows.pop(person_id, deque()))
    for row in rows:
        row = _refresh_manual_labels(row, person_id)
        try:
            blendshape_queue.put_nowait(row)
        except queue.Full:
            break
    with pending_micro_lock:
        events = list(pending_micro_events.pop(person_id, deque()))
    for event in events:
        event = _refresh_manual_labels(event, person_id)
        _write_micro_event_now(event)


def register_current_batch():
    ids = registration_batch_ids()
    if not ids:
        return []
    REGISTRATION_DIR.mkdir(parents=True, exist_ok=True)
    registered=[]
    for person_id in ids:
        state=get_person_conditions(person_id)
        with registration_lock:
            item=person_registration.get(person_id)
            if not item or item.get("status") != "PENDING":
                continue
            item["status"]="REGISTERED"
            rgb_crop=None if item.get("best_rgb_crop") is None else item["best_rgb_crop"].copy()
            noir_crop=None if item.get("best_noir_crop") is None else item["best_noir_crop"].copy()
        person_dir=REGISTRATION_DIR / person_id
        person_dir.mkdir(parents=True, exist_ok=True)
        if rgb_crop is not None:
            p=person_dir / "{}_RGB_reference.png".format(person_id)
            save_image(p, rgb_crop)
            with registration_lock:
                person_registration[person_id]["rgb_reference_path"] = str(p)
        if noir_crop is not None:
            p=person_dir / "{}_NOIR_reference.png".format(person_id)
            save_image(p, noir_crop)
            with registration_lock:
                person_registration[person_id]["noir_reference_path"] = str(p)
        meta={
            "person_id":person_id, "registered_at":datetime.now().isoformat(timespec="milliseconds"),
            "lentes":int(bool(state.get("lentes"))), "barba":int(bool(state.get("barba"))),
            "cabello_frente":int(bool(state.get("cabello_frente"))),
            "condiciones_manual":person_conditions_summary(state),
        }
        (person_dir / "registro.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        flush_pending_person(person_id)
        registered.append(person_id)
    if registered:
        print("[REGISTRO V16.11] lote listo y captura solicitada:", ", ".join(registered))
    return registered


def default_person_conditions():
    return {
        "lentes": False,
        "barba": False,
        "cabello_frente": False,
    }


def ensure_person_conditions(
    person_id
):
    if not person_id:
        return default_person_conditions()

    with person_conditions_lock:
        if person_id not in person_conditions:
            person_conditions[
                person_id
            ] = default_person_conditions()

        return dict(
            person_conditions[
                person_id
            ]
        )


def get_person_conditions(
    person_id
):
    if not person_id:
        return default_person_conditions()

    with person_conditions_lock:
        state = person_conditions.get(
            person_id
        )

        if state is None:
            state = default_person_conditions()
            person_conditions[
                person_id
            ] = dict(
                state
            )

        return dict(
            state
        )


def person_conditions_summary(
    state
):
    selected = []

    if state.get(
        "lentes"
    ):
        selected.append(
            "LENTES"
        )

    if state.get(
        "barba"
    ):
        selected.append(
            "BARBA"
        )

    if state.get(
        "cabello_frente"
    ):
        selected.append(
            "CABELLO_FRENTE"
        )

    return (
        "+".join(
            selected
        )
        if selected
        else "NINGUNA"
    )


def person_conditions_summary_by_id(
    person_id
):
    return person_conditions_summary(
        get_person_conditions(
            person_id
        )
    )


def capture_conditions_summary(
    person_ids
):
    parts = []

    seen = set()

    for person_id in person_ids:
        if (
            not person_id
            or person_id in seen
        ):
            continue

        seen.add(
            person_id
        )

        parts.append(
            "{}:{}".format(
                person_id,
                person_conditions_summary_by_id(
                    person_id
                )
            )
        )

    return (
        " | ".join(
            parts
        )
        if parts
        else "SIN_ROSTROS"
    )


def register_rgb_track_person(
    track_id
):
    global next_rgb_person_number

    if track_id is None:
        return None

    track_id = int(
        track_id
    )

    with rgb_track_person_lock:
        person_id = rgb_track_to_person_id.get(
            track_id
        )

        if person_id is None:
            person_id = "P{:02d}".format(
                next_rgb_person_number
            )

            next_rgb_person_number += 1

            rgb_track_to_person_id[
                track_id
            ] = person_id

    ensure_person_conditions(
        person_id
    )
    ensure_person_registration(
        person_id
    )

    return person_id


def set_source_track_person(
    source,
    track_id,
    person_id
):
    if (
        source is None
        or track_id is None
        or not person_id
    ):
        return

    key = (
        str(
            source
        ),
        int(
            track_id
        ),
    )

    with source_track_person_lock:
        source_track_to_person_id[
            key
        ] = str(
            person_id
        )


def get_source_track_person(
    source,
    track_id
):
    if (
        source is None
        or track_id is None
    ):
        return None

    key = (
        str(
            source
        ),
        int(
            track_id
        ),
    )

    with source_track_person_lock:
        return source_track_to_person_id.get(
            key
        )


def normalized_face_center(
    face,
    frame_shape
):
    bbox = face.get(
        "bbox",
        [0, 0, 0, 0]
    )

    x, y, w, h = [
        float(v)
        for v in bbox
    ]

    frame_h = max(
        1.0,
        float(
            frame_shape[0]
        )
    )

    frame_w = max(
        1.0,
        float(
            frame_shape[1]
        )
    )

    return np.asarray(
        [
            (
                x
                + 0.5 * w
            )
            / frame_w,
            (
                y
                + 0.5 * h
            )
            / frame_h,
        ],
        dtype=np.float32
    )


def face_reference_list(
    faces,
    frame_shape
):
    refs = []

    for face in faces:
        person_id = face.get(
            "person_id"
        )

        if not person_id:
            continue

        refs.append(
            {
                "person_id": person_id,
                "center_norm": normalized_face_center(
                    face,
                    frame_shape
                ).tolist(),
                "bbox": list(
                    face.get(
                        "bbox",
                        [0, 0, 0, 0]
                    )
                ),
            }
        )

    return refs


def assign_faces_from_references(
    faces,
    frame_shape,
    references,
    fallback_new=False
):
    """
    Asocia rostros por posicion normalizada, de forma uno-a-uno.

    Sirve para:
      - transferir P01/P02... de RGB preview a RGB captura
      - transferir los mismos Pxx de RGB a NoIR
      - transferir Pxx de NoIR original a cada filtro digital

    Si no hay referencia suficiente, usa orden izquierda->derecha.
    """
    if not faces:
        return []

    out = [
        copy.deepcopy(
            face
        )
        for face in faces
    ]

    refs = [
        copy.deepcopy(
            ref
        )
        for ref in (
            references
            or []
        )
        if ref.get(
            "person_id"
        )
    ]

    available_refs = set(
        range(
            len(
                refs
            )
        )
    )

    # Rostros grandes primero: reduce cruces cuando hay solapamiento.
    face_indices = sorted(
        range(
            len(
                out
            )
        ),
        key=lambda idx: (
            float(
                out[idx].get(
                    "bbox",
                    [0, 0, 0, 0]
                )[2]
            )
            * float(
                out[idx].get(
                    "bbox",
                    [0, 0, 0, 0]
                )[3]
            )
        ),
        reverse=True
    )

    unmatched = []

    for face_idx in face_indices:
        center = normalized_face_center(
            out[
                face_idx
            ],
            frame_shape
        )

        best_ref_idx = None
        best_dist = 999.0

        for ref_idx in list(
            available_refs
        ):
            ref_center = np.asarray(
                refs[
                    ref_idx
                ].get(
                    "center_norm",
                    [0.5, 0.5]
                ),
                dtype=np.float32
            )

            dist = float(
                np.linalg.norm(
                    center
                    - ref_center
                )
            )

            if dist < best_dist:
                best_dist = dist
                best_ref_idx = ref_idx

        # Tolerancia relativamente amplia para diferencia de FOV RGB/NoIR.
        if (
            best_ref_idx is not None
            and best_dist <= 0.34
        ):
            person_id = refs[
                best_ref_idx
            ][
                "person_id"
            ]

            out[
                face_idx
            ][
                "person_id"
            ] = person_id

            available_refs.remove(
                best_ref_idx
            )

            ensure_person_conditions(
                person_id
            )
            ensure_person_registration(
                person_id
            )

        else:
            unmatched.append(
                face_idx
            )

    if unmatched:
        # Si no fue posible empatar por referencia, conserva todos los
        # rostros y crea/recicla IDs por orden horizontal.
        already_used = {
            face.get(
                "person_id"
            )
            for face in out
            if face.get(
                "person_id"
            )
        }

        sorted_unmatched = sorted(
            unmatched,
            key=lambda idx: normalized_face_center(
                out[
                    idx
                ],
                frame_shape
            )[0]
        )

        for idx in sorted_unmatched:
            person_id = None

            if fallback_new:
                # Buscar el primer Pxx libre.
                for number in range(
                    1,
                    100
                ):
                    candidate = "P{:02d}".format(
                        number
                    )

                    if candidate not in already_used:
                        person_id = candidate
                        break

            if person_id is None:
                # V16.19: cuando fallback_new=False NO inventar PX01/PX02.
                # El rostro queda sin ID para que el proceso de captura lo
                # asigne despues a un Pxx seleccionado de forma uno-a-uno.
                out[idx].pop("person_id", None)
                continue

            out[
                idx
            ][
                "person_id"
            ] = person_id

            already_used.add(
                person_id
            )

            ensure_person_conditions(
                person_id
            )
            ensure_person_registration(
                person_id
            )

    return out


# V16.3 - tracker espacial ligero para registrar Pxx TAN PRONTO MediaPipe ve la cara.
# No espera a que el analizador de microexpresiones cree/estabilice su track.
rgb_spatial_person_lock = threading.Lock()
rgb_spatial_person_tracks = {}


def _bbox_match_score(a, b):
    ca = _bbox_center(a)
    cb = _bbox_center(b)
    norm = max(_bbox_diag(a), _bbox_diag(b), 1.0)
    return float(np.linalg.norm(ca - cb) / norm)


def assign_rgb_person_ids_immediate(faces, timestamp_ms):
    global next_rgb_person_number
    out = [copy.deepcopy(f) for f in (faces or [])]
    used = set()
    now_ms = int(timestamp_ms)

    with rgb_spatial_person_lock:
        # Eliminar tracks muy viejos, pero conservar breves perdidas por movimiento.
        stale = [pid for pid, tr in rgb_spatial_person_tracks.items()
                 if now_ms - int(tr.get("last_seen_ms", 0)) > 1800]
        for pid in stale:
            rgb_spatial_person_tracks.pop(pid, None)

        order = sorted(range(len(out)),
                       key=lambda i: float(out[i].get("bbox", [0,0,0,0])[2]) *
                                     float(out[i].get("bbox", [0,0,0,0])[3]),
                       reverse=True)

        for idx in order:
            bbox = out[idx].get("bbox", [0,0,1,1])
            best_pid = None
            best_score = 999.0
            for pid, tr in rgb_spatial_person_tracks.items():
                if pid in used:
                    continue
                score = _bbox_match_score(bbox, tr.get("bbox", bbox))
                if score < 0.75 and score < best_score:
                    best_score = score
                    best_pid = pid

            if best_pid is None:
                with rgb_track_person_lock:
                    best_pid = "P{:02d}".format(next_rgb_person_number)
                    next_rgb_person_number += 1
                ensure_person_conditions(best_pid)
                ensure_person_registration(best_pid)

            out[idx]["person_id"] = best_pid
            used.add(best_pid)
            rgb_spatial_person_tracks[best_pid] = {
                "bbox": list(bbox),
                "last_seen_ms": now_ms,
            }
            ensure_person_registration(best_pid)

    return out


def link_rgb_micro_tracks(faces, micro_faces, frame_shape):
    for idx, face in enumerate(faces):
        person_id = face.get("person_id")
        if idx < len(micro_faces):
            micro_faces[idx]["person_id"] = person_id
            track_id = micro_faces[idx].get("track_id")
            if track_id is not None and person_id:
                with rgb_track_person_lock:
                    rgb_track_to_person_id[int(track_id)] = person_id
                set_source_track_person("RGB", track_id, person_id)

    refs = face_reference_list(faces, frame_shape)
    with latest_rgb_person_refs_lock:
        latest_rgb_person_refs.clear()
        latest_rgb_person_refs.extend(copy.deepcopy(refs))
    return faces, micro_faces


def annotate_rgb_faces_with_person_ids(
    faces,
    micro_faces,
    frame_shape
):
    for idx, face in enumerate(
        faces
    ):
        track_id = None

        if idx < len(
            micro_faces
        ):
            track_id = micro_faces[
                idx
            ].get(
                "track_id"
            )

        person_id = register_rgb_track_person(
            track_id
        )

        if person_id is None:
            continue

        face[
            "person_id"
        ] = person_id

        if idx < len(
            micro_faces
        ):
            micro_faces[
                idx
            ][
                "person_id"
            ] = person_id

        set_source_track_person(
            "RGB",
            track_id,
            person_id
        )

    refs = face_reference_list(
        faces,
        frame_shape
    )

    with latest_rgb_person_refs_lock:
        latest_rgb_person_refs.clear()
        latest_rgb_person_refs.extend(
            copy.deepcopy(
                refs
            )
        )

    return faces, micro_faces


def annotate_nonrgb_faces_with_person_ids(
    source,
    faces,
    micro_faces,
    frame_shape
):
    with latest_rgb_person_refs_lock:
        refs = copy.deepcopy(
            latest_rgb_person_refs
        )

    assigned = assign_faces_from_references(
        faces,
        frame_shape,
        refs,
        fallback_new=True
    )

    for idx, face in enumerate(
        assigned
    ):
        person_id = face.get(
            "person_id"
        )

        if idx < len(
            micro_faces
        ):
            track_id = micro_faces[
                idx
            ].get(
                "track_id"
            )

            micro_faces[
                idx
            ][
                "person_id"
            ] = person_id

            set_source_track_person(
                source,
                track_id,
                person_id
            )

    return assigned, micro_faces


def get_last_event_for_person(
    source,
    person_id
):
    if (
        not source
        or not person_id
    ):
        return None

    key = (
        str(
            source
        ),
        str(
            person_id
        ),
    )

    with last_event_person_lock:
        value = last_event_by_source_person.get(
            key
        )

        return (
            copy.deepcopy(
                value
            )
            if value is not None
            else None
        )


def draw_person_condition_cards(
    canvas,
    rgb_frame,
    rgb_faces
):
    """V16: muestra SOLO los primeros 3 PENDING de la cola."""
    global registration_save_rect

    card_y1 = DISPLAY_EACH_H + 126
    card_y2 = DISPLAY_H - 58

    counts = registration_counts()
    batch_ids = registration_batch_ids()

    cv2.putText(
        canvas,
        "REGISTRO V16 3-EN-3 | PENDIENTES {} | REGISTRADAS {}".format(
            counts["pending"], counts["registered"]
        ),
        (15, DISPLAY_EACH_H + 118),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA
    )

    rects = []
    card_gap = 10
    card_w = int((DISPLAY_W - 20 - 2*card_gap) / 3)

    live_by_id = {
        f.get("person_id"): f
        for f in rgb_faces
        if f.get("person_id")
    }

    for idx in range(3):
        x1 = 5 + idx*(card_w+card_gap)
        x2 = x1 + card_w
        cv2.rectangle(canvas,(x1,card_y1),(x2,card_y2),(43,43,43),-1)
        cv2.rectangle(canvas,(x1,card_y1),(x2,card_y2),(120,120,120),1)

        if idx >= len(batch_ids):
            cv2.putText(canvas,"EN ESPERA...",(x1+18,card_y1+70),cv2.FONT_HERSHEY_SIMPLEX,0.48,GRAY,1,cv2.LINE_AA)
            continue

        person_id = batch_ids[idx]
        state = get_person_conditions(person_id)
        thumb, quality = registration_thumbnail(person_id)

        cv2.putText(canvas,person_id,(x1+8,card_y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.56,CYAN,2,cv2.LINE_AA)
        qcolor = GREEN if quality == "GREEN" else (YELLOW if quality == "YELLOW" else RED)
        cv2.putText(canvas,"BEST {}".format(quality),(x1+68,card_y1+20),cv2.FONT_HERSHEY_SIMPLEX,0.34,qcolor,1,cv2.LINE_AA)

        if thumb is None:
            live_face = live_by_id.get(person_id)
            if live_face is not None:
                thumb = crop_face_for_dataset(rgb_frame, live_face.get("bbox"), output_size=104)
        if thumb is not None:
            thumb=cv2.resize(thumb,(104,104),interpolation=cv2.INTER_AREA)
            ty=card_y1+30; tx=x1+8
            canvas[ty:ty+104,tx:tx+104]=thumb

        button_x1=x1+122; button_x2=x2-8
        for bidx,(key,label) in enumerate([("lentes","LENTES"),("barba","BARBA"),("cabello_frente","CABELLO")]):
            by1=card_y1+35+bidx*36; by2=by1+27
            active=bool(state.get(key))
            fill=(0,150,0) if active else (70,70,70)
            border=GREEN if active else (150,150,150)
            cv2.rectangle(canvas,(button_x1,by1),(button_x2,by2),fill,-1)
            cv2.rectangle(canvas,(button_x1,by1),(button_x2,by2),border,1)
            cv2.putText(canvas,label,(button_x1+5,by1+18),cv2.FONT_HERSHEY_SIMPLEX,0.35,WHITE,1,cv2.LINE_AA)
            rects.append({"action":"toggle","person_id":person_id,"key":key,"rect":(button_x1,by1,button_x2,by2)})

        cv2.putText(canvas,person_conditions_summary(state),(x1+8,card_y2-9),cv2.FONT_HERSHEY_SIMPLEX,0.31,GREEN if person_conditions_summary(state)!="NINGUNA" else GRAY,1,cv2.LINE_AA)

    save_x1=DISPLAY_W-260; save_y1=DISPLAY_H-48; save_x2=DISPLAY_W-12; save_y2=DISPLAY_H-12
    enabled=bool(batch_ids)
    cv2.rectangle(canvas,(save_x1,save_y1),(save_x2,save_y2),(0,145,0) if enabled else (60,60,60),-1)
    cv2.rectangle(canvas,(save_x1,save_y1),(save_x2,save_y2),GREEN if enabled else GRAY,2)
    cv2.putText(canvas,"GUARDAR LOTE ({})".format(len(batch_ids)),(save_x1+25,save_y1+24),cv2.FONT_HERSHEY_SIMPLEX,0.48,WHITE,1,cv2.LINE_AA)
    registration_save_rect=(save_x1,save_y1,save_x2,save_y2)
    rects.append({"action":"save_batch","rect":registration_save_rect})

    cv2.putText(canvas,"Las fotos guardadas salen del frame original limpio, no de la interfaz.",(15,DISPLAY_H-22),cv2.FONT_HERSHEY_SIMPLEX,0.34,GRAY,1,cv2.LINE_AA)

    with person_button_rects_lock:
        person_button_rects.clear()
        person_button_rects.extend(rects)


def person_conditions_mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    with person_button_rects_lock:
        rects = copy.deepcopy(person_button_rects)
    for item in rects:
        x1,y1,x2,y2=item["rect"]
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        if item.get("action") == "save_batch":
            registered = register_current_batch()
            if registered:
                request_local_dataset_capture(
                    registered
                )
            return
        person_id=item.get("person_id")
        key=item.get("key")
        if not person_id or not key:
            return
        with person_conditions_lock:
            if person_id not in person_conditions:
                person_conditions[person_id]=default_person_conditions()
            person_conditions[person_id][key]=not bool(person_conditions[person_id].get(key,False))
            state=dict(person_conditions[person_id])
        print("[CONDICION {}] {} = {}".format(person_id,key.upper(),"SI" if state.get(key) else "NO"))
        return


latest_live_faces_lock = threading.Lock()
latest_live_faces = {
    0: [],
    1: [],
}

latest_face_input_lock = threading.Lock()
latest_face_input = {
    0: None,
    1: None,
}

# V16.3: conserva el frame recibido a resolucion original del stream
# (960x540 por camara con Jetson V11.3). El analisis LIVE usa una
# copia reducida 480x270 para no aumentar la latencia.
latest_dataset_input_lock = threading.Lock()
latest_dataset_input = {
    0: None,
    1: None,
}

# Cada camara recibe su propia secuencia/timestamp.
# Esto evita procesar varias veces el mismo frame.
latest_face_seq = {
    0: 0,
    1: 0,
}

latest_face_timestamp_ms = {
    0: 0,
    1: 0,
}

# V16.11 - cache del MISMO frame que MediaPipe ya valido en vivo.
# Evita volver a detectar el rostro al presionar GUARDAR LOTE.
live_capture_cache_lock = threading.Lock()
live_rgb_capture_cache = {}   # seq -> {frame, faces, timestamp_ms}
LIVE_CAPTURE_CACHE_MAX = 180

latest_noir_cycle_lock = threading.Lock()
latest_noir_cycle_snapshot = {
    "seq": -1,
    "timestamp_ms": 0,
    "frame": None,
    "filter_faces": {},
    "filter_micro": {},
}

# Estado RGB independiente.
latest_micro_lock = threading.Lock()
latest_micro = {
    0: {
        "camera": "RGB",
        "faces": [],
        "last_event": None,
        "processed_fps": 0.0,
    },
}

# Cada filtro NoIR tiene SU PROPIO estado temporal.
latest_noir_filter_lock = threading.Lock()
latest_noir_filter_faces = {
    i: []
    for i in range(
        len(FILTERS_NOIR)
    )
}

latest_noir_filter_micro = {
    i: {
        "source": NOIR_FILTER_IDS[i],
        "filter_name": FILTERS_NOIR[i],
        "faces": [],
        "last_event": None,
        "processed_fps": 0.0,
    }
    for i in range(
        len(FILTERS_NOIR)
    )
}

MICRO_EVENTS_FILE = (
    DATASET_DIR
    / "microexpresiones_eventos.jsonl"
)

MICRO_EVENTS_CSV = (
    DATASET_DIR
    / "microexpresiones_eventos.csv"
)

micro_log_lock = threading.Lock()
micro_event_buffer_lock = threading.Lock()
micro_event_buffer = deque(
    maxlen=10000
)

# Serie temporal cruda de blendshapes. Se escribe en un hilo separado
# para no frenar el preview ni el analisis facial.
blendshape_queue = queue.Queue(
    maxsize=10000
)

stop_event = threading.Event()

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (0, 165, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GRAY = (145, 145, 145)
CYAN = (255, 255, 0)

# ============================================================
# LANDMARK ENGINE V8-STYLE
# ============================================================

def ensure_face_landmarker_model():
    if FACE_LANDMARKER_MODEL.exists():
        return True

    print("")
    print("No existe face_landmarker.task.")
    print("Descargando modelo oficial MediaPipe Face Landmarker...")
    print("")

    try:
        urllib.request.urlretrieve(
            FACE_LANDMARKER_URL,
            str(FACE_LANDMARKER_MODEL)
        )
        return True
    except Exception as e:
        print(
            "ERROR descargando Face Landmarker:",
            e
        )
        return False



def dist(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float32)
            -
            np.asarray(b, dtype=np.float32)
        )
    )


def eye_aspect_ratio(eye):
    p = np.asarray(
        eye,
        dtype=np.float32
    )

    if len(p) != 6:
        return float("nan")

    vertical = (
        dist(p[1], p[5])
        +
        dist(p[2], p[4])
    )

    horizontal = (
        2.0 * dist(p[0], p[3])
        + 1e-6
    )

    return float(
        vertical
        / horizontal
    )


def slope_deg(a, b):
    dx = float(b[0] - a[0])
    dy = float(b[1] - a[1])

    return float(
        math.degrees(
            math.atan2(dy, dx)
        )
    )


def extract_face_metrics(
    pts,
    bbox
):
    pts = np.asarray(
        pts,
        dtype=np.float32
    )

    x, y, w, h = bbox

    fw = max(
        1.0,
        float(w)
    )

    fh = max(
        1.0,
        float(h)
    )

    left_brow = pts[17:22]
    right_brow = pts[22:27]

    left_eye = pts[36:42]
    right_eye = pts[42:48]

    left_eye_center = np.mean(
        left_eye,
        axis=0
    )

    right_eye_center = np.mean(
        right_eye,
        axis=0
    )

    left_brow_center = np.mean(
        left_brow,
        axis=0
    )

    right_brow_center = np.mean(
        right_brow,
        axis=0
    )

    def r(v):
        return round(
            float(v),
            6
        )

    return {
        # CEJA IZQUIERDA
        "left_brow_height": r(
            abs(
                left_eye_center[1]
                - left_brow_center[1]
            )
            / fh
        ),
        "left_brow_width": r(
            dist(
                pts[17],
                pts[21]
            )
            / fw
        ),
        "left_brow_slope_deg": r(
            slope_deg(
                pts[17],
                pts[21]
            )
        ),
        "left_brow_eye_distance": r(
            dist(
                left_brow_center,
                left_eye_center
            )
            / fh
        ),

        # CEJA DERECHA
        "right_brow_height": r(
            abs(
                right_eye_center[1]
                - right_brow_center[1]
            )
            / fh
        ),
        "right_brow_width": r(
            dist(
                pts[22],
                pts[26]
            )
            / fw
        ),
        "right_brow_slope_deg": r(
            slope_deg(
                pts[22],
                pts[26]
            )
        ),
        "right_brow_eye_distance": r(
            dist(
                right_brow_center,
                right_eye_center
            )
            / fh
        ),

        # OJOS
        "left_eye_width": r(
            dist(
                pts[36],
                pts[39]
            )
            / fw
        ),
        "left_eye_height": r(
            (
                dist(
                    pts[37],
                    pts[41]
                )
                +
                dist(
                    pts[38],
                    pts[40]
                )
            )
            / 2.0
            / fh
        ),
        "left_eye_ear": r(
            eye_aspect_ratio(
                left_eye
            )
        ),

        "right_eye_width": r(
            dist(
                pts[42],
                pts[45]
            )
            / fw
        ),
        "right_eye_height": r(
            (
                dist(
                    pts[43],
                    pts[47]
                )
                +
                dist(
                    pts[44],
                    pts[46]
                )
            )
            / 2.0
            / fh
        ),
        "right_eye_ear": r(
            eye_aspect_ratio(
                right_eye
            )
        ),

        "inter_eye_distance": r(
            dist(
                left_eye_center,
                right_eye_center
            )
            / fw
        ),

        # NARIZ
        "nose_width": r(
            dist(
                pts[31],
                pts[35]
            )
            / fw
        ),
        "nose_height": r(
            dist(
                pts[27],
                pts[33]
            )
            / fh
        ),

        # BOCA
        "mouth_width": r(
            dist(
                pts[48],
                pts[54]
            )
            / fw
        ),
        "mouth_height": r(
            dist(
                pts[51],
                pts[57]
            )
            / fh
        ),
        "mouth_open": r(
            dist(
                pts[62],
                pts[66]
            )
            / fh
        ),
        "smile_width": r(
            dist(
                pts[48],
                pts[54]
            )
            / fw
        ),

        # CARA
        "face_width_px": r(fw),
        "face_height_px": r(fh),
        "face_ratio": r(
            fw / fh
        ),
        "jaw_width": r(
            dist(
                pts[4],
                pts[12]
            )
            / fw
        ),
        "head_roll_deg": r(
            slope_deg(
                left_eye_center,
                right_eye_center
            )
        ),
    }


class MediaPipeMicroEngine(object):
    """
    Motor facial independiente por camara.

    MediaPipe detecta una malla facial densa y esta clase extrae
    68 puntos equivalentes para conservar el dataset existente.

    No se comparten resultados entre RGB y NoIR.
    """

    def __init__(
        self,
        name,
        running_mode="VIDEO",
        num_faces=4,
    ):
        self.name = name
        self.running_mode_name = running_mode
        self.num_faces = int(num_faces)

        self.landmarker = None
        self.lock = threading.Lock()
        self.ready = False
        self.last_timestamp_ms = 0

    def initialize(self):
        if not ensure_face_landmarker_model():
            return False

        try:
            if self.running_mode_name == "VIDEO":
                running_mode = (
                    mp_vision.RunningMode.VIDEO
                )
            else:
                running_mode = (
                    mp_vision.RunningMode.IMAGE
                )

            options = (
                mp_vision.FaceLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(
                            FACE_LANDMARKER_MODEL
                        )
                    ),
                    running_mode=running_mode,
                    num_faces=self.num_faces,
                    min_face_detection_confidence=0.22,
                    min_face_presence_confidence=0.22,
                    min_tracking_confidence=0.30,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False,
                )
            )

            self.landmarker = (
                mp_vision.FaceLandmarker
                .create_from_options(
                    options
                )
            )

        except Exception as e:
            print(
                "ERROR [{}] MediaPipe: {}"
                .format(
                    self.name,
                    e
                )
            )
            return False

        self.ready = True

        print(
            "MediaPipe 68 [{}]: LISTO | modo={} | caras={}"
            .format(
                self.name,
                self.running_mode_name,
                self.num_faces,
            )
        )

        return True

    @staticmethod
    def _bbox_from_mesh(
        mesh,
        frame_w,
        frame_h
    ):
        xs = np.asarray(
            [float(p.x) for p in mesh],
            dtype=np.float32
        )

        ys = np.asarray(
            [float(p.y) for p in mesh],
            dtype=np.float32
        )

        x1 = int(
            np.clip(
                np.min(xs) * frame_w,
                0,
                frame_w - 1
            )
        )

        y1 = int(
            np.clip(
                np.min(ys) * frame_h,
                0,
                frame_h - 1
            )
        )

        x2 = int(
            np.clip(
                np.max(xs) * frame_w,
                1,
                frame_w
            )
        )

        y2 = int(
            np.clip(
                np.max(ys) * frame_h,
                1,
                frame_h
            )
        )

        # Pequeño margen, calculado SOLO desde esta camara.
        bw = max(
            1,
            x2 - x1
        )

        bh = max(
            1,
            y2 - y1
        )

        mx = int(
            0.06 * bw
        )

        my_top = int(
            0.10 * bh
        )

        my_bottom = int(
            0.05 * bh
        )

        x1 = max(
            0,
            x1 - mx
        )

        x2 = min(
            frame_w,
            x2 + mx
        )

        y1 = max(
            0,
            y1 - my_top
        )

        y2 = min(
            frame_h,
            y2 + my_bottom
        )

        return [
            int(x1),
            int(y1),
            int(max(1, x2 - x1)),
            int(max(1, y2 - y1)),
        ]

    @staticmethod
    def _mesh_to_68(
        mesh,
        frame_w,
        frame_h
    ):
        pts = np.zeros(
            (68, 2),
            dtype=np.float32
        )

        for out_idx, mp_idx in enumerate(
            MP_TO_68
        ):
            p = mesh[
                mp_idx
            ]

            pts[
                out_idx,
                0
            ] = float(
                np.clip(
                    p.x * frame_w,
                    0,
                    frame_w - 1
                )
            )

            pts[
                out_idx,
                1
            ] = float(
                np.clip(
                    p.y * frame_h,
                    0,
                    frame_h - 1
                )
            )

        return pts

    def detect_all(
        self,
        frame,
        timestamp_ms=None
    ):
        if (
            not self.ready
            or self.landmarker is None
        ):
            return []

        if (
            frame is None
            or not isinstance(
                frame,
                np.ndarray
            )
            or frame.size == 0
        ):
            return []

        try:
            bgr = np.ascontiguousarray(
                frame
            )

            rgb = cv2.cvtColor(
                bgr,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            with self.lock:
                if self.running_mode_name == "VIDEO":
                    if timestamp_ms is None:
                        timestamp_ms = int(
                            time.monotonic_ns()
                            // 1_000_000
                        )
                    else:
                        timestamp_ms = int(
                            timestamp_ms
                        )

                    if (
                        timestamp_ms
                        <= self.last_timestamp_ms
                    ):
                        timestamp_ms = (
                            self.last_timestamp_ms
                            + 1
                        )

                    self.last_timestamp_ms = (
                        timestamp_ms
                    )

                    result = (
                        self.landmarker
                        .detect_for_video(
                            mp_image,
                            timestamp_ms
                        )
                    )

                else:
                    result = (
                        self.landmarker.detect(
                            mp_image
                        )
                    )

            meshes = getattr(
                result,
                "face_landmarks",
                []
            )

            blendshape_sets = getattr(
                result,
                "face_blendshapes",
                []
            )

            if not meshes:
                return []

            frame_h, frame_w = (
                bgr.shape[:2]
            )

            faces = []

            for face_index, mesh in enumerate(meshes):
                if (
                    mesh is None
                    or len(mesh) <= max(
                        MP_TO_68
                    )
                ):
                    continue

                pts = self._mesh_to_68(
                    mesh,
                    frame_w,
                    frame_h
                )

                bbox = self._bbox_from_mesh(
                    mesh,
                    frame_w,
                    frame_h
                )

                blendshapes = {}

                if (
                    face_index
                    < len(blendshape_sets)
                ):
                    categories = (
                        blendshape_sets[
                            face_index
                        ]
                    )

                    for cat in categories:
                        name = str(
                            getattr(
                                cat,
                                "category_name",
                                ""
                            )
                            or getattr(
                                cat,
                                "display_name",
                                ""
                            )
                        )

                        if not name:
                            continue

                        blendshapes[
                            name
                        ] = float(
                            getattr(
                                cat,
                                "score",
                                0.0
                            )
                        )

                faces.append(
                    {
                        "bbox": bbox,
                        "landmarks_detected": True,
                        "landmarks": pts,
                        "metrics": extract_face_metrics(
                            pts,
                            bbox
                        ),
                        "blendshapes": blendshapes,
                    }
                )

            faces.sort(
                key=lambda f: (
                    f["bbox"][0]
                )
            )

            return faces

        except Exception as e:
            print(
                "[{}] MediaPipe detect: {}"
                .format(
                    self.name,
                    e
                )
            )
            return []


# RGB LIVE conserva su tracking de video.
LIVE_RGB_ENGINE = MediaPipeMicroEngine(
    "LIVE_RGB",
    running_mode="VIDEO",
    num_faces=6,
)

# Los 6 filtros NoIR se infieren como IMAGENES independientes.
# Se comparte solamente el modelo para ahorrar memoria.
# NO se comparte baseline, historial, onset/apex/offset ni eventos.
FILTER_FACE_ENGINE = MediaPipeMicroEngine(
    "NOIR_FILTER_IMAGE_ENGINE",
    running_mode="IMAGE",
    num_faces=6,
)

# Motores de captura.
DATA_RGB_ENGINE = MediaPipeMicroEngine(
    "DATA_RGB",
    running_mode="IMAGE",
    num_faces=6,
)

DATA_NOIR_ENGINE = MediaPipeMicroEngine(
    "DATA_NOIR",
    running_mode="IMAGE",
    num_faces=6,
)


# ============================================================
# MICRO-EXPRESSION TEMPORAL ANALYZER
# ============================================================
#
# IMPORTANTE:
# - No etiqueta emociones.
# - Detecta CAMBIOS FACIALES BREVES como candidatos.
# - RGB y NoIR tienen analizadores/estados separados.
# - Usa blendshapes de MediaPipe, no copia datos entre camaras.
#
# Ventana candidata:
#   40 ms - 500 ms
#
# Esto es deliberadamente amplio para investigacion. La literatura
# no usa un unico limite universal para todas las bases de datos.
# ============================================================

MICRO_MIN_DURATION_MS = 40
MICRO_MAX_DURATION_MS = 500

# Activacion minima sobre la linea base.
MICRO_MIN_ACTIVITY = 0.085

# Una expresion debe superar el ruido de fondo del propio rostro.
MICRO_NOISE_GAIN = 4.0

# Tiempo que se muestra el ultimo evento en pantalla.
MICRO_EVENT_HOLD_MS = 1800


MICRO_CHANNELS = [
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",

    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",

    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",

    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",

    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",

    "noseSneerLeft",
    "noseSneerRight",
]


MICRO_REGIONS = {
    "CEJAS": [
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
    ],

    "OJOS": [
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
    ],

    "MEJILLAS": [
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
    ],

    "NARIZ": [
        "noseSneerLeft",
        "noseSneerRight",
    ],

    "MANDIBULA": [
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
    ],

    "BOCA": [
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
    ],
}



BLENDSHAPE_SERIES_HEADERS = [
    "schema_version",
    "session_id",
    "frame_seq",
    "timestamp_ms",
    "timestamp_iso",
    "source",
    "filter_name",
    "person_id",
    "track_id",
    "distance_cm",
    "distance_valid",
    "effective_fps",
    "sampling_status",
    "registration_status",
    "quality_status",
    "training_valid",
    "ir_active",
    "lentes",
    "barba",
    "cabello_frente",
    "condiciones_manual",
] + MICRO_CHANNELS


def micro_sampling_status(
    fps
):
    fps = float(
        fps or 0.0
    )

    if fps >= 29.0:
        return "EXPLORATORIO_>=29FPS"

    if fps >= 20.0:
        return "LIMITADO_20_28FPS"

    return "INSUFICIENTE_<20FPS"


def enqueue_blendshape_series(
    source,
    filter_name,
    frame_seq,
    timestamp_ms,
    faces,
    micro_faces,
    effective_fps,
):
    """
    Guarda una fila por rostro/fuente/frame.

    Cada rostro conserva:
      person_id
      track_id
      etiquetas manuales propias
      52 blendshapes
    """
    with latest_telemetry_lock:
        telemetry = dict(
            latest_telemetry
        )

    distance_cm = telemetry.get(
        "distance_cm"
    )

    distance_valid = bool(
        telemetry.get(
            "valid",
            False
        )
    )

    timestamp_iso = datetime.now().isoformat(
        timespec="milliseconds"
    )

    for idx, face in enumerate(
        faces
    ):
        blendshapes = face.get(
            "blendshapes",
            {}
        )

        if not blendshapes:
            continue

        track_id = None

        if idx < len(
            micro_faces
        ):
            track_id = micro_faces[
                idx
            ].get(
                "track_id"
            )

        person_id = face.get(
            "person_id"
        )

        if (
            not person_id
            and track_id is not None
        ):
            person_id = get_source_track_person(
                source,
                track_id
            )

        state = get_person_conditions(
            person_id
        )
        quality = face.get(
            "quality_gate",
            {}
        )
        registration_status = get_registration_status(
            person_id
        )

        row = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "session_id": SESSION_ID,
            "frame_seq": int(
                frame_seq
            ),
            "timestamp_ms": int(
                timestamp_ms
            ),
            "timestamp_iso": timestamp_iso,
            "source": str(
                source
            ),
            "filter_name": str(
                filter_name
            ),
            "person_id": person_id,
            "track_id": track_id,
            "distance_cm": distance_cm,
            "distance_valid": int(
                distance_valid
            ),
            "effective_fps": round(
                float(
                    effective_fps or 0.0
                ),
                3
            ),
            "sampling_status": micro_sampling_status(
                effective_fps
            ),
            "registration_status": registration_status,
            "quality_status": quality.get(
                "status",
                "UNKNOWN"
            ),
            "training_valid": int(
                bool(quality.get("training_valid", 0))
                and registration_status == "REGISTERED"
            ),
            "ir_active": int(
                bool(telemetry.get("ir_active", False))
            ),
            "lentes": int(
                bool(
                    state.get(
                        "lentes"
                    )
                )
            ),
            "barba": int(
                bool(
                    state.get(
                        "barba"
                    )
                )
            ),
            "cabello_frente": int(
                bool(
                    state.get(
                        "cabello_frente"
                    )
                )
            ),
            "condiciones_manual": person_conditions_summary(
                state
            ),
        }

        for name in MICRO_CHANNELS:
            value = blendshapes.get(
                name
            )

            row[
                name
            ] = (
                round(
                    float(value),
                    6
                )
                if value is not None
                else None
            )

        if person_is_registered(person_id):
            row["registration_status"] = "REGISTERED"
            row["training_valid"] = int(
                bool(quality.get("training_valid", 0))
            )
            try:
                blendshape_queue.put_nowait(row)
            except queue.Full:
                pass
        else:
            row["registration_status"] = "PENDING"
            row["training_valid"] = 0
            with pending_blendshape_lock:
                dq = pending_blendshape_rows.setdefault(
                    person_id,
                    deque(maxlen=PENDING_BLENDSHAPE_MAX)
                )
                dq.append(copy.deepcopy(row))


def blendshape_writer_worker():
    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    file_exists = (
        BLENDSHAPE_SERIES_CSV.exists()
        and BLENDSHAPE_SERIES_CSV.stat().st_size > 0
    )

    with BLENDSHAPE_SERIES_CSV.open(
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=BLENDSHAPE_SERIES_HEADERS,
            extrasaction="ignore"
        )

        if not file_exists:
            writer.writeheader()
            f.flush()

        pending = 0

        while not stop_event.is_set():
            try:
                item = blendshape_queue.get(
                    timeout=0.25
                )
            except queue.Empty:
                if pending:
                    f.flush()
                    pending = 0
                continue

            if item is None:
                blendshape_queue.task_done()
                break

            writer.writerow(
                item
            )

            blendshape_queue.task_done()

            pending += 1

            if pending >= 100:
                f.flush()
                pending = 0

        f.flush()


def _bbox_center(box):
    x, y, w, h = [
        float(v)
        for v in box
    ]
    return np.asarray(
        [
            x + 0.5 * w,
            y + 0.5 * h,
        ],
        dtype=np.float32
    )


def _bbox_diag(box):
    w = float(box[2])
    h = float(box[3])
    return max(
        1.0,
        float(
            np.hypot(
                w,
                h
            )
        )
    )


def _micro_region_scores(deltas):
    scores = {}

    for region, names in MICRO_REGIONS.items():
        values = [
            float(
                deltas.get(
                    name,
                    0.0
                )
            )
            for name in names
        ]

        values.sort(
            reverse=True
        )

        if values:
            # Dos canales mas activos de la region.
            scores[
                region
            ] = float(
                np.mean(
                    values[:2]
                )
            )
        else:
            scores[
                region
            ] = 0.0

    return scores


def _top_micro_channels(
    deltas,
    count=5
):
    items = [
        (
            name,
            float(
                deltas.get(
                    name,
                    0.0
                )
            )
        )
        for name in MICRO_CHANNELS
    ]

    items.sort(
        key=lambda kv: kv[1],
        reverse=True
    )

    return items[
        :count
    ]


MICRO_EVENT_HEADERS = [
    "schema_version",
    "session_id",
    "created_at",
    "source",
    "filter_name",
    "person_id",
    "track_id",
    "type",
    "is_micro_candidate",
    "duration_class",
    "onset_ms",
    "apex_ms",
    "offset_ms",
    "duration_ms",
    "peak_activity",
    "threshold",
    "top_region",
    "top_channels",
    "lentes",
    "barba",
    "cabello_frente",
    "condiciones_manual",
    "registration_status",
    "ir_active",
]


def _write_micro_event_now(event):
    event = copy.deepcopy(
        event
    )

    source = str(
        event.get(
            "camera",
            event.get(
                "source",
                ""
            )
        )
    )

    event[
        "source"
    ] = source

    event[
        "schema_version"
    ] = DATASET_SCHEMA_VERSION

    event[
        "session_id"
    ] = SESSION_ID

    track_id = event.get(
        "track_id"
    )

    person_id = event.get(
        "person_id"
    )

    if (
        not person_id
        and track_id is not None
    ):
        person_id = get_source_track_person(
            source,
            track_id
        )

    event[
        "person_id"
    ] = person_id
    event[
        "registration_status"
    ] = get_registration_status(
        person_id
    )
    if "ir_active" not in event:
        with latest_telemetry_lock:
            _event_telem = dict(latest_telemetry)
        event[
            "ir_active"
        ] = int(bool(_event_telem.get("ir_active", False)))

    state = get_person_conditions(
        person_id
    )

    event[
        "lentes"
    ] = int(
        bool(
            state.get(
                "lentes"
            )
        )
    )

    event[
        "barba"
    ] = int(
        bool(
            state.get(
                "barba"
            )
        )
    )

    event[
        "cabello_frente"
    ] = int(
        bool(
            state.get(
                "cabello_frente"
            )
        )
    )

    event[
        "condiciones_manual"
    ] = person_conditions_summary(
        state
    )

    if source in NOIR_FILTER_IDS:
        try:
            idx = NOIR_FILTER_IDS.index(
                source
            )

            event[
                "filter_name"
            ] = FILTERS_NOIR[
                idx
            ]

        except Exception:
            event[
                "filter_name"
            ] = ""

    elif source == "RGB":
        event[
            "filter_name"
        ] = "ORIGINAL RGB"

    else:
        event.setdefault(
            "filter_name",
            ""
        )

    if person_id:
        key = (
            source,
            str(
                person_id
            ),
        )

        with last_event_person_lock:
            last_event_by_source_person[
                key
            ] = copy.deepcopy(
                event
            )

    try:
        with micro_log_lock:
            DATASET_DIR.mkdir(
                parents=True,
                exist_ok=True
            )

            with MICRO_EVENTS_FILE.open(
                "a",
                encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        event,
                        ensure_ascii=False
                    )
                    + "\n"
                )

            csv_exists = (
                MICRO_EVENTS_CSV.exists()
                and MICRO_EVENTS_CSV.stat().st_size > 0
            )

            with MICRO_EVENTS_CSV.open(
                "a",
                newline="",
                encoding="utf-8-sig"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=MICRO_EVENT_HEADERS,
                    extrasaction="ignore"
                )

                if not csv_exists:
                    writer.writeheader()

                csv_row = dict(
                    event
                )

                csv_row[
                    "top_channels"
                ] = json.dumps(
                    event.get(
                        "top_channels",
                        []
                    ),
                    ensure_ascii=False
                )

                writer.writerow(
                    csv_row
                )

        with micro_event_buffer_lock:
            micro_event_buffer.append(
                event
            )

    except Exception as e:
        print(
            "Micro log:",
            str(e)[:120]
        )


def log_micro_event(event):
    event = copy.deepcopy(event)
    source = str(event.get("camera", event.get("source", "")))
    track_id = event.get("track_id")
    person_id = event.get("person_id")
    if not person_id and track_id is not None:
        person_id = get_source_track_person(source, track_id)
    event["person_id"] = person_id
    event["registration_status"] = get_registration_status(person_id)
    with latest_telemetry_lock:
        telem = dict(latest_telemetry)
    event["ir_active"] = int(bool(telem.get("ir_active", False)))
    if person_id and not person_is_registered(person_id):
        with pending_micro_lock:
            dq = pending_micro_events.setdefault(
                person_id,
                deque(maxlen=PENDING_MICRO_MAX)
            )
            dq.append(copy.deepcopy(event))
        return
    _write_micro_event_now(event)


def drain_micro_event_buffer():
    with micro_event_buffer_lock:
        items = list(
            micro_event_buffer
        )
        micro_event_buffer.clear()

    return items


class MicroExpressionAnalyzer(object):
    """
    Detector temporal de CANDIDATOS a microexpresion.

    Cada instancia corresponde a UNA camara.
    Nunca recibe blendshapes de la otra camara.
    """

    def __init__(
        self,
        camera_name
    ):
        self.camera_name = str(
            camera_name
        )

        self.tracks = {}
        self.next_track_id = 1

    def _new_track(
        self,
        face,
        timestamp_ms
    ):
        bs = {
            name: float(
                face.get(
                    "blendshapes",
                    {}
                ).get(
                    name,
                    0.0
                )
            )
            for name in MICRO_CHANNELS
        }

        track = {
            "id": self.next_track_id,
            "bbox": list(
                face.get(
                    "bbox",
                    [0, 0, 1, 1]
                )
            ),
            "baseline": dict(bs),
            "noise": {
                name: 0.01
                for name in MICRO_CHANNELS
            },
            "active": False,
            "onset_ms": None,
            "peak_ms": None,
            "peak_score": 0.0,
            "peak_deltas": {},
            "below_count": 0,
            "last_seen_ms": int(
                timestamp_ms
            ),
            "last_activity": 0.0,
            "last_region_scores": {},
        }

        self.tracks[
            self.next_track_id
        ] = track

        self.next_track_id += 1

        return track

    def _match_track(
        self,
        face,
        timestamp_ms,
        excluded_track_ids=None
    ):
        if excluded_track_ids is None:
            excluded_track_ids = set()

        bbox = face.get(
            "bbox"
        )

        if not bbox:
            return self._new_track(
                face,
                timestamp_ms
            )

        center = _bbox_center(
            bbox
        )

        best = None
        best_score = 999.0

        for track in self.tracks.values():
            if int(
                track.get(
                    "id",
                    -1
                )
            ) in excluded_track_ids:
                continue

            old_box = track.get(
                "bbox"
            )

            if not old_box:
                continue

            age_ms = (
                int(timestamp_ms)
                - int(
                    track.get(
                        "last_seen_ms",
                        0
                    )
                )
            )

            if age_ms > 1200:
                continue

            old_center = _bbox_center(
                old_box
            )

            norm = max(
                _bbox_diag(
                    bbox
                ),
                _bbox_diag(
                    old_box
                ),
            )

            score = float(
                np.linalg.norm(
                    center
                    - old_center
                )
                / norm
            )

            if (
                score < 0.55
                and score < best_score
            ):
                best = track
                best_score = score

        if best is None:
            best = self._new_track(
                face,
                timestamp_ms
            )

        return best

    @staticmethod
    def _activity_score(
        deltas
    ):
        values = sorted(
            [
                max(
                    0.0,
                    float(
                        deltas.get(
                            name,
                            0.0
                        )
                    )
                )
                for name in MICRO_CHANNELS
            ],
            reverse=True
        )

        if not values:
            return 0.0

        # Top-3 evita que un unico canal ruidoso dispare todo.
        return float(
            np.mean(
                values[:3]
            )
        )

    def _update_track(
        self,
        track,
        face,
        timestamp_ms
    ):
        current = {
            name: float(
                face.get(
                    "blendshapes",
                    {}
                ).get(
                    name,
                    0.0
                )
            )
            for name in MICRO_CHANNELS
        }

        # V16: un frame YELLOW/RED puede mantener el TRACK, pero NO
        # modifica baseline ni genera eventos de entrenamiento.
        if not bool(face.get("training_valid", 1)):
            track["bbox"] = list(face.get("bbox", track.get("bbox", [0,0,1,1])))
            track["last_seen_ms"] = int(timestamp_ms)
            return {
                "track_id": int(track["id"]),
                "active": bool(track.get("active", False)),
                "activity": round(float(track.get("last_activity", 0.0)), 5),
                "threshold": None,
                "region_scores": dict(track.get("last_region_scores", {})),
                "top_channels": [],
                "bbox": list(track["bbox"]),
                "quality_status": face.get("quality_status", "UNKNOWN"),
                "training_valid": 0,
            }, None

        baseline = track[
            "baseline"
        ]

        noise = track[
            "noise"
        ]

        deltas = {
            name: max(
                0.0,
                current[name]
                - baseline.get(
                    name,
                    current[name]
                )
            )
            for name in MICRO_CHANNELS
        }

        activity = (
            self._activity_score(
                deltas
            )
        )

        noise_level = float(
            np.mean(
                [
                    noise.get(
                        name,
                        0.01
                    )
                    for name in MICRO_CHANNELS
                ]
            )
        )

        threshold = max(
            MICRO_MIN_ACTIVITY,
            min(
                0.20,
                0.035
                + MICRO_NOISE_GAIN
                * noise_level
            )
        )

        region_scores = (
            _micro_region_scores(
                deltas
            )
        )

        event = None

        if not track[
            "active"
        ]:
            if activity >= threshold:
                track[
                    "active"
                ] = True

                track[
                    "onset_ms"
                ] = int(
                    timestamp_ms
                )

                track[
                    "peak_ms"
                ] = int(
                    timestamp_ms
                )

                track[
                    "peak_score"
                ] = float(
                    activity
                )

                track[
                    "peak_deltas"
                ] = dict(
                    deltas
                )

                track[
                    "below_count"
                ] = 0

            else:
                # Linea base se adapta lentamente SOLO cuando no
                # hay un evento facial activo.
                for name in MICRO_CHANNELS:
                    diff = abs(
                        current[name]
                        - baseline.get(
                            name,
                            current[name]
                        )
                    )

                    noise[name] = (
                        0.97
                        * noise.get(
                            name,
                            0.01
                        )
                        + 0.03
                        * diff
                    )

                    baseline[name] = (
                        0.985
                        * baseline.get(
                            name,
                            current[name]
                        )
                        + 0.015
                        * current[name]
                    )

        else:
            if activity > track[
                "peak_score"
            ]:
                track[
                    "peak_score"
                ] = float(
                    activity
                )

                track[
                    "peak_ms"
                ] = int(
                    timestamp_ms
                )

                track[
                    "peak_deltas"
                ] = dict(
                    deltas
                )

            if activity < (
                threshold
                * 0.45
            ):
                track[
                    "below_count"
                ] += 1
            else:
                track[
                    "below_count"
                ] = 0

            active_duration = (
                int(timestamp_ms)
                - int(
                    track[
                        "onset_ms"
                    ]
                )
            )

            should_end = (
                track[
                    "below_count"
                ] >= 2
                or active_duration
                > 700
            )

            if should_end:
                onset = int(
                    track[
                        "onset_ms"
                    ]
                )

                apex = int(
                    track[
                        "peak_ms"
                    ]
                )

                offset = int(
                    timestamp_ms
                )

                duration_ms = (
                    offset
                    - onset
                )

                peak_deltas = dict(
                    track[
                        "peak_deltas"
                    ]
                )

                peak_regions = (
                    _micro_region_scores(
                        peak_deltas
                    )
                )

                top_regions = sorted(
                    peak_regions.items(),
                    key=lambda kv: kv[1],
                    reverse=True
                )

                top_channels = (
                    _top_micro_channels(
                        peak_deltas,
                        count=5
                    )
                )

                is_micro_candidate = (
                    MICRO_MIN_DURATION_MS
                    <= duration_ms
                    <= MICRO_MAX_DURATION_MS
                    and track[
                        "peak_score"
                    ]
                    >= threshold
                )

                event = {
                    "type": (
                        "MICRO_CANDIDATE"
                        if is_micro_candidate
                        else "FACIAL_CHANGE"
                    ),
                    "is_micro_candidate": int(
                        bool(
                            is_micro_candidate
                        )
                    ),
                    "duration_class": (
                        "MICRO_40_500MS"
                        if is_micro_candidate
                        else (
                            "CAMBIO_LARGO_>500MS"
                            if duration_ms > MICRO_MAX_DURATION_MS
                            else "CAMBIO_CORTO_NO_MICRO"
                        )
                    ),
                    "camera": self.camera_name,
                    "track_id": int(
                        track[
                            "id"
                        ]
                    ),
                    "onset_ms": onset,
                    "apex_ms": apex,
                    "offset_ms": offset,
                    "duration_ms": int(
                        duration_ms
                    ),
                    "peak_activity": round(
                        float(
                            track[
                                "peak_score"
                            ]
                        ),
                        5
                    ),
                    "threshold": round(
                        float(
                            threshold
                        ),
                        5
                    ),
                    "top_region": (
                        top_regions[0][0]
                        if top_regions
                        else "SIN_REGION"
                    ),
                    "region_scores": {
                        k: round(
                            float(v),
                            5
                        )
                        for k, v
                        in peak_regions.items()
                    },
                    "top_channels": [
                        {
                            "name": name,
                            "delta": round(
                                float(value),
                                5
                            ),
                        }
                        for name, value
                        in top_channels
                    ],
                    "created_at": (
                        datetime.now()
                        .isoformat(
                            timespec="milliseconds"
                        )
                    ),
                }

                # Guardar TODO evento temporal.
                # is_micro_candidate indica si cumple el criterio
                # de microexpresion; FACIAL_CHANGE tambien se conserva.
                log_micro_event(
                    event
                )

                # Reset del evento. La linea base no se reemplaza
                # bruscamente por el pico.
                track[
                    "active"
                ] = False

                track[
                    "onset_ms"
                ] = None

                track[
                    "peak_ms"
                ] = None

                track[
                    "peak_score"
                ] = 0.0

                track[
                    "peak_deltas"
                ] = {}

                track[
                    "below_count"
                ] = 0

        track[
            "bbox"
        ] = list(
            face.get(
                "bbox",
                track.get(
                    "bbox",
                    [0, 0, 1, 1]
                )
            )
        )

        track[
            "last_seen_ms"
        ] = int(
            timestamp_ms
        )

        track[
            "last_activity"
        ] = float(
            activity
        )

        track[
            "last_region_scores"
        ] = dict(
            region_scores
        )

        return {
            "track_id": int(
                track[
                    "id"
                ]
            ),
            "active": bool(
                track[
                    "active"
                ]
            ),
            "activity": round(
                float(
                    activity
                ),
                5
            ),
            "threshold": round(
                float(
                    threshold
                ),
                5
            ),
            "region_scores": {
                k: round(
                    float(v),
                    5
                )
                for k, v
                in region_scores.items()
            },
            "top_channels": [
                {
                    "name": name,
                    "delta": round(
                        float(value),
                        5
                    ),
                }
                for name, value
                in _top_micro_channels(
                    deltas,
                    count=3
                )
            ],
            "bbox": list(
                track[
                    "bbox"
                ]
            ),
        }, event

    def update(
        self,
        faces,
        timestamp_ms
    ):
        outputs = []
        events = []

        # Un track solo puede pertenecer a un rostro por frame.
        used_track_ids = set()

        for face in faces:
            blendshapes = face.get(
                "blendshapes",
                {}
            )

            if not blendshapes:
                continue

            track = self._match_track(
                face,
                timestamp_ms,
                excluded_track_ids=used_track_ids
            )

            used_track_ids.add(
                int(
                    track[
                        "id"
                    ]
                )
            )

            state, event = (
                self._update_track(
                    track,
                    face,
                    timestamp_ms
                )
            )

            outputs.append(
                state
            )

            if event is not None:
                events.append(
                    event
                )

        # Limpiar tracks viejos.
        stale = [
            track_id
            for track_id, track
            in self.tracks.items()
            if (
                int(timestamp_ms)
                - int(
                    track.get(
                        "last_seen_ms",
                        0
                    )
                )
            ) > 2000
        ]

        for track_id in stale:
            self.tracks.pop(
                track_id,
                None
            )

        return outputs, events


MICRO_ANALYZERS = {
    0: MicroExpressionAnalyzer(
        "RGB"
    ),
}

NOIR_FILTER_ANALYZERS = {
    i: MicroExpressionAnalyzer(
        NOIR_FILTER_IDS[i]
    )
    for i in range(
        len(FILTERS_NOIR)
    )
}


def micro_summary_text(
    camera_id
):
    with latest_micro_lock:
        state = copy.deepcopy(
            latest_micro.get(
                camera_id,
                {}
            )
        )

    faces = state.get(
        "faces",
        []
    )

    active = [
        item
        for item in faces
        if item.get(
            "active"
        )
    ]

    if active:
        best = max(
            active,
            key=lambda item: item.get(
                "activity",
                0.0
            )
        )

        regions = best.get(
            "region_scores",
            {}
        )

        top_region = (
            max(
                regions,
                key=regions.get
            )
            if regions
            else "SIN_REGION"
        )

        return (
            "MICRO ACTIVA | {} | act {:.3f}"
            .format(
                top_region,
                float(
                    best.get(
                        "activity",
                        0.0
                    )
                )
            )
        ), ORANGE

    event = state.get(
        "last_event"
    )

    if event:
        age_ms = int(
            time.monotonic_ns()
            // 1_000_000
        ) - int(
            event.get(
                "_shown_at_ms",
                0
            )
        )

        if age_ms <= MICRO_EVENT_HOLD_MS:
            return (
                "MICRO CANDIDATA | {} ms | {}"
                .format(
                    event.get(
                        "duration_ms",
                        0
                    ),
                    event.get(
                        "top_region",
                        "SIN_REGION"
                    )
                )
            ), GREEN

    return (
        "MICRO: esperando cambio breve..."
    ), GRAY


def noir_filter_micro_summary_text(
    filter_index
):
    with latest_noir_filter_lock:
        state = copy.deepcopy(
            latest_noir_filter_micro.get(
                filter_index,
                {}
            )
        )

    faces = state.get(
        "faces",
        []
    )

    active = [
        item
        for item in faces
        if item.get(
            "active"
        )
    ]

    if active:
        best = max(
            active,
            key=lambda item: item.get(
                "activity",
                0.0
            )
        )

        regions = best.get(
            "region_scores",
            {}
        )

        top_region = (
            max(
                regions,
                key=regions.get
            )
            if regions
            else "SIN_REGION"
        )

        return (
            "MICRO ACTIVA | {} | {:.3f}"
            .format(
                top_region,
                float(
                    best.get(
                        "activity",
                        0.0
                    )
                )
            )
        ), ORANGE

    event = state.get(
        "last_event"
    )

    if event:
        age_ms = (
            int(
                time.monotonic_ns()
                // 1_000_000
            )
            - int(
                event.get(
                    "_shown_at_ms",
                    0
                )
            )
        )

        if age_ms <= MICRO_EVENT_HOLD_MS:
            return (
                "MICRO CANDIDATA | {} ms | {}"
                .format(
                    event.get(
                        "duration_ms",
                        0
                    ),
                    event.get(
                        "top_region",
                        "SIN_REGION"
                    )
                )
            ), GREEN

    return (
        "MICRO: esperando..."
    ), GRAY


def noir_filters_status_line():
    parts = []

    with latest_noir_filter_lock:
        states = copy.deepcopy(
            latest_noir_filter_micro
        )

    for idx, short_name in enumerate(
        NOIR_FILTER_SHORT
    ):
        state = states.get(
            idx,
            {}
        )

        faces = state.get(
            "faces",
            []
        )

        active_faces = [
            f
            for f in faces
            if f.get(
                "active"
            )
        ]

        if active_faces:
            activity = max(
                float(
                    f.get(
                        "activity",
                        0.0
                    )
                )
                for f in active_faces
            )

            parts.append(
                "{}:{:.2f}".format(
                    short_name,
                    activity
                )
            )
        else:
            event = state.get(
                "last_event"
            )

            if event:
                age_ms = (
                    int(
                        time.monotonic_ns()
                        // 1_000_000
                    )
                    - int(
                        event.get(
                            "_shown_at_ms",
                            0
                        )
                    )
                )

                if age_ms <= MICRO_EVENT_HOLD_MS:
                    parts.append(
                        "{}:EVENT".format(
                            short_name
                        )
                    )
                    continue

            parts.append(
                "{}:--".format(
                    short_name
                )
            )

    return " | ".join(
        parts
    )


def draw_micro_points(
    frame,
    faces
):
    """Dibuja 68 puntos con color del Quality Gate."""
    out = frame.copy()

    for face in faces:
        quality = str(
            face.get(
                "quality_status",
                face.get("quality_gate", {}).get("status", "UNKNOWN")
            )
        )

        if quality == "GREEN":
            point_color = GREEN
        elif quality == "YELLOW":
            point_color = YELLOW
        else:
            point_color = RED

        pts = np.asarray(
            face.get("landmarks", []),
            dtype=np.float32
        )

        if pts.shape == (68, 2):
            for px, py in pts:
                cv2.circle(
                    out,
                    (int(round(px)), int(round(py))),
                    1,
                    point_color,
                    -1,
                    cv2.LINE_AA
                )

        person_id = face.get("person_id")
        bbox = face.get("bbox")

        if person_id and bbox and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            reg = get_registration_status(person_id)
            cv2.putText(
                out,
                "{} | {} | {}".format(person_id, quality, reg),
                (max(0,x), max(18,y-6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                point_color,
                1,
                cv2.LINE_AA
            )

    return out


# ============================================================
# LANDMARK DRAWING
# ============================================================

FACIAL_CHAINS = [
    (list(range(0, 17)), (0, 255, 255)),
    (list(range(17, 22)), (255, 0, 255)),
    (list(range(22, 27)), (0, 165, 255)),
    (list(range(27, 31)), (255, 255, 0)),
    (list(range(31, 36)), (255, 255, 0)),
    (list(range(36, 42)) + [36], (0, 255, 0)),
    (list(range(42, 48)) + [42], (255, 0, 0)),
    (list(range(48, 60)) + [48], (0, 0, 255)),
    (list(range(60, 68)) + [60], (255, 0, 128)),
]


def landmark_region(idx):
    if idx <= 16:
        return "MANDIBULA"
    if idx <= 21:
        return "CEJA_IZQUIERDA"
    if idx <= 26:
        return "CEJA_DERECHA"
    if idx <= 35:
        return "NARIZ"
    if idx <= 41:
        return "OJO_IZQUIERDO"
    if idx <= 47:
        return "OJO_DERECHO"
    if idx <= 59:
        return "BOCA_EXTERIOR"
    return "BOCA_INTERIOR"


def draw_faces(
    frame,
    faces
):
    return draw_micro_points(
        frame,
        faces
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality(
    frame,
    bbox=None
):
    roi = frame

    if bbox is not None:
        x, y, w, h = [
            int(v)
            for v in bbox
        ]

        x = max(0, x)
        y = max(0, y)

        x2 = min(
            frame.shape[1],
            x + max(1, w)
        )

        y2 = min(
            frame.shape[0],
            y + max(1, h)
        )

        if x2 > x and y2 > y:
            roi = frame[
                y:y2,
                x:x2
            ]

    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "brightness": round(
            float(
                np.mean(gray)
            ),
            4
        ),
        "contrast": round(
            float(
                np.std(gray)
            ),
            4
        ),
        "sharpness": round(
            float(
                cv2.Laplacian(
                    gray,
                    cv2.CV_64F
                ).var()
            ),
            4
        ),
    }



# ============================================================
# MULTI-FACE DATASET HELPERS
# ============================================================

FACE_CROP_SIZE = 256


def crop_face_for_dataset(
    frame,
    bbox,
    output_size=None
):
    """
    Recorte facial LIMPIO conservando los pixeles y la relacion de aspecto.
    No fuerza 256x256 para los archivos del dataset.

    output_size se usa solamente cuando la interfaz necesita una miniatura.
    """
    if frame is None or bbox is None or len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]
    if w <= 1 or h <= 1:
        return None

    frame_h, frame_w = frame.shape[:2]

    # Margen proporcional: mas frente/cejas arriba, sin volverlo cuadrado.
    mx = 0.16 * w
    mt = 0.24 * h
    mb = 0.16 * h

    x1 = max(0, int(round(x - mx)))
    y1 = max(0, int(round(y - mt)))
    x2 = min(frame_w, int(round(x + w + mx)))
    y2 = min(frame_h, int(round(y + h + mb)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2].copy()

    if output_size is not None:
        # Solo para UI: limitar el lado mayor sin deformar la proporcion.
        target = max(16, int(output_size))
        ch, cw = crop.shape[:2]
        scale = min(float(target) / max(1, cw), float(target) / max(1, ch))
        nw = max(1, int(round(cw * scale)))
        nh = max(1, int(round(ch * scale)))
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    return crop



def aggregate_face_quality(
    frame,
    faces
):
    """
    Promedio de brillo/contraste/nitidez de TODOS los rostros detectados.
    Devuelve None por metrica si no hay rostros validos.
    """
    metrics = []

    for face in (faces or []):
        bbox = face.get("bbox") if isinstance(face, dict) else None
        if not bbox:
            continue

        try:
            q = image_quality(frame, bbox)
            if isinstance(q, dict):
                metrics.append(q)
        except Exception:
            pass

    if not metrics:
        return {
            "brightness": None,
            "contrast": None,
            "sharpness": None,
        }

    result = {}

    for key in [
        "brightness",
        "contrast",
        "sharpness",
    ]:
        values = []
        for item in metrics:
            value = item.get(key)
            if value is None:
                continue
            try:
                values.append(float(value))
            except Exception:
                pass

        result[key] = (
            round(float(np.mean(values)), 4)
            if values
            else None
        )

    return result

def create_face_strip_from_records(
    person_records,
    output_path,
    title
):
    """
    Crea una sola imagen SIN FONDO que contiene los recortes
    P01/P02/P03... de una camara.
    """
    items = []

    for record in person_records:
        path = record.get(
            "face_crop_path"
        )

        if not path:
            continue

        image = cv2.imread(
            str(
                path
            )
        )

        if image is None:
            continue

        items.append(
            (
                str(
                    record.get(
                        "person_id",
                        "P??"
                    )
                ),
                image,
                person_conditions_summary_by_id(
                    record.get(
                        "person_id"
                    )
                ),
            )
        )

    if not items:
        return None

    tile_w = 180
    tile_h = 210
    top_h = 34

    canvas = np.zeros(
        (
            tile_h,
            tile_w
            * len(
                items
            ),
            3
        ),
        dtype=np.uint8
    )

    for idx, (
        person_id,
        image,
        summary,
    ) in enumerate(
        items
    ):
        x1 = idx * tile_w

        face = cv2.resize(
            image,
            (
                150,
                150
            ),
            interpolation=cv2.INTER_AREA
        )

        canvas[
            top_h:top_h + 150,
            x1 + 15:x1 + 165
        ] = face

        cv2.putText(
            canvas,
            "{} | {}".format(
                title,
                person_id
            ),
            (
                x1 + 10,
                23
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            summary[:22],
            (
                x1 + 8,
                tile_h - 10
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            GREEN
            if summary != "NINGUNA"
            else GRAY,
            1,
            cv2.LINE_AA
        )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    save_image(
        output_path,
        canvas
    )

    return str(
        output_path
    )


# ============================================================
# YOLO HELPERS
# ============================================================

def yolo_settings(metadata):
    y = metadata.get(
        "yolo11n",
        {}
    )

    return (
        float(
            y.get(
                "valid_conf",
                VALID_CONF_DEFAULT
            )
        ),
        float(
            y.get(
                "weak_conf",
                WEAK_CONF_DEFAULT
            )
        )
    )



def scale_detections(
    detections,
    source_w,
    source_h,
    target_w,
    target_h
):
    sx = float(target_w) / float(source_w)
    sy = float(target_h) / float(source_h)

    scaled = []

    for det in detections:
        item = dict(det)
        box = det.get(
            "box",
            []
        )

        if len(box) == 4:
            item["box"] = [
                int(round(float(box[0]) * sx)),
                int(round(float(box[1]) * sy)),
                int(round(float(box[2]) * sx)),
                int(round(float(box[3]) * sy)),
            ]

        scaled.append(item)

    return scaled


def effective_valid_conf_for_detection(
    det,
    valid_conf
):
    name = str(
        det.get(
            "class_name",
            ""
        )
    ).lower()

    if name == "person":
        return max(
            float(valid_conf),
            float(PERSON_VALID_CONF)
        )

    return float(
        valid_conf
    )


def draw_yolo_boxes(
    frame,
    detections,
    valid_conf,
    weak_conf
):
    out = frame.copy()

    for det in detections:
        box = det.get(
            "box",
            []
        )

        if len(box) != 4:
            continue

        conf = float(
            det.get(
                "confidence",
                0.0
            )
        )

        if conf < weak_conf:
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

        effective_valid_conf = (
            effective_valid_conf_for_detection(
                det,
                valid_conf
            )
        )

        if conf >= effective_valid_conf:
            color = (
                GREEN
                if name.lower()
                == "person"
                else YELLOW
            )
            prefix = ""
        else:
            color = GRAY
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
                max(
                    18,
                    y1 - 5
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA
        )

    return out


def person_detections(
    detections,
    valid_conf
):
    persons = []

    person_conf = max(
        float(valid_conf),
        float(PERSON_VALID_CONF)
    )

    for det in detections:
        if (
            str(
                det.get(
                    "class_name",
                    ""
                )
            ).lower()
            == "person"
            and float(
                det.get(
                    "confidence",
                    0.0
                )
            )
            >= person_conf
        ):
            persons.append(det)

    persons.sort(
        key=lambda d: (
            d.get(
                "box",
                [0, 0, 0, 0]
            )[0]
        )
    )

    return persons



def select_faces_for_detected_persons(
    detections,
    faces,
    valid_conf
):
    """
    Cuando YOLO detecta personas, conserva como maximo UNA cara
    MediaPipe dentro de cada persona. Esto elimina falsos rostros
    NoIR como reflejos/luces sin impedir varias personas reales.

    RGB y NoIR se procesan por separado.
    """
    persons = person_detections(
        detections,
        valid_conf
    )

    if not persons:
        return list(
            faces
        )

    selected_faces = []
    used = set()

    for person in persons:
        pbox = [
            int(v)
            for v in person.get(
                "box",
                [0, 0, 0, 0]
            )
        ]

        candidates = []

        for fidx, face in enumerate(
            faces
        ):
            if fidx in used:
                continue

            bbox = face.get(
                "bbox"
            )

            if not bbox:
                continue

            x, y, w, h = [
                float(v)
                for v in bbox
            ]

            cx = x + 0.5 * w
            cy = y + 0.5 * h

            # Pequeño margen para tolerar que la cabeza toque
            # el borde superior de la caja de persona.
            pw = max(
                1.0,
                float(
                    pbox[2]
                    - pbox[0]
                )
            )

            ph = max(
                1.0,
                float(
                    pbox[3]
                    - pbox[1]
                )
            )

            mx = 0.08 * pw
            my = 0.10 * ph

            if (
                pbox[0] - mx
                <= cx
                <= pbox[2] + mx
                and
                pbox[1] - my
                <= cy
                <= pbox[3] + my
            ):
                candidates.append(
                    (
                        w * h,
                        fidx,
                        face
                    )
                )

        if candidates:
            candidates.sort(
                key=lambda item: item[0],
                reverse=True
            )

            _area, fidx, face = (
                candidates[0]
            )

            used.add(
                fidx
            )

            selected_faces.append(
                face
            )

    return selected_faces


def associate_faces_to_persons(
    detections,
    faces,
    valid_conf
):
    """
    PERSONAS representa sujetos con ROSTRO visible.

    Las detecciones YOLO 'person' sin rostro asociado (por ejemplo
    alguien al fondo, de espaldas o cortado) se conservan en OBJETOS,
    pero NO reciben P01/P02 dentro de PERSONAS.

    Esto evita:
        RGB: fondo=P01, sujeto=P02
        NoIR: sujeto=P01

    Si hay un solo rostro real en ambas camaras, ese sujeto sera P01
    en RGB y P01 en NoIR.
    """
    persons = person_detections(
        detections,
        valid_conf
    )

    # Orden estable de los rostros por posicion horizontal.
    indexed_faces = list(
        enumerate(
            faces
        )
    )

    indexed_faces.sort(
        key=lambda item: (
            float(
                item[1].get(
                    "bbox",
                    [0, 0, 0, 0]
                )[0]
            )
            + 0.5
            * float(
                item[1].get(
                    "bbox",
                    [0, 0, 0, 0]
                )[2]
            )
        )
    )

    rows = []
    used_person_indices = set()

    for subject_idx, (
        _face_idx,
        face
    ) in enumerate(
        indexed_faces,
        start=1
    ):
        bbox = face.get(
            "bbox",
            [0, 0, 0, 0]
        )

        x, y, w, h = [
            float(v)
            for v in bbox
        ]

        cx = x + w / 2.0
        cy = y + h / 2.0

        candidates = []

        for pidx, person in enumerate(
            persons
        ):
            if pidx in used_person_indices:
                continue

            pbox = [
                int(v)
                for v in person.get(
                    "box",
                    [0, 0, 0, 0]
                )
            ]

            pw = max(
                1.0,
                float(
                    pbox[2]
                    - pbox[0]
                )
            )

            ph = max(
                1.0,
                float(
                    pbox[3]
                    - pbox[1]
                )
            )

            # Margen pequeño para tolerar cabeza tocando el borde.
            mx = 0.08 * pw
            my = 0.10 * ph

            if (
                pbox[0] - mx
                <= cx
                <= pbox[2] + mx
                and
                pbox[1] - my
                <= cy
                <= pbox[3] + my
            ):
                area = pw * ph
                conf = float(
                    person.get(
                        "confidence",
                        0.0
                    )
                )

                # Prefiere una caja ajustada; confianza desempata.
                candidates.append(
                    (
                        area,
                        -conf,
                        pidx,
                        person,
                        pbox,
                    )
                )

        matched_person = None
        matched_box = None
        matched_conf = None

        if candidates:
            candidates.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                )
            )

            (
                _area,
                _neg_conf,
                pidx,
                matched_person,
                matched_box,
            ) = candidates[0]

            used_person_indices.add(
                pidx
            )

            matched_conf = float(
                matched_person.get(
                    "confidence",
                    0.0
                )
            )

        rows.append(
            {
                "person_id": (
                    face.get(
                        "person_id"
                    )
                    or "P{:02d}".format(
                        subject_idx
                    )
                ),
                "source": (
                    "YOLO_PERSON_FACE"
                    if matched_person is not None
                    else "FACE_ONLY"
                ),
                "yolo_person_confidence": (
                    matched_conf
                    if matched_person is not None
                    else None
                ),
                "person_box": (
                    matched_box
                    if matched_person is not None
                    else None
                ),
                "face": face,
            }
        )

    return rows


# ============================================================
# V16.19 - CONTROL DE CALIDAD DEL DATASET
# ============================================================

def _v1619_load_qc_history():
    try:
        if QC_HISTORY_FILE.exists():
            data = json.loads(QC_HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print("[V16.19 QC] historial no pudo leerse:", repr(exc))
    return {}


def _v1619_save_qc_history(history):
    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        tmp = QC_HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        tmp.replace(QC_HISTORY_FILE)
    except Exception as exc:
        print("[V16.19 QC] historial no pudo guardarse:", repr(exc))


def _v1619_iqr_outlier(value, samples):
    clean = []
    for item in samples or []:
        try:
            fv = float(item)
            if np.isfinite(fv):
                clean.append(fv)
        except Exception:
            pass
    if value is None or len(clean) < QC_HISTORY_MIN_SAMPLES:
        return False, len(clean)
    try:
        v = float(value)
        if not np.isfinite(v):
            return True, len(clean)
        q1 = float(np.percentile(clean, 25))
        q3 = float(np.percentile(clean, 75))
        iqr = q3 - q1
        if iqr <= 1e-9:
            # Sin dispersion historica suficiente para juzgar.
            return False, len(clean)
        low = q1 - QC_IQR_MULTIPLIER * iqr
        high = q3 + QC_IQR_MULTIPLIER * iqr
        return bool(v < low or v > high), len(clean)
    except Exception:
        return False, len(clean)


def _v1619_enrich_landmark_qc(landmark_records):
    """
    Compara cada landmark normalizado con el historial del MISMO
    camera/filter + point_index. No mezcla RGB con NoIR ni filtros.
    Empieza a marcar outliers solo con >=8 referencias previas.
    """
    history = _v1619_load_qc_history()

    for rec in landmark_records:
        camera = str(rec.get("camera") or "UNKNOWN")
        try:
            pidx = int(rec.get("point_index"))
        except Exception:
            pidx = -1
        key = "{}|{}".format(camera, pidx)
        h = history.get(key, {}) if isinstance(history.get(key, {}), dict) else {}
        xs = h.get("x", []) or []
        ys = h.get("y", []) or []

        ox, nx = _v1619_iqr_outlier(rec.get("x_norm_face"), xs)
        oy, ny = _v1619_iqr_outlier(rec.get("y_norm_face"), ys)
        rec["iqr_reference_n"] = min(nx, ny)
        rec["is_outlier_iqr"] = int(bool(ox or oy))

    # Actualizar historial DESPUES de evaluar, para no compararse consigo mismo.
    for rec in landmark_records:
        camera = str(rec.get("camera") or "UNKNOWN")
        try:
            pidx = int(rec.get("point_index"))
            xv = float(rec.get("x_norm_face"))
            yv = float(rec.get("y_norm_face"))
            if not (np.isfinite(xv) and np.isfinite(yv)):
                continue
        except Exception:
            continue
        key = "{}|{}".format(camera, pidx)
        h = history.setdefault(key, {"x": [], "y": []})
        h.setdefault("x", []).append(xv)
        h.setdefault("y", []).append(yv)
        h["x"] = h["x"][-QC_HISTORY_MAX_SAMPLES:]
        h["y"] = h["y"][-QC_HISTORY_MAX_SAMPLES:]

    _v1619_save_qc_history(history)


def _v1619_quality_status(valid_ratio, outlier_ratio, enough_history):
    if valid_ratio >= 0.999:
        base = "COMPLETO"
    elif valid_ratio >= 0.75:
        base = "BUENO"
    elif valid_ratio >= 0.50:
        base = "PARCIAL"
    else:
        base = "BAJO"

    if enough_history and outlier_ratio > 0.15:
        return "REVISAR_OUTLIERS"
    return base


def _v1619_apply_quality_summary(
    selected_ids,
    capture_id,
    person_records,
    landmark_records,
    noir_filter_records,
):
    """Agrega grupos, resumen por Pxx y devuelve filas CALIDAD_DATASET."""
    group_id = str(capture_id)
    selected = [str(x) for x in selected_ids]

    # Indexar outliers por (camera, person_id).
    lm_index = {}
    for lm in landmark_records:
        key = (str(lm.get("camera")), str(lm.get("person_id")))
        lm_index.setdefault(key, []).append(lm)

    def stats(camera, pid):
        rows = lm_index.get((str(camera), str(pid)), [])
        total = len(rows)
        outliers = sum(int(r.get("is_outlier_iqr") or 0) for r in rows)
        refs = [int(r.get("iqr_reference_n") or 0) for r in rows]
        ref_min = min(refs) if refs else 0
        ratio = (float(outliers) / float(total)) if total else 0.0
        return total, outliers, ratio, ref_min

    for rec in person_records:
        pid = str(rec.get("person_id"))
        cam = str(rec.get("camera"))
        total, outliers, ratio, ref_min = stats(cam, pid)
        detected = int(rec.get("landmarks_detected") or 0) == 1 and total >= 68
        enough = ref_min >= QC_HISTORY_MIN_SAMPLES
        rec["subject_group"] = pid
        rec["landmark_outlier_count"] = outliers if total else None
        rec["landmark_outlier_ratio"] = round(ratio, 6) if total else None
        rec["normalized_valid"] = int(detected and (not enough or ratio <= 0.15))

    for rec in noir_filter_records:
        pid = str(rec.get("person_id"))
        fid = str(rec.get("filter_id"))
        total, outliers, ratio, ref_min = stats(fid, pid)
        detected = int(rec.get("landmarks_detected") or 0) == 1 and total >= 68
        enough = ref_min >= QC_HISTORY_MIN_SAMPLES
        rec["group_id"] = group_id
        rec["subject_group"] = pid
        rec["landmark_outlier_count"] = outliers if total else None
        rec["landmark_outlier_ratio"] = round(ratio, 6) if total else None
        rec["normalized_valid"] = int(detected and (not enough or ratio <= 0.15))

    quality_rows = []
    channels = ["RGB", "NOIR"] + list(NOIR_FILTER_IDS)
    for pid in selected:
        valid_count = 0
        total_outliers = 0
        total_landmarks = 0
        ref_ns = []
        for ch in channels:
            total, outliers, ratio, ref_min = stats(ch, pid)
            if total >= 68:
                valid_count += 1
            total_outliers += outliers
            total_landmarks += total
            if total:
                ref_ns.append(ref_min)

        total_channels = len(channels)
        valid_ratio = float(valid_count) / float(total_channels) if total_channels else 0.0
        outlier_ratio = (
            float(total_outliers) / float(total_landmarks)
            if total_landmarks else 0.0
        )
        ref_min = min(ref_ns) if ref_ns else 0
        enough_history = ref_min >= QC_HISTORY_MIN_SAMPLES
        status = _v1619_quality_status(valid_ratio, outlier_ratio, enough_history)
        recommended = int(
            valid_ratio >= 0.75 and
            (not enough_history or outlier_ratio <= 0.15)
        )
        quality_rows.append({
            "schema_version": DATASET_SCHEMA_VERSION,
            "session_id": SESSION_ID,
            "group_id": group_id,
            "capture_id": capture_id,
            "person_id": pid,
            "subject_group": pid,
            "valid_channels_count": valid_count,
            "valid_channels_total": total_channels,
            "valid_channels_ratio": round(valid_ratio, 6),
            "landmark_rows": total_landmarks,
            "outlier_count": total_outliers,
            "outlier_ratio": round(outlier_ratio, 6),
            "history_reference_min_n": ref_min,
            "quality_status": status,
            "recommended_for_training": recommended,
        })

    return quality_rows


# ============================================================
# EXCEL
# ============================================================

CAPTURE_HEADERS = [
    "schema_version",
    "session_id",
    "capture_id",
    "timestamp",
    "distance_cm",
    "tfmini_strength",
    "distance_status",
    "ir_active",
    "ir_control_enabled",
    "ir_reason",
    "condiciones_personas",
    "rgb_personas_yolo",
    "noir_personas_yolo",
    "rgb_faces",
    "noir_faces",
    "rgb_objetos_validos",
    "noir_objetos_validos",
    "rgb_objetos_debiles",
    "noir_objetos_debiles",
    "rgb_brightness",
    "rgb_contrast",
    "rgb_sharpness",
    "noir_brightness",
    "noir_contrast",
    "noir_sharpness",
    "rgb_yolo_ms",
    "noir_yolo_ms",
    "rgb_original_path",
    "noir_original_path",
    "rgb_faces_strip_path",
    "noir_faces_strip_path",
    "rgb_foto",
    "noir_foto",
    "capture_source",
    "images_saved_count",
    "filters_saved_count",
    "analysis_status",
    "training_complete",
    "group_id",
    "selected_people_count",
    "valid_channels_count",
    "valid_channels_total",
    "valid_channels_ratio",
    "dataset_quality_status",
    "analysis_seq",
    "rgb_highres_archive_path",
    "noir_highres_archive_path",
]

PERSON_HEADERS = [
    "capture_id",
    "camera",
    "person_id",
    "source",
    "distance_cm",
    "yolo_person_confidence",

    "person_x1",
    "person_y1",
    "person_x2",
    "person_y2",

    "face_detected",
    "face_x",
    "face_y",
    "face_w",
    "face_h",
    "landmarks_detected",
    "landmarks_count",
    "landmark_source",

    "left_brow_height",
    "left_brow_width",
    "left_brow_slope_deg",
    "left_brow_eye_distance",

    "right_brow_height",
    "right_brow_width",
    "right_brow_slope_deg",
    "right_brow_eye_distance",

    "left_eye_width",
    "left_eye_height",
    "left_eye_ear",

    "right_eye_width",
    "right_eye_height",
    "right_eye_ear",

    "inter_eye_distance",

    "nose_width",
    "nose_height",

    "mouth_width",
    "mouth_height",
    "mouth_open",
    "smile_width",

    "face_width_px",
    "face_height_px",
    "face_ratio",
    "jaw_width",
    "head_roll_deg",

    "face_brightness",
    "face_contrast",
    "face_sharpness",

    "lentes",
    "barba",
    "cabello_frente",
    "condiciones_manual",
    "registration_status",
    "quality_status",
    "training_valid",
    "subject_group",
    "landmark_outlier_count",
    "landmark_outlier_ratio",
    "normalized_valid",
    "ir_active",

    "face_crop_path",
    "face_foto",
]

LANDMARK_HEADERS = [
    "capture_id",
    "camera",
    "person_id",
    "point_index",
    "region",
    "x_px",
    "y_px",
    "x_norm_face",
    "y_norm_face",

    # V16.15 - normalizacion geometrica automatica, sin etiquetas.
    "x_norm_eyes",
    "y_norm_eyes",

    # V16.19 - control IQR historico por landmark/fuente.
    "iqr_reference_n",
    "is_outlier_iqr",
]

OBJECT_HEADERS = [
    "capture_id",
    "camera",
    "detection_id",
    "distance_cm",
    "class_id",
    "class_name",
    "confidence",
    "detection_status",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "center_x",
    "center_y",
    "width",
    "height",
    "area_px",
    "is_person",
]

FILTER_NOIR_HEADERS = [
    "capture_id",
    "timestamp",
    "filter_index",
    "filter_id",
    "filter_name",
    "person_id",
    "faces_mediapipe",
    "landmarks_detected",
    "landmarks_count",
    "blendshape_count",
    "landmark_source",
    "left_eye_ear",
    "right_eye_ear",
    "mouth_open",
    "smile_width",
    "head_roll_deg",
    "brightness",
    "contrast",
    "sharpness",
    "lentes",
    "barba",
    "cabello_frente",
    "condiciones_manual",
    "registration_status",
    "quality_status",
    "training_valid",
    "group_id",
    "subject_group",
    "landmark_outlier_count",
    "landmark_outlier_ratio",
    "normalized_valid",
    "ir_active",
    "event_active",
    "activity_score",
    "last_event_type",
    "last_event_is_micro",
    "last_event_duration_ms",
    "last_event_region",
    "effective_fps",
    "full_image_path",
    "face_crop_path",
    "foto",
]

QUALITY_DATASET_HEADERS = [
    "schema_version",
    "session_id",
    "group_id",
    "capture_id",
    "person_id",
    "subject_group",
    "valid_channels_count",
    "valid_channels_total",
    "valid_channels_ratio",
    "landmark_rows",
    "outlier_count",
    "outlier_ratio",
    "history_reference_min_n",
    "quality_status",
    "recommended_for_training",
]


MICRO_EXCEL_HEADERS = [
    "schema_version",
    "session_id",
    "created_at",
    "source",
    "filter_name",
    "person_id",
    "track_id",
    "type",
    "is_micro_candidate",
    "duration_class",
    "onset_ms",
    "apex_ms",
    "offset_ms",
    "duration_ms",
    "peak_activity",
    "threshold",
    "top_region",
    "top_channels",
    "lentes",
    "barba",
    "cabello_frente",
    "condiciones_manual",
    "registration_status",
    "ir_active",
]



def style_header(ws):
    fill = PatternFill(
        "solid",
        fgColor="1F4E78"
    )

    font = Font(
        color="FFFFFF",
        bold=True
    )

    thin = Side(
        style="thin",
        color="D9E2F3"
    )

    for cell in ws[1]:
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True
        )
        cell.border = Border(
            left=thin,
            right=thin,
            top=thin,
            bottom=thin
        )

    ws.freeze_panes = "A2"


def _configure_filter_sheet(
    ws
):
    for col in range(
        1,
        len(
            FILTER_NOIR_HEADERS
        ) + 1
    ):
        ws.column_dimensions[
            get_column_letter(
                col
            )
        ].width = 18

    ws.column_dimensions[
        "A"
    ].width = 28

    ws.column_dimensions[
        "B"
    ].width = 24

    # Columnas principales por nombre para que si cambia el orden
    # no se rompa el ancho de rutas/fotos.
    for header_name, width in [
        ("filter_name", 30),
        ("filter_id", 20),
        ("micro_last_region", 20),
        ("full_image_path", 48),
        ("face_crop_path", 48),
        ("foto", 44),
    ]:
        if header_name in FILTER_NOIR_HEADERS:
            col_letter = get_column_letter(
                FILTER_NOIR_HEADERS.index(
                    header_name
                ) + 1
            )

            ws.column_dimensions[
                col_letter
            ].width = width


def create_workbook():
    wb = Workbook()

    captures = wb.active
    captures.title = "CAPTURAS"
    captures.append(
        CAPTURE_HEADERS
    )

    persons = wb.create_sheet(
        "PERSONAS"
    )
    persons.append(
        PERSON_HEADERS
    )

    landmarks = wb.create_sheet(
        "LANDMARKS"
    )
    landmarks.append(
        LANDMARK_HEADERS
    )

    objects = wb.create_sheet(
        "OBJETOS"
    )
    objects.append(
        OBJECT_HEADERS
    )

    filters_ws = wb.create_sheet(
        "FILTROS_NOIR"
    )
    filters_ws.append(
        FILTER_NOIR_HEADERS
    )

    quality_ws = wb.create_sheet(
        "CALIDAD_DATASET"
    )
    quality_ws.append(
        QUALITY_DATASET_HEADERS
    )

    micro_ws = wb.create_sheet(
        "MICROEXPRESIONES"
    )
    micro_ws.append(
        MICRO_EXCEL_HEADERS
    )

    for ws in [
        captures,
        persons,
        landmarks,
        objects,
        filters_ws,
        quality_ws,
        micro_ws,
    ]:
        style_header(ws)

    for col in range(
        1,
        len(CAPTURE_HEADERS) + 1
    ):
        captures.column_dimensions[
            get_column_letter(col)
        ].width = 16

    captures.column_dimensions[
        "A"
    ].width = 28

    captures.column_dimensions[
        "B"
    ].width = 24

    for header_name, width in [
        ("rgb_original_path", 42),
        ("noir_original_path", 42),
        ("rgb_faces_strip_path", 42),
        ("noir_faces_strip_path", 42),
        ("rgb_foto", 44),
        ("noir_foto", 44),
    ]:
        if header_name in CAPTURE_HEADERS:
            col_letter = get_column_letter(
                CAPTURE_HEADERS.index(
                    header_name
                ) + 1
            )

            captures.column_dimensions[
                col_letter
            ].width = width

    for ws, headers in [
        (
            persons,
            PERSON_HEADERS
        ),
        (
            landmarks,
            LANDMARK_HEADERS
        ),
        (
            objects,
            OBJECT_HEADERS
        ),
    ]:
        for col in range(
            1,
            len(headers) + 1
        ):
            ws.column_dimensions[
                get_column_letter(col)
            ].width = 18

        ws.column_dimensions[
            "A"
        ].width = 28

    if "face_crop_path" in PERSON_HEADERS:
        persons.column_dimensions[
            get_column_letter(
                PERSON_HEADERS.index(
                    "face_crop_path"
                ) + 1
            )
        ].width = 44

    if "face_foto" in PERSON_HEADERS:
        persons.column_dimensions[
            get_column_letter(
                PERSON_HEADERS.index(
                    "face_foto"
                ) + 1
            )
        ].width = 26

    _configure_filter_sheet(
        filters_ws
    )

    for col in range(1, len(QUALITY_DATASET_HEADERS) + 1):
        quality_ws.column_dimensions[get_column_letter(col)].width = 20
    quality_ws.column_dimensions["A"].width = 24
    quality_ws.column_dimensions["B"].width = 24
    quality_ws.column_dimensions["C"].width = 30
    quality_ws.column_dimensions["D"].width = 30

    for col in range(
        1,
        len(MICRO_EXCEL_HEADERS) + 1
    ):
        micro_ws.column_dimensions[
            get_column_letter(
                col
            )
        ].width = 20

    micro_ws.column_dimensions[
        "A"
    ].width = 26

    micro_ws.column_dimensions[
        "B"
    ].width = 20

    micro_ws.column_dimensions[
        "C"
    ].width = 28

    micro_ws.column_dimensions[
        "M"
    ].width = 55

    wb.save(
        str(EXCEL_FILE)
    )


def ensure_workbook():
    if not EXCEL_FILE.exists():
        create_workbook()
        return

    # Actualiza un Excel V9/V11 existente sin borrar sus filas.
    wb = load_workbook(
        str(EXCEL_FILE)
    )

    changed = False

    if "FILTROS_NOIR" not in wb.sheetnames:
        ws = wb.create_sheet(
            "FILTROS_NOIR"
        )

        ws.append(
            FILTER_NOIR_HEADERS
        )

        style_header(
            ws
        )

        _configure_filter_sheet(
            ws
        )

        changed = True

    if "CALIDAD_DATASET" not in wb.sheetnames:
        ws = wb.create_sheet(
            "CALIDAD_DATASET"
        )
        ws.append(QUALITY_DATASET_HEADERS)
        style_header(ws)
        for col in range(1, len(QUALITY_DATASET_HEADERS) + 1):
            ws.column_dimensions[get_column_letter(col)].width = 20
        changed = True

    if "MICROEXPRESIONES" not in wb.sheetnames:
        ws = wb.create_sheet(
            "MICROEXPRESIONES"
        )

        ws.append(
            MICRO_EXCEL_HEADERS
        )

        style_header(
            ws
        )

        changed = True

    if changed:
        wb.save(
            str(EXCEL_FILE)
        )

    wb.close()



def _excel_column_pixels(ws, col_index):
    """Aproximacion estable de ancho Excel (caracteres) a pixeles."""
    letter = get_column_letter(int(col_index))
    width = ws.column_dimensions[letter].width
    if width is None:
        width = 8.43
    return max(24, int(round(float(width) * 7.0 + 5.0)))


def _excel_row_pixels(ws, row_index):
    """Convierte la altura de fila (puntos) a pixeles."""
    height = ws.row_dimensions[int(row_index)].height
    if height is None:
        height = getattr(ws.sheet_format, "defaultRowHeight", None) or 15.0
    return max(20, int(points_to_pixels(float(height))))


def add_excel_image(
    ws,
    cell_name,
    image_path
):
    """
    Inserta la foto DENTRO de su celda, centrada horizontal y verticalmente,
    sin deformar la relacion de aspecto.
    """
    img = XLImage(str(image_path))

    original_w = max(1.0, float(img.width))
    original_h = max(1.0, float(img.height))

    row_index, col_index = coordinate_to_tuple(cell_name)
    cell_w = _excel_column_pixels(ws, col_index)
    cell_h = _excel_row_pixels(ws, row_index)

    # Deja un borde interior pequeño para que la imagen no toque la cuadricula.
    padding = 6
    max_w = max(20, cell_w - 2 * padding)
    max_h = max(20, cell_h - 2 * padding)

    scale = min(
        float(max_w) / original_w,
        float(max_h) / original_h,
        1.0,
    )

    display_w = max(1, int(round(original_w * scale)))
    display_h = max(1, int(round(original_h * scale)))

    img.width = display_w
    img.height = display_h

    offset_x = max(0, int((cell_w - display_w) / 2))
    offset_y = max(0, int((cell_h - display_h) / 2))

    marker = AnchorMarker(
        col=col_index - 1,
        row=row_index - 1,
        colOff=pixels_to_EMU(offset_x),
        rowOff=pixels_to_EMU(offset_y),
    )
    extent = XDRPositiveSize2D(
        cx=pixels_to_EMU(display_w),
        cy=pixels_to_EMU(display_h),
    )
    img.anchor = OneCellAnchor(_from=marker, ext=extent)
    ws.add_image(img)


def append_to_excel(
    capture_record,
    person_records,
    landmark_records,
    object_records,
    rgb_faces_strip_path,
    noir_faces_strip_path,
    noir_filter_records,
    quality_records,
):
    """
    V16.15:
    - una sola escritura Excel a la vez;
    - las fotos embebidas nunca cancelan las filas;
    - verifica que capture_id realmente exista despues del save;
    - si el XLSX esta abierto/bloqueado, continua en un XLSX RECUPERACION.
    """
    capture_id = str(capture_record.get("capture_id", "SIN_ID"))

    with EXCEL_WRITE_LOCK:
        target = _active_excel_path()

        # Primera ejecucion: crear estructura normal.
        if target == EXCEL_FILE:
            ensure_workbook()
        elif not target.exists():
            # Caso extremo: se perdio la ruta activa. Recuperar desde el principal.
            if EXCEL_FILE.exists():
                shutil.copy2(str(EXCEL_FILE), str(target))
            else:
                ensure_workbook()
                target = EXCEL_FILE

        wb = load_workbook(str(target))

        try:
            captures = wb["CAPTURAS"]
            capture_row = captures.max_row + 1

            capture_values = [capture_record.get(key) for key in CAPTURE_HEADERS]
            if "rgb_foto" in CAPTURE_HEADERS:
                capture_values[CAPTURE_HEADERS.index("rgb_foto")] = ""
            if "noir_foto" in CAPTURE_HEADERS:
                capture_values[CAPTURE_HEADERS.index("noir_foto")] = ""
            captures.append(capture_values)
            captures.row_dimensions[capture_row].height = 145

            # Las filas ya existen en memoria. Las fotos son opcionales.
            if "rgb_foto" in CAPTURE_HEADERS:
                col = get_column_letter(CAPTURE_HEADERS.index("rgb_foto") + 1)
                _safe_excel_image(
                    captures,
                    "{}{}".format(col, capture_row),
                    capture_record.get("rgb_original_path")
                )
            if "noir_foto" in CAPTURE_HEADERS:
                col = get_column_letter(CAPTURE_HEADERS.index("noir_foto") + 1)
                _safe_excel_image(
                    captures,
                    "{}{}".format(col, capture_row),
                    capture_record.get("noir_original_path")
                )

            for cell in captures[capture_row]:
                cell.alignment = Alignment(
                    horizontal="center", vertical="center", wrap_text=True
                )

            persons_ws = wb["PERSONAS"]
            for record in person_records:
                rr = persons_ws.max_row + 1
                values = [record.get(key) for key in PERSON_HEADERS]
                if "face_foto" in PERSON_HEADERS:
                    values[PERSON_HEADERS.index("face_foto")] = ""
                persons_ws.append(values)
                persons_ws.row_dimensions[rr].height = 125

                if "face_foto" in PERSON_HEADERS:
                    col = get_column_letter(PERSON_HEADERS.index("face_foto") + 1)
                    _safe_excel_image(
                        persons_ws,
                        "{}{}".format(col, rr),
                        record.get("face_crop_path")
                    )

                for cell in persons_ws[rr]:
                    cell.alignment = Alignment(
                        horizontal="center", vertical="center", wrap_text=True
                    )

            landmarks_ws = wb["LANDMARKS"]
            for record in landmark_records:
                landmarks_ws.append([record.get(key) for key in LANDMARK_HEADERS])

            objects_ws = wb["OBJETOS"]
            for record in object_records:
                objects_ws.append([record.get(key) for key in OBJECT_HEADERS])

            filters_ws = wb["FILTROS_NOIR"]
            for record in noir_filter_records:
                rr = filters_ws.max_row + 1
                values = [record.get(key) for key in FILTER_NOIR_HEADERS]
                if "foto" in FILTER_NOIR_HEADERS:
                    values[FILTER_NOIR_HEADERS.index("foto")] = ""
                filters_ws.append(values)
                filters_ws.row_dimensions[rr].height = 145

                if "foto" in FILTER_NOIR_HEADERS:
                    col = get_column_letter(FILTER_NOIR_HEADERS.index("foto") + 1)
                    _safe_excel_image(
                        filters_ws,
                        "{}{}".format(col, rr),
                        record.get("full_image_path")
                    )

                for cell in filters_ws[rr]:
                    cell.alignment = Alignment(
                        horizontal="center", vertical="center", wrap_text=True
                    )

            quality_ws = wb["CALIDAD_DATASET"]
            for record in quality_records:
                quality_ws.append([record.get(key) for key in QUALITY_DATASET_HEADERS])

            micro_ws = wb["MICROEXPRESIONES"]
            micro_events = drain_micro_event_buffer()
            for event in micro_events:
                excel_row = []
                for key in MICRO_EXCEL_HEADERS:
                    value = event.get(key)
                    if key == "top_channels":
                        value = json.dumps(
                            value if value is not None else [],
                            ensure_ascii=False
                        )
                    excel_row.append(value)
                micro_ws.append(excel_row)

            saved_path = _save_workbook_robust(
                wb,
                target,
                capture_id
            )

        finally:
            try:
                wb.close()
            except Exception:
                pass

        # VERIFICACION REAL DEL ARCHIVO YA GUARDADO.
        verify = load_workbook(str(saved_path), read_only=True, data_only=True)
        try:
            ws = verify["CAPTURAS"]
            capture_col = CAPTURE_HEADERS.index("capture_id") + 1
            found = False
            for rr in range(max(2, ws.max_row - 10), ws.max_row + 1):
                if str(ws.cell(rr, capture_col).value) == capture_id:
                    found = True
                    break
            if not found:
                raise RuntimeError(
                    "Excel guardado pero capture_id {} no aparece en CAPTURAS".format(
                        capture_id
                    )
                )
        finally:
            verify.close()

        print(
            "[EXCEL OK]",
            capture_id,
            "| CAPTURAS fila", capture_row,
            "| PERSONAS +", len(person_records),
            "| LANDMARKS +", len(landmark_records),
            "| OBJETOS +", len(object_records),
            "| FILTROS +", len(noir_filter_records),
            "| MICRO +", len(micro_events),
            "|", saved_path
        )

        try:
            LAST_EXCEL_PATH_FILE.write_text(str(saved_path), encoding="utf-8")
        except Exception:
            pass

        return str(saved_path)


# ============================================================
# RECORD BUILDERS
# ============================================================

def build_object_records(
    capture_id,
    camera_name,
    distance_cm,
    detections,
    valid_conf,
    weak_conf
):
    records = []

    dets = sorted(
        detections,
        key=lambda d: float(
            d.get(
                "confidence",
                0.0
            )
        ),
        reverse=True
    )

    for idx, det in enumerate(
        dets,
        start=1
    ):
        conf = float(
            det.get(
                "confidence",
                0.0
            )
        )

        if conf < weak_conf:
            continue

        box = det.get(
            "box",
            []
        )

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]

        width = max(
            0,
            x2 - x1
        )

        height = max(
            0,
            y2 - y1
        )

        name = str(
            det.get(
                "class_name",
                ""
            )
        )

        records.append(
            {
                "capture_id": capture_id,
                "camera": camera_name,
                "detection_id": (
                    "D{:03d}"
                    .format(idx)
                ),
                "distance_cm": distance_cm,
                "class_id": int(
                    det.get(
                        "class_id",
                        -1
                    )
                ),
                "class_name": name,
                "confidence": round(
                    conf,
                    5
                ),
                "detection_status": (
                    "VALID"
                    if conf >= effective_valid_conf_for_detection(
                        det,
                        valid_conf
                    )
                    else "WEAK"
                ),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "center_x": round(
                    (
                        x1 + x2
                    )
                    / 2.0,
                    3
                ),
                "center_y": round(
                    (
                        y1 + y2
                    )
                    / 2.0,
                    3
                ),
                "width": width,
                "height": height,
                "area_px": (
                    width * height
                ),
                "is_person": int(
                    name.lower()
                    == "person"
                ),
            }
        )

    return records


def build_person_records(
    capture_id,
    camera_name,
    distance_cm,
    frame,
    detections,
    faces,
    valid_conf,
    event_dir,
):
    associations = (
        associate_faces_to_persons(
            detections,
            faces,
            valid_conf
        )
    )

    person_records = []
    landmark_records = []

    for item in associations:
        record = {
            key: None
            for key in PERSON_HEADERS
        }

        record[
            "capture_id"
        ] = capture_id

        record[
            "camera"
        ] = camera_name

        record[
            "person_id"
        ] = item[
            "person_id"
        ]

        person_state = get_person_conditions(
            item[
                "person_id"
            ]
        )

        record[
            "source"
        ] = item[
            "source"
        ]

        record[
            "distance_cm"
        ] = distance_cm

        conf = item.get(
            "yolo_person_confidence"
        )

        record[
            "yolo_person_confidence"
        ] = (
            round(
                conf,
                5
            )
            if conf is not None
            else None
        )

        pbox = item.get(
            "person_box"
        )

        if pbox:
            record[
                "person_x1"
            ] = pbox[0]

            record[
                "person_y1"
            ] = pbox[1]

            record[
                "person_x2"
            ] = pbox[2]

            record[
                "person_y2"
            ] = pbox[3]

        face = item.get(
            "face"
        )

        record[
            "face_detected"
        ] = int(
            face is not None
        )

        if face is None:
            record[
                "landmarks_detected"
            ] = 0

            record[
                "landmarks_count"
            ] = 0
            record["landmark_source"] = "NONE"

            person_records.append(
                record
            )
            continue

        x, y, w, h = face[
            "bbox"
        ]

        record[
            "face_x"
        ] = x

        record[
            "face_y"
        ] = y

        record[
            "face_w"
        ] = w

        record[
            "face_h"
        ] = h

        lm_ok = bool(
            face.get(
                "landmarks_detected"
            )
        )

        record[
            "landmarks_detected"
        ] = int(lm_ok)

        landmarks = face.get(
            "landmarks",
            []
        )

        record[
            "landmarks_count"
        ] = (
            len(landmarks)
            if lm_ok
            else 0
        )

        record["landmark_source"] = (
            str(face.get("landmark_recovery") or "LIVE")
            if lm_ok else "NONE"
        )

        for key, value in face.get(
            "metrics",
            {}
        ).items():
            if key in record:
                record[
                    key
                ] = value

        quality = image_quality(
            frame,
            [
                x, y, w, h
            ]
        )

        record[
            "face_brightness"
        ] = quality[
            "brightness"
        ]

        record[
            "face_contrast"
        ] = quality[
            "contrast"
        ]

        record[
            "face_sharpness"
        ] = quality[
            "sharpness"
        ]

        record[
            "lentes"
        ] = int(
            bool(
                person_state.get(
                    "lentes"
                )
            )
        )

        record[
            "barba"
        ] = int(
            bool(
                person_state.get(
                    "barba"
                )
            )
        )

        record[
            "cabello_frente"
        ] = int(
            bool(
                person_state.get(
                    "cabello_frente"
                )
            )
        )

        record[
            "condiciones_manual"
        ] = person_conditions_summary(
            person_state
        )
        face_q = face.get("quality_gate", {})
        record["registration_status"] = get_registration_status(item["person_id"])
        record["quality_status"] = face_q.get("status", "UNKNOWN")
        record["training_valid"] = int(bool(face_q.get("training_valid", 0)) and person_is_registered(item["person_id"]))
        with latest_telemetry_lock:
            _person_telem = dict(latest_telemetry)
        record["ir_active"] = int(bool(_person_telem.get("ir_active", False)))

        face_crop = crop_face_for_dataset(
            frame,
            [x, y, w, h],
            output_size=None
        )

        face_crop_path = None

        if face_crop is not None:
            faces_dir = (
                event_dir
                / "faces"
            )

            faces_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            face_crop_path = (
                faces_dir
                / "{}_{}_face.png".format(
                    camera_name,
                    item[
                        "person_id"
                    ]
                )
            )

            save_image(
                face_crop_path,
                face_crop
            )

        record[
            "face_crop_path"
        ] = (
            str(
                face_crop_path
            )
            if face_crop_path is not None
            else None
        )

        record[
            "face_foto"
        ] = ""

        if lm_ok:
            pts = np.asarray(
                landmarks,
                dtype=np.float32
            )

            fw = max(
                1.0,
                float(w)
            )

            fh = max(
                1.0,
                float(h)
            )

            for idx, (
                px,
                py
            ) in enumerate(pts):
                landmark_records.append(
                    {
                        "capture_id": capture_id,
                        "camera": camera_name,
                        "person_id": item[
                            "person_id"
                        ],
                        "point_index": idx,
                        "region": landmark_region(
                            idx
                        ),
                        "x_px": round(
                            float(px),
                            3
                        ),
                        "y_px": round(
                            float(py),
                            3
                        ),
                        "x_norm_face": round(
                            (
                                float(px)
                                - x
                            )
                            / fw,
                            6
                        ),
                        "y_norm_face": round(
                            (
                                float(py)
                                - y
                            )
                            / fh,
                            6
                        ),
                        "x_norm_eyes": None,
                        "y_norm_eyes": None,
                    }
                )

        person_records.append(
            record
        )

    return (
        person_records,
        landmark_records,
        associations
    )


# ============================================================
# DATASET PROCESSING
# ============================================================

def save_image(
    path,
    image
):
    if not cv2.imwrite(
        str(path),
        image
    ):
        raise RuntimeError(
            "No se pudo guardar {}".format(
                path
            )
        )





# ============================================================
# V16.11 ESTABLE - CAPTURE FIRST, ANALYSIS SECOND
# ============================================================
# Esta definicion reemplaza la funcion legacy anterior en tiempo de ejecucion.
# Regla principal:
#   1) guardar RGB + NoIR + 6 filtros SIEMPRE primero;
#   2) construir CAPTURAS aunque falle el analisis facial;
#   3) usar placeholders con training_valid=0; nunca inventar medidas.

V1610_FILTER_EXPORT_FILENAMES = [
    "01_NOIR_original.png",
    "02_NOIR_red_bandpass_virtual.png",
    "03_NOIR_ir_cut_rgb_virtual.png",
    "04_NOIR_blue_bandpass_virtual.png",
    "05_NOIR_ir_bandpass_virtual.png",
    "06_NOIR_uv_bandpass_virtual.png",
]


def _v1610_ir_value(tfmini, incoming):
    value = tfmini.get("ir_active", incoming.get("ir_active"))
    if value is None:
        return None
    return int(bool(value))


def _v1610_detect_robust(frame, engine):
    """Detecta en original; si falla, prueba 480 px y reescala al original."""
    if frame is None:
        return []

    try:
        faces = engine.detect_all(frame)
    except Exception as exc:
        print("[V16.11] MediaPipe original fallo:", repr(exc))
        faces = []

    if faces:
        try:
            return apply_quality_gate(frame, faces)
        except Exception:
            return faces

    h, w = frame.shape[:2]
    if w <= 500:
        return []

    target_w = 480
    scale = float(target_w) / float(w)
    target_h = max(1, int(round(h * scale)))
    small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    try:
        small_faces = engine.detect_all(small)
    except Exception as exc:
        print("[V16.11] MediaPipe reducido fallo:", repr(exc))
        return []

    if not small_faces:
        return []

    inv = 1.0 / scale
    out = []
    for face in small_faces:
        f = copy.deepcopy(face)
        bbox = f.get("bbox")
        if bbox and len(bbox) == 4:
            f["bbox"] = [
                int(round(float(bbox[0]) * inv)),
                int(round(float(bbox[1]) * inv)),
                int(round(float(bbox[2]) * inv)),
                int(round(float(bbox[3]) * inv)),
            ]
        landmarks = f.get("landmarks")
        if landmarks is not None:
            try:
                f["landmarks"] = [
                    [float(p[0]) * inv, float(p[1]) * inv]
                    for p in landmarks
                ]
            except Exception:
                pass
        out.append(f)

    try:
        out = apply_quality_gate(frame, out)
    except Exception:
        pass
    return out



def _v1617_has_landmarks(face):
    if not isinstance(face, dict):
        return False
    landmarks = face.get("landmarks")
    if landmarks is None:
        return False
    try:
        return bool(face.get("landmarks_detected", False)) and len(landmarks) >= 68
    except Exception:
        return False


def _v1617_face_area(face):
    try:
        x, y, w, h = face.get("bbox", [0, 0, 0, 0])
        return max(0.0, float(w)) * max(0.0, float(h))
    except Exception:
        return 0.0


def _v1617_roi_from_person_box(frame_shape, person_box):
    """
    Convierte una caja YOLO [x1,y1,x2,y2] de persona en una ROI enfocada
    en cabeza/parte superior del torso. La ROI solo se usa para MediaPipe;
    los puntos se remapean despues al frame completo.
    """
    if not person_box or len(person_box) != 4:
        return None
    fh, fw = frame_shape[:2]
    try:
        x1, y1, x2, y2 = [float(v) for v in person_box]
    except Exception:
        return None
    pw = max(1.0, x2 - x1)
    ph = max(1.0, y2 - y1)

    # Cabeza suele ocupar la zona superior de la caja de persona.
    # Se deja margen lateral y superior para pelo/orejas y algo de hombros.
    rx1 = int(round(x1 - 0.16 * pw))
    rx2 = int(round(x2 + 0.16 * pw))
    ry1 = int(round(y1 - 0.10 * ph))
    ry2 = int(round(y1 + 0.62 * ph))

    rx1 = max(0, min(fw - 1, rx1))
    ry1 = max(0, min(fh - 1, ry1))
    rx2 = max(rx1 + 2, min(fw, rx2))
    ry2 = max(ry1 + 2, min(fh, ry2))

    if (rx2 - rx1) < 40 or (ry2 - ry1) < 40:
        return None
    return [rx1, ry1, rx2, ry2]


def _v1617_map_face_from_roi(face, roi_box, scale, frame):
    """Remapea bbox + 68 landmarks de la ROI al sistema del frame original."""
    f = copy.deepcopy(face)
    rx1, ry1, _rx2, _ry2 = roi_box
    inv = 1.0 / float(scale)

    bbox = f.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    bx, by, bw, bh = [float(v) for v in bbox]
    mapped_bbox = [
        int(round(rx1 + bx * inv)),
        int(round(ry1 + by * inv)),
        int(round(bw * inv)),
        int(round(bh * inv)),
    ]
    f["bbox"] = mapped_bbox

    landmarks = f.get("landmarks")
    if landmarks is None:
        return None
    try:
        pts = np.asarray(landmarks, dtype=np.float32).reshape((-1, 2))
    except Exception:
        return None
    if len(pts) < 68:
        return None

    pts[:, 0] = float(rx1) + pts[:, 0] * inv
    pts[:, 1] = float(ry1) + pts[:, 1] * inv
    fh, fw = frame.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, max(0, fw - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0, max(0, fh - 1))
    f["landmarks"] = pts
    f["landmarks_detected"] = True
    try:
        f["metrics"] = extract_face_metrics(pts, mapped_bbox)
    except Exception:
        pass
    f["landmark_recovery"] = "YOLO_ROI"
    return f


def _v1617_detect_in_yolo_roi(frame, engine, person_box, person_id, source_name):
    """
    Intenta MediaPipe sobre una ROI de YOLO. Primero a tamano natural y,
    si no hay rostro, amplia la ROI para darle mas pixeles al landmarker.
    """
    roi_box = _v1617_roi_from_person_box(frame.shape, person_box)
    if roi_box is None:
        return None
    x1, y1, x2, y2 = roi_box
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    # 1x primero. Si la cara es pequena, 1.6x y 2.0x ayudan a MediaPipe.
    attempts = [1.0, 1.6, 2.0]
    for scale in attempts:
        if scale == 1.0:
            work = crop
        else:
            work = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        try:
            faces = engine.detect_all(work)
        except Exception as exc:
            print("[V16.17 ROI] {} {} x{} fallo: {}".format(
                source_name, person_id, scale, repr(exc)
            ))
            faces = []
        if not faces:
            continue

        # En una ROI de una persona, el rostro real suele ser la cara mas grande.
        faces = sorted(faces, key=_v1617_face_area, reverse=True)
        for face in faces:
            mapped = _v1617_map_face_from_roi(face, roi_box, scale, frame)
            if mapped is None or not _v1617_has_landmarks(mapped):
                continue
            mapped["person_id"] = str(person_id)
            try:
                mapped = apply_quality_gate(frame, [mapped])[0]
            except Exception:
                pass
            print("[V16.17 ROI] {} {} recuperado con ROI x{}".format(
                source_name, person_id, scale
            ))
            return mapped
    return None


def _v1617_merge_recovered(result, recovered_by_id):
    """Reemplaza solamente Pxx sin landmarks y conserva todo resultado valido."""
    merged = []
    seen = set()
    for face in result:
        pid = str(face.get("person_id") or "")
        if pid in recovered_by_id and not _v1617_has_landmarks(face):
            merged.append(copy.deepcopy(recovered_by_id[pid]))
            seen.add(pid)
        else:
            merged.append(face)
            if pid:
                seen.add(pid)
    for pid, face in recovered_by_id.items():
        if pid not in seen:
            merged.append(copy.deepcopy(face))
    return merged


def _v1617_recover_landmarks(
    frame,
    current_faces,
    engine,
    selected_ids,
    references,
    source_name,
    detections=None,
    valid_conf=None,
):
    """
    V16.17 LANDMARK ROI FIX.

    Orden:
      1) conservar landmarks LIVE;
      2) reintentar MediaPipe en el frame completo (V16.16);
      3) para Pxx que aun falten, usar caja YOLO -> ROI cabeza/torso superior;
      4) si hace falta, ampliar ROI 1.6x/2x;
      5) remapear los 68 puntos al frame original.

    Nunca inventa landmarks y nunca reemplaza un resultado LIVE valido.
    """
    result = [copy.deepcopy(f) for f in (current_faces or []) if isinstance(f, dict)]
    selected = [str(x) for x in (selected_ids or []) if x]
    if not selected or frame is None:
        return result

    by_id = {str(f.get("person_id")): f for f in result if f.get("person_id")}
    missing = [pid for pid in selected if not _v1617_has_landmarks(by_id.get(pid))]
    if not missing:
        return result

    # --------------------------------------------------------
    # NIVEL 2: segundo intento sobre frame completo.
    # --------------------------------------------------------
    try:
        detected = _v1610_detect_robust(frame, engine)
    except Exception as exc:
        print("[V16.17 LANDMARK] {} frame completo fallo: {}".format(source_name, repr(exc)))
        detected = []

    full_recovered = {}
    if detected:
        refs = [copy.deepcopy(r) for r in (references or []) if r.get("person_id")]
        refs = [r for r in refs if str(r.get("person_id")) in set(selected)]
        try:
            assigned = assign_faces_from_references(
                detected, frame.shape, refs, fallback_new=False
            )
        except Exception as exc:
            print("[V16.17 LANDMARK] {} asignacion frame fallo: {}".format(source_name, repr(exc)))
            assigned = [copy.deepcopy(f) for f in detected]

        leftovers = []
        for face in assigned:
            pid = str(face.get("person_id") or "")
            if pid in selected and _v1617_has_landmarks(face):
                full_recovered[pid] = face
            elif _v1617_has_landmarks(face):
                leftovers.append(face)
        still = [pid for pid in missing if pid not in full_recovered]
        leftovers.sort(key=lambda f: float((f.get("bbox") or [0, 0, 0, 0])[0]))
        for pid, face in zip(still, leftovers):
            face = copy.deepcopy(face)
            face["person_id"] = pid
            face["landmark_recovery"] = "FULL_FRAME_RETRY"
            full_recovered[pid] = face

    if full_recovered:
        result = _v1617_merge_recovered(result, full_recovered)

    by_id = {str(f.get("person_id")): f for f in result if f.get("person_id")}
    missing = [pid for pid in selected if not _v1617_has_landmarks(by_id.get(pid))]
    if not missing:
        print("[V16.17 LANDMARK] {} recuperados frame completo: {}".format(
            source_name, ", ".join(sorted(full_recovered))
        ))
        return result

    # --------------------------------------------------------
    # NIVEL 3: ROI de la persona detectada por YOLO.
    # --------------------------------------------------------
    detections = detections or []
    if not detections or valid_conf is None:
        return result

    try:
        yolo_map = _v1614_yolo_map_for_ids(
            detections, valid_conf, missing, frame.shape
        )
    except Exception as exc:
        print("[V16.17 ROI] {} mapeo YOLO fallo: {}".format(source_name, repr(exc)))
        yolo_map = {}

    roi_recovered = {}
    for pid in missing:
        info = yolo_map.get(pid)
        if not info:
            continue
        face = _v1617_detect_in_yolo_roi(
            frame,
            engine,
            info.get("box"),
            pid,
            source_name,
        )
        if face is not None and _v1617_has_landmarks(face):
            roi_recovered[pid] = face

    if roi_recovered:
        result = _v1617_merge_recovered(result, roi_recovered)
        print("[V16.17 LANDMARK] {} recuperados ROI: {}".format(
            source_name, ", ".join(sorted(roi_recovered))
        ))

    return result

def _v1610_person_placeholder(capture_id, camera, person_id, distance_cm, ir_value):
    row = {key: None for key in PERSON_HEADERS}
    state = get_person_conditions(person_id)
    row.update({
        "capture_id": capture_id,
        "camera": camera,
        "person_id": person_id,
        "source": "FACE_NOT_DETECTED",
        "distance_cm": distance_cm,
        "face_detected": 0,
        "landmarks_detected": 0,
        "landmarks_count": 0,
        "landmark_source": "NONE",
        "lentes": int(bool(state.get("lentes"))),
        "barba": int(bool(state.get("barba"))),
        "cabello_frente": int(bool(state.get("cabello_frente"))),
        "condiciones_manual": person_conditions_summary(state),
        "registration_status": get_registration_status(person_id),
        "quality_status": "RED",
        "training_valid": 0,
        "ir_active": ir_value,
        "face_crop_path": None,
        "face_foto": "",
    })
    return row


def _v1610_filter_placeholder(
    capture_id, created, filter_index, filter_id, filter_name,
    person_id, full_image_path, ir_value
):
    state = get_person_conditions(person_id) if person_id else default_person_conditions()
    row = {key: None for key in FILTER_NOIR_HEADERS}
    row.update({
        "capture_id": capture_id,
        "timestamp": created,
        "filter_index": filter_index + 1,
        "filter_id": filter_id,
        "filter_name": filter_name,
        "person_id": person_id,
        "faces_mediapipe": 0,
        "landmarks_detected": 0,
        "landmarks_count": 0,
        "blendshape_count": 0,
        "landmark_source": "NONE",
        "lentes": int(bool(state.get("lentes"))) if person_id else None,
        "barba": int(bool(state.get("barba"))) if person_id else None,
        "cabello_frente": int(bool(state.get("cabello_frente"))) if person_id else None,
        "condiciones_manual": person_conditions_summary(state) if person_id else "SIN_ROSTRO",
        "registration_status": get_registration_status(person_id) if person_id else "SIN_ROSTRO",
        "quality_status": "RED",
        "training_valid": 0,
        "ir_active": ir_value,
        "event_active": False,
        "activity_score": 0.0,
        "last_event_type": "",
        "last_event_is_micro": 0,
        "last_event_duration_ms": "",
        "last_event_region": "",
        "effective_fps": 0.0,
        "full_image_path": str(full_image_path),
        "face_crop_path": None,
        "foto": "",
    })
    return row


def _v1610_ensure_person_rows(records, capture_id, camera, selected_ids, distance_cm, ir_value):
    existing = {str(r.get("person_id")) for r in records if r.get("person_id")}
    for pid in selected_ids:
        if str(pid) not in existing:
            records.append(
                _v1610_person_placeholder(
                    capture_id, camera, str(pid), distance_cm, ir_value
                )
            )
    return records


def _v1610_safe_count_status(records, status):
    return sum(1 for r in (records or []) if r.get("detection_status") == status)




# ============================================================
# V16.19 - PERSONAS + FILTROS NOIR: ASIGNACION ESTRICTA 1-A-1
# ============================================================

def _v1618_is_real_person_id(value):
    """Acepta P01, P02... y rechaza IDs temporales PX01."""
    pid = str(value or "")
    return (
        len(pid) >= 2
        and pid.startswith("P")
        and not pid.startswith("PX")
        and pid[1:].isdigit()
    )


def _v1618_face_bbox(face):
    try:
        x, y, w, h = [float(v) for v in face.get("bbox", [0, 0, 0, 0])]
        return [x, y, max(0.0, w), max(0.0, h)]
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def _v1618_face_center(face):
    x, y, w, h = _v1618_face_bbox(face)
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)


def _v1618_bbox_iou_face(a, b):
    ax, ay, aw, ah = _v1618_face_bbox(a)
    bx, by, bw, bh = _v1618_face_bbox(b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 1e-6 else 0.0


def _v1618_dedupe_faces(faces):
    """
    MediaPipe puede devolver la misma cara dos veces en variantes NoIR.
    Conserva una sola si las cajas se superponen casi por completo.
    """
    candidates = [
        copy.deepcopy(f)
        for f in (faces or [])
        if isinstance(f, dict)
    ]
    candidates.sort(
        key=lambda f: (
            1 if _v1617_has_landmarks(f) else 0,
            _v1617_face_area(f),
        ),
        reverse=True,
    )

    kept = []
    for face in candidates:
        duplicate = False
        for prev in kept:
            if _v1618_bbox_iou_face(face, prev) >= 0.72:
                duplicate = True
                break
        if not duplicate:
            kept.append(face)
    return kept


def _v1618_strict_assign_selected(
    frame,
    faces,
    selected_ids,
    references,
    detections,
    valid_conf,
    source_name,
):
    """
    Asigna rostros SOLO a Pxx seleccionados y de forma uno-a-uno.

    Prioridad:
      1. ID Pxx valido ya presente.
      2. Referencia espacial del RGB LIVE.
      3. Caja YOLO de esa persona.
      4. Si queda exactamente 1 rostro y 1 Pxx, emparejar.

    Nunca crea PX01 y nunca usa un mismo rostro para dos Pxx.
    """
    selected = []
    for value in (selected_ids or []):
        pid = str(value)
        if _v1618_is_real_person_id(pid) and pid not in selected:
            selected.append(pid)

    if not selected:
        return []

    pool = _v1618_dedupe_faces(faces)
    if not pool:
        return []

    fh, fw = frame.shape[:2]
    diag = max(1.0, math.hypot(float(fw), float(fh)))

    refs = {}
    for ref in (references or []):
        pid = str(ref.get("person_id") or "")
        cn = ref.get("center_norm")
        if pid in selected and cn is not None and len(cn) >= 2:
            try:
                refs[pid] = np.array(
                    [float(cn[0]) * fw, float(cn[1]) * fh],
                    dtype=np.float32,
                )
            except Exception:
                pass

    try:
        yolo_map = _v1614_yolo_map_for_ids(
            detections or [], valid_conf, selected, frame.shape
        )
    except Exception:
        yolo_map = {}

    assigned = {}
    used = set()

    # 1) Conservar IDs Pxx validos solo una vez.
    for idx, face in enumerate(pool):
        pid = str(face.get("person_id") or "")
        if pid in selected and pid not in assigned:
            assigned[pid] = idx
            used.add(idx)

    # 2/3) Referencias + YOLO.
    for pid in selected:
        if pid in assigned:
            continue

        best = None
        for idx, face in enumerate(pool):
            if idx in used:
                continue
            center = _v1618_face_center(face)
            score = 10.0
            evidence = False

            if pid in refs:
                d = float(np.linalg.norm(center - refs[pid]) / diag)
                score = min(score, d)
                if d <= 0.30:
                    evidence = True

            info = yolo_map.get(pid)
            if info:
                box = info.get("box") or [0, 0, 0, 0]
                x1, y1, x2, y2 = [float(v) for v in box]
                pw = max(1.0, x2 - x1)
                ph = max(1.0, y2 - y1)
                inside = (
                    x1 - 0.12 * pw <= center[0] <= x2 + 0.12 * pw
                    and y1 - 0.12 * ph <= center[1] <= y2 + 0.12 * ph
                )
                head_center = np.array(
                    [(x1 + x2) * 0.5, y1 + 0.30 * ph],
                    dtype=np.float32,
                )
                yd = float(np.linalg.norm(center - head_center) / diag)
                if inside:
                    evidence = True
                    score = min(score, 0.05 + yd)

            if evidence and (best is None or score < best[0]):
                best = (score, idx)

        if best is not None:
            assigned[pid] = best[1]
            used.add(best[1])

    # 4) Solo un caso inequívoco restante.
    remaining_pids = [pid for pid in selected if pid not in assigned]
    remaining_faces = [idx for idx in range(len(pool)) if idx not in used]
    if len(remaining_pids) == 1 and len(remaining_faces) == 1:
        assigned[remaining_pids[0]] = remaining_faces[0]
        used.add(remaining_faces[0])

    out = []
    for pid in selected:
        idx = assigned.get(pid)
        if idx is None:
            continue
        face = copy.deepcopy(pool[idx])
        face["person_id"] = pid
        face["identity_assignment"] = "STRICT_1_TO_1"
        out.append(face)

    if len(pool) != len(out):
        print(
            "[V16.18 PERSONAS] {} rostros={} asignados={} Pxx={}".format(
                source_name, len(pool), len(out), ",".join(selected)
            )
        )
    return out


def _v1618_filter_placeholder(
    capture_id, created, filter_index, filter_id, filter_name,
    person_id, full_image_path, ir_value, filtered, faces_count
):
    row = _v1610_filter_placeholder(
        capture_id, created, filter_index, filter_id, filter_name,
        person_id, full_image_path, ir_value
    )
    # Calidad a nivel de imagen: es real incluso si no hubo rostro.
    try:
        q = image_quality(filtered)
    except Exception:
        q = {}
    row["brightness"] = q.get("brightness")
    row["contrast"] = q.get("contrast")
    row["sharpness"] = q.get("sharpness")
    row["faces_mediapipe"] = int(faces_count or 0)
    row["landmark_source"] = "NONE"
    return row


def process_dataset_zip(payload):
    """
    UNA SOLA ruta de guardado V16.18.

    No vuelve a correr MediaPipe. Los landmarks/blendshapes vienen del mismo
    frame LIVE que ya fue validado antes de crear el payload.
    """
    with zipfile.ZipFile(io.BytesIO(payload), "r") as zf:
        incoming = json.loads(zf.read("metadata.json").decode("utf-8"))
        capture_id = str(incoming["capture_id"])
        created = incoming.get(
            "created_at", datetime.now().isoformat(timespec="milliseconds")
        )

        event_dir = CAPTURES_DIR / capture_id
        if event_dir.exists():
            raise FileExistsError("ID ya existe: " + capture_id)
        event_dir.mkdir(parents=True)

        rgb = cv2.imdecode(
            np.frombuffer(zf.read("images/" + incoming["rgb_filename"]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        noir = cv2.imdecode(
            np.frombuffer(zf.read("images/" + incoming["noir_filename"]), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if rgb is None or noir is None:
            raise RuntimeError("Frame validado RGB/NoIR invalido")

        # Archivos de alta resolucion son SOLO archivo; los datos faciales se
        # miden sobre rgb/noir, que son exactamente los frames validados.
        rgb_highres_path = None
        noir_highres_path = None
        raw_rgb_name = incoming.get("raw_rgb_filename")
        raw_noir_name = incoming.get("raw_noir_filename")
        if raw_rgb_name and ("images/" + raw_rgb_name) in zf.namelist():
            raw = cv2.imdecode(
                np.frombuffer(zf.read("images/" + raw_rgb_name), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if raw is not None:
                rgb_highres_path = event_dir / "RGB_highres_archive.png"
                save_image(rgb_highres_path, raw)
        if raw_noir_name and ("images/" + raw_noir_name) in zf.namelist():
            raw = cv2.imdecode(
                np.frombuffer(zf.read("images/" + raw_noir_name), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if raw is not None:
                noir_highres_path = event_dir / "NOIR_highres_archive.png"
                save_image(noir_highres_path, raw)

        tfmini = incoming.get("tfmini", {}) or {}
        distance_cm = tfmini.get("distance_cm")
        ir_value = _v1610_ir_value(tfmini, incoming)
        selected_ids = [str(x) for x in incoming.get("selected_person_ids", []) if x]
        selected_set = set(selected_ids)

        analysis = incoming.get("analysis_faces", {}) or {}
        rgb_faces = copy.deepcopy(analysis.get("RGB", []) or [])
        filter_faces_map = copy.deepcopy(analysis.get("filters", {}) or {})
        filter_micro_map = copy.deepcopy(analysis.get("filter_micro", {}) or {})
        noir_faces = copy.deepcopy(filter_faces_map.get("NOIR_ORIGINAL", []) or [])

        if selected_ids:
            rgb_faces = [f for f in rgb_faces if str(f.get("person_id")) in selected_set]
            noir_faces = [f for f in noir_faces if str(f.get("person_id")) in selected_set]

        # La ausencia de landmarks NO cancela las 8 imagenes ni CAPTURAS.
        # Se documenta como dato faltante y training_valid=0.

        yolo = incoming.get("yolo11n", {}) or {}
        valid_conf = float(yolo.get("valid_conf", VALID_CONF_DEFAULT))
        weak_conf = float(yolo.get("weak_conf", WEAK_CONF_DEFAULT))
        rgb_yolo = yolo.get("rgb", {}) or {}
        noir_yolo = yolo.get("noir", {}) or {}
        rgb_dets = rgb_yolo.get("detections", []) or []
        noir_dets = noir_yolo.get("detections", []) or []

        # ====================================================
        # 1) Guardar primero las dos camaras limpias.
        # ====================================================
        rgb_original_path = event_dir / "RGB_original.png"
        noir_original_path = event_dir / "NOIR_original.png"
        save_image(rgb_original_path, rgb)
        save_image(noir_original_path, noir)

        # ====================================================
        # 2) Guardar los seis filtros sobre EXACTAMENTE ese NoIR.
        # ====================================================
        filters_dir = event_dir / "filtros_noir"
        filters_dir.mkdir(parents=True, exist_ok=True)
        filter_images = {}
        filter_paths = {}
        export_names = [
            "01_NOIR_original.png",
            "02_NOIR_red_bandpass_virtual.png",
            "03_NOIR_ir_cut_rgb_virtual.png",
            "04_NOIR_blue_bandpass_virtual.png",
            "05_NOIR_ir_bandpass_virtual.png",
            "06_NOIR_uv_bandpass_virtual.png",
        ]
        for idx, fid in enumerate(NOIR_FILTER_IDS):
            filtered = apply_noir_filter(noir, idx)
            path = filters_dir / export_names[idx]
            save_image(path, filtered)
            filter_images[fid] = filtered
            filter_paths[fid] = path

        required_paths = [rgb_original_path, noir_original_path] + [
            filter_paths[fid] for fid in NOIR_FILTER_IDS
        ]
        if not all(p.exists() and p.stat().st_size > 0 for p in required_paths):
            raise RuntimeError("No se guardaron las 8 imagenes obligatorias")

        # ====================================================
        # V16.18 LANDMARK + ASIGNACION PERSONAS/FILTROS
        # LIVE manda. Si LIVE no trajo 68 puntos, MediaPipe hace UN segundo
        # intento sobre estas mismas imagenes guardadas.
        # ====================================================
        with latest_rgb_person_refs_lock:
            _v1616_live_refs = copy.deepcopy(latest_rgb_person_refs)

        rgb_faces = _v1617_recover_landmarks(
            rgb, rgb_faces, DATA_RGB_ENGINE, selected_ids,
            _v1616_live_refs, "RGB",
            detections=rgb_dets, valid_conf=valid_conf
        )
        rgb_faces = _v1618_strict_assign_selected(
            rgb, rgb_faces, selected_ids, _v1616_live_refs,
            rgb_dets, valid_conf, "RGB"
        )

        _v1616_rgb_refs = face_reference_list(rgb_faces, rgb.shape)
        if not _v1616_rgb_refs:
            _v1616_rgb_refs = _v1616_live_refs

        noir_faces = _v1617_recover_landmarks(
            noir, noir_faces, DATA_NOIR_ENGINE, selected_ids,
            _v1616_rgb_refs, "NOIR_ORIGINAL",
            detections=noir_dets, valid_conf=valid_conf
        )
        noir_faces = _v1618_strict_assign_selected(
            noir, noir_faces, selected_ids, _v1616_rgb_refs,
            noir_dets, valid_conf, "NOIR_ORIGINAL"
        )
        filter_faces_map["NOIR_ORIGINAL"] = copy.deepcopy(noir_faces)

        _v1616_noir_refs = face_reference_list(noir_faces, noir.shape)
        if not _v1616_noir_refs:
            _v1616_noir_refs = _v1616_rgb_refs

        # ====================================================
        # 3) PERSONAS + LANDMARKS usando LIVE + fallback MediaPipe.
        # ====================================================
        rgb_person_records, rgb_landmarks, _ = build_person_records(
            capture_id, "RGB", distance_cm, rgb, rgb_dets,
            rgb_faces, valid_conf, event_dir
        )
        noir_person_records, noir_landmarks, _ = build_person_records(
            capture_id, "NOIR", distance_cm, noir, noir_dets,
            noir_faces, valid_conf, event_dir
        )

        rgb_person_records = _v1610_ensure_person_rows(
            rgb_person_records, capture_id, "RGB", selected_ids, distance_cm, ir_value
        )
        noir_person_records = _v1610_ensure_person_rows(
            noir_person_records, capture_id, "NOIR", selected_ids, distance_cm, ir_value
        )

        # V16.15: si MediaPipe falla, PERSONAS conserva la caja/confianza YOLO real.
        rgb_person_records = _v1614_enrich_person_rows_with_yolo(
            rgb_person_records, rgb_dets, valid_conf, selected_ids, rgb.shape
        )
        noir_person_records = _v1614_enrich_person_rows_with_yolo(
            noir_person_records, noir_dets, valid_conf, selected_ids, noir.shape
        )

        # V16.15: normalizacion geometrica automatica, sin etiquetas manuales.
        _v1615_enrich_normalized_landmarks(rgb_landmarks)
        _v1615_enrich_normalized_landmarks(noir_landmarks)

        # ====================================================
        # 4) FILTROS_NOIR + LANDMARKS de cada filtro.
        # ====================================================
        noir_filter_records = []
        filter_landmark_records = []
        filter_faces_dir = event_dir / "filter_faces"
        filter_faces_dir.mkdir(parents=True, exist_ok=True)

        for idx, (fid, fname) in enumerate(zip(NOIR_FILTER_IDS, FILTERS_NOIR)):
            filtered = filter_images[fid]
            faces = copy.deepcopy(filter_faces_map.get(fid, []) or [])
            if selected_ids:
                faces = [f for f in faces if str(f.get("person_id")) in selected_set]

            # V16.19: si este filtro no trajo landmarks desde LIVE, reintenta
            # MediaPipe sobre la imagen del filtro que ya fue guardada.
            _filter_engine = DATA_NOIR_ENGINE if fid == "NOIR_ORIGINAL" else FILTER_FACE_ENGINE
            faces = _v1617_recover_landmarks(
                filtered, faces, _filter_engine, selected_ids,
                _v1616_noir_refs, fid,
                detections=noir_dets, valid_conf=valid_conf
            )
            faces = _v1618_strict_assign_selected(
                filtered, faces, selected_ids, _v1616_noir_refs,
                noir_dets, valid_conf, fid
            )

            micro_state = filter_micro_map.get(str(idx), filter_micro_map.get(idx, {})) or {}
            micro_faces = micro_state.get("faces", []) or []
            active_faces = [m for m in micro_faces if isinstance(m, dict) and m.get("active")]
            micro_active = bool(active_faces)
            micro_activity = max(
                [float(m.get("activity", 0.0)) for m in active_faces] or [0.0]
            )
            last_event = micro_state.get("last_event") or {}
            effective_fps = float(micro_state.get("processed_fps", 0.0) or 0.0)

            for pid in selected_ids:
                matching = [f for f in faces if str(f.get("person_id")) == pid]
                if not matching:
                    noir_filter_records.append(_v1618_filter_placeholder(
                        capture_id, created, idx, fid, fname, pid,
                        filter_paths[fid], ir_value, filtered, len(faces)
                    ))
                    continue
                face = matching[0]
                landmarks = face.get("landmarks")
                if landmarks is None:
                    landmarks = []
                if not bool(face.get("landmarks_detected", False)) or len(landmarks) < 68:
                    noir_filter_records.append(_v1618_filter_placeholder(
                        capture_id, created, idx, fid, fname, pid,
                        filter_paths[fid], ir_value, filtered, len(faces)
                    ))
                    continue

                bbox = face.get("bbox")
                state = get_person_conditions(pid)
                qgate = face.get("quality_gate", {}) or {}
                metrics = face.get("metrics", {}) or {}
                blendshapes = face.get("blendshapes", {}) or {}
                q = image_quality(filtered, bbox) if bbox else {
                    "brightness": None, "contrast": None, "sharpness": None
                }

                crop = crop_face_for_dataset(filtered, bbox, output_size=None)
                crop_path = None
                if crop is not None:
                    crop_path = filter_faces_dir / "{}_{}_face.png".format(fid, pid)
                    save_image(crop_path, crop)

                bx, by, bw, bh = [float(v) for v in bbox]
                bw = max(1.0, bw); bh = max(1.0, bh)
                for point_idx, point in enumerate(landmarks):
                    px, py = float(point[0]), float(point[1])
                    filter_landmark_records.append({
                        "capture_id": capture_id,
                        "camera": fid,
                        "person_id": pid,
                        "point_index": point_idx,
                        "region": landmark_region(point_idx),
                        "x_px": round(px, 3),
                        "y_px": round(py, 3),
                        "x_norm_face": round((px - bx) / bw, 6),
                        "y_norm_face": round((py - by) / bh, 6),
                        "x_norm_eyes": None,
                        "y_norm_eyes": None,
                    })

                noir_filter_records.append({
                    "capture_id": capture_id,
                    "timestamp": created,
                    "filter_index": idx + 1,
                    "filter_id": fid,
                    "filter_name": fname,
                    "person_id": pid,
                    "faces_mediapipe": len(faces),
                    "landmarks_detected": 1,
                    "landmarks_count": len(landmarks),
                    "blendshape_count": len(blendshapes),
                    "landmark_source": str(face.get("landmark_recovery") or "LIVE"),
                    "left_eye_ear": metrics.get("left_eye_ear"),
                    "right_eye_ear": metrics.get("right_eye_ear"),
                    "mouth_open": metrics.get("mouth_open"),
                    "smile_width": metrics.get("smile_width"),
                    "head_roll_deg": metrics.get("head_roll_deg"),
                    "brightness": q.get("brightness"),
                    "contrast": q.get("contrast"),
                    "sharpness": q.get("sharpness"),
                    "lentes": int(bool(state.get("lentes"))),
                    "barba": int(bool(state.get("barba"))),
                    "cabello_frente": int(bool(state.get("cabello_frente"))),
                    "condiciones_manual": person_conditions_summary(state),
                    "registration_status": get_registration_status(pid),
                    "quality_status": qgate.get("status", "UNKNOWN"),
                    "training_valid": int(
                        bool(qgate.get("training_valid", 0)) and person_is_registered(pid)
                    ),
                    "ir_active": ir_value,
                    "event_active": int(micro_active),
                    "activity_score": round(micro_activity, 5),
                    "last_event_type": last_event.get("type", ""),
                    "last_event_is_micro": int(bool(last_event.get("is_micro_candidate", 0))),
                    "last_event_duration_ms": last_event.get("duration_ms", ""),
                    "last_event_region": last_event.get("top_region", ""),
                    "effective_fps": effective_fps,
                    "full_image_path": str(filter_paths[fid]),
                    "face_crop_path": str(crop_path) if crop_path else None,
                    "foto": "",
                })

        # Normalizar tambien los 68 puntos de cada filtro NoIR.
        _v1615_enrich_normalized_landmarks(filter_landmark_records)

        # Exactamente 6 filas por persona seleccionada.
        expected_filter_rows = len(selected_ids) * 6
        if len(noir_filter_records) != expected_filter_rows:
            raise RuntimeError(
                "FILTROS_NOIR incompleto: {}/{} filas".format(
                    len(noir_filter_records), expected_filter_rows
                )
            )

        # ====================================================
        # 5) YOLO y resumen CAPTURAS.
        # ====================================================
        rgb_objects = build_object_records(
            capture_id, "RGB", distance_cm, rgb_dets, valid_conf, weak_conf
        )
        noir_objects = build_object_records(
            capture_id, "NOIR", distance_cm, noir_dets, valid_conf, weak_conf
        )

        rgb_q = aggregate_face_quality(rgb, rgb_faces)
        noir_q = aggregate_face_quality(noir, noir_faces)

        # CAPTURAS es nivel de imagen. Si MediaPipe no encontro un rostro,
        # no dejamos vacia la calidad de la captura: medimos el frame completo.
        # Esto NO inventa landmarks ni metricas faciales.
        if rgb_q.get("brightness") is None:
            rgb_q = image_quality(rgb)
        if noir_q.get("brightness") is None:
            noir_q = image_quality(noir)

        rgb_faces_strip_path = event_dir / "RGB_faces_strip.png"
        noir_faces_strip_path = event_dir / "NOIR_faces_strip.png"
        rgb_faces_strip_path = create_face_strip_from_records(
            rgb_person_records, rgb_faces_strip_path, "RGB"
        )
        noir_faces_strip_path = create_face_strip_from_records(
            noir_person_records, noir_faces_strip_path, "NOIR"
        )

        def _count_status(records, status):
            return sum(1 for r in records if r.get("detection_status") == status)

        # training_complete = 1 solo si TODOS los canales son GREEN.
        training_complete = 1
        all_person_records = rgb_person_records + noir_person_records
        for pid in selected_ids:
            for cam in ("RGB", "NOIR"):
                rows = [r for r in all_person_records if r.get("camera") == cam and str(r.get("person_id")) == pid]
                if not rows or not any(int(r.get("training_valid") or 0) == 1 for r in rows):
                    training_complete = 0
            for fid in NOIR_FILTER_IDS:
                rows = [r for r in noir_filter_records if r.get("filter_id") == fid and str(r.get("person_id")) == pid]
                if not rows or not any(int(r.get("training_valid") or 0) == 1 for r in rows):
                    training_complete = 0

        # V16.19 - calidad del dataset sin alterar la deteccion.
        person_records = rgb_person_records + noir_person_records
        landmark_records = rgb_landmarks + noir_landmarks + filter_landmark_records
        object_records = rgb_objects + noir_objects

        _v1619_enrich_landmark_qc(landmark_records)
        quality_records = _v1619_apply_quality_summary(
            selected_ids, capture_id, person_records, landmark_records, noir_filter_records
        )
        valid_channels_count = sum(int(r.get("valid_channels_count") or 0) for r in quality_records)
        valid_channels_total = sum(int(r.get("valid_channels_total") or 0) for r in quality_records)
        valid_channels_ratio = (
            float(valid_channels_count) / float(valid_channels_total)
            if valid_channels_total else 0.0
        )
        capture_outliers = sum(int(r.get("outlier_count") or 0) for r in quality_records)
        capture_landmarks = sum(int(r.get("landmark_rows") or 0) for r in quality_records)
        capture_outlier_ratio = (
            float(capture_outliers) / float(capture_landmarks)
            if capture_landmarks else 0.0
        )
        history_ready = any(
            int(r.get("history_reference_min_n") or 0) >= QC_HISTORY_MIN_SAMPLES
            for r in quality_records
        )
        dataset_quality_status = _v1619_quality_status(
            valid_channels_ratio, capture_outlier_ratio, history_ready
        )

        capture_record = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "session_id": SESSION_ID,
            "capture_id": capture_id,
            "timestamp": created,
            "distance_cm": distance_cm,
            "tfmini_strength": tfmini.get("strength"),
            "distance_status": tfmini.get("status"),
            "ir_active": ir_value,
            "ir_control_enabled": int(bool(tfmini.get("ir_control_enabled", incoming.get("ir_control_enabled", False)))),
            "ir_reason": tfmini.get("ir_reason", incoming.get("ir_reason", "")),
            "condiciones_personas": capture_conditions_summary(selected_ids),
            "rgb_personas_yolo": len(person_detections(rgb_dets, valid_conf)),
            "noir_personas_yolo": len(person_detections(noir_dets, valid_conf)),
            "rgb_faces": len(rgb_faces),
            "noir_faces": len(noir_faces),
            "rgb_objetos_validos": _count_status(rgb_objects, "VALID"),
            "noir_objetos_validos": _count_status(noir_objects, "VALID"),
            "rgb_objetos_debiles": _count_status(rgb_objects, "WEAK"),
            "noir_objetos_debiles": _count_status(noir_objects, "WEAK"),
            "rgb_brightness": rgb_q.get("brightness"),
            "rgb_contrast": rgb_q.get("contrast"),
            "rgb_sharpness": rgb_q.get("sharpness"),
            "noir_brightness": noir_q.get("brightness"),
            "noir_contrast": noir_q.get("contrast"),
            "noir_sharpness": noir_q.get("sharpness"),
            "rgb_yolo_ms": rgb_yolo.get("inference_ms"),
            "noir_yolo_ms": noir_yolo.get("inference_ms"),
            "rgb_original_path": str(rgb_original_path),
            "noir_original_path": str(noir_original_path),
            "rgb_faces_strip_path": str(rgb_faces_strip_path) if rgb_faces_strip_path else None,
            "noir_faces_strip_path": str(noir_faces_strip_path) if noir_faces_strip_path else None,
            "rgb_foto": "",
            "noir_foto": "",
            "capture_source": incoming.get("source", "LIVE_RGB_NOIR_CAPTURE_5051"),
            "images_saved_count": len(required_paths),
            "filters_saved_count": len(filter_paths),
            "analysis_status": "PENDIENTE_VERIFICACION",
            "training_complete": int(training_complete),
            "group_id": capture_id,
            "selected_people_count": len(selected_ids),
            "valid_channels_count": valid_channels_count,
            "valid_channels_total": valid_channels_total,
            "valid_channels_ratio": round(valid_channels_ratio, 6),
            "dataset_quality_status": dataset_quality_status,
            "analysis_seq": incoming.get("analysis_seq"),
            "rgb_highres_archive_path": str(rgb_highres_path) if rgb_highres_path else None,
            "noir_highres_archive_path": str(noir_highres_path) if noir_highres_path else None,
        }

        # Completitud fisica != completitud de entrenamiento.
        expected_person_rows = len(selected_ids) * 2
        expected_landmark_min = len(selected_ids) * 68 * 8
        measured_complete = (
            len(landmark_records) >= expected_landmark_min
            and all(any(str(r.get("person_id")) == pid
                        and int(r.get("landmarks_detected") or 0) == 1
                        for r in person_records if r.get("camera") == cam)
                    for pid in selected_ids for cam in ("RGB", "NOIR"))
            and all(any(str(r.get("person_id")) == pid
                        and r.get("filter_id") == fid
                        and int(r.get("landmarks_detected") or 0) == 1
                        for r in noir_filter_records)
                    for pid in selected_ids for fid in NOIR_FILTER_IDS)
        )
        capture_record["analysis_status"] = (
            "COMPLETO" if measured_complete else "PARCIAL_SIN_LANDMARKS"
        )
        capture_record["training_complete"] = int(
            bool(training_complete) and measured_complete
        )

        metadata_out = {
            "capture": capture_record,
            "incoming": incoming,
            "persons": person_records,
            "landmarks": landmark_records,
            "objects": object_records,
            "noir_filters_v8_1": noir_filter_records,
            "dataset_quality": quality_records,
        }
        (event_dir / "metadata_v16_19.json").write_text(
            json.dumps(metadata_out, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        excel_saved_path = append_to_excel(
            capture_record,
            person_records,
            landmark_records,
            object_records,
            rgb_faces_strip_path,
            noir_faces_strip_path,
            noir_filter_records,
            quality_records,
        )
        (event_dir / "EXCEL_GUARDADO.txt").write_text(
            "capture_id={}\nexcel={}\n".format(capture_id, excel_saved_path),
            encoding="utf-8"
        )

        status = {
            "capture_id": capture_id,
            "selected_person_ids": selected_ids,
            "images_saved": 8,
            "package_complete": 1,
            "analysis_status": capture_record["analysis_status"],
            "person_rows": len(person_records),
            "landmark_rows": len(landmark_records),
            "filter_rows": len(noir_filter_records),
            "analysis_seq": incoming.get("analysis_seq"),
            "training_complete": capture_record["training_complete"],
            "valid_channels_ratio": capture_record["valid_channels_ratio"],
            "dataset_quality_status": capture_record["dataset_quality_status"],
            "excel": str(excel_saved_path),
        }
        (event_dir / "CAPTURA_COMPLETA.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        print(
            "[V16.19 GUARDADO]", capture_id,
            "| imagenes 8/8",
            "| personas", len(person_records),
            "| landmarks", len(landmark_records),
            "| filtros", len(noir_filter_records),
            "| analysis_status", capture_record["analysis_status"],
            "| training_complete", capture_record["training_complete"],
            "| quality", capture_record["dataset_quality_status"],
            "| valid_channels", "{}/{}".format(
                capture_record["valid_channels_count"], capture_record["valid_channels_total"]
            ),
        )
        return capture_id, event_dir


# ============================================================
# SOCKET SERVERS
# ============================================================

def recv_exact(
    sock,
    count
):
    data = bytearray()

    while len(data) < count:
        chunk = sock.recv(
            count - len(data)
        )

        if not chunk:
            raise ConnectionError(
                "Conexion cerrada"
            )

        data.extend(chunk)

    return bytes(data)


def dataset_server():
    print(
        "Dataset server:",
        DATASET_PORT
    )

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind(
            (
                HOST,
                DATASET_PORT
            )
        )

        server.listen(5)

        while not stop_event.is_set():
            conn, addr = server.accept()

            with conn:
                try:
                    magic = recv_exact(
                        conn,
                        4
                    )

                    if magic == b"PING":
                        conn.sendall(
                            b"PONG"
                        )
                        continue

                    if magic != b"DSET":
                        raise ValueError(
                            "Protocolo desconocido"
                        )

                    size = struct.unpack(
                        "!Q",
                        recv_exact(
                            conn,
                            8
                        )
                    )[0]

                    if (
                        size <= 0
                        or size
                        > 100 * 1024 * 1024
                    ):
                        raise ValueError(
                            "Tamano invalido"
                        )

                    payload = recv_exact(
                        conn,
                        size
                    )

                    capture_id, event_dir = (
                        process_dataset_zip(
                            payload
                        )
                    )

                    response = {
                        "ok": True,
                        "message": (
                            "Guardado {}"
                            .format(
                                capture_id
                            )
                        ),
                        "path": str(
                            event_dir
                        ),
                    }

                except Exception as e:
                    print(
                        "Dataset error:",
                        e
                    )

                    response = {
                        "ok": False,
                        "message": str(e),
                    }

                conn.sendall(
                    (
                        json.dumps(
                            response,
                            ensure_ascii=False
                        )
                        + "\n"
                    ).encode(
                        "utf-8"
                    )
                )


def preview_client(
    conn,
    addr
):
    global latest_frame
    global latest_telemetry

    print(
        "Preview Jetson conectado:",
        addr[0]
    )

    try:
        while not stop_event.is_set():
            magic = recv_exact(
                conn,
                4
            )

            if magic != b"FPR8":
                raise ValueError(
                    "Preview invalido"
                )

            meta_size = struct.unpack(
                "!I",
                recv_exact(
                    conn,
                    4
                )
            )[0]

            if (
                meta_size <= 0
                or meta_size
                > 256 * 1024
            ):
                raise ValueError(
                    "Telemetria invalida"
                )

            meta_bytes = recv_exact(
                conn,
                meta_size
            )

            telemetry = json.loads(
                meta_bytes.decode(
                    "utf-8"
                )
            )

            size = struct.unpack(
                "!I",
                recv_exact(
                    conn,
                    4
                )
            )[0]

            if (
                size <= 0
                or size
                > 5 * 1024 * 1024
            ):
                raise ValueError(
                    "Preview invalido"
                )

            jpg = recv_exact(
                conn,
                size
            )

            frame = cv2.imdecode(
                np.frombuffer(
                    jpg,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if frame is not None:
                with latest_frame_lock:
                    latest_frame = frame

                with latest_telemetry_lock:
                    latest_telemetry = telemetry

                # Enviar mitades al motor LIVE.
                h, w = frame.shape[:2]
                mid = w // 2

                rgb_raw = frame[:, :mid].copy()
                noir_raw = frame[:, mid:].copy()

                # Puerto 5051 queda EXCLUSIVO para tracking rapido 30 FPS.
                # El formato original llega por 5052 y se guarda aparte.
                rgb = cv2.resize(
                    rgb_raw, (DISPLAY_EACH_W, DISPLAY_EACH_H),
                    interpolation=cv2.INTER_AREA
                )
                noir = cv2.resize(
                    noir_raw, (DISPLAY_EACH_W, DISPLAY_EACH_H),
                    interpolation=cv2.INTER_AREA
                )

                arrival_ms = int(
                    time.monotonic_ns()
                    // 1_000_000
                )

                with latest_face_input_lock:
                    latest_face_input[0] = rgb
                    latest_face_input[1] = noir

                    latest_face_seq[0] += 1
                    latest_face_seq[1] += 1

                    latest_face_timestamp_ms[0] = (
                        arrival_ms
                    )

                    latest_face_timestamp_ms[1] = (
                        arrival_ms
                    )

    except Exception as e:
        print(
            "Preview:",
            str(e)[:120]
        )

    print(
        "Preview Jetson desconectado."
    )


def preview_server():
    print(
        "Preview server:",
        PREVIEW_PORT
    )

    with socket.socket(
        socket.AF_INET,
        socket.SOCK_STREAM
    ) as server:
        server.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )

        server.bind(
            (
                HOST,
                PREVIEW_PORT
            )
        )

        server.listen(2)

        while not stop_event.is_set():
            conn, addr = server.accept()

            preview_client(
                conn,
                addr
            )

            try:
                conn.close()
            except Exception:
                pass



def original_stream_client(conn, addr):
    """Recibe 960x540 por camara desde Jetson: solo para dataset/filtros."""
    print("Original stream Jetson:", addr)
    while not stop_event.is_set():
        try:
            magic = recv_exact(conn, 4)
            if magic != b"FORG":
                raise ValueError("Cabecera original invalida")
            meta_size = struct.unpack("!I", recv_exact(conn, 4))[0]
            if meta_size <= 0 or meta_size > 64 * 1024:
                raise ValueError("Metadata original invalida")
            _meta = json.loads(recv_exact(conn, meta_size).decode("utf-8"))
            size = struct.unpack("!I", recv_exact(conn, 4))[0]
            if size <= 0 or size > 12 * 1024 * 1024:
                raise ValueError("Frame original invalido")
            jpg = recv_exact(conn, size)
            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            if frame is None:
                continue
            h, w = frame.shape[:2]
            mid = w // 2
            rgb_original = frame[:, :mid].copy()
            noir_original = frame[:, mid:].copy()
            with latest_dataset_input_lock:
                latest_dataset_input[0] = rgb_original
                latest_dataset_input[1] = noir_original
        except Exception as e:
            print("Original stream:", str(e)[:120])
            break
    print("Original stream Jetson desconectado.")


def original_stream_server():
    print("Original stream server:", ORIGINAL_STREAM_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, ORIGINAL_STREAM_PORT))
        server.listen(2)
        while not stop_event.is_set():
            conn, addr = server.accept()
            original_stream_client(conn, addr)
            try:
                conn.close()
            except Exception:
                pass


# ============================================================
# LIVE LANDMARK WORKER
# ============================================================

def live_rgb_worker(
    camera_id,
    engine
):
    analyzer = MICRO_ANALYZERS[
        camera_id
    ]

    last_seq = -1
    frames = 0
    fps_t0 = time.perf_counter()
    processed_fps = 0.0

    while not stop_event.is_set():
        frame = None
        seq = last_seq
        timestamp_ms = 0

        with latest_face_input_lock:
            seq = int(
                latest_face_seq.get(
                    camera_id,
                    0
                )
            )

            if seq != last_seq:
                src = latest_face_input.get(
                    camera_id
                )

                if src is not None:
                    frame = src.copy()

                    timestamp_ms = int(
                        latest_face_timestamp_ms.get(
                            camera_id,
                            int(
                                time.monotonic_ns()
                                // 1_000_000
                            )
                        )
                    )

        if (
            frame is None
            or seq == last_seq
        ):
            time.sleep(
                0.002
            )
            continue

        last_seq = seq

        faces = engine.detect_all(
            frame,
            timestamp_ms=timestamp_ms
        )
        faces = apply_quality_gate(
            frame,
            faces
        )

        # Alta inmediata a la fila PENDING antes del analisis temporal.
        faces = assign_rgb_person_ids_immediate(
            faces,
            timestamp_ms
        )

        # PUBLICAR INMEDIATAMENTE: si ya hay rasgos, ya aparece PENDING.
        with latest_live_faces_lock:
            latest_live_faces[camera_id] = copy.deepcopy(faces)

        micro_faces, events = (
            analyzer.update(
                faces,
                timestamp_ms
            )
        )

        (faces, micro_faces) = link_rgb_micro_tracks(
            faces, micro_faces, frame.shape
        )
        for face in faces:
            person_id = face.get("person_id")
            ensure_person_registration(person_id)
            update_best_reference(
                person_id,
                frame,
                face,
                source="RGB"
            )
            maybe_save_auto_face_sample(
                "RGB",
                frame,
                face,
                timestamp_ms
            )

        frames += 1
        now = time.perf_counter()

        if now - fps_t0 >= 1.0:
            processed_fps = (
                frames
                / (
                    now
                    - fps_t0
                )
            )

            frames = 0
            fps_t0 = now

        enqueue_blendshape_series(
            source="RGB",
            filter_name="ORIGINAL RGB",
            frame_seq=seq,
            timestamp_ms=timestamp_ms,
            faces=faces,
            micro_faces=micro_faces,
            effective_fps=processed_fps,
        )

        last_event = None

        if events:
            candidates = [
                ev
                for ev in events
                if ev.get(
                    "type"
                ) == "MICRO_CANDIDATE"
            ]

            last_event = (
                candidates[-1]
                if candidates
                else events[-1]
            )

            last_event[
                "_shown_at_ms"
            ] = int(
                time.monotonic_ns()
                // 1_000_000
            )

        with latest_live_faces_lock:
            latest_live_faces[
                camera_id
            ] = faces

        # V16.11: guardar el MISMO frame y caras que ya funcionaron en LIVE.
        with live_capture_cache_lock:
            live_rgb_capture_cache[int(seq)] = {
                "frame": frame.copy(),
                "faces": copy.deepcopy(faces),
                "timestamp_ms": int(timestamp_ms),
            }
            if len(live_rgb_capture_cache) > LIVE_CAPTURE_CACHE_MAX:
                for old_seq in sorted(live_rgb_capture_cache.keys())[:-LIVE_CAPTURE_CACHE_MAX]:
                    live_rgb_capture_cache.pop(old_seq, None)

        with latest_micro_lock:
            previous_event = (
                latest_micro[
                    camera_id
                ].get(
                    "last_event"
                )
            )

            latest_micro[
                camera_id
            ] = {
                "camera": "RGB",
                "faces": micro_faces,
                "last_event": (
                    last_event
                    if last_event is not None
                    else previous_event
                ),
                "processed_fps": round(
                    float(
                        processed_fps
                    ),
                    2
                ),
            }


def live_noir_filters_worker():
    """
    Una misma captura NoIR genera 6 imagenes.
    CADA FILTRO:
      - ejecuta MediaPipe sobre SU imagen filtrada
      - usa SU MicroExpressionAnalyzer
      - mantiene SU baseline/onset/apex/offset
      - genera SUS propios eventos

    No se copian blendshapes ni microexpresiones entre filtros.
    """
    last_seq = -1

    cycles = 0
    fps_t0 = time.perf_counter()
    cycle_fps = 0.0

    while not stop_event.is_set():
        noir = None
        seq = last_seq
        timestamp_ms = 0

        with latest_face_input_lock:
            seq = int(
                latest_face_seq.get(
                    1,
                    0
                )
            )

            if seq != last_seq:
                src = latest_face_input.get(
                    1
                )

                if src is not None:
                    noir = src.copy()

                    timestamp_ms = int(
                        latest_face_timestamp_ms.get(
                            1,
                            int(
                                time.monotonic_ns()
                                // 1_000_000
                            )
                        )
                    )

        if (
            noir is None
            or seq == last_seq
        ):
            time.sleep(
                0.002
            )
            continue

        last_seq = seq

        # Procesar los seis filtros sobre EXACTAMENTE el mismo
        # frame NoIR de origen y el mismo timestamp.
        for filter_index in range(
            len(FILTERS_NOIR)
        ):
            filtered = apply_noir_filter(
                noir,
                filter_index
            )

            faces = FILTER_FACE_ENGINE.detect_all(
                filtered
            )
            faces = apply_quality_gate(
                filtered,
                faces
            )

            analyzer = NOIR_FILTER_ANALYZERS[
                filter_index
            ]

            micro_faces, events = (
                analyzer.update(
                    faces,
                    timestamp_ms
                )
            )

            (
                faces,
                micro_faces,
            ) = annotate_nonrgb_faces_with_person_ids(
                NOIR_FILTER_IDS[
                    filter_index
                ],
                faces,
                micro_faces,
                filtered.shape
            )
            if filter_index == 0:
                for face in faces:
                    update_best_reference(
                        face.get("person_id"),
                        filtered,
                        face,
                        source="NOIR"
                    )
                    maybe_save_auto_face_sample(
                        "NOIR_ORIGINAL",
                        filtered,
                        face,
                        timestamp_ms
                    )

            enqueue_blendshape_series(
                source=NOIR_FILTER_IDS[
                    filter_index
                ],
                filter_name=FILTERS_NOIR[
                    filter_index
                ],
                frame_seq=seq,
                timestamp_ms=timestamp_ms,
                faces=faces,
                micro_faces=micro_faces,
                effective_fps=cycle_fps,
            )

            last_event = None

            if events:
                candidates = [
                    ev
                    for ev in events
                    if ev.get(
                        "type"
                    ) == "MICRO_CANDIDATE"
                ]

                last_event = (
                    candidates[-1]
                    if candidates
                    else events[-1]
                )

                last_event[
                    "_shown_at_ms"
                ] = int(
                    time.monotonic_ns()
                    // 1_000_000
                )

            with latest_noir_filter_lock:
                previous_event = (
                    latest_noir_filter_micro[
                        filter_index
                    ].get(
                        "last_event"
                    )
                )

                latest_noir_filter_faces[
                    filter_index
                ] = faces

                latest_noir_filter_micro[
                    filter_index
                ] = {
                    "source": NOIR_FILTER_IDS[
                        filter_index
                    ],
                    "filter_name": FILTERS_NOIR[
                        filter_index
                    ],
                    "faces": micro_faces,
                    "last_event": (
                        last_event
                        if last_event is not None
                        else previous_event
                    ),
                    "processed_fps": round(
                        float(
                            cycle_fps
                        ),
                        2
                    ),
                }

        # Al terminar los 6 filtros, todos corresponden al MISMO seq/frame NoIR.
        with latest_noir_filter_lock:
            _cycle_faces_copy = copy.deepcopy(latest_noir_filter_faces)
            _cycle_micro_copy = copy.deepcopy(latest_noir_filter_micro)
        with latest_noir_cycle_lock:
            latest_noir_cycle_snapshot.update({
                "seq": int(seq),
                "timestamp_ms": int(timestamp_ms),
                "frame": noir.copy(),
                "filter_faces": _cycle_faces_copy,
                "filter_micro": _cycle_micro_copy,
            })

        cycles += 1
        now = time.perf_counter()

        if now - fps_t0 >= 1.0:
            cycle_fps = (
                cycles
                / (
                    now
                    - fps_t0
                )
            )

            cycles = 0
            fps_t0 = now

            with latest_noir_filter_lock:
                for filter_index in range(
                    len(FILTERS_NOIR)
                ):
                    latest_noir_filter_micro[
                        filter_index
                    ][
                        "processed_fps"
                    ] = round(
                        float(
                            cycle_fps
                        ),
                        2
                    )



# ============================================================
# V16.3 - VENTANA COMPARATIVA 6 FILTROS NOIR
# ============================================================

def make_noir_filter_grid():
    with latest_face_input_lock:
        noir = latest_face_input.get(
            1
        )

        noir = (
            None
            if noir is None
            else noir.copy()
        )

    canvas = np.zeros(
        (
            FILTER_GRID_H,
            FILTER_GRID_W,
            3
        ),
        dtype=np.uint8
    )

    if noir is None:
        cv2.putText(
            canvas,
            "ESPERANDO NOIR...",
            (320, 290),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            ORANGE,
            2,
            cv2.LINE_AA
        )
        return canvas

    with latest_noir_filter_lock:
        all_faces = copy.deepcopy(
            latest_noir_filter_faces
        )

        all_micro = copy.deepcopy(
            latest_noir_filter_micro
        )

    with latest_telemetry_lock:
        telem = copy.deepcopy(
            latest_telemetry
        )

    cell_w = FILTER_GRID_W // 3
    cell_h = FILTER_GRID_H // 2

    for idx in range(
        len(
            FILTERS_NOIR
        )
    ):
        row = idx // 3
        col = idx % 3

        x0 = col * cell_w
        y0 = row * cell_h

        filtered = apply_noir_filter(
            noir,
            idx
        )

        faces = all_faces.get(
            idx,
            []
        )

        panel = draw_faces(
            filtered,
            faces
        )

        panel = cv2.resize(
            panel,
            (
                cell_w,
                cell_h - 58
            ),
            interpolation=cv2.INTER_AREA
        )

        canvas[
            y0:y0 + cell_h - 58,
            x0:x0 + cell_w
        ] = panel

        state = all_micro.get(
            idx,
            {}
        )

        micro_faces = state.get(
            "faces",
            []
        )

        active = [
            f
            for f in micro_faces
            if f.get(
                "active"
            )
        ]

        if active:
            best = max(
                active,
                key=lambda f: float(
                    f.get(
                        "activity",
                        0.0
                    )
                )
            )

            micro_text = (
                "MICRO ACTIVA {:.3f}"
                .format(
                    float(
                        best.get(
                            "activity",
                            0.0
                        )
                    )
                )
            )

            micro_color = ORANGE

        else:
            ev = state.get(
                "last_event"
            )

            if ev:
                micro_text = "{} {}ms".format(
                    ev.get(
                        "type",
                        "EVENTO"
                    ),
                    ev.get(
                        "duration_ms",
                        ""
                    )
                )

                micro_color = (
                    GREEN
                    if ev.get(
                        "is_micro_candidate"
                    )
                    else YELLOW
                )

            else:
                micro_text = "MICRO: --"
                micro_color = GRAY

        cv2.rectangle(
            canvas,
            (
                x0,
                y0 + cell_h - 58
            ),
            (
                x0 + cell_w - 1,
                y0 + cell_h - 1
            ),
            (
                20,
                20,
                20
            ),
            -1
        )

        people_parts = []

        for mf in micro_faces[:4]:
            pid = (
                mf.get(
                    "person_id"
                )
                or "PX"
            )

            activity = float(
                mf.get(
                    "activity",
                    0.0
                )
            )

            people_parts.append(
                "{}:{:.2f}".format(
                    pid,
                    activity
                )
            )

        cv2.putText(
            canvas,
            "{} | caras {} | {}".format(
                NOIR_FILTER_SHORT[
                    idx
                ],
                len(
                    faces
                ),
                " ".join(
                    people_parts
                )
            ),
            (
                x0 + 8,
                y0 + cell_h - 36
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            WHITE,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            micro_text,
            (
                x0 + 8,
                y0 + cell_h - 14
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            micro_color,
            1,
            cv2.LINE_AA
        )

        cv2.rectangle(
            canvas,
            (
                x0,
                y0
            ),
            (
                x0 + cell_w - 1,
                y0 + cell_h - 1
            ),
            (
                90,
                90,
                90
            ),
            1
        )

    ir_enabled = bool(
        telem.get(
            "ir_control_enabled",
            False
        )
    )

    ir_active = telem.get(
        "ir_active"
    )

    if ir_enabled:
        ir_text = (
            "ON"
            if bool(
                ir_active
            )
            else "OFF"
        )
    else:
        ir_text = "SIN CONTROL"

    cv2.putText(
        canvas,
        "IR {} | TFmini {} cm | filtros con landmarks/blendshapes/micro independientes".format(
            ir_text,
            telem.get(
                "distance_cm",
                "---"
            )
        ),
        (
            12,
            18
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        CYAN,
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# UI
# ============================================================

def make_display():
    with latest_frame_lock:
        frame = (
            None
            if latest_frame is None
            else latest_frame.copy()
        )

    with latest_telemetry_lock:
        telemetry = copy.deepcopy(
            latest_telemetry
        )

    canvas = np.zeros(
        (
            DISPLAY_H,
            DISPLAY_W,
            3
        ),
        dtype=np.uint8
    )

    if frame is None:
        cv2.putText(
            canvas,
            "ESPERANDO PREVIEW DE JETSON...",
            (210, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            ORANGE,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            "Los botones por persona apareceran cuando MediaPipe detecte rostros RGB.",
            (
                70,
                DISPLAY_EACH_H + 170
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            GRAY,
            1,
            cv2.LINE_AA
        )

        return canvas

    h, w = frame.shape[:2]
    mid = w // 2

    rgb = frame[
        :,
        :mid
    ].copy()

    rgb_for_cards = rgb.copy()

    noir = frame[
        :,
        mid:
    ].copy()

    with latest_live_faces_lock:
        rgb_faces = copy.deepcopy(
            latest_live_faces[0]
        )

        with latest_noir_filter_lock:
            noir_faces = copy.deepcopy(
                latest_noir_filter_faces[0]
            )

    yolo = telemetry.get(
        "yolo11n",
        {}
    )

    valid_conf = float(
        yolo.get(
            "valid_conf",
            VALID_CONF_DEFAULT
        )
    )

    weak_conf = float(
        yolo.get(
            "weak_conf",
            WEAK_CONF_DEFAULT
        )
    )

    rgb_dets = (
        yolo.get(
            "rgb",
            {}
        ).get(
            "detections",
            []
        )
    )

    noir_dets = (
        yolo.get(
            "noir",
            {}
        ).get(
            "detections",
            []
        )
    )

    # Las detecciones YOLO se calcularon sobre 960x540 en Jetson,
    # mientras el preview viaja a 480x270 por camara.
    rgb_dets_preview = scale_detections(
        rgb_dets,
        960,
        540,
        rgb.shape[1],
        rgb.shape[0]
    )

    noir_dets_preview = scale_detections(
        noir_dets,
        960,
        540,
        noir.shape[1],
        noir.shape[0]
    )

    rgb = draw_yolo_boxes(
        rgb,
        rgb_dets_preview,
        valid_conf,
        weak_conf
    )

    noir = draw_yolo_boxes(
        noir,
        noir_dets_preview,
        valid_conf,
        weak_conf
    )

    rgb = draw_faces(
        rgb,
        rgb_faces
    )

    noir = draw_faces(
        noir,
        noir_faces
    )

    rgb_micro_text, rgb_micro_color = (
        micro_summary_text(
            0
        )
    )

    noir_micro_text, noir_micro_color = (
        noir_filter_micro_summary_text(
            0
        )
    )

    cv2.putText(
        rgb,
        rgb_micro_text,
        (
            8,
            max(
                42,
                rgb.shape[0] - 12
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        rgb_micro_color,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        noir,
        noir_micro_text,
        (
            8,
            max(
                42,
                noir.shape[0] - 12
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        noir_micro_color,
        1,
        cv2.LINE_AA
    )

    rgb = cv2.resize(
        rgb,
        (
            DISPLAY_EACH_W,
            DISPLAY_EACH_H
        ),
        interpolation=cv2.INTER_AREA
    )

    noir = cv2.resize(
        noir,
        (
            DISPLAY_EACH_W,
            DISPLAY_EACH_H
        ),
        interpolation=cv2.INTER_AREA
    )

    cv2.putText(
        rgb,
        "RGB | MULTI-FACE + MICROEXPRESIONES",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        noir,
        "NoIR | MULTI-FACE + MICROEXPRESIONES",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        WHITE,
        1,
        cv2.LINE_AA
    )

    canvas[
        :DISPLAY_EACH_H,
        :DISPLAY_EACH_W
    ] = rgb

    canvas[
        :DISPLAY_EACH_H,
        DISPLAY_EACH_W:
    ] = noir

    distance = telemetry.get(
        "distance_cm"
    )

    zone_min = telemetry.get(
        "zone_min_cm",
        30
    )

    zone_max = telemetry.get(
        "zone_max_cm",
        60
    )

    status = telemetry.get(
        "status",
        "SIN DATOS"
    )

    valid = bool(
        telemetry.get(
            "valid",
            False
        )
    )

    color = (
        GREEN
        if valid
        else RED
    )

    cv2.putText(
        canvas,
        "TFmini: {} cm | {} | ZONA {}-{} cm | IR {}".format(
            distance if distance is not None else "---",
            status,
            zone_min,
            zone_max,
            (
                "ON"
                if bool(
                    telemetry.get(
                        "ir_active"
                    )
                )
                else (
                    "OFF"
                    if bool(
                        telemetry.get(
                            "ir_control_enabled",
                            False
                        )
                    )
                    else "SIN CONTROL"
                )
            )
        ),
        (
            15,
            DISPLAY_EACH_H + 30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )

    with latest_micro_lock:
        rgb_micro_fps = float(
            latest_micro[0].get(
                "processed_fps",
                0.0
            )
        )

    with latest_noir_filter_lock:
        noir_micro_fps = float(
            latest_noir_filter_micro[0].get(
                "processed_fps",
                0.0
            )
        )

    cv2.putText(
        canvas,
        "Micro: RGB + 6 filtros NoIR | cada filtro tiene baseline/eventos independientes",
        (
            15,
            DISPLAY_EACH_H + 64
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        WHITE,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Procesado: RGB {:.1f} FPS | cada filtro NoIR {:.1f} FPS | {}".format(
            rgb_micro_fps,
            noir_micro_fps,
            noir_filters_status_line()
        ),
        (
            15,
            DISPLAY_EACH_H + 92
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        CYAN,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "YOLO11n Jetson | CAPTURA LOCAL: {}".format(
            local_capture_status_text()
        ),
        (
            15,
            DISPLAY_EACH_H + 118
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        GRAY,
        1,
        cv2.LINE_AA
    )

    draw_person_condition_cards(
        canvas,
        rgb_for_cards,
        rgb_faces
    )

    return canvas


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Dataset RGB/NoIR V16 NATURAL MULTI-FACE con microexpresiones "
            "independientes por rostro y por filtro."
        )
    )

    parser.add_argument(
        "--reset-dataset",
        action="store_true",
        help=(
            "Ignorado en esta version segura; no borra datasets."
        )
    )

    args = parser.parse_args()

    prepare_dataset(
        reset=args.reset_dataset
    )

    print("")
    print("========================================================")
    print(" LAPTOP v16 NATURAL MULTI-FACE - COLA 3 + QUALITY GATE + IR TELEMETRIA")
    print("========================================================")
    print("Dataset:", DATASET_DIR)
    print("Excel:", EXCEL_FILE)
    print("Capturas:", DATASET_PORT)
    print("Preview:", PREVIEW_PORT)
    print("Eventos micro JSONL:", MICRO_EVENTS_FILE)
    print("Eventos micro CSV:", MICRO_EVENTS_CSV)
    print("Serie temporal 52 blendshapes:", BLENDSHAPE_SERIES_CSV)
    print("Manifest dataset:", DATASET_MANIFEST)
    print("Filtros NoIR V8.1: 6 analizadores temporales independientes")
    print("  1) NOIR ORIGINAL")
    print("  2) RED BANDPASS VIRTUAL")
    print("  3) IR-CUT / RGB VIRTUAL")
    print("  4) BLUE BANDPASS VIRTUAL")
    print("  5) IR BANDPASS VIRTUAL")
    print("  6) UV BANDPASS VIRTUAL")
    print("Hoja Excel: FILTROS_NOIR")
    print("Dataset V16 NATURAL:", DATASET_DIR)
    print("NO se borra ningun dataset anterior.")
    print("PERSON valid conf:", PERSON_VALID_CONF)
    print("Dataset emocional: TODOS los rostros MediaPipe validos (maximo 4).")
    print("Fotos Excel: tira de rostros + recorte 256x256 por Pxx; NO fondo.")
    print("Persona con rostro visible -> Pxx; YOLO sin rostro -> solo OBJETOS.")
    print("Eventos faciales: se guardan TODOS; micro se marca aparte.")
    print("Registro V16: cola automatica; se muestran 3 PENDING a la vez")
    print("GUARDAR LOTE: registra hasta 3 personas Y CREA captura completa del dataset")
    print("BEST FRAME: foto limpia original; nunca captura de la interfaz")
    print("AUTO FACE SAMPLES: RGB + NoIR original a {:.1f} FPS solo GREEN/REGISTERED".format(AUTO_FACE_SAMPLE_FPS))
    print("QUALITY GATE: GREEN valido; YELLOW/RED tracking sin entrenar")
    print("Etiquetas manuales POR Pxx: LENTES / BARBA / CABELLO EN FRENTE")
    print("  Verde = seleccionado | Gris = no seleccionado")
    print("  Se guardan por persona en PERSONAS, FILTROS_NOIR, MICROEXPRESIONES y blendshapes CSV")
    print("OpenCV:", cv2.__version__)
    print("Motor facial: MediaPipe Face Landmarker + 52 blendshapes temporales")
    print("========================================================")
    print("")

    print(
        "MediaPipe:",
        getattr(
            mp,
            "__version__",
            "DESCONOCIDA"
        )
    )

    if not LIVE_RGB_ENGINE.initialize():
        input(
            "ERROR LIVE RGB. ENTER..."
        )
        return

    if not FILTER_FACE_ENGINE.initialize():
        input(
            "ERROR MOTOR FILTROS NOIR. ENTER..."
        )
        return

    if not DATA_RGB_ENGINE.initialize():
        input(
            "ERROR DATA RGB. ENTER..."
        )
        return

    if not DATA_NOIR_ENGINE.initialize():
        input(
            "ERROR DATA NOIR. ENTER..."
        )
        return

    ensure_workbook()

    threading.Thread(
        target=dataset_server,
        daemon=True
    ).start()

    threading.Thread(
        target=preview_server,
        daemon=True
    ).start()

    threading.Thread(
        target=original_stream_server,
        daemon=True
    ).start()

    threading.Thread(
        target=blendshape_writer_worker,
        daemon=True
    ).start()

    threading.Thread(
        target=live_rgb_worker,
        args=(
            0,
            LIVE_RGB_ENGINE
        ),
        daemon=True
    ).start()

    threading.Thread(
        target=live_noir_filters_worker,
        daemon=True
    ).start()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        person_conditions_mouse_callback
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        DISPLAY_W,
        DISPLAY_H
    )

    cv2.namedWindow(
        FILTER_WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        FILTER_WINDOW_NAME,
        FILTER_GRID_W,
        FILTER_GRID_H
    )

    try:
        while True:
            cv2.imshow(
                WINDOW_NAME,
                make_display()
            )

            cv2.imshow(
                FILTER_WINDOW_NAME,
                make_noir_filter_grid()
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                ord("Q")
            ):
                break

            time.sleep(0.01)

    finally:
        stop_event.set()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()