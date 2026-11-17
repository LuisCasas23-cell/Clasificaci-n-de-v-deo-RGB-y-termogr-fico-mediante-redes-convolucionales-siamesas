import cv2
import numpy as np
import json
import os
import time
from datetime import datetime

# ============================================================
# NOIR FILTER LAB v8.1 - COLOR BOOST / VIDEO LOOK
# Jetson Nano + OV5647 NoIR
#
# OBJETIVO DE ESTA VERSION:
# - Usar la configuracion que ya viste que ilumina mejor:
#       exposurecompensation = 1.2
#       gainrange = 1 a 9
#       ispdigitalgainrange = 1 a 1
#       exposuretimerange = 20 a 33.3 ms
#
# - Quitar el aclarado agresivo de versiones anteriores.
# - Mantener los filtros parecidos al video.
# - Reducir ruido de forma ligera sin volver borrosa la imagen.
# - Conservar la nitidez y los FPS lo mejor posible.
# ============================================================

CONFIG_FILE = "noir_config.json"
CAPTURE_DIR = "capturas_noir"

SENSOR_ID = 1
WIDTH = 960
HEIGHT = 540

FILTERS = [
    "ORIGINAL NOIR",
    "RED BANDPASS VIRTUAL",
    "IR-CUT / RGB VIRTUAL",
    "BLUE BANDPASS VIRTUAL",
    "IR BANDPASS VIRTUAL",
    "UV BANDPASS VIRTUAL"
]

current_filter = 0
button_rects = []

# ============================================================
# AJUSTE DE COLOR PARA EL MODO 3 (IR-CUT / RGB)
# ============================================================

# En la V8 original la saturacion terminaba multiplicada por 0.68,
# lo que apagaba bastante los colores.
# Aqui iniciamos en 1.20 para ver mas color.
RGB_COLOR_BOOST = 1.20
RGB_COLOR_MIN = 0.50
RGB_COLOR_MAX = 2.00
RGB_COLOR_STEP = 0.10

# ============================================================
# PARAMETROS DE CAMARA
# ============================================================

EXPOSURE_COMPENSATION = 1.2

# Si por alguna razon Argus no acepta 9, el programa intenta 8.
GAIN_MAX_PRIMARY = 9
GAIN_MAX_FALLBACK = 8

EXPOSURE_MIN_NS = 20000000
EXPOSURE_MAX_NS = 33333333

# ============================================================
# LUT SUAVE DE SOMBRAS
# ============================================================

# Esta curva es MUCHO mas suave que en v7.
# Como ahora la camara trabaja con mas ganancia, ya no necesitamos
# levantar tanto las sombras en software.
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

# Contraste muy leve para no exagerar el ruido
CONTRAST_LUT = np.array(
    [
        np.clip((i - 128.0) * 1.035 + 128.0, 0, 255)
        for i in range(256)
    ],
    dtype=np.uint8
)


# ============================================================
# MEMORIA DEL ULTIMO FILTRO
# ============================================================

def load_config():
    global current_filter
    global RGB_COLOR_BOOST

    if not os.path.exists(CONFIG_FILE):
        current_filter = 0
        return

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)

        current_filter = int(
            data.get("last_filter", 0)
        )

        if current_filter < 0 or current_filter >= len(FILTERS):
            current_filter = 0

        RGB_COLOR_BOOST = float(
            data.get("rgb_color_boost", RGB_COLOR_BOOST)
        )

        RGB_COLOR_BOOST = float(
            np.clip(
                RGB_COLOR_BOOST,
                RGB_COLOR_MIN,
                RGB_COLOR_MAX
            )
        )

    except Exception:
        current_filter = 0


def save_config():
    with open(CONFIG_FILE, "w") as f:
        json.dump(
            {
                "last_filter": current_filter,
                "filter_name": FILTERS[current_filter],
                "rgb_color_boost": round(RGB_COLOR_BOOST, 2)
            },
            f,
            indent=4
        )


def change_filter(index):
    global current_filter

    if index == current_filter:
        return

    current_filter = index
    save_config()

    print(
        "Filtro seleccionado:",
        FILTERS[current_filter]
    )


def change_rgb_color(delta):
    global RGB_COLOR_BOOST

    RGB_COLOR_BOOST = round(
        float(
            np.clip(
                RGB_COLOR_BOOST + delta,
                RGB_COLOR_MIN,
                RGB_COLOR_MAX
            )
        ),
        2
    )

    save_config()

    print(
        "RGB COLOR BOOST:",
        RGB_COLOR_BOOST
    )


def reset_rgb_color():
    global RGB_COLOR_BOOST

    RGB_COLOR_BOOST = 1.20
    save_config()

    print(
        "RGB COLOR BOOST reiniciado:",
        RGB_COLOR_BOOST
    )


# ============================================================
# REDUCCION DE RUIDO LIGERA
# ============================================================

def light_chroma_denoise(frame):
    """
    Reduce principalmente el ruido de color (puntos verdes/magenta)
    sin suavizar demasiado la luminancia.

    Esto conserva mejor los detalles del rostro que un blur fuerte.
    """

    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(ycrcb)

    # Solo limpiamos crominancia
    cr = cv2.medianBlur(cr, 3)
    cb = cv2.medianBlur(cb, 3)

    return cv2.cvtColor(
        cv2.merge([y, cr, cb]),
        cv2.COLOR_YCrCb2BGR
    )


def mild_shadow_lift(frame):
    """
    Levanta un poco las zonas oscuras.
    Ya no es el aclarado agresivo de v7.
    """

    ycrcb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2YCrCb
    )

    y, cr, cb = cv2.split(ycrcb)

    y = cv2.LUT(
        y,
        SHADOW_LUT
    )

    y = cv2.LUT(
        y,
        CONTRAST_LUT
    )

    return cv2.cvtColor(
        cv2.merge([y, cr, cb]),
        cv2.COLOR_YCrCb2BGR
    )


def sharpen_light(frame):
    """
    Enfoque ligero.
    No buscamos sobre-enfocar porque la ganancia alta tambien
    aumenta el ruido.
    """

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
    """
    Base comun para todos los filtros:
    1) limpia ruido de color
    2) levanta sombras suavemente
    3) recupera un poco de nitidez
    """

    out = light_chroma_denoise(frame)
    out = mild_shadow_lift(out)
    out = sharpen_light(out)

    return out


# ============================================================
# BASE EN ESCALA DE GRISES
# ============================================================

def clear_gray(frame):
    """
    Crea una luminancia clara y util usando los 3 canales.
    """

    base = prepare_base(frame)

    b, g, r = cv2.split(base)

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

    # Limpieza minima
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
    f = gray.astype(np.float32)

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


# ============================================================
# IR-CUT / RGB VIRTUAL
# ============================================================

def gray_world_limited(frame):
    """
    Balance de blancos limitado para evitar que la imagen
    se vaya completamente a verde.
    """

    work = frame.astype(np.float32)

    means = (
        work
        .reshape(-1, 3)
        .mean(axis=0)
        + 1e-6
    )

    target = float(np.mean(means))

    desired = target / means

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
    """
    Opcion 3:
    intenta parecerse visualmente al IR-CUT/RGB del video.

    Ya parte de una imagen mas clara por la ganancia 1-9,
    asi que la correccion de software es mas moderada.
    """

    out = prepare_base(frame)

    out = gray_world_limited(out)

    work = out.astype(np.float32)

    b = work[:, :, 0]
    g = work[:, :, 1]
    r = work[:, :, 2]

    # Mezcla de canales para reducir magenta y recuperar verde.
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
            np.clip(b2, 0, 255).astype(np.uint8),
            np.clip(g2, 0, 255).astype(np.uint8),
            np.clip(r2, 0, 255).astype(np.uint8)
        ]
    )

    # V8.1: aumentar color en lugar de apagarlo.
    # Puedes ajustar este valor en vivo con + y -.
    hsv = cv2.cvtColor(
        out,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    s = np.clip(
        s.astype(np.float32) * RGB_COLOR_BOOST,
        0,
        255
    ).astype(np.uint8)

    out = cv2.cvtColor(
        cv2.merge([h, s, v]),
        cv2.COLOR_HSV2BGR
    )

    return sharpen_light(out)


# ============================================================
# FILTROS
# ============================================================

def filter_original(frame):
    """
    Original NoIR.
    Lo dejamos como referencia de la camara.
    """
    return frame.copy()


def filter_red(frame):
    gray = clear_gray(frame)

    out = tint_gray(
        gray,
        b_gain=0.10,
        g_gain=0.20,
        r_gain=1.00
    )

    return sharpen_light(out)


def filter_blue(frame):
    gray = clear_gray(frame)

    out = tint_gray(
        gray,
        b_gain=1.00,
        g_gain=0.24,
        r_gain=0.10
    )

    return sharpen_light(out)


def filter_ir(frame):
    gray = clear_gray(frame)

    out = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR
    )

    return sharpen_light(out)


def filter_uv(frame):
    gray = clear_gray(frame)

    out = tint_gray(
        gray,
        b_gain=0.82,
        g_gain=0.12,
        r_gain=1.00
    )

    return sharpen_light(out)


def apply_filter(
    frame,
    index
):
    if index == 0:
        return filter_original(frame)

    if index == 1:
        return filter_red(frame)

    if index == 2:
        return rgb_virtual(frame)

    if index == 3:
        return filter_blue(frame)

    if index == 4:
        return filter_ir(frame)

    if index == 5:
        return filter_uv(frame)

    return frame.copy()


# ============================================================
# GUARDAR
# ============================================================

def save_frame(original_frame):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filter_name = FILTERS[
        current_filter
    ]

    safe_name = (
        filter_name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "_")
    )

    original_dir = os.path.join(
        CAPTURE_DIR,
        "original"
    )

    filtered_dir = os.path.join(
        CAPTURE_DIR,
        safe_name
    )

    os.makedirs(
        original_dir,
        exist_ok=True
    )

    os.makedirs(
        filtered_dir,
        exist_ok=True
    )

    original_path = os.path.join(
        original_dir,
        timestamp + ".png"
    )

    filtered_path = os.path.join(
        filtered_dir,
        timestamp + ".png"
    )

    filtered = apply_filter(
        original_frame,
        current_filter
    )

    cv2.imwrite(
        original_path,
        original_frame
    )

    cv2.imwrite(
        filtered_path,
        filtered
    )

    print("")
    print("FRAME GUARDADO")
    print("Original :", original_path)
    print("Filtrado :", filtered_path)
    print("Modo     :", filter_name)
    print("")


# ============================================================
# INTERFAZ
# ============================================================

def draw_interface(
    frame,
    fps
):
    global button_rects

    button_rects = []

    names = [
        "1 ORIGINAL",
        "2 RED",
        "3 IR-CUT",
        "4 BLUE",
        "5 IR",
        "6 UV"
    ]

    x = 8
    y1 = 8
    y2 = 43
    width = 145

    for i, name in enumerate(names):
        x1 = x
        x2 = x + width

        button_rects.append(
            (x1, y1, x2, y2)
        )

        if i == current_filter:
            color = (
                60,
                180,
                60
            )
        else:
            color = (
                45,
                45,
                45
            )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            -1
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (
                225,
                225,
                225
            ),
            1
        )

        cv2.putText(
            frame,
            name,
            (
                x1 + 7,
                y1 + 23
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )

        x += width + 7

    h = frame.shape[0]

    cv2.rectangle(
        frame,
        (
            8,
            h - 58
        ),
        (
            810,
            h - 8
        ),
        (
            0,
            0,
            0
        ),
        -1
    )

    cv2.putText(
        frame,
        "Filtro: "
        + FILTERS[current_filter],
        (
            18,
            h - 32
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "FPS: {:.1f}".format(fps),
        (
            420,
            h - 32
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (
            255,
            255,
            255
        ),
        1,
        cv2.LINE_AA
    )

    if current_filter == 2:
        cv2.putText(
            frame,
            "COLOR x{:.2f}  +/- ajustar  0 reset".format(
                RGB_COLOR_BOOST
            ),
            (
                520,
                h - 32
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            frame,
            "S=Guardar Q=Salir",
            (
                535,
                h - 32
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (
                255,
                255,
                255
            ),
            1,
            cv2.LINE_AA
        )


# ============================================================
# MOUSE
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    for i, rect in enumerate(
        button_rects
    ):
        x1, y1, x2, y2 = rect

        if (
            x1 <= x <= x2
            and y1 <= y <= y2
        ):
            change_filter(i)
            break


# ============================================================
# GSTREAMER
# ============================================================

def make_pipeline(gain_max):
    return (
        'nvarguscamerasrc sensor-id='
        + str(SENSOR_ID)
        + ' '
        'exposurecompensation='
        + str(EXPOSURE_COMPENSATION)
        + ' '
        'gainrange="1 '
        + str(gain_max)
        + '" '
        'ispdigitalgainrange="1 1" '
        'exposuretimerange="'
        + str(EXPOSURE_MIN_NS)
        + ' '
        + str(EXPOSURE_MAX_NS)
        + '" ! '
        'video/x-raw(memory:NVMM), '
        'width=1920, height=1080, '
        'framerate=30/1, format=NV12 ! '
        'nvvidconv ! '
        'video/x-raw, '
        'width=960, height=540, '
        'format=BGRx ! '
        'videoconvert ! '
        'video/x-raw, format=BGR ! '
        'appsink '
        'drop=true '
        'max-buffers=1 '
        'sync=false'
    )


def open_camera():
    print("")
    print(
        "Intentando OV5647 con gainrange 1-9..."
    )

    cap = cv2.VideoCapture(
        make_pipeline(
            GAIN_MAX_PRIMARY
        ),
        cv2.CAP_GSTREAMER
    )

    if cap.isOpened():
        print(
            "Camara abierta con gainrange 1-9."
        )
        return cap

    cap.release()

    print(
        "1-9 no fue aceptado."
    )
    print(
        "Probando gainrange 1-8..."
    )

    return cv2.VideoCapture(
        make_pipeline(
            GAIN_MAX_FALLBACK
        ),
        cv2.CAP_GSTREAMER
    )


# ============================================================
# MAIN
# ============================================================

load_config()

print("")
print("======================================")
print("  NOIR FILTER LAB v8 - GAIN 1-9")
print("======================================")
print(
    "Exposure compensation:",
    EXPOSURE_COMPENSATION
)
print(
    "Gain range: 1-",
    GAIN_MAX_PRIMARY
)
print(
    "Exposure:",
    EXPOSURE_MIN_NS,
    "a",
    EXPOSURE_MAX_NS,
    "ns"
)
print("")
print("1 = Original NoIR")
print("2 = Red Bandpass virtual")
print("3 = IR-Cut / RGB virtual")
print("4 = Blue Bandpass virtual")
print("5 = IR Bandpass virtual")
print("6 = UV Bandpass virtual")
print("")
print("+ / =  Mas color en IR-CUT/RGB")
print("-      Menos color en IR-CUT/RGB")
print("0      Reset de color a 1.20")
print("S = Guardar")
print("Q = Salir")
print("======================================")
print("")

cap = open_camera()

if not cap.isOpened():
    print(
        "ERROR: no se pudo abrir la OV5647."
    )
    raise SystemExit(1)

window_name = (
    "NOIR FILTER LAB"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    WIDTH,
    HEIGHT
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)

fps = 0.0
frames = 0
t0 = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print(
            "ERROR al recibir frame."
        )
        break

    filtered = apply_filter(
        frame,
        current_filter
    )

    frames += 1
    now = time.time()

    if (
        now - t0
        >= 1.0
    ):
        fps = (
            frames
            / (now - t0)
        )

        frames = 0
        t0 = now

    draw_interface(
        filtered,
        fps
    )

    cv2.imshow(
        window_name,
        filtered
    )

    key = (
        cv2.waitKey(1)
        & 0xFF
    )

    if key == ord("1"):
        change_filter(0)

    elif key == ord("2"):
        change_filter(1)

    elif key == ord("3"):
        change_filter(2)

    elif key == ord("4"):
        change_filter(3)

    elif key == ord("5"):
        change_filter(4)

    elif key == ord("6"):
        change_filter(5)

    elif key in (
        ord("+"),
        ord("=")
    ):
        change_rgb_color(
            RGB_COLOR_STEP
        )

    elif key == ord("-"):
        change_rgb_color(
            -RGB_COLOR_STEP
        )

    elif key == ord("0"):
        reset_rgb_color()

    elif key in (
        ord("s"),
        ord("S")
    ):
        save_frame(frame)

    elif key in (
        ord("q"),
        ord("Q")
    ):
        break

save_config()

cap.release()
cv2.destroyAllWindows()

print("")
print(
    "Programa cerrado. Ultimo filtro:",
    FILTERS[current_filter]
)