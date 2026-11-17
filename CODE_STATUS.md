# Estado de los códigos

## Flujo principal recomendado

| Equipo | Archivo | Estado | Función |
|---|---|---|---|
| Jetson Nano | `code/main/jetson/JETSON_V11_10_ESTABLE.py` | Base estable | RGB + NoIR + TFmini + telemetría YOLO + preview 5051 + stream original 5052 |
| Jetson Nano / Docker CUDA | `code/main/jetson/yolo11_v9_server_docker.py` | Compatible | YOLO11n local por socket `127.0.0.1:5070`, protocolo `FRM2` |
| Laptop | `code/main/laptop/V19.py` | Dataset V16.19 | MediaPipe, 68 landmarks, 52 blendshapes, filtros NoIR, Excel y control de calidad IQR |
| Jetson Nano | `code/tools/noir_interface_v8_1.py` | Versión canónica | NOIR FILTER LAB v8.1 con seis representaciones |

## Compatibilidad de red

- Jetson -> YOLO Docker: puerto **5070**, cabecera **FRM2**.
- Jetson -> laptop:
  - **5050** `PING/PONG`.
  - **5051** preview + telemetría / `FPR8`.
  - **5052** stream original RGB+NoIR / `FORG`.
- `V19.py` escucha **5050**, **5051** y **5052**.

## Importante

Los archivos corresponden a las versiones recuperadas del proyecto. Las funciones que dependen físicamente de CSI, TFmini, CUDA/TensorRT, GPIO/IR y cámaras requieren ese hardware para una prueba completa.
