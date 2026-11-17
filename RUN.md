# Ejecución del flujo principal

## 1. Laptop

Instalar dependencias:

```powershell
py -3.13 -m pip install -r requirements/laptop.txt
```

Ejecutar el receptor/dataset:

```powershell
py -3.13 code/main/laptop/V19.py
```

Puertos usados por la laptop:

- 5050: servidor de compatibilidad y prueba `PING/PONG` (también acepta `DSET` de clientes anteriores).
- 5051: preview y telemetría `FPR8`.
- 5052: stream original RGB + NoIR `FORG`.

El firewall de Windows debe permitir Python en la red privada utilizada entre la Jetson y la laptop.

## 2. Jetson Nano - servidor YOLO

Ejecutar dentro del entorno/Docker CUDA donde ya funciona Ultralytics:

```bash
python3 code/main/jetson/yolo11_v9_server_docker.py
```

El servidor utiliza:

```text
127.0.0.1:5070
modelo: yolo11n.pt
protocolo: FRM2
```

## 3. Jetson Nano - adquisición RGB + NoIR

En otra terminal de la Jetson:

```bash
python3 code/main/jetson/JETSON_V11_10_ESTABLE.py IP_DE_LA_LAPTOP
```

Ejemplo:

```bash
python3 code/main/jetson/JETSON_V11_10_ESTABLE.py 192.168.1.50
```

La IP se guarda localmente en `laptop_transfer_config.json`, archivo excluido del repositorio por `.gitignore`.

## 4. NOIR FILTER LAB v8.1

Para probar la cámara NoIR de forma independiente:

```bash
python3 code/tools/noir_interface_v8_1.py
```

Configuración base conservada:

- OV5647 NoIR.
- `sensor-id=1`.
- 960 x 540.
- `exposurecompensation=1.2`.
- ganancia máxima primaria 9, fallback 8.
- exposición 20–33.3 ms.
- seis representaciones digitales NoIR.

## Archivos que NO deben subirse

No subir datasets de personas, capturas faciales, archivos Excel generados, IPs locales ni pesos/modelos binarios.
