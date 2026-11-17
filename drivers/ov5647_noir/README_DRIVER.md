# Driver OV5647 NoIR para NVIDIA Jetson Nano

> Documentación del driver y Device Tree utilizados en este proyecto para habilitar una Raspberry Pi Camera Rev. 1.3 con sensor OmniVision OV5647 NoIR junto con una cámara IMX219 RGB.

## Plataforma utilizada

| Elemento | Configuración |
|---|---|
| Equipo | NVIDIA Jetson Nano Developer Kit |
| JetPack | 4.6.1 |
| L4T | R32.7.1 |
| Kernel | 4.9.253-tegra |
| Arquitectura | aarch64 |
| Cámara RGB | IMX219 |
| Cámara NoIR | OV5647 / Raspberry Pi Camera Rev. 1.3 |

## Estado final

La configuración funcional utilizada en el proyecto quedó:

| sensor-id | Sensor | Rol | Nodo lógico |
|---:|---|---|---|
| 0 | IMX219 | RGB | /dev/video0 |
| 1 | OV5647 | NoIR | /dev/video1 |

Mensajes de detección observados:

```text
vi-output, imx219 7-0010
ov5647 8-0036: tegracam sensor driver: ov5647_v2.0.6
vi 54080000.vi: subdev ov5647 8-0036 bound
```

## Archivos utilizados durante el port

Los archivos principales empleados durante la implementación fueron:

```text
ov5647.c
ov5647_mode_tbls.h
tegra210-porg-camera-imx219-ov5647.dtsi
ov5647.ko
tegra210-p3448-0000-p3449-0000-b00-imx219-ov5647.dtb
```

### Importante

Los archivos fuente exactos `ov5647.c`, `ov5647_mode_tbls.h` y el `.dtsi` personalizado no se encuentran actualmente recuperados en este repositorio. No se incluyen versiones reconstruidas para evitar presentar como originales archivos que no han sido verificados byte a byte.

Esta carpeta documenta únicamente la configuración que sí quedó registrada y utilizada durante el proyecto.

## Árbol de fuentes utilizado

Kernel:

```text
~/ov5647_work/Linux_for_Tegra/source/public/kernel_src/kernel/kernel-4.9
```

Drivers NVIDIA:

```text
~/ov5647_work/Linux_for_Tegra/source/public/kernel_src/kernel/nvidia/drivers/media/i2c
```

## Cambios realizados durante el port

- Adaptación del driver OV5647 a la API de cámara de NVIDIA para kernel 4.9.
- Corrección de funciones duplicadas y errores de compilación.
- Ajustes relacionados con `gain`, `set_mode` y `fine integration time`.
- Corrección de includes, indentación y warnings.
- Preparación del árbol con `Module.symvers`.
- Uso de `CONFIG_LOCALVERSION="-tegra"`.
- Integración del OV5647 en un Device Tree mixto IMX219 + OV5647.
- Registro del OV5647 en I²C `0x36`.
- Asociación del OV5647 al segundo canal CSI / `serial_e`.

## Módulo compilado

Ruta de salida:

```text
kernel/nvidia/drivers/media/i2c/ov5647.ko
```

`vermagic` utilizado:

```text
4.9.253-tegra SMP preempt mod_unload modversions aarch64
```

## Device Tree

DTS base de la placa:

```text
hardware/nvidia/platform/t210/porg/kernel-dts/
tegra210-p3448-0000-p3449-0000-b00.dts
```

Configuración dual IMX219 usada como base:

```text
hardware/nvidia/platform/t210/porg/kernel-dts/porg-platforms/
tegra210-porg-camera-rbpcv2-dual-imx219.dtsi
```

Configuración final:

- IMX219 conservada en `i2c@0`, `serial_a`.
- Segunda IMX219 de la configuración dual original deshabilitada.
- OV5647 agregado en `i2c@1`.
- Dirección I²C OV5647: `0x36`.
- OV5647 asociado a `serial_e`.
- Compatible utilizado: `nvidia,ov5647`.

## DTB personalizado

```text
/boot/tegra210-p3448-0000-p3449-0000-b00-imx219-ov5647.dtb
```

## Captura final

RGB:

```text
nvarguscamerasrc sensor-id=0 !
video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12 !
nvvidconv !
video/x-raw,width=960,height=540,format=BGRx !
videoconvert !
video/x-raw,format=BGR !
appsink drop=true max-buffers=1 sync=false
```

NoIR:

```text
nvarguscamerasrc sensor-id=1
exposurecompensation=1.2
gainrange="1 9"
ispdigitalgainrange="1 1"
exposuretimerange="20000000 33333333" !
video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12 !
nvvidconv !
video/x-raw,width=960,height=540,format=BGRx !
videoconvert !
video/x-raw,format=BGR !
appsink drop=true max-buffers=1 sync=false
```

La configuración NoIR anterior es la base utilizada posteriormente por **NOIR FILTER LAB v8.1**.
