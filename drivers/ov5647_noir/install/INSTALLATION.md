# Instalación documentada del módulo OV5647

## Compatibilidad

Este procedimiento corresponde a:

```text
Jetson Nano
JetPack 4.6.1
L4T R32.7.1
Kernel 4.9.253-tegra
aarch64
```

No debe asumirse que el mismo módulo binario funciona en otra versión de kernel.

## 1. Módulo compilado

El resultado final de compilación fue:

```text
kernel/nvidia/drivers/media/i2c/ov5647.ko
```

Con:

```text
vermagic:
4.9.253-tegra SMP preempt mod_unload modversions aarch64
```

## 2. Instalación

```bash
sudo cp ov5647.ko /lib/modules/4.9.253-tegra/extra/ov5647.ko
sudo depmod -a
```

## 3. Carga automática

Se utilizó:

```text
/etc/modules-load.d/ov5647.conf
```

Contenido:

```text
ov5647
```

El archivo equivalente incluido en este repositorio está en:

```text
drivers/ov5647_noir/config/ov5647.conf
```

## 4. DTB personalizado

El DTB de prueba fue instalado como:

```text
/boot/tegra210-p3448-0000-p3449-0000-b00-imx219-ov5647.dtb
```

Antes de modificar archivos de arranque se conservaron respaldos de:

- `/boot/Image`
- DTB anterior
- `/boot/extlinux/extlinux.conf`

## 5. Entrada alternativa de extlinux

Se utilizó una entrada de arranque independiente para conservar una ruta de recuperación:

```text
LABEL ov5647-test
  MENU LABEL IMX219 + OV5647 test
  LINUX /boot/Image
  INITRD /boot/initrd
  FDT /boot/tegra210-p3448-0000-p3449-0000-b00-imx219-ov5647.dtb
  APPEND ${cbootargs} quiet root=/dev/mmcblk0p1 rw rootwait rootfstype=ext4 console=ttyS0,115200n8 console=tty0 fbcon=map:0 net.ifnames=0
```

Existe una copia de referencia en:

```text
drivers/ov5647_noir/config/extlinux_ov5647-test.conf.example
```

## 6. Comprobación del módulo

```bash
uname -r
modinfo ov5647
lsmod | grep ov5647
dmesg | grep -i ov5647
```

Resultado esperado durante el proyecto:

```text
ov5647 8-0036: tegracam sensor driver: ov5647_v2.0.6
vi 54080000.vi: subdev ov5647 8-0036 bound
```

## 7. Resultado lógico esperado

```text
sensor-id=0 -> IMX219 RGB
sensor-id=1 -> OV5647 NoIR
```

## Advertencia

El mensaje indicando que un módulo externo **taints the kernel** puede aparecer al cargar un módulo fuera del árbol oficial. En las pruebas del proyecto, ese mensaje por sí solo no indicó un fallo del sensor.
