#!/usr/bin/env bash
set -e

echo "=== Kernel ==="
uname -r

echo
echo "=== Modulo OV5647 ==="
modinfo ov5647 2>/dev/null || true
lsmod | grep ov5647 || true

echo
echo "=== Deteccion kernel ==="
dmesg | grep -i -E "ov5647|imx219" | tail -n 50 || true

echo
echo "=== Dispositivos video ==="
ls -l /dev/video* 2>/dev/null || true

echo
echo "=== Prueba Argus RGB sensor-id=0 ==="
gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 num-buffers=60 !   'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12' !   nvvidconv ! fakesink

echo
echo "=== Prueba Argus NoIR OV5647 sensor-id=1 ==="
gst-launch-1.0 -e nvarguscamerasrc sensor-id=1 num-buffers=60   exposurecompensation=1.2 gainrange="1 9"   ispdigitalgainrange="1 1"   exposuretimerange="20000000 33333333" !   'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1,format=NV12' !   nvvidconv ! fakesink
