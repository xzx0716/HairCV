# -*- coding: utf-8 -*-
"""
WiFi 头皮检测仪（MaxSee / Inskam 类）直连抓帧模块

原理：
  这类检测仪开机后会发出自己的 WiFi 热点（名称通常含 MaxSee/inskam/microscope）。
  手机/电脑连上该热点后，设备在局域网内提供一个 HTTP 摄像头服务，
  MaxSee App 本质上就是访问设备的局域网地址拉取画面。
  本模块在电脑端做同样的事——自动探测常见设备地址，直接抓拍一帧。

常见设备地址（不同厂商固件不同，全部尝试一遍）：
  http://192.168.10.123:8080/?action=snapshot   (快照接口)
  http://192.168.10.123:8080/?action=stream      (MJPEG 视频流)
  http://192.168.1.1:8080/ ...

使用前提：
  电脑已连接到检测仪发出的 WiFi 热点。
"""

import socket
import io
import urllib.request
import urllib.error
from typing import Optional, List, Tuple
from PIL import Image
import numpy as np

# 常见候选地址（IP, 端口）
CANDIDATE_HOSTS: List[Tuple[str, int]] = [
    ("192.168.10.123", 8080),  # MaxSee / inskam 类最常见
    ("192.168.10.123", 80),
    ("192.168.1.1", 8080),
    ("192.168.1.1", 80),
    ("192.168.10.1", 8080),
    ("192.168.10.1", 80),
    ("192.168.22.1", 8080),
    ("192.168.100.1", 8080),
]

# 快照接口路径（HTTP GET 直接返回一张 JPEG）
SNAPSHOT_PATHS = [
    "/?action=snapshot",
    "/snapshot.jpg",
    "/photo.jpg",
    "/cam.jpg",
    "/capture",
    "/snapshot.cgi",
    "/image/jpeg.cgi",
    "/?action=capture",
]

# MJPEG 视频流路径（cv2.VideoCapture 可直接读取）
STREAM_PATHS = [
    "/?action=stream",
    "/stream",
    "/video",
    "/mjpegfeed",
    "/live",
    "/videostream.cgi",
]


def _tcp_open(ip: str, port: int, timeout: float = 1.0) -> bool:
    """快速探测端口是否开放"""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, socket.timeout):
        return False


def _http_get_jpeg(url: str, timeout: float = 4.0) -> Optional[np.ndarray]:
    """HTTP GET 一个快照接口，返回 RGB ndarray，失败返回 None"""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HairCV/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(2 * 1024 * 1024)  # 最多读2MB
            ctype = resp.headers.get("Content-Type", "")
        if len(data) < 1000:
            return None
        # 快照接口通常直接返回 JPEG；MJPEG 也取第一段 JPEG 试试
        img = _decode_jpeg(data)
        if img is None and "multipart" in ctype:
            img = _decode_mjpeg_first_frame(data)
        return img
    except Exception:
        return None


def _decode_jpeg(data: bytes) -> Optional[np.ndarray]:
    """解码字节流中的第一张 JPEG"""
    try:
        pil = Image.open(io.BytesIO(data))
        pil = pil.convert("RGB")
        return np.array(pil)
    except Exception:
        return None


def _decode_mjpeg_first_frame(data: bytes) -> Optional[np.ndarray]:
    """从 MJPEG multipart 字节流里提取第一帧 JPEG"""
    soi = data.find(b"\xff\xd8")  # JPEG 起始标记
    eoi = data.find(b"\xff\xd9", soi if soi >= 0 else 0)  # JPEG 结束标记
    if soi >= 0 and eoi > soi:
        return _decode_jpeg(data[soi:eoi + 2])
    return None


def _grab_stream_frame(url: str, timeout: float = 6.0) -> Optional[np.ndarray]:
    """用 OpenCV 从 MJPEG 流读取一帧"""
    try:
        import cv2
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        _prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
        if _prop is not None:
            cap.set(_prop, int(timeout * 1000))
        ok = cap.isOpened()
        frame = None
        if ok:
            for _ in range(10):
                ret, fr = cap.read()
                if ret and fr is not None:
                    frame = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    break
        cap.release()
        return frame
    except Exception:
        return None


def discover_device(verbose: bool = True) -> Optional[str]:
    """
    Auto-discover the scanner address.
    Returns: base URL e.g. "http://192.168.10.123:8080", or None if not found.
    """
    if verbose:
        print("[probe] Scanning common scanner addresses (make sure this computer is on the scanner's WiFi hotspot)...")
    for ip, port in CANDIDATE_HOSTS:
        if not _tcp_open(ip, port):
            continue
        base = f"http://{ip}:{port}"
        if verbose:
            print(f"[probe] Open port found: {base}")
        for path in SNAPSHOT_PATHS:
            img = _http_get_jpeg(base + path)
            if img is not None:
                if verbose:
                    print(f"[probe] Snapshot endpoint OK: {base}{path}")
                return base
        for path in STREAM_PATHS:
            img = _grab_stream_frame(base + path)
            if img is not None:
                if verbose:
                    print(f"[probe] Video stream OK: {base}{path}")
                return base
    if verbose:
        print("[probe] Scanner not found. Check: 1) computer on scanner hotspot 2) MaxSee shows live view on phone")
    return None


def grab_frame(base_url: Optional[str] = None) -> Tuple[Optional[np.ndarray], str]:
    """
    Capture one frame from the scanner.

    Args:
        base_url: e.g. "http://192.168.10.123:8080"; None for auto-detect
    Returns:
        (RGB ndarray or None, status message)
    """
    if base_url is None:
        base_url = discover_device()
        if base_url is None:
            return None, ("Scanner not found. Connect this computer to the scanner's WiFi hotspot "
                          "(same hotspot your phone uses for MaxSee) and try again.")

    # Try snapshots first (fast)
    for path in SNAPSHOT_PATHS:
        img = _http_get_jpeg(base_url + path)
        if img is not None:
            return img, f"Frame captured (snapshot {base_url}{path})."

    # Then try video streams (slower)
    for path in STREAM_PATHS:
        img = _grab_stream_frame(base_url + path)
        if img is not None:
            return img, f"Frame captured (stream {base_url}{path})."

    return None, (f"Reached device {base_url} but no snapshot/stream endpoint responded "
                  "(this device may use a proprietary protocol).")


if __name__ == "__main__":
    frame, msg = grab_frame()
    print(msg)
    if frame is not None:
        Image.fromarray(frame).save("_device_test.jpg")
        print(f"Saved _device_test.jpg, shape {frame.shape}")
