import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


@dataclass
class CameraSettings:
    """
    Best-effort settings. The phone will apply what it supports.
    Values may be ignored if the device/camera doesn't support them.
    """

    camera_id: Optional[str] = None  # e.g. "0" back, "1" front (varies by device)
    jpeg_quality: int = 95  # 1..100

    # Exposure / gain (Camera2 style)
    ae_mode: Optional[str] = None  # "on" | "off"
    exposure_time_ns: Optional[int] = None  # e.g. 10000000 (10ms)
    iso: Optional[int] = None  # e.g. 200
    exposure_compensation: Optional[int] = None  # EV steps (if AE on)

    # Focus
    af_mode: Optional[str] = None  # "auto" | "off"
    focus_distance: Optional[float] = None  # diopters; 0 = infinity

    # Zoom (best-effort)
    zoom_ratio: Optional[float] = None  # 1.0..max

    # White balance
    awb_mode: Optional[str] = None  # "auto" | "incandescent" | "daylight" ...

    # Stabilization (best-effort)
    ois: Optional[bool] = None
    eis: Optional[bool] = None


class AndroidCameraHostError(RuntimeError):
    pass


class AndroidCameraHostClient:
    """
    Client for the Android "Camera Host" app.

    Works over:
      - Wi-Fi: base_url like "http://PHONE_IP:8765"
      - USB (recommended): use `adb reverse tcp:8765 tcp:8765`, then base_url "http://127.0.0.1:8765"
    """

    def __init__(self, base_url: str, timeout_s: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def ping(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/status", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def get_capabilities(self) -> Dict[str, Any]:
        r = self.session.get(f"{self.base_url}/capabilities", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def apply_settings(self, settings: CameraSettings) -> Dict[str, Any]:
        payload = {k: v for k, v in settings.__dict__.items() if v is not None}
        r = self.session.post(
            f"{self.base_url}/settings",
            json=payload,
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def capture_jpeg(self, settings: Optional[CameraSettings] = None) -> Tuple[bytes, Dict[str, Any]]:
        payload = None
        if settings is not None:
            payload = {k: v for k, v in settings.__dict__.items() if v is not None}

        r = self.session.post(
            f"{self.base_url}/capture/jpeg",
            json=payload,
            timeout=self.timeout_s,
        )
        if r.status_code != 200:
            raise AndroidCameraHostError(f"Capture failed: {r.status_code} {r.text[:500]}")

        meta_header = r.headers.get("X-Capture-Meta", "{}")
        try:
            import json as _json
            meta = _json.loads(meta_header)
        except Exception:
            meta = {"raw_header": meta_header}

        return r.content, meta

    def capture_to_path(self, path: str, settings: Optional[CameraSettings] = None) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data, meta = self.capture_jpeg(settings=settings)
        with open(path, "wb") as f:
            f.write(data)
        return meta

