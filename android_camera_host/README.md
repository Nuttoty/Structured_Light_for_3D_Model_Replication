# Android Camera Host (USB/Wi‑Fi) for Structured Light

This folder will contain an Android app that:

- Runs a small HTTP server on port **8765**
- Lets the PC control camera settings (best-effort)
- Lets the PC trigger capture and receive the **JPEG bytes directly**

The Python controller in `server/scanner_controller_android.py` talks to this app.

## Quick USB setup (recommended)

1. Enable **Developer options** on the phone.
2. Enable **USB debugging**.
3. Install Android platform tools (`adb`) on the PC.
4. Connect phone via USB and approve the debugging prompt.
5. Run:

```bash
adb devices
adb reverse tcp:8765 tcp:8765
```

Now the phone server will be reachable from the PC as:

- `http://127.0.0.1:8765`

## Wi‑Fi setup

- Ensure phone + PC are on the same Wi‑Fi.
- Use the phone’s IP shown in the app, e.g. `http://192.168.1.50:8765`

## Build & install the Android app

Open `android_camera_host/CameraHostApp` in **Android Studio**, then:

- **Build** → **Build APK(s)**
- Install on device, or run via **Run** (green play button)

