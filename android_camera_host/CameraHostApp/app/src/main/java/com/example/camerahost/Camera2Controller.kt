package com.example.camerahost

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Range
import android.util.Size
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import kotlin.math.max
import kotlin.math.min

data class AppliedSettings(
    val cameraId: String,
    val jpegQuality: Int,
    val aeMode: String?,
    val exposureTimeNs: Long?,
    val iso: Int?,
    val exposureCompensation: Int?,
    val afMode: String?,
    val focusDistance: Float?,
    val awbMode: String?,
    val eis: Boolean?,
    val ois: Boolean?,
    val zoomRatio: Float?,
)

class Camera2Controller(private val context: Context) {
    private val tag = "Camera2Controller"

    private val cameraManager: CameraManager =
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var imageReader: ImageReader? = null

    private var thread: HandlerThread? = null
    private var handler: Handler? = null

    private var activeCameraId: String? = null

    private var current: AppliedSettings? = null

    fun listCameraIds(): List<String> = cameraManager.cameraIdList.toList()

    fun getActiveCameraId(): String? = activeCameraId

    fun getCapabilities(): Map<String, Any?> {
        val cams = mutableListOf<Map<String, Any?>>()
        for (id in cameraManager.cameraIdList) {
            val cc = cameraManager.getCameraCharacteristics(id)
            val facing = cc.get(CameraCharacteristics.LENS_FACING)
            val facingStr = when (facing) {
                CameraCharacteristics.LENS_FACING_BACK -> "back"
                CameraCharacteristics.LENS_FACING_FRONT -> "front"
                CameraCharacteristics.LENS_FACING_EXTERNAL -> "external"
                else -> "unknown"
            }
            val caps = cc.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)?.toList() ?: emptyList()
            val rawSupported = caps.contains(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)
            val aeRange = cc.get(CameraCharacteristics.CONTROL_AE_COMPENSATION_RANGE)
            val isoRange = cc.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            val expRange = cc.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val maxZoom = cc.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM)
            cams.add(
                mapOf(
                    "cameraId" to id,
                    "facing" to facingStr,
                    "rawSupported" to rawSupported,
                    "aeCompensationRange" to (aeRange?.let { listOf(it.lower, it.upper) }),
                    "isoRange" to (isoRange?.let { listOf(it.lower, it.upper) }),
                    "exposureTimeNsRange" to (expRange?.let { listOf(it.lower, it.upper) }),
                    "maxDigitalZoom" to maxZoom,
                )
            )
        }
        return mapOf(
            "cameras" to cams,
            "notes" to "Support varies by device. RAW requires camera capability RAW.",
        )
    }

    @SuppressLint("MissingPermission")
    @Synchronized
    fun ensureOpen(cameraId: String) {
        if (activeCameraId == cameraId && cameraDevice != null && captureSession != null && handler != null) return
        close()

        thread = HandlerThread("camera2").also { it.start() }
        handler = Handler(thread!!.looper)

        val latch = CountDownLatch(1)
        var openErr: Exception? = null

        cameraManager.openCamera(
            cameraId,
            object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    cameraDevice = camera
                    activeCameraId = cameraId
                    latch.countDown()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    openErr = RuntimeException("Camera disconnected")
                    latch.countDown()
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    openErr = RuntimeException("Camera open error=$error")
                    latch.countDown()
                }
            },
            handler
        )

        if (!latch.await(5, TimeUnit.SECONDS)) {
            close()
            throw RuntimeException("Timeout opening camera")
        }
        if (openErr != null) {
            close()
            throw openErr!!
        }

        // Create a simple still-capture session with ImageReader.
        // Use a conservative size to avoid huge latency on mid phones.
        val size = pickJpegSize(cameraId)
        imageReader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, 2)

        val sessionLatch = CountDownLatch(1)
        var sessErr: Exception? = null
        cameraDevice!!.createCaptureSession(
            listOf(imageReader!!.surface),
            object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    sessionLatch.countDown()
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    sessErr = RuntimeException("CaptureSession configure failed")
                    sessionLatch.countDown()
                }
            },
            handler
        )

        if (!sessionLatch.await(5, TimeUnit.SECONDS)) {
            close()
            throw RuntimeException("Timeout configuring camera session")
        }
        if (sessErr != null) {
            close()
            throw sessErr!!
        }
    }

    fun applySettings(s: Map<String, Any?>): AppliedSettings {
        val camId = (s["camera_id"] as? String) ?: activeCameraId ?: listCameraIds().first()
        ensureOpen(camId)

        val jpegQ = (s["jpeg_quality"] as? Number)?.toInt()?.let { min(100, max(1, it)) } ?: (current?.jpegQuality ?: 95)

        val aeMode = (s["ae_mode"] as? String)?.lowercase()
        val exposureTimeNs = (s["exposure_time_ns"] as? Number)?.toLong()
        val iso = (s["iso"] as? Number)?.toInt()
        val expComp = (s["exposure_compensation"] as? Number)?.toInt()

        val afMode = (s["af_mode"] as? String)?.lowercase()
        val focusDistance = (s["focus_distance"] as? Number)?.toFloat()
        val awbMode = (s["awb_mode"] as? String)?.lowercase()

        val eis = (s["eis"] as? Boolean)
        val ois = (s["ois"] as? Boolean)
        val zoomRatio = (s["zoom_ratio"] as? Number)?.toFloat()

        current = AppliedSettings(
            cameraId = camId,
            jpegQuality = jpegQ,
            aeMode = aeMode,
            exposureTimeNs = exposureTimeNs,
            iso = iso,
            exposureCompensation = expComp,
            afMode = afMode,
            focusDistance = focusDistance,
            awbMode = awbMode,
            eis = eis,
            ois = ois,
            zoomRatio = zoomRatio,
        )
        return current!!
    }

    fun captureJpeg(optionalOverrides: Map<String, Any?>?): Pair<ByteArray, Map<String, Any?>> {
        val merged = mutableMapOf<String, Any?>()
        // Start from current settings
        current?.let {
            merged["camera_id"] = it.cameraId
            merged["jpeg_quality"] = it.jpegQuality
            merged["ae_mode"] = it.aeMode
            merged["exposure_time_ns"] = it.exposureTimeNs
            merged["iso"] = it.iso
            merged["exposure_compensation"] = it.exposureCompensation
            merged["af_mode"] = it.afMode
            merged["focus_distance"] = it.focusDistance
            merged["awb_mode"] = it.awbMode
            merged["eis"] = it.eis
            merged["ois"] = it.ois
            merged["zoom_ratio"] = it.zoomRatio
        }
        optionalOverrides?.forEach { (k, v) -> merged[k] = v }

        val applied = applySettings(merged)

        val reader = imageReader ?: throw RuntimeException("ImageReader not ready")
        val session = captureSession ?: throw RuntimeException("CaptureSession not ready")
        val cam = cameraDevice ?: throw RuntimeException("Camera not ready")

        val captureLatch = CountDownLatch(1)
        var jpegBytes: ByteArray? = null
        var err: Exception? = null

        reader.setOnImageAvailableListener({ ir ->
            try {
                ir.acquireLatestImage()?.use { img ->
                    val buf: ByteBuffer = img.planes[0].buffer
                    val arr = ByteArray(buf.remaining())
                    buf.get(arr)
                    jpegBytes = arr
                }
            } catch (e: Exception) {
                err = e
            } finally {
                captureLatch.countDown()
            }
        }, handler)

        val req = cam.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(reader.surface)
            set(CaptureRequest.JPEG_QUALITY, applied.jpegQuality.toByte())

            // AE / manual exposure
            when (applied.aeMode) {
                "off" -> set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_OFF)
                "on" -> set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
                null -> { /* leave default */ }
            }
            applied.exposureTimeNs?.let { set(CaptureRequest.SENSOR_EXPOSURE_TIME, it) }
            applied.iso?.let { set(CaptureRequest.SENSOR_SENSITIVITY, it) }
            applied.exposureCompensation?.let { set(CaptureRequest.CONTROL_AE_EXPOSURE_COMPENSATION, it) }

            // AF / manual focus distance
            when (applied.afMode) {
                "off" -> set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_OFF)
                "auto" -> set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_AUTO)
                null -> { /* leave default */ }
            }
            applied.focusDistance?.let { set(CaptureRequest.LENS_FOCUS_DISTANCE, it) }

            // AWB
            when (applied.awbMode) {
                "auto" -> set(CaptureRequest.CONTROL_AWB_MODE, CaptureRequest.CONTROL_AWB_MODE_AUTO)
                else -> { /* keep default */ }
            }

            // EIS (video stabilization) - some devices ignore for stills
            applied.eis?.let { enabled ->
                set(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    if (enabled) CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                    else CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_OFF
                )
            }

            // OIS (optical) - may be unsupported or ignored
            applied.ois?.let { enabled ->
                set(
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    if (enabled) CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
                    else CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                )
            }

            // Zoom (digital) via crop region: needs characteristics, best-effort
            applied.zoomRatio?.let { zr ->
                val cc = cameraManager.getCameraCharacteristics(applied.cameraId)
                val rect = cc.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                if (rect != null) {
                    val maxZoom = cc.get(CameraCharacteristics.SCALER_AVAILABLE_MAX_DIGITAL_ZOOM) ?: 1f
                    val z = min(max(1f, zr), maxZoom)
                    val cropW = (rect.width() / z).toInt()
                    val cropH = (rect.height() / z).toInt()
                    val left = (rect.width() - cropW) / 2
                    val top = (rect.height() - cropH) / 2
                    val crop = android.graphics.Rect(left, top, left + cropW, top + cropH)
                    set(CaptureRequest.SCALER_CROP_REGION, crop)
                }
            }
        }

        session.capture(req.build(), object : CameraCaptureSession.CaptureCallback() {}, handler)

        if (!captureLatch.await(10, TimeUnit.SECONDS)) {
            throw RuntimeException("Timeout waiting for JPEG")
        }
        if (err != null) throw err!!
        val out = jpegBytes ?: throw RuntimeException("No JPEG received")

        val meta = mapOf(
            "cameraId" to applied.cameraId,
            "jpegQuality" to applied.jpegQuality,
            "aeMode" to applied.aeMode,
            "exposureTimeNs" to applied.exposureTimeNs,
            "iso" to applied.iso,
            "afMode" to applied.afMode,
            "focusDistance" to applied.focusDistance,
            "zoomRatio" to applied.zoomRatio,
        )
        return Pair(out, meta)
    }

    @Synchronized
    fun close() {
        try {
            imageReader?.close()
        } catch (_: Exception) {
        }
        imageReader = null
        try {
            captureSession?.close()
        } catch (_: Exception) {
        }
        captureSession = null
        try {
            cameraDevice?.close()
        } catch (_: Exception) {
        }
        cameraDevice = null
        activeCameraId = null

        try {
            handler = null
            thread?.quitSafely()
            thread = null
        } catch (e: Exception) {
            Log.w(tag, "Error closing thread: $e")
        }
    }

    private fun pickJpegSize(cameraId: String): Size {
        val cc = cameraManager.getCameraCharacteristics(cameraId)
        val cfg = cc.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
            ?: return Size(1280, 720)
        val sizes = cfg.getOutputSizes(ImageFormat.JPEG)?.toList() ?: return Size(1280, 720)

        // Prefer ~1600px wide class images (good for scanning + not too slow).
        val targetW = 1600
        return sizes.minByOrNull { s -> kotlin.math.abs(s.width - targetW) } ?: sizes.first()
    }
}

