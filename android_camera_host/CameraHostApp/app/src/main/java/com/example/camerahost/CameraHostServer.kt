package com.example.camerahost

import android.content.Context
import android.util.Log
import fi.iki.elonen.NanoHTTPD

class CameraHostServer(
    private val context: Context,
    private val controller: Camera2Controller,
    port: Int = 8765,
) : NanoHTTPD(port) {
    private val tag = "CameraHostServer"

    override fun serve(session: IHTTPSession): Response {
        return try {
            val method = session.method?.name ?: "GET"
            val path = session.uri ?: "/"
            Log.i(tag, "$method $path")

            when {
                method == "GET" && path == "/status" -> json(
                    mapOf(
                        "ok" to true,
                        "device" to android.os.Build.MODEL,
                        "sdkInt" to android.os.Build.VERSION.SDK_INT,
                        "activeCameraId" to controller.getActiveCameraId(),
                        "cameraIds" to controller.listCameraIds(),
                        "port" to listeningPort,
                    )
                )

                method == "GET" && path == "/capabilities" -> json(controller.getCapabilities())

                method == "POST" && path == "/settings" -> {
                    val body = readBody(session)
                    val map = BodyJson.parseToMap(body)
                    val applied = controller.applySettings(map)
                    json(
                        mapOf(
                            "ok" to true,
                            "applied" to mapOf(
                                "cameraId" to applied.cameraId,
                                "jpegQuality" to applied.jpegQuality,
                                "aeMode" to applied.aeMode,
                                "exposureTimeNs" to applied.exposureTimeNs,
                                "iso" to applied.iso,
                                "exposureCompensation" to applied.exposureCompensation,
                                "afMode" to applied.afMode,
                                "focusDistance" to applied.focusDistance,
                                "awbMode" to applied.awbMode,
                                "eis" to applied.eis,
                                "ois" to applied.ois,
                                "zoomRatio" to applied.zoomRatio,
                            )
                        )
                    )
                }

                method == "POST" && path == "/capture/jpeg" -> {
                    val body = readBody(session)
                    val overrides = BodyJson.parseToMapOrNull(body)
                    val (jpeg, meta) = controller.captureJpeg(overrides)
                    val resp = newFixedLengthResponse(Response.Status.OK, "image/jpeg", jpeg.inputStream(), jpeg.size.toLong())
                    resp.addHeader("X-Capture-Meta", Json.obj(meta))
                    resp
                }

                else -> newFixedLengthResponse(
                    Response.Status.NOT_FOUND,
                    "text/plain",
                    "Not found. Endpoints: GET /status, GET /capabilities, POST /settings, POST /capture/jpeg"
                )
            }
        } catch (e: Exception) {
            Log.e(tag, "Error serving request", e)
            newFixedLengthResponse(Response.Status.INTERNAL_ERROR, "application/json", Json.obj(mapOf("ok" to false, "error" to (e.message ?: e.toString()))))
        }
    }

    private fun json(map: Map<String, Any?>): Response {
        val s = Json.obj(map)
        val r = newFixedLengthResponse(Response.Status.OK, "application/json", s)
        r.addHeader("Access-Control-Allow-Origin", "*")
        r.addHeader("Access-Control-Allow-Headers", "*")
        return r
    }

    private fun readBody(session: IHTTPSession): String {
        val files = HashMap<String, String>()
        session.parseBody(files)
        // NanoHTTPD stores the raw body in a tmp file under key "postData"
        val raw = files["postData"] ?: ""
        return raw
    }
}

/**
 * Very small JSON body parser for flat objects only (string/number/bool/null).
 * Enough for our simple settings/capture override payloads.
 */
private object BodyJson {
    fun parseToMapOrNull(body: String?): Map<String, Any?>? {
        val b = body?.trim().orEmpty()
        if (b.isEmpty()) return null
        return parseToMap(b)
    }

    fun parseToMap(body: String): Map<String, Any?> {
        val b = body.trim()
        if (b.isEmpty() || b == "null") return emptyMap()
        if (!b.startsWith("{") || !b.endsWith("}")) return emptyMap()

        // Extremely simple parser: split top-level commas not inside quotes.
        val inner = b.substring(1, b.length - 1).trim()
        if (inner.isEmpty()) return emptyMap()

        val pairs = splitTopLevel(inner)
        val out = mutableMapOf<String, Any?>()
        for (p in pairs) {
            val idx = p.indexOf(":")
            if (idx <= 0) continue
            val k = unquote(p.substring(0, idx).trim())
            val rawV = p.substring(idx + 1).trim()
            out[k] = parseValue(rawV)
        }
        return out
    }

    private fun splitTopLevel(s: String): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        var inQuotes = false
        var esc = false
        for (ch in s) {
            if (esc) {
                sb.append(ch)
                esc = false
                continue
            }
            when (ch) {
                '\\' -> {
                    sb.append(ch)
                    esc = true
                }
                '"' -> {
                    sb.append(ch)
                    inQuotes = !inQuotes
                }
                ',' -> {
                    if (inQuotes) sb.append(ch)
                    else {
                        out.add(sb.toString().trim())
                        sb.setLength(0)
                    }
                }
                else -> sb.append(ch)
            }
        }
        if (sb.isNotEmpty()) out.add(sb.toString().trim())
        return out
    }

    private fun unquote(s: String): String {
        val t = s.trim()
        if (t.startsWith("\"") && t.endsWith("\"") && t.length >= 2) {
            return t.substring(1, t.length - 1).replace("\\\"", "\"").replace("\\\\", "\\")
        }
        return t
    }

    private fun parseValue(v: String): Any? {
        val t = v.trim()
        if (t == "null") return null
        if (t == "true") return true
        if (t == "false") return false
        if (t.startsWith("\"") && t.endsWith("\"")) return unquote(t)
        // number?
        return t.toLongOrNull() ?: t.toDoubleOrNull() ?: t
    }
}

