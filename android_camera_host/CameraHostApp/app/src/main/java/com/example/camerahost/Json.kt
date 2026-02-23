package com.example.camerahost

/**
 * Minimal JSON helpers to avoid adding heavy deps.
 * This is NOT a full JSON serializer; it's enough for our simple API.
 */
object Json {
    fun escape(s: String): String =
        s.replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t")

    fun obj(map: Map<String, Any?>): String {
        val parts = map.entries.map { (k, v) ->
            "\"${escape(k)}\":${value(v)}"
        }
        return "{${parts.joinToString(",")}}"
    }

    fun arr(values: List<Any?>): String =
        "[${values.joinToString(",") { value(it) }}]"

    private fun value(v: Any?): String =
        when (v) {
            null -> "null"
            is Boolean -> if (v) "true" else "false"
            is Int, is Long, is Float, is Double -> v.toString()
            is String -> "\"${escape(v)}\""
            is Map<*, *> -> obj(v.entries.associate { it.key.toString() to it.value })
            is List<*> -> arr(v as List<Any?>)
            else -> "\"${escape(v.toString())}\""
        }
}

