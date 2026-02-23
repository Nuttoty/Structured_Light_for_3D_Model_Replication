package com.example.camerahost

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import fi.iki.elonen.NanoHTTPD

class MainActivity : AppCompatActivity() {
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var txtStatus: TextView
    private lateinit var txtUrl: TextView

    private var server: CameraHostServer? = null
    private var controller: Camera2Controller? = null

    private val port = 8765

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        txtStatus = findViewById(R.id.txtStatus)
        txtUrl = findViewById(R.id.txtUrl)

        btnStart.setOnClickListener { startServerWithPermission() }
        btnStop.setOnClickListener { stopServer() }

        updateUi(false)
    }

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }

    private fun startServerWithPermission() {
        val granted = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (!granted) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1001)
            return
        }
        startServer()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1001) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startServer()
            } else {
                txtStatus.text = "Status: camera permission denied"
            }
        }
    }

    private fun startServer() {
        if (server != null) return

        controller = Camera2Controller(this)
        server = CameraHostServer(this, controller!!, port)
        server!!.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)

        val ip = Net.getLikelyDeviceIp(this)
        val url = if (ip != null) "http://$ip:$port" else "http://(your-phone-ip):$port"
        txtUrl.text = "URL: $url"
        txtStatus.text = "Status: running (port $port)"
        updateUi(true)
    }

    private fun stopServer() {
        try {
            server?.stop()
        } catch (_: Exception) {
        }
        server = null
        controller?.close()
        controller = null
        updateUi(false)
        txtStatus.text = "Status: stopped"
        txtUrl.text = "URL: (start server)"
    }

    private fun updateUi(running: Boolean) {
        btnStart.isEnabled = !running
        btnStop.isEnabled = running
    }
}

