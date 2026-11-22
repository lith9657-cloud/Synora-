package com.synora.android

import android.app.Activity
import android.os.Bundle
import android.webkit.WebChromeClient
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.TextView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import java.io.File

class MainActivity : Activity() {
    private lateinit var webView: WebView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        webView = WebView(this).apply {
            settings.javaScriptEnabled = true
            settings.domStorageEnabled = true
            settings.allowFileAccess = false
            settings.allowContentAccess = false
            webViewClient = WebViewClient()
            webChromeClient = WebChromeClient()
        }
        setContentView(webView)

        Thread({
            try {
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(this))
                }

                // Copy the existing web UI into writable app-private storage.
                val indexFile = File(filesDir, "synora-index.html")
                assets.open("index.html").use { input ->
                    indexFile.outputStream().use { output -> input.copyTo(output) }
                }

                // Starts the Python engine/API on loopback only.
                Python.getInstance()
                    .getModule("android_engine")
                    .callAttr("start_server", filesDir.absolutePath, 8765)

                runOnUiThread {
                    webView.loadUrl("http://127.0.0.1:8765/")
                }
            } catch (error: Exception) {
                val message = error.message ?: error.javaClass.simpleName
                runOnUiThread {
                    val errorView = TextView(this).apply {
                        text = "تعذر تشغيل محرك Synora داخل التطبيق.\n\n$message"
                        textSize = 16f
                        setPadding(32, 48, 32, 32)
                    }
                    setContentView(errorView)
                }
            }
        }, "synora-engine-bootstrap").start()
    }

    @Deprecated("Deprecated in Android API")
    override fun onBackPressed() {
        if (::webView.isInitialized && webView.canGoBack()) {
            webView.goBack()
        } else {
            super.onBackPressed()
        }
    }
}
