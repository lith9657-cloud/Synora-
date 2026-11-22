plugins {
    id("com.android.application")
    id("com.chaquo.python")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.synora.android"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.synora.android"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"

        // First target: ARM64 Android phones.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    sourceSets {
        getByName("main") {
            // Reuse the existing web interface without changing its source.
            assets.srcDirs("src/main/assets", "../../synora19_web")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

chaquopy {
    defaultConfig {
        version = "3.12"
        buildPython("python3.12")
        // PyTorch is not included in this first Android build.
        pyc {
            src = false
        }
    }

}
