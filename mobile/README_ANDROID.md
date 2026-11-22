# Synora Android — first local build

This project packages the existing Synora web interface in an Android WebView
and runs the existing Python WebMind engine inside the app using Chaquopy.
The local API binds only to `127.0.0.1:8765`; it is not exposed to the LAN.

## Build an APK

1. Open [GitHub Actions](https://github.com/lith9657-cloud/Synora-/actions).
2. Select **Build Synora Android APK**.
3. Select **Run workflow** on the `main` branch.
4. Open the completed run and download the `synora-android-debug-apk` artifact.
5. Extract the artifact ZIP and install `app-debug.apk` on an ARM64 Android device.

A push changing `mobile/**`, the Python tools, or the web UI also triggers a build.

## Current scope — read before installing

- The Android app embeds the current `synora19_web/index.html` interface.
- The Python WebMind core and its JSON knowledge bases are packaged from
  `synora16_model_clean/tools`.
- Memory and the learned bilingual lexicon are redirected to app-private
  storage, so they persist between launches and are removed when the app is
  uninstalled.
- The optional neural layer is deliberately disabled in this first build.
  The current implementation requires PyTorch plus the packed encoder/model
  assets; PyTorch has not been ported or validated in this Android package.
- The app can use internet retrieval when the device is online. Local skills,
  packaged knowledge, and app-private memory are available without a server.
- This is an ARM64 debug APK, not a signed Play Store release.

## Technical choices

- Chaquopy 17.2 / Python 3.12
- Android Gradle Plugin 8.13.2 / Gradle 8.13 / JDK 17
- minSdk 24, targetSdk 35
- ARM64 ABI
- Existing web UI and engine source are reused without editing their files.

## Remaining work for full neural parity

1. Port or replace the PyTorch encoder with an Android-compatible runtime.
2. Package and validate the exact neural assets (encoder weights, anchors,
   whitening and projection files).
3. Run clean-room Android tests on the target phone, including memory
   persistence, offline skills, online retrieval and dialect understanding.
4. Sign a release APK only after those tests pass.
