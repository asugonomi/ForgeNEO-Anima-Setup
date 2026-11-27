# Third-party notices

## Android application

The Android application uses components published under the Apache License, Version 2.0. A copy of the license and the Maven coordinates recorded for the release are included. [License text](third-party/Apache-2.0.txt) · [Dependency inventory](third-party/android-release-dependencies.txt)

| Component | Attribution and upstream project |
| --- | --- |
| AndroidX, including Compose, Material 3, Activity, Lifecycle, WindowManager and Graphics Path | The Android Open Source Project and AndroidX contributors. [AndroidX](https://developer.android.com/jetpack/androidx) |
| Kotlin standard library and JetBrains annotations | JetBrains and contributors. [Kotlin](https://github.com/JetBrains/kotlin) · [JetBrains annotations](https://github.com/JetBrains/java-annotations) |
| kotlinx.coroutines for Android / JVM | JetBrains and contributors. [kotlinx.coroutines](https://github.com/Kotlin/kotlinx.coroutines) |
| Guava ListenableFuture | Google and Guava contributors. [Guava](https://github.com/google/guava) |

The dependency inventory includes the version-selection BOMs and platform metadata used by Gradle.

## Gradle Wrapper

The source distribution includes the Gradle 8.7 Wrapper JAR, its configuration and launcher scripts. Gradle and its contributors supply these files under the Apache License, Version 2.0. The launcher scripts retain their upstream copyright headers. The Gradle distribution downloaded at build time provides its own license and notice files, reproduced here for reference. [Gradle license file](third-party/Gradle-8.7-LICENSE.txt) · [Gradle notice](third-party/Gradle-8.7-NOTICE.txt) · [Gradle Wrapper documentation](https://docs.gradle.org/8.7/userguide/gradle_wrapper.html)

## PC companion and integration

The companion imports FastAPI, Starlette, Pillow, Requests, Uvicorn and cryptography from the PC's Python environment. The optional Forge integrations use the corresponding projects' installed software, models and dictionaries. Each project distributes its own license and notices with its installation.

The offline pairing card uses Project Nayuki's QR Code generator, version 1.8.0, under the MIT License. Copyright and license terms are retained in the distributed Python file. [Upstream source](https://github.com/nayuki/QR-Code-generator/blob/v1.8.0/python/qrcodegen.py)
