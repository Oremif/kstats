package org.oremif.kstats.core

import kotlin.random.Random

/**
 * Returns a cryptographically secure random number generator for the current platform.
 *
 * Each platform uses its native secure random source: `java.security.SecureRandom` on JVM
 * and Android, `SecRandomCopyBytes` on Apple platforms, `/dev/urandom` on Linux,
 * `BCryptGenRandom` on Windows, and `crypto.getRandomValues` on JS and Wasm.
 *
 * The returned [Random] instance is suitable for generating random samples in distribution
 * `sample()` methods. It should not be used for high-throughput scenarios where performance
 * is critical — use `kotlin.random.Random` instead in those cases.
 *
 * ### Example:
 * ```kotlin
 * val rng = secureRandom()
 * rng.nextDouble() // cryptographically random value in [0, 1)
 * ```
 *
 * @return a platform-specific [Random] backed by a cryptographic random source.
 */
public expect fun secureRandom(): Random
