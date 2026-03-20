package org.oremif.kstats.core

import kotlin.random.Random

// Random.Default on wasmWasi is backed by the WASI `random_get` syscall,
// which is specified to return cryptographically secure random bytes.
public actual fun secureRandom(): Random = Random.Default
