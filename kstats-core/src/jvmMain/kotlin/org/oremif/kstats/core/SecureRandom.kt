package org.oremif.kstats.core

import kotlin.random.Random
import kotlin.random.asKotlinRandom

private val secureRandomInstance: Random by lazy { java.security.SecureRandom().asKotlinRandom() }

public actual fun secureRandom(): Random = secureRandomInstance
