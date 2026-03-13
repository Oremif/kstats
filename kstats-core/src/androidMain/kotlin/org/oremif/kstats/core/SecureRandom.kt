package org.oremif.kstats.core

import kotlin.random.Random
import kotlin.random.asKotlinRandom

public actual fun secureRandom(): Random = java.security.SecureRandom().asKotlinRandom()
