package org.oremif.kstats.core

import kotlin.random.Random

public actual fun secureRandom(): Random = Random.Default
