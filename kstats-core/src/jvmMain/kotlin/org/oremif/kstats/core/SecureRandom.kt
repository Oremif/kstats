package org.oremif.kstats.core

import java.util.concurrent.ThreadLocalRandom
import kotlin.random.Random
import kotlin.random.asKotlinRandom

public actual fun secureRandom(): Random = ThreadLocalRandom.current().asKotlinRandom()
