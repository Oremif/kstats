@file:OptIn(ExperimentalForeignApi::class)

package org.oremif.kstats.core

import kotlin.random.Random
import kotlinx.cinterop.ExperimentalForeignApi
import platform.posix.arc4random

public actual fun secureRandom(): Random = Arc4Random

private object Arc4Random : Random() {
    override fun nextBits(bitCount: Int): Int = arc4random().toInt().ushr(32 - bitCount)
}
