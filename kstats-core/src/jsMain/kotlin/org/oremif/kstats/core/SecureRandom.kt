package org.oremif.kstats.core

import kotlin.random.Random

public actual fun secureRandom(): Random = JsSecureRandom

private object JsSecureRandom : Random() {
    private val buffer = Int32Array(64)
    private var position = buffer.length

    override fun nextBits(bitCount: Int): Int {
        if (position >= buffer.length) {
            fillBuffer()
        }
        val value = buffer[position++]
        return value.ushr(32 - bitCount)
    }

    private fun fillBuffer() {
        crypto.getRandomValues(buffer)
        position = 0
    }
}

private external object crypto {
    fun getRandomValues(array: Int32Array)
}

private external class Int32Array(size: Int) {
    val length: Int
    operator fun get(index: Int): Int
}
