package org.oremif.kstats.core

import org.khronos.webgl.Int32Array
import org.khronos.webgl.get
import kotlin.random.Random

/**
 * Returns a cryptographically secure random number generator backed by the Web Crypto API.
 *
 * Uses `crypto.getRandomValues` to fill an internal buffer of 64 integers, refilling
 * when exhausted. This avoids calling into the native crypto API on every random number
 * request.
 *
 * @return a [Random] instance backed by `crypto.getRandomValues`.
 */
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
