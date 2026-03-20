@file:OptIn(ExperimentalForeignApi::class)

package org.oremif.kstats.core

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.reinterpret
import kotlinx.cinterop.usePinned
import platform.windows.BCRYPT_USE_SYSTEM_PREFERRED_RNG
import platform.windows.BCryptGenRandom
import kotlin.random.Random

public actual fun secureRandom(): Random = WinSecureRandom()

private class WinSecureRandom : Random() {
    private val buffer = ByteArray(256)
    private var position = buffer.size

    override fun nextBits(bitCount: Int): Int {
        if (position + 4 > buffer.size) {
            fillBuffer()
        }
        val value = (buffer[position].toInt() and 0xFF shl 24) or
            (buffer[position + 1].toInt() and 0xFF shl 16) or
            (buffer[position + 2].toInt() and 0xFF shl 8) or
            (buffer[position + 3].toInt() and 0xFF)
        position += 4
        return value.ushr(32 - bitCount)
    }

    private fun fillBuffer() {
        buffer.usePinned { pinned ->
            val status = BCryptGenRandom(
                null,
                pinned.addressOf(0).reinterpret(),
                buffer.size.toUInt(),
                BCRYPT_USE_SYSTEM_PREFERRED_RNG.toUInt(),
            )
            check(status >= 0) { "BCryptGenRandom failed with status: $status" }
        }
        position = 0
    }
}
