@file:OptIn(ExperimentalForeignApi::class)

package org.oremif.kstats.core

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.posix.O_RDONLY
import platform.posix.close
import platform.posix.open
import platform.posix.read
import kotlin.random.Random

public actual fun secureRandom(): Random = DevUrandom

private object DevUrandom : Random() {
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
        val fd = open("/dev/urandom", O_RDONLY)
        check(fd >= 0) { "Failed to open /dev/urandom" }
        try {
            var totalRead = 0
            buffer.usePinned { pinned ->
                while (totalRead < buffer.size) {
                    val bytesRead = read(fd, pinned.addressOf(totalRead), (buffer.size - totalRead).toULong())
                    check(bytesRead > 0) { "Failed to read from /dev/urandom" }
                    totalRead += bytesRead.toInt()
                }
            }
        } finally {
            close(fd)
        }
        position = 0
    }
}
