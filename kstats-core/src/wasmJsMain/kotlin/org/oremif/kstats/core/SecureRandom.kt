@file:OptIn(ExperimentalWasmJsInterop::class)

package org.oremif.kstats.core

import kotlin.js.ExperimentalWasmJsInterop
import kotlin.random.Random

public actual fun secureRandom(): Random = WasmJsSecureRandom

private object WasmJsSecureRandom : Random() {
    private val buffer = IntArray(64)
    private var position = buffer.size

    override fun nextBits(bitCount: Int): Int {
        if (position >= buffer.size) {
            fillBuffer()
        }
        val value = buffer[position++]
        return value.ushr(32 - bitCount)
    }

    private fun fillBuffer() {
        val jsArray = createInt32Array(buffer.size)
        cryptoGetRandomValues(jsArray)
        for (i in buffer.indices) {
            buffer[i] = readInt32Array(jsArray, i)
        }
        position = 0
    }
}

@JsFun("(size) => new Int32Array(size)")
private external fun createInt32Array(size: Int): JsAny

@JsFun("(array) => crypto.getRandomValues(array)")
private external fun cryptoGetRandomValues(array: JsAny): JsAny

@JsFun("(array, index) => array[index]")
private external fun readInt32Array(array: JsAny, index: Int): Int
