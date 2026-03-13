package org.oremif.kstats.core

import kotlin.test.Test
import kotlin.test.assertTrue

class SecureRandomTest {

    @Test
    fun returnsWorkingRandom() {
        val rng = secureRandom()
        rng.nextDouble()
    }

    @Test
    fun nextDoubleInRange() {
        val rng = secureRandom()
        repeat(1000) {
            val value = rng.nextDouble()
            assertTrue(value >= 0.0 && value < 1.0, "nextDouble() out of range: $value")
        }
    }

    @Test
    fun nextIntInRange() {
        val rng = secureRandom()
        repeat(1000) {
            val value = rng.nextInt(10, 100)
            assertTrue(value in 10 until 100, "nextInt(10, 100) out of range: $value")
        }
    }

    @Test
    fun producesNonConstantValues() {
        val rng = secureRandom()
        val values = DoubleArray(100) { rng.nextDouble() }
        val distinct = values.toSet().size
        assertTrue(distinct > 50, "Expected >50 distinct values, got $distinct")
    }

    @Test
    fun nextBitsProducesValidRange() {
        val rng = secureRandom()
        repeat(1000) {
            val value = rng.nextBits(16)
            assertTrue(value in 0 until 65536, "nextBits(16) out of range: $value")
        }
    }
}
