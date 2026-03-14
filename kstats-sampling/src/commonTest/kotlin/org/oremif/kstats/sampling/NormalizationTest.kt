package org.oremif.kstats.sampling

import kotlin.test.Test
import kotlin.test.assertEquals

class NormalizationTest {

    @Test
    fun testZScore() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val z = data.zScore()
        // Mean of z-scores should be ~0
        assertEquals(0.0, z.average(), 1e-10)
    }

    @Test
    fun testMinMaxNormalize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val norm = data.minMaxNormalize()
        assertEquals(0.0, norm[0], 1e-10)
        assertEquals(1.0, norm[4], 1e-10)
        assertEquals(0.5, norm[2], 1e-10)
    }

    @Test
    fun testMinMaxNormalizeCustomRange() {
        val data = doubleArrayOf(0.0, 5.0, 10.0)
        val norm = data.minMaxNormalize(-1.0, 1.0)
        assertEquals(-1.0, norm[0], 1e-10)
        assertEquals(0.0, norm[1], 1e-10)
        assertEquals(1.0, norm[2], 1e-10)
    }
}
