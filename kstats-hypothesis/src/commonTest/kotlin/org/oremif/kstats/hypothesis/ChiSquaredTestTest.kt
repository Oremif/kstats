package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ChiSquaredTestTest {

    @Test
    fun testGoodnessOfFitUniform() {
        // Fair die: 6 outcomes, 60 rolls total, ~10 each
        val observed = intArrayOf(8, 12, 11, 9, 10, 10)
        val result = chiSquaredTest(observed)
        assertFalse(result.isSignificant(), "Near-uniform should not be significant")
    }

    @Test
    fun testGoodnessOfFitSignificant() {
        // Very unequal
        val observed = intArrayOf(50, 5, 5, 5, 5, 30)
        val result = chiSquaredTest(observed)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testIndependence() {
        // R: chisq.test(matrix(c(10,20,30,40), nrow=2))
        val table = arrayOf(intArrayOf(10, 30), intArrayOf(20, 40))
        val result = chiSquaredIndependenceTest(table)
        assertEquals(1.0, result.degreesOfFreedom, 1e-10)
    }
}
