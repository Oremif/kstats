package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class NonParametricTest {

    @Test
    fun testMannWhitneyUSignificant() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testWilcoxonSignedRank() {
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val result = wilcoxonSignedRankTest(s1, s2)
        assertTrue(result.statistic > 0)
    }

    @Test
    fun testKolmogorovSmirnovOneSample() {
        // Test standard normal sample against normal distribution
        val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
        val result = kolmogorovSmirnovTest(sample, org.oremif.kstats.distributions.NormalDistribution.STANDARD)
        assertFalse(result.isSignificant(), "Normal-looking data should not be significant against normal")
    }

    @Test
    fun testKolmogorovSmirnovTwoSample() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = kolmogorovSmirnovTest(s1, s2)
        assertTrue(result.statistic > 0.5)
    }

    @Test
    fun testShapiroWilk() {
        // Normal-ish data
        val data = doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5)
        val result = shapiroWilkTest(data)
        assertTrue(result.statistic > 0.8) // Should be close to 1 for normal data
    }
}
