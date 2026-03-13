package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TTestTest {

    @Test
    fun testOneSampleTTest() {
        // Data: sample mean close to 0 with large variance -> not significant
        val sample = doubleArrayOf(1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
        val result = tTest(sample, mu = 0.0)
        assertFalse(result.isSignificant())
        assertEquals("One-Sample t-Test", result.testName)
    }

    @Test
    fun testOneSampleTTestSignificant() {
        // Data: clearly above 0
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 0.0)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testTwoSampleWelch() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = tTest(s1, s2)
        assertTrue(result.isSignificant(), "Clearly different means should be significant")
        assertTrue(result.testName.contains("Welch"))
    }

    @Test
    fun testTwoSampleEqualVariances() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = tTest(s1, s2, equalVariances = true)
        assertTrue(result.isSignificant())
        assertTrue(result.testName.contains("Equal Variances"))
    }

    @Test
    fun testPairedTTest() {
        val before = doubleArrayOf(200.0, 190.0, 210.0, 180.0, 195.0)
        val after = doubleArrayOf(190.0, 180.0, 195.0, 170.0, 185.0)
        val result = pairedTTest(before, after)
        assertEquals("Paired t-Test", result.testName)
        assertTrue(result.statistic > 0) // before > after
    }

    @Test
    fun testConfidenceInterval() {
        val sample = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0)
        val result = tTest(sample, mu = 0.0, confidenceLevel = 0.95)
        val ci = result.confidenceInterval!!
        val mean = sample.average()
        assertTrue(ci.first < mean && mean < ci.second)
    }
}

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

class AnovaTest {

    @Test
    fun testOneWayAnovaSignificant() {
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)
        val result = oneWayAnova(g1, g2, g3)
        assertTrue(result.pValue < 0.001, "Very different groups should have low p-value")
        assertEquals(2, result.dfBetween)
        assertEquals(12, result.dfWithin)
    }

    @Test
    fun testOneWayAnovaNotSignificant() {
        val g1 = doubleArrayOf(5.0, 5.1, 4.9, 5.0, 5.2)
        val g2 = doubleArrayOf(5.0, 5.0, 5.1, 4.9, 5.0)
        val result = oneWayAnova(g1, g2)
        assertFalse(result.pValue < 0.05)
    }
}

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

class FisherExactTestTest {

    @Test
    fun testTwoByTwo() {
        // Classic example: [[1, 9], [11, 3]]
        val table = arrayOf(intArrayOf(1, 9), intArrayOf(11, 3))
        val result = fisherExactTest(table)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testNotSignificant() {
        val table = arrayOf(intArrayOf(5, 5), intArrayOf(5, 5))
        val result = fisherExactTest(table)
        assertFalse(result.isSignificant())
    }
}
