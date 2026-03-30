package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.distributions.NormalDistribution
import kotlin.math.abs
import kotlin.test.*

class KolmogorovSmirnovTestTest {

    // ===== One-sample KS test =====

    @Test
    fun testOneSampleNormalData() {
        val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        assertFalse(result.isSignificant(), "Normal-looking data should not be significant against normal")
        assertEquals("Kolmogorov-Smirnov Test (One-Sample)", result.testName)
    }

    @Test
    fun testOneSampleScipyReference() {
        // scipy.stats.ks_1samp([-1.0,-0.5,0.0,0.5,1.0,1.5,-1.5,-0.3,0.3,0.8], norm.cdf)
        // D=0.1193, p~0.9858
        val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        assertTrue(
            abs(result.statistic - 0.1193) < 0.01,
            "D statistic: expected~0.1193, actual=${result.statistic}"
        )
        assertTrue(
            abs(result.pValue - 0.9858) < 0.05,
            "p-value: expected~0.9858, actual=${result.pValue}"
        )
    }

    @Test
    fun testOneSampleDPlusDMinus() {
        val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        val dPlus = result.additionalInfo["dPlus"]!!
        val dMinus = result.additionalInfo["dMinus"]!!
        assertTrue(dPlus >= 0.0, "dPlus should be non-negative")
        assertTrue(dMinus >= 0.0, "dMinus should be non-negative")
        assertEquals(result.statistic, maxOf(dPlus, dMinus), 1e-15, "D = max(dPlus, dMinus)")
    }

    @Test
    fun testOneSampleSignificant() {
        // Data clearly not from standard normal (all values large)
        val sample = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        assertTrue(result.isSignificant(), "Data far from normal should be significant")
        assertEquals(1.0, result.statistic, 1e-10, "D should be ~1 for extreme data")
    }

    @Test
    fun testOneSampleSingleElement() {
        val sample = doubleArrayOf(0.0)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        assertTrue(result.statistic in 0.0..1.0, "D should be in [0, 1]")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    // ===== Two-sample KS test =====

    @Test
    fun testTwoSampleDisjoint() {
        // scipy.stats.ks_2samp([1,2,3,4,5], [6,7,8,9,10])
        // D=1.0, p=0.00396825 (with correct effective N)
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = kolmogorovSmirnovTest(s1, s2)
        assertEquals(1.0, result.statistic, 1e-10, "D should be 1.0 for completely separated samples")
        assertTrue(
            abs(result.pValue - 0.00396825) < 0.005,
            "p-value: expected~0.00397, actual=${result.pValue}"
        )
        assertTrue(result.isSignificant(), "Completely separated samples should be significant")
        assertEquals("Kolmogorov-Smirnov Test (Two-Sample)", result.testName)
    }

    @Test
    fun testTwoSampleIdentical() {
        // Identical samples: D=0.0, p=1.0
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kolmogorovSmirnovTest(s1, s1.copyOf())
        assertEquals(0.0, result.statistic, 1e-15, "D should be 0 for identical samples")
        assertEquals(1.0, result.pValue, 1e-6, "p should be 1 for identical samples")
    }

    @Test
    fun testTwoSampleOverlapping() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0)
        val result = kolmogorovSmirnovTest(s1, s2)
        assertTrue(result.statistic > 0.0, "D should be > 0 for different samples")
        assertTrue(result.statistic < 1.0, "D should be < 1 for overlapping samples")
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testTwoSampleUnequalSizes() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0)
        val s2 = doubleArrayOf(4.0, 5.0, 6.0, 7.0, 8.0)
        val result = kolmogorovSmirnovTest(s1, s2)
        assertEquals(1.0, result.statistic, 1e-10, "D should be 1 for disjoint samples")
        assertTrue(result.isSignificant())
    }

    // ===== Validation =====

    @Test
    fun testOneSampleEmptyValidation() {
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(), NormalDistribution.STANDARD)
        }
    }

    @Test
    fun testTwoSampleEmptyFirstValidation() {
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(), doubleArrayOf(1.0))
        }
    }

    @Test
    fun testTwoSampleEmptySecondValidation() {
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(1.0), doubleArrayOf())
        }
    }

    @Test
    fun testTwoSampleBothEmptyValidation() {
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(), doubleArrayOf())
        }
    }

    // ===== NaN / Infinity handling =====

    @Test
    fun testOneSampleWithNaN() {
        val result = kolmogorovSmirnovTest(TestData.WITH_NAN, NormalDistribution.STANDARD)
        TestAssertions.assertNaNResult(result, "when input contains NaN")
        assertTrue(result.additionalInfo["dPlus"]!!.isNaN(), "dPlus should be NaN")
        assertTrue(result.additionalInfo["dMinus"]!!.isNaN(), "dMinus should be NaN")
    }

    @Test
    fun testOneSampleWithInfinity() {
        val result = kolmogorovSmirnovTest(TestData.WITH_POS_INF, NormalDistribution.STANDARD)
        TestAssertions.assertNaNResult(result, "when input contains Infinity")
    }

    @Test
    fun testOneSampleWithNegativeInfinity() {
        val sample = doubleArrayOf(1.0, 2.0, Double.NEGATIVE_INFINITY, 4.0, 5.0)
        val result = kolmogorovSmirnovTest(sample, NormalDistribution.STANDARD)
        TestAssertions.assertNaNResult(result, "when input contains -Infinity")
    }

    @Test
    fun testTwoSampleWithNaN() {
        val result = kolmogorovSmirnovTest(TestData.WITH_NAN, TestData.SEQUENTIAL_6_10)
        TestAssertions.assertNaNResult(result, "when sample1 contains NaN")
    }

    @Test
    fun testTwoSampleSecondWithNaN() {
        val s2 = doubleArrayOf(6.0, Double.NaN, 8.0, 9.0, 10.0)
        val result = kolmogorovSmirnovTest(TestData.SEQUENTIAL_1_5, s2)
        TestAssertions.assertNaNResult(result, "when sample2 contains NaN")
    }

    @Test
    fun testTwoSampleWithInfinity() {
        val s1 = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0)
        val result = kolmogorovSmirnovTest(s1, TestData.SHORT_3.map { it + 3.0 }.toDoubleArray())
        TestAssertions.assertNaNResult(result, "when sample contains Infinity")
    }
}
