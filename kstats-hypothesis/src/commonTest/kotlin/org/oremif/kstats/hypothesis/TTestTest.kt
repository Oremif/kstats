package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.*

class TTestTest {

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-6, message: String = "") {
        TestAssertions.assertPValue(expected, actual, tol, message)
    }

    // ===== One-sample t-test: basic correctness =====

    @Test
    fun testOneSampleTTest() {
        val sample = doubleArrayOf(1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
        val result = tTest(sample, mu = 0.0)
        assertFalse(result.isSignificant())
        assertEquals("One-Sample t-Test", result.testName)
    }

    @Test
    fun testOneSampleTTestSignificant() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 0.0)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testOneSampleReference() {
        // mean=6.0, sd=sqrt(0.625), se=sqrt(0.125), t=1/sqrt(0.125)=2*sqrt(2)≈2.8284
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0)
        assertEquals(2.8284271247, result.statistic, 1e-6, "t-statistic")
        assertEquals(4.0, result.degreesOfFreedom)
        assertTrue(result.isSignificant(), "p should be < 0.05")
    }

    @Test
    fun testOneSampleAlternatives() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val two = tTest(sample, mu = 5.0, alternative = Alternative.TWO_SIDED)
        val less = tTest(sample, mu = 5.0, alternative = Alternative.LESS)
        val greater = tTest(sample, mu = 5.0, alternative = Alternative.GREATER)
        // mean > mu, so LESS p-value should be large, GREATER small
        assertTrue(less.pValue > 0.5, "LESS p-value should be > 0.5, got ${less.pValue}")
        assertTrue(greater.pValue < 0.05, "GREATER p-value should be small, got ${greater.pValue}")
        // Two-sided = 2 * min(less, greater)
        assertTrue(two.pValue > greater.pValue, "Two-sided p should be > one-sided")
    }

    // ===== Two-sample t-test =====

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
    fun testTwoSampleWelchScipyReference() {
        // scipy: ttest_ind([1,2,3,4,5], [6,7,8,9,10])
        // → statistic=-5.0, pvalue=0.0010528257933665
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = tTest(s1, s2)
        assertEquals(-5.0, result.statistic, 1e-10, "t-statistic")
        assertP(0.0010528258, result.pValue, message = "welch two-sample")
    }

    // ===== Paired t-test =====

    @Test
    fun testPairedTTest() {
        val before = doubleArrayOf(200.0, 190.0, 210.0, 180.0, 195.0)
        val after = doubleArrayOf(190.0, 180.0, 195.0, 170.0, 185.0)
        val result = pairedTTest(before, after)
        assertEquals("Paired t-Test", result.testName)
        assertTrue(result.statistic > 0)
    }

    // ===== Confidence interval: two-sided =====

    @Test
    fun testConfidenceInterval() {
        val sample = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0)
        val result = tTest(sample, mu = 0.0, confidenceLevel = 0.95)
        val ci = result.confidenceInterval!!
        val mean = sample.average()
        assertTrue(ci.lower < mean && mean < ci.upper)
    }

    // ===== Confidence interval: one-sided =====

    @Test
    fun testOneSidedCILess() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0, alternative = Alternative.LESS)
        val ci = result.confidenceInterval!!
        assertEquals(Double.NEGATIVE_INFINITY, ci.lower, "LESS CI lower should be -Inf")
        assertTrue(ci.upper.isFinite(), "LESS CI upper should be finite")
    }

    @Test
    fun testOneSidedCIGreater() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0, alternative = Alternative.GREATER)
        val ci = result.confidenceInterval!!
        assertTrue(ci.lower.isFinite(), "GREATER CI lower should be finite")
        assertEquals(Double.POSITIVE_INFINITY, ci.upper, "GREATER CI upper should be +Inf")
    }

    @Test
    fun testTwoSampleOneSidedCI() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val less = tTest(s1, s2, alternative = Alternative.LESS)
        val greater = tTest(s1, s2, alternative = Alternative.GREATER)
        val lessCI = less.confidenceInterval!!
        val greaterCI = greater.confidenceInterval!!
        assertEquals(Double.NEGATIVE_INFINITY, lessCI.lower)
        assertTrue(lessCI.upper.isFinite())
        assertTrue(greaterCI.lower.isFinite())
        assertEquals(Double.POSITIVE_INFINITY, greaterCI.upper)
    }

    // ===== Validation =====

    @Test
    fun testOneSampleTooFewElements() {
        assertFailsWith<InsufficientDataException> { tTest(doubleArrayOf(1.0), mu = 0.0) }
        assertFailsWith<InsufficientDataException> { tTest(doubleArrayOf(), mu = 0.0) }
    }

    @Test
    fun testTwoSampleTooFewElements() {
        assertFailsWith<InsufficientDataException> {
            tTest(doubleArrayOf(1.0), doubleArrayOf(2.0, 3.0))
        }
        assertFailsWith<InsufficientDataException> {
            tTest(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        }
    }

    @Test
    fun testPairedTTestMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            pairedTTest(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        }
    }

    @Test
    fun testPairedTTestTooFewElements() {
        assertFailsWith<InsufficientDataException> {
            pairedTTest(doubleArrayOf(1.0), doubleArrayOf(2.0))
        }
    }

    @Test
    fun testConfidenceLevelValidation() {
        val sample = doubleArrayOf(1.0, 2.0, 3.0)
        assertFailsWith<InvalidParameterException> { tTest(sample, confidenceLevel = 0.0) }
        assertFailsWith<InvalidParameterException> { tTest(sample, confidenceLevel = 1.0) }
        assertFailsWith<InvalidParameterException> { tTest(sample, confidenceLevel = -0.5) }
        assertFailsWith<InvalidParameterException> { tTest(sample, confidenceLevel = 1.5) }
    }

    @Test
    fun testTwoSampleConfidenceLevelValidation() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0)
        val s2 = doubleArrayOf(4.0, 5.0, 6.0)
        assertFailsWith<InvalidParameterException> { tTest(s1, s2, confidenceLevel = 0.0) }
        assertFailsWith<InvalidParameterException> { tTest(s1, s2, confidenceLevel = 1.0) }
    }

    @Test
    fun testPairedConfidenceLevelValidation() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0)
        val s2 = doubleArrayOf(4.0, 5.0, 6.0)
        assertFailsWith<InvalidParameterException> { pairedTTest(s1, s2, confidenceLevel = 0.0) }
        assertFailsWith<InvalidParameterException> { pairedTTest(s1, s2, confidenceLevel = 1.0) }
    }

    // ===== Degenerate cases: constant samples =====

    @Test
    fun testOneSampleConstantEqualsMu() {
        // All values = mu: se=0, diff=0 → t=NaN, p=NaN
        val sample = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = tTest(sample, mu = 5.0)
        assertTrue(result.statistic.isNaN(), "t should be NaN when all values equal mu")
        assertTrue(result.pValue.isNaN(), "p should be NaN when all values equal mu")
        assertEquals(4.0, result.degreesOfFreedom, 1e-10)
        // Confidence interval collapses to a point
        val ci = result.confidenceInterval!!
        assertEquals(5.0, ci.lower, 1e-10, "CI lower = mean when se=0")
        assertEquals(5.0, ci.upper, 1e-10, "CI upper = mean when se=0")
    }

    @Test
    fun testOneSampleConstantNotEqualToMu() {
        // All values = 5.0, mu = 3.0: se=0, diff>0 → t=+Inf, p=0 (two-sided)
        val sample = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = tTest(sample, mu = 3.0)
        assertEquals(Double.POSITIVE_INFINITY, result.statistic, "t should be +Inf when mean > mu and se=0")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0 for two-sided when mean != mu and se=0")
    }

    @Test
    fun testOneSampleConstantBelowMu() {
        // All values = 2.0, mu = 5.0: se=0, diff<0 → t=-Inf, p=0 (two-sided)
        val sample = doubleArrayOf(2.0, 2.0, 2.0, 2.0)
        val result = tTest(sample, mu = 5.0)
        assertEquals(Double.NEGATIVE_INFINITY, result.statistic, "t should be -Inf when mean < mu and se=0")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0 for two-sided when mean != mu and se=0")
    }

    @Test
    fun testOneSampleConstantAlternatives() {
        val sample = doubleArrayOf(5.0, 5.0, 5.0, 5.0)
        // mean > mu, so GREATER → p=0, LESS → p=1
        val greater = tTest(sample, mu = 3.0, alternative = Alternative.GREATER)
        assertEquals(0.0, greater.pValue, 1e-15, "GREATER p should be 0 when mean > mu and se=0")
        val less = tTest(sample, mu = 3.0, alternative = Alternative.LESS)
        assertEquals(1.0, less.pValue, 1e-15, "LESS p should be 1 when mean > mu and se=0")
    }

    @Test
    fun testTwoSampleBothConstantSameMean() {
        // Both samples constant with same mean: se=0, diff=0 → t=NaN, p=NaN
        val s1 = doubleArrayOf(5.0, 5.0, 5.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = tTest(s1, s2)
        assertTrue(result.statistic.isNaN(), "t should be NaN when both constant and same mean")
        assertTrue(result.pValue.isNaN(), "p should be NaN when both constant and same mean")
        val ci = result.confidenceInterval!!
        assertEquals(0.0, ci.lower, 1e-10, "CI lower = 0 when diff=0 and se=0")
        assertEquals(0.0, ci.upper, 1e-10, "CI upper = 0 when diff=0 and se=0")
    }

    @Test
    fun testTwoSampleBothConstantDifferentMeans() {
        // Both constant but different means: se=0, diff!=0 → t=+/-Inf, p=0
        val s1 = doubleArrayOf(10.0, 10.0, 10.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = tTest(s1, s2)
        assertEquals(Double.POSITIVE_INFINITY, result.statistic, "t should be +Inf when mean1 > mean2 and se=0")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0 when means differ and se=0")
        val ci = result.confidenceInterval!!
        assertEquals(5.0, ci.lower, 1e-10, "CI should be a point at the difference")
        assertEquals(5.0, ci.upper, 1e-10, "CI should be a point at the difference")
    }

    @Test
    fun testTwoSampleBothConstantDifferentMeansEqualVariances() {
        // Equal variances variant
        val s1 = doubleArrayOf(10.0, 10.0, 10.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = tTest(s1, s2, equalVariances = true)
        assertEquals(Double.POSITIVE_INFINITY, result.statistic, "t should be +Inf")
        assertEquals(0.0, result.pValue, 1e-15, "p should be 0")
    }

    @Test
    fun testTwoSampleBothConstantDifferentMeansAlternatives() {
        val s1 = doubleArrayOf(10.0, 10.0, 10.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        // mean1 > mean2, so GREATER → p=0, LESS → p=1
        val greater = tTest(s1, s2, alternative = Alternative.GREATER)
        assertEquals(0.0, greater.pValue, 1e-15, "GREATER p should be 0 when mean1 > mean2 and se=0")
        val less = tTest(s1, s2, alternative = Alternative.LESS)
        assertEquals(1.0, less.pValue, 1e-15, "LESS p should be 1 when mean1 > mean2 and se=0")
    }
}
