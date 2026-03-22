package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.*

class TTestTest {

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-6, message: String = "") {
        assertTrue(abs(expected - actual) < tol, "p-value $message: expected=$expected, actual=$actual")
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
        assertTrue(ci.first < mean && mean < ci.second)
    }

    // ===== Confidence interval: one-sided =====

    @Test
    fun testOneSidedCILess() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0, alternative = Alternative.LESS)
        val ci = result.confidenceInterval!!
        assertEquals(Double.NEGATIVE_INFINITY, ci.first, "LESS CI lower should be -Inf")
        assertTrue(ci.second.isFinite(), "LESS CI upper should be finite")
    }

    @Test
    fun testOneSidedCIGreater() {
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0, alternative = Alternative.GREATER)
        val ci = result.confidenceInterval!!
        assertTrue(ci.first.isFinite(), "GREATER CI lower should be finite")
        assertEquals(Double.POSITIVE_INFINITY, ci.second, "GREATER CI upper should be +Inf")
    }

    @Test
    fun testTwoSampleOneSidedCI() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val less = tTest(s1, s2, alternative = Alternative.LESS)
        val greater = tTest(s1, s2, alternative = Alternative.GREATER)
        val lessCI = less.confidenceInterval!!
        val greaterCI = greater.confidenceInterval!!
        assertEquals(Double.NEGATIVE_INFINITY, lessCI.first)
        assertTrue(lessCI.second.isFinite())
        assertTrue(greaterCI.first.isFinite())
        assertEquals(Double.POSITIVE_INFINITY, greaterCI.second)
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
}
