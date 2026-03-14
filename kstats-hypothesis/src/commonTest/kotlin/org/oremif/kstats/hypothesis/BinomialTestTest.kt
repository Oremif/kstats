package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class BinomialTestTest {

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-10, message: String = "") {
        assertEquals(expected, actual, tol, "p-value $message")
    }

    private fun assertCI(expectedLow: Double, expectedHigh: Double, ci: Pair<Double, Double>?, tol: Double = 1e-8, message: String = "") {
        requireNotNull(ci) { "CI should not be null $message" }
        assertEquals(expectedLow, ci.first, tol, "CI lower $message")
        assertEquals(expectedHigh, ci.second, tol, "CI upper $message")
    }

    // ===== Basic correctness: fair coin =====

    @Test
    fun testFairCoin5of10() {
        // scipy: binomtest(5, 10, 0.5)
        val result = binomialTest(successes = 5, trials = 10)
        assertEquals(0.5, result.statistic, 1e-15)
        assertP(1.0, result.pValue, message = "5/10 two-sided")
        assertCI(0.1870860284474045, 0.8129139715525955, result.confidenceInterval)
        assertFalse(result.isSignificant())
    }

    @Test
    fun testFairCoin7of10() {
        // scipy: binomtest(7, 10, 0.5)
        val two = binomialTest(successes = 7, trials = 10)
        val less = binomialTest(successes = 7, trials = 10, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 7, trials = 10, alternative = Alternative.GREATER)
        assertEquals(0.7, two.statistic, 1e-15)
        assertP(0.34375, two.pValue, message = "7/10 two-sided")
        assertP(0.9453125, less.pValue, message = "7/10 less")
        assertP(0.171875, greater.pValue, message = "7/10 greater")
        assertCI(0.3475471499399921, 0.9332604888222655, two.confidenceInterval)
    }

    @Test
    fun testFairCoin60of100() {
        // scipy: binomtest(60, 100, 0.5)
        val two = binomialTest(successes = 60, trials = 100)
        val less = binomialTest(successes = 60, trials = 100, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 60, trials = 100, alternative = Alternative.GREATER)
        assertEquals(0.6, two.statistic, 1e-15)
        assertP(0.05688793364098078, two.pValue, message = "60/100 two-sided")
        assertP(0.9823998998911476, less.pValue, message = "60/100 less")
        assertP(0.02844396682049039, greater.pValue, message = "60/100 greater")
        assertCI(0.4972091504223334, 0.696705231297155, two.confidenceInterval)
        assertFalse(two.isSignificant())
    }

    // ===== Basic correctness: biased coin =====

    @Test
    fun testBiasedCoin8of10() {
        // scipy: binomtest(8, 10, 0.3)
        val two = binomialTest(successes = 8, trials = 10, probability = 0.3)
        val less = binomialTest(successes = 8, trials = 10, probability = 0.3, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 8, trials = 10, probability = 0.3, alternative = Alternative.GREATER)
        assertEquals(0.8, two.statistic, 1e-15)
        assertP(0.0015903864, two.pValue, message = "8/10 p=0.3 two-sided")
        assertP(0.9998563141, less.pValue, message = "8/10 p=0.3 less")
        assertP(0.0015903864, greater.pValue, message = "8/10 p=0.3 greater")
        assertCI(0.4439045376923585, 0.9747892736731665, two.confidenceInterval)
        assertTrue(two.isSignificant())
    }

    @Test
    fun testBiasedCoin4of20() {
        // scipy: binomtest(4, 20, 0.5)
        val two = binomialTest(successes = 4, trials = 20)
        val less = binomialTest(successes = 4, trials = 20, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 4, trials = 20, alternative = Alternative.GREATER)
        assertEquals(0.2, two.statistic, 1e-15)
        assertP(0.01181793212890625, two.pValue, message = "4/20 two-sided")
        assertP(0.005908966064453125, less.pValue, message = "4/20 less")
        assertP(0.9987115859985352, greater.pValue, message = "4/20 greater")
        assertCI(0.05733399705002387, 0.4366140029966687, two.confidenceInterval)
        assertTrue(two.isSignificant())
    }

    // ===== Large n =====

    @Test
    fun testLargeNNotSignificant() {
        // scipy: binomtest(500, 1000, 0.5) -> p=1.0
        val result = binomialTest(successes = 500, trials = 1000)
        assertEquals(0.5, result.statistic, 1e-15)
        assertP(1.0, result.pValue, message = "500/1000 two-sided")
        assertCI(0.4685491729717919, 0.531450827028208, result.confidenceInterval)
        assertFalse(result.isSignificant())
    }

    @Test
    fun testLargeNHighlySignificant() {
        // scipy: binomtest(700, 1000, 0.5) -> p≈1.77e-37
        val two = binomialTest(successes = 700, trials = 1000)
        val less = binomialTest(successes = 700, trials = 1000, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 700, trials = 1000, alternative = Alternative.GREATER)
        assertEquals(0.7, two.statistic, 1e-15)
        assertTrue(two.pValue < 1e-35, "700/1000 two-sided p should be very small, got ${two.pValue}")
        assertP(1.0, less.pValue, message = "700/1000 less")
        assertTrue(greater.pValue < 1e-35, "700/1000 greater p should be very small, got ${greater.pValue}")
        assertCI(0.6705383213026351, 0.7282788878708513, two.confidenceInterval)
        assertTrue(two.isSignificant())
    }

    // ===== Edge cases =====

    @Test
    fun testZeroTrials() {
        val result = binomialTest(successes = 0, trials = 0)
        assertTrue(result.statistic.isNaN(), "statistic should be NaN for n=0")
        assertP(1.0, result.pValue, message = "0/0")
        assertCI(0.0, 1.0, result.confidenceInterval, message = "0/0")
    }

    @Test
    fun testZeroSuccesses() {
        // scipy: binomtest(0, 10, 0.5)
        val two = binomialTest(successes = 0, trials = 10)
        val less = binomialTest(successes = 0, trials = 10, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 0, trials = 10, alternative = Alternative.GREATER)
        assertEquals(0.0, two.statistic, 1e-15)
        assertP(0.001953125, two.pValue, message = "0/10 two-sided")
        assertP(0.0009765625, less.pValue, message = "0/10 less")
        assertP(1.0, greater.pValue, message = "0/10 greater")
        assertCI(0.0, 0.3084971078187629, two.confidenceInterval)
    }

    @Test
    fun testAllSuccesses() {
        // scipy: binomtest(10, 10, 0.5)
        val two = binomialTest(successes = 10, trials = 10)
        val less = binomialTest(successes = 10, trials = 10, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 10, trials = 10, alternative = Alternative.GREATER)
        assertEquals(1.0, two.statistic, 1e-15)
        assertP(0.001953125, two.pValue, message = "10/10 two-sided")
        assertP(1.0, less.pValue, message = "10/10 less")
        assertP(0.0009765625, greater.pValue, message = "10/10 greater")
        assertCI(0.6915028921812371, 1.0, two.confidenceInterval)
    }

    @Test
    fun testSingleTrialSuccess() {
        // scipy: binomtest(1, 1, 0.5)
        val two = binomialTest(successes = 1, trials = 1)
        val less = binomialTest(successes = 1, trials = 1, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 1, trials = 1, alternative = Alternative.GREATER)
        assertEquals(1.0, two.statistic, 1e-15)
        assertP(1.0, two.pValue, message = "1/1 two-sided")
        assertP(1.0, less.pValue, message = "1/1 less")
        assertP(0.5, greater.pValue, message = "1/1 greater")
        assertCI(0.025, 1.0, two.confidenceInterval, tol = 1e-6)
    }

    @Test
    fun testSingleTrialFailure() {
        // scipy: binomtest(0, 1, 0.5)
        val two = binomialTest(successes = 0, trials = 1)
        val less = binomialTest(successes = 0, trials = 1, alternative = Alternative.LESS)
        val greater = binomialTest(successes = 0, trials = 1, alternative = Alternative.GREATER)
        assertEquals(0.0, two.statistic, 1e-15)
        assertP(1.0, two.pValue, message = "0/1 two-sided")
        assertP(0.5, less.pValue, message = "0/1 less")
        assertP(1.0, greater.pValue, message = "0/1 greater")
        assertCI(0.0, 0.975, two.confidenceInterval, tol = 1e-6)
    }

    @Test
    fun testProbabilityZeroWithZeroSuccesses() {
        // scipy: binomtest(0, 5, 0.0) -> p=1.0
        val result = binomialTest(successes = 0, trials = 5, probability = 0.0)
        assertEquals(0.0, result.statistic, 1e-15)
        assertP(1.0, result.pValue, message = "p=0 k=0")
    }

    @Test
    fun testProbabilityZeroWithPositiveSuccesses() {
        // scipy: binomtest(3, 5, 0.0) -> p=0.0
        val result = binomialTest(successes = 3, trials = 5, probability = 0.0)
        assertEquals(0.6, result.statistic, 1e-15)
        assertP(0.0, result.pValue, message = "p=0 k>0")
    }

    @Test
    fun testProbabilityOneWithAllSuccesses() {
        // scipy: binomtest(5, 5, 1.0) -> p=1.0
        val result = binomialTest(successes = 5, trials = 5, probability = 1.0)
        assertEquals(1.0, result.statistic, 1e-15)
        assertP(1.0, result.pValue, message = "p=1 k=n")
    }

    @Test
    fun testProbabilityOneWithFewerSuccesses() {
        // scipy: binomtest(0, 5, 1.0) -> p=0.0
        val result = binomialTest(successes = 0, trials = 5, probability = 1.0)
        assertEquals(0.0, result.statistic, 1e-15)
        assertP(0.0, result.pValue, message = "p=1 k=0")
    }

    // ===== Custom confidence level =====

    @Test
    fun testConfidenceLevel99() {
        // scipy: binomtest(7, 10, 0.5).proportion_ci(confidence_level=0.99, method='exact')
        val result = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.99)
        assertCI(0.2648860147128693, 0.9629927789037926, result.confidenceInterval, message = "99% CI")
    }

    @Test
    fun testConfidenceLevel90() {
        // Verify that narrower CI at 90% is a subset of 95% CI
        val ci95 = binomialTest(successes = 60, trials = 100).confidenceInterval!!
        val ci90 = binomialTest(successes = 60, trials = 100, confidenceLevel = 0.90).confidenceInterval!!
        assertTrue(ci90.first >= ci95.first, "90% lower should be >= 95% lower")
        assertTrue(ci90.second <= ci95.second, "90% upper should be <= 95% upper")
    }

    // ===== Validation =====

    @Test
    fun testNegativeTrials() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 0, trials = -1)
        }
    }

    @Test
    fun testNegativeSuccesses() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = -1, trials = 10)
        }
    }

    @Test
    fun testSuccessesExceedTrials() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 11, trials = 10)
        }
    }

    @Test
    fun testProbabilityBelowZero() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, probability = -0.1)
        }
    }

    @Test
    fun testProbabilityAboveOne() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, probability = 1.1)
        }
    }

    @Test
    fun testNaNProbability() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, probability = Double.NaN)
        }
    }

    @Test
    fun testConfidenceLevelZero() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, confidenceLevel = 0.0)
        }
    }

    @Test
    fun testConfidenceLevelOne() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, confidenceLevel = 1.0)
        }
    }

    @Test
    fun testConfidenceLevelNegative() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, confidenceLevel = -0.5)
        }
    }

    @Test
    fun testConfidenceLevelNaN() {
        assertFailsWith<InvalidParameterException> {
            binomialTest(successes = 5, trials = 10, confidenceLevel = Double.NaN)
        }
    }

    // ===== isSignificant consistency =====

    @Test
    fun testIsSignificantConsistency() {
        val cases = listOf(
            binomialTest(successes = 5, trials = 10),       // p=1.0, not significant
            binomialTest(successes = 0, trials = 10),       // p≈0.002, significant
            binomialTest(successes = 8, trials = 10, probability = 0.3), // p≈0.0016, significant
            binomialTest(successes = 60, trials = 100),     // p≈0.057, not significant
        )
        for (result in cases) {
            assertEquals(result.pValue < 0.05, result.isSignificant(), "isSignificant for p=${result.pValue}")
        }
    }

    // ===== Metadata =====

    @Test
    fun testTestName() {
        val result = binomialTest(successes = 5, trials = 10)
        assertEquals("Binomial Test", result.testName)
    }

    @Test
    fun testAdditionalInfo() {
        val result = binomialTest(successes = 7, trials = 10, probability = 0.3)
        assertEquals(7.0, result.additionalInfo["successes"])
        assertEquals(10.0, result.additionalInfo["trials"])
        assertEquals(0.3, result.additionalInfo["hypothesizedProbability"])
    }

    @Test
    fun testAlternativeInResult() {
        assertEquals(Alternative.TWO_SIDED, binomialTest(successes = 5, trials = 10).alternative)
        assertEquals(Alternative.LESS, binomialTest(successes = 5, trials = 10, alternative = Alternative.LESS).alternative)
        assertEquals(Alternative.GREATER, binomialTest(successes = 5, trials = 10, alternative = Alternative.GREATER).alternative)
    }
}
