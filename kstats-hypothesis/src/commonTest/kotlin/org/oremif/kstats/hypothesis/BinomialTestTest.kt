package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.*

class BinomialTestTest {

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-10, message: String = "") {
        TestAssertions.assertPValue(expected, actual, tol, message)
    }

    private fun assertCI(
        expectedLow: Double, expectedHigh: Double,
        ci: ConfidenceInterval?, tol: Double = 1e-8, message: String = ""
    ) {
        TestAssertions.assertCI(expectedLow, expectedHigh, ci, tol, message)
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
        assertTrue(ci90.lower >= ci95.lower, "90% lower should be >= 95% lower")
        assertTrue(ci90.upper <= ci95.upper, "90% upper should be <= 95% upper")
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
        assertEquals(
            Alternative.LESS,
            binomialTest(successes = 5, trials = 10, alternative = Alternative.LESS).alternative
        )
        assertEquals(
            Alternative.GREATER,
            binomialTest(successes = 5, trials = 10, alternative = Alternative.GREATER).alternative
        )
    }

    // ===== Wilson CI: Basic correctness =====

    @Test
    fun testWilsonCIKnownValues() {
        // scipy: binomtest(7, 10, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.396778147461145, 0.892208732593699,
            binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 7/10"
        )
        // scipy: binomtest(5, 10, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.236593090512564, 0.763406909487436,
            binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 5/10"
        )
        // scipy: binomtest(60, 100, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.502002586791062, 0.690598713567541,
            binomialTest(successes = 60, trials = 100, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 60/100"
        )
        // scipy: binomtest(8, 10, 0.3).proportion_ci(0.95, method='wilson')
        assertCI(
            0.490162471536642, 0.943317848545625,
            binomialTest(successes = 8, trials = 10, probability = 0.3, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 8/10 p=0.3"
        )
        // scipy: binomtest(4, 20, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.0806576625797981, 0.416017432251894,
            binomialTest(successes = 4, trials = 20, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 4/20"
        )
    }

    @Test
    fun testWilsonCILargeN() {
        // scipy: binomtest(500, 1000, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.469069600368104, 0.530930399631896,
            binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 500/1000"
        )
        // scipy: binomtest(700, 1000, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.670876139082783, 0.727593157522995,
            binomialTest(successes = 700, trials = 1000, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 700/1000"
        )
    }

    // ===== Wilson CI: Edge cases =====

    @Test
    fun testWilsonCIZeroSuccesses() {
        // scipy: binomtest(0, 10, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.0, 0.277532799862889,
            binomialTest(successes = 0, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 0/10"
        )
    }

    @Test
    fun testWilsonCIAllSuccesses() {
        // scipy: binomtest(10, 10, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.722467200137111, 1.0,
            binomialTest(successes = 10, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 10/10"
        )
    }

    @Test
    fun testWilsonCISingleTrialSuccess() {
        // scipy: binomtest(1, 1, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.206549314377237, 1.0,
            binomialTest(successes = 1, trials = 1, ciMethod = CIMethod.WILSON).confidenceInterval,
            tol = 1e-6, message = "Wilson 1/1"
        )
    }

    @Test
    fun testWilsonCISingleTrialFailure() {
        // scipy: binomtest(0, 1, 0.5).proportion_ci(0.95, method='wilson')
        assertCI(
            0.0, 0.793450685622763,
            binomialTest(successes = 0, trials = 1, ciMethod = CIMethod.WILSON).confidenceInterval,
            tol = 1e-6, message = "Wilson 0/1"
        )
    }

    @Test
    fun testWilsonCIZeroTrials() {
        // n=0 should return (0, 1) regardless of ciMethod
        val result = binomialTest(successes = 0, trials = 0, ciMethod = CIMethod.WILSON)
        assertCI(0.0, 1.0, result.confidenceInterval, message = "Wilson 0/0")
    }

    // ===== Wilson CI: Custom confidence levels =====

    @Test
    fun testWilsonCIConfidenceLevel99() {
        // scipy: binomtest(7, 10, 0.5).proportion_ci(0.99, method='wilson')
        assertCI(
            0.320024679201103, 0.920433683476934,
            binomialTest(successes = 7, trials = 10, confidenceLevel = 0.99, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 7/10 99%"
        )
    }

    @Test
    fun testWilsonCIConfidenceLevel90() {
        // scipy: binomtest(7, 10, 0.5).proportion_ci(0.90, method='wilson')
        assertCI(
            0.441699795869835, 0.873123416096802,
            binomialTest(successes = 7, trials = 10, confidenceLevel = 0.90, ciMethod = CIMethod.WILSON).confidenceInterval,
            message = "Wilson 7/10 90%"
        )
    }

    // ===== Wilson CI: Properties =====

    @Test
    fun testWilsonCINarrowerThanClopperPearson() {
        // Wilson CI is generally narrower than Clopper-Pearson for the same confidence level
        val cases = listOf(
            Pair(7, 10), Pair(60, 100), Pair(500, 1000)
        )
        for ((k, n) in cases) {
            val cp = binomialTest(successes = k, trials = n, ciMethod = CIMethod.CLOPPER_PEARSON).confidenceInterval!!
            val w = binomialTest(successes = k, trials = n, ciMethod = CIMethod.WILSON).confidenceInterval!!
            val cpWidth = cp.upper - cp.lower
            val wWidth = w.upper - w.lower
            assertTrue(wWidth <= cpWidth, "Wilson CI ($k/$n) width $wWidth should be <= CP width $cpWidth")
        }
    }

    @Test
    fun testWilsonCIContainsObservedProportion() {
        // The observed proportion should be within the Wilson CI
        val cases = listOf(
            Triple(5, 10, 0.5), Triple(7, 10, 0.7), Triple(60, 100, 0.6),
            Triple(0, 10, 0.0), Triple(10, 10, 1.0), Triple(4, 20, 0.2)
        )
        for ((k, n, pHat) in cases) {
            val ci = binomialTest(successes = k, trials = n, ciMethod = CIMethod.WILSON).confidenceInterval!!
            assertTrue(
                pHat >= ci.lower - 1e-10 && pHat <= ci.upper + 1e-10,
                "Wilson CI ($k/$n): $pHat should be in [${ci.lower}, ${ci.upper}]"
            )
        }
    }

    @Test
    fun testWilsonCISymmetryAtHalf() {
        // When p_hat = 0.5, Wilson CI should be symmetric around 0.5
        // scipy: binomtest(50, 100, 0.5).proportion_ci(0.95, method='wilson')
        val ci = binomialTest(successes = 50, trials = 100, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val center = (ci.lower + ci.upper) / 2.0
        assertEquals(0.5, center, 1e-10, "Wilson CI center should be 0.5 when p_hat=0.5")
    }

    @Test
    fun testWilsonCIBoundsInZeroOne() {
        // Wilson CI bounds should always be in [0, 1]
        val cases = listOf(
            Pair(0, 10), Pair(10, 10), Pair(0, 1), Pair(1, 1),
            Pair(5, 10), Pair(60, 100), Pair(500, 1000)
        )
        for ((k, n) in cases) {
            val ci = binomialTest(successes = k, trials = n, ciMethod = CIMethod.WILSON).confidenceInterval!!
            assertTrue(ci.lower >= 0.0, "Wilson CI ($k/$n) lower ${ci.lower} >= 0")
            assertTrue(ci.upper <= 1.0, "Wilson CI ($k/$n) upper ${ci.upper} <= 1")
            assertTrue(ci.lower <= ci.upper, "Wilson CI ($k/$n) lower <= upper")
        }
    }

    @Test
    fun testWilsonCIWiderWithHigherConfidence() {
        // Higher confidence level should produce a wider CI
        val ci90 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.90, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ci95 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.95, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ci99 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.99, ciMethod = CIMethod.WILSON).confidenceInterval!!
        assertTrue(ci90.upper - ci90.lower < ci95.upper - ci95.lower, "90% CI narrower than 95%")
        assertTrue(ci95.upper - ci95.lower < ci99.upper - ci99.lower, "95% CI narrower than 99%")
    }

    @Test
    fun testWilsonCINarrowsWithLargerN() {
        // With p_hat constant at 0.5, increasing n should narrow the Wilson CI
        val ci10 = binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ci100 = binomialTest(successes = 50, trials = 100, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ci1000 = binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val w10 = ci10.upper - ci10.lower
        val w100 = ci100.upper - ci100.lower
        val w1000 = ci1000.upper - ci1000.lower
        assertTrue(w10 > w100, "Wilson CI narrows from n=10 to n=100")
        assertTrue(w100 > w1000, "Wilson CI narrows from n=100 to n=1000")
    }

    // ===== Wilson CI: Does not affect p-value or statistic =====

    @Test
    fun testWilsonCIDoesNotAffectPValue() {
        // Changing ciMethod should not affect statistic or p-value
        val cp = binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.CLOPPER_PEARSON)
        val w = binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.WILSON)
        assertEquals(cp.statistic, w.statistic, 0.0, "statistic unchanged by ciMethod")
        assertEquals(cp.pValue, w.pValue, 0.0, "p-value unchanged by ciMethod")
    }

    // ===== Agresti-Coull CI: Basic correctness =====

    @Test
    fun testAgrestiCoullCIKnownValues() {
        // statsmodels: proportion_confint(7, 10, alpha=0.05, method='agresti_coull')
        assertCI(
            0.39232529797727, 0.896661582077575,
            binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 7/10"
        )
        // statsmodels: proportion_confint(5, 10, alpha=0.05, method='agresti_coull')
        assertCI(
            0.236593090512564, 0.763406909487436,
            binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 5/10"
        )
        // statsmodels: proportion_confint(60, 100, alpha=0.05, method='agresti_coull')
        assertCI(
            0.501932733571748, 0.690668566786855,
            binomialTest(successes = 60, trials = 100, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 60/100"
        )
        // statsmodels: proportion_confint(8, 10, alpha=0.05, method='agresti_coull')
        assertCI(
            0.479367590566151, 0.954112729516116,
            binomialTest(successes = 8, trials = 10, probability = 0.3, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 8/10 p=0.3"
        )
        // statsmodels: proportion_confint(4, 20, alpha=0.05, method='agresti_coull')
        assertCI(
            0.0749115102767071, 0.421763584554985,
            binomialTest(successes = 4, trials = 20, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 4/20"
        )
    }

    @Test
    fun testAgrestiCoullCILargeN() {
        // statsmodels: proportion_confint(500, 1000, alpha=0.05, method='agresti_coull')
        assertCI(
            0.469069600368104, 0.530930399631896,
            binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 500/1000"
        )
        // statsmodels: proportion_confint(700, 1000, alpha=0.05, method='agresti_coull')
        assertCI(
            0.670865852649341, 0.727603443956437,
            binomialTest(successes = 700, trials = 1000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 700/1000"
        )
    }

    // ===== Agresti-Coull CI: Edge cases =====

    @Test
    fun testAgrestiCoullCIZeroSuccesses() {
        // statsmodels: proportion_confint(0, 10, alpha=0.05, method='agresti_coull')
        assertCI(
            0.0, 0.320887305750546,
            binomialTest(successes = 0, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 0/10"
        )
    }

    @Test
    fun testAgrestiCoullCIAllSuccesses() {
        // statsmodels: proportion_confint(10, 10, alpha=0.05, method='agresti_coull')
        assertCI(
            0.679112694249454, 1.0,
            binomialTest(successes = 10, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 10/10"
        )
    }

    @Test
    fun testAgrestiCoullCISingleTrialSuccess() {
        // statsmodels: proportion_confint(1, 1, alpha=0.05, method='agresti_coull')
        assertCI(
            0.167499485479413, 1.0,
            binomialTest(successes = 1, trials = 1, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            tol = 1e-6, message = "AC 1/1"
        )
    }

    @Test
    fun testAgrestiCoullCISingleTrialFailure() {
        // statsmodels: proportion_confint(0, 1, alpha=0.05, method='agresti_coull')
        assertCI(
            0.0, 0.832500514520587,
            binomialTest(successes = 0, trials = 1, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            tol = 1e-6, message = "AC 0/1"
        )
    }

    @Test
    fun testAgrestiCoullCIZeroTrials() {
        // n=0 should return (0, 1) regardless of ciMethod
        val result = binomialTest(successes = 0, trials = 0, ciMethod = CIMethod.AGRESTI_COULL)
        assertCI(0.0, 1.0, result.confidenceInterval, message = "AC 0/0")
    }

    // ===== Agresti-Coull CI: Custom confidence levels =====

    @Test
    fun testAgrestiCoullCIConfidenceLevel99() {
        // statsmodels: proportion_confint(7, 10, alpha=0.01, method='agresti_coull')
        assertCI(
            0.313719695264693, 0.926738667413344,
            binomialTest(successes = 7, trials = 10, confidenceLevel = 0.99, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 7/10 99%"
        )
    }

    @Test
    fun testAgrestiCoullCIConfidenceLevel90() {
        // statsmodels: proportion_confint(7, 10, alpha=0.10, method='agresti_coull')
        assertCI(
            0.438415879330022, 0.876407332636615,
            binomialTest(successes = 7, trials = 10, confidenceLevel = 0.90, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval,
            message = "AC 7/10 90%"
        )
    }

    // ===== Agresti-Coull CI: Properties =====

    @Test
    fun testAgrestiCoullCIContainsObservedProportion() {
        // The observed proportion should be within the AC CI (for reasonable n)
        val cases = listOf(
            Triple(5, 10, 0.5), Triple(7, 10, 0.7), Triple(60, 100, 0.6),
            Triple(4, 20, 0.2)
        )
        for ((k, n, pHat) in cases) {
            val ci = binomialTest(successes = k, trials = n, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
            assertTrue(
                pHat >= ci.lower - 1e-10 && pHat <= ci.upper + 1e-10,
                "AC CI ($k/$n): $pHat should be in [${ci.lower}, ${ci.upper}]"
            )
        }
    }

    @Test
    fun testAgrestiCoullCIBoundsInZeroOne() {
        // AC CI bounds should always be in [0, 1] (coerced)
        val cases = listOf(
            Pair(0, 10), Pair(10, 10), Pair(0, 1), Pair(1, 1),
            Pair(5, 10), Pair(60, 100), Pair(500, 1000)
        )
        for ((k, n) in cases) {
            val ci = binomialTest(successes = k, trials = n, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
            assertTrue(ci.lower >= 0.0, "AC CI ($k/$n) lower ${ci.lower} >= 0")
            assertTrue(ci.upper <= 1.0, "AC CI ($k/$n) upper ${ci.upper} <= 1")
            assertTrue(ci.lower <= ci.upper, "AC CI ($k/$n) lower <= upper")
        }
    }

    @Test
    fun testAgrestiCoullCISymmetryAtHalf() {
        // When p_hat = 0.5, AC CI should be symmetric around 0.5
        // statsmodels: proportion_confint(50, 100, alpha=0.05, method='agresti_coull')
        val ci = binomialTest(successes = 50, trials = 100, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val center = (ci.lower + ci.upper) / 2.0
        assertEquals(0.5, center, 1e-10, "AC CI center should be 0.5 when p_hat=0.5")
    }

    @Test
    fun testAgrestiCoullCIWiderWithHigherConfidence() {
        // Higher confidence level should produce a wider CI
        val ci90 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.90, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val ci95 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.95, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val ci99 = binomialTest(successes = 7, trials = 10, confidenceLevel = 0.99, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        assertTrue(ci90.upper - ci90.lower < ci95.upper - ci95.lower, "AC 90% CI narrower than 95%")
        assertTrue(ci95.upper - ci95.lower < ci99.upper - ci99.lower, "AC 95% CI narrower than 99%")
    }

    @Test
    fun testAgrestiCoullCINarrowsWithLargerN() {
        // With p_hat constant at 0.5, increasing n should narrow the AC CI
        val ci10 = binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val ci100 = binomialTest(successes = 50, trials = 100, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val ci1000 = binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        val w10 = ci10.upper - ci10.lower
        val w100 = ci100.upper - ci100.lower
        val w1000 = ci1000.upper - ci1000.lower
        assertTrue(w10 > w100, "AC CI narrows from n=10 to n=100")
        assertTrue(w100 > w1000, "AC CI narrows from n=100 to n=1000")
    }

    @Test
    fun testAgrestiCoullCIDoesNotAffectPValue() {
        // Changing ciMethod should not affect statistic or p-value
        val cp = binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.CLOPPER_PEARSON)
        val ac = binomialTest(successes = 7, trials = 10, ciMethod = CIMethod.AGRESTI_COULL)
        assertEquals(cp.statistic, ac.statistic, 0.0, "statistic unchanged by ciMethod")
        assertEquals(cp.pValue, ac.pValue, 0.0, "p-value unchanged by ciMethod")
    }

    // ===== Cross-method comparisons =====

    @Test
    fun testAllMethodsConvergeForLargeN() {
        // For large n with p_hat near 0.5, all three methods should give similar results
        val cp = binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.CLOPPER_PEARSON).confidenceInterval!!
        val w = binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ac = binomialTest(successes = 500, trials = 1000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        // All three should agree within ~0.01 for n=1000
        assertEquals(cp.lower, w.lower, 0.01, "CP and Wilson lower converge for large n")
        assertEquals(cp.upper, w.upper, 0.01, "CP and Wilson upper converge for large n")
        assertEquals(w.lower, ac.lower, 0.01, "Wilson and AC lower converge for large n")
        assertEquals(w.upper, ac.upper, 0.01, "Wilson and AC upper converge for large n")
    }

    @Test
    fun testWilsonAndAgrestiCoullAgreeAtHalf() {
        // When p_hat = 0.5, Wilson and Agresti-Coull produce identical CIs
        // (because the adjustment is symmetric)
        // scipy/statsmodels confirm: both give (0.236593090512564, 0.763406909487436) for 5/10
        val w = binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.WILSON).confidenceInterval!!
        val ac = binomialTest(successes = 5, trials = 10, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        assertEquals(w.lower, ac.lower, 1e-10, "Wilson and AC identical at p_hat=0.5 (lower)")
        assertEquals(w.upper, ac.upper, 1e-10, "Wilson and AC identical at p_hat=0.5 (upper)")
    }

    // ===== Extreme parameters for CI methods =====

    @Test
    fun testWilsonCIVeryLargeN() {
        // scipy: binomtest(1, 10000, 0.5).proportion_ci(0.95, method='wilson')
        val ci = binomialTest(successes = 1, trials = 10000, ciMethod = CIMethod.WILSON).confidenceInterval!!
        assertEquals(1.76526736011223e-05, ci.lower, 1e-8, "Wilson 1/10000 lower")
        assertEquals(0.000566268897401338, ci.upper, 1e-8, "Wilson 1/10000 upper")
        assertTrue(ci.lower >= 0.0)
        assertTrue(ci.upper <= 1.0)
    }

    @Test
    fun testAgrestiCoullCIVeryLargeN() {
        // statsmodels: proportion_confint(1, 10000, alpha=0.05, method='agresti_coull')
        val ci = binomialTest(successes = 1, trials = 10000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        // lower is clamped to 0.0 since raw value would be negative
        assertEquals(0.0, ci.lower, 1e-10, "AC 1/10000 lower (clamped)")
        assertEquals(0.000626743899606934, ci.upper, 1e-8, "AC 1/10000 upper")
    }

    @Test
    fun testWilsonCIVeryLargeNNearOne() {
        // scipy: binomtest(9999, 10000, 0.5).proportion_ci(0.95, method='wilson')
        val ci = binomialTest(successes = 9999, trials = 10000, ciMethod = CIMethod.WILSON).confidenceInterval!!
        assertEquals(0.999433731102599, ci.lower, 1e-8, "Wilson 9999/10000 lower")
        assertEquals(0.999982347326399, ci.upper, 1e-8, "Wilson 9999/10000 upper")
    }

    @Test
    fun testAgrestiCoullCIVeryLargeNNearOne() {
        // statsmodels: proportion_confint(9999, 10000, alpha=0.05, method='agresti_coull')
        val ci = binomialTest(successes = 9999, trials = 10000, ciMethod = CIMethod.AGRESTI_COULL).confidenceInterval!!
        assertEquals(0.999373256100393, ci.lower, 1e-8, "AC 9999/10000 lower")
        // upper is clamped to 1.0 since raw value would exceed 1
        assertEquals(1.0, ci.upper, 1e-10, "AC 9999/10000 upper (clamped)")
    }

    // ===== Regression: very large trials (regularizedBeta convergence) =====

    @Test
    fun testVeryLargeTrialsRegression() {
        // Regression: binomialTest with ~290K trials previously threw ConvergenceException
        // in regularizedBeta because the fixed 200-iteration limit was insufficient.
        // The fix adds dynamic iteration limits via betaMaxIterations(a, b).
        // scipy: binomtest(145274, 290585, 0.5)
        // statistic = 0.499936335323571, pvalue = 0.946754474820775
        val result = binomialTest(successes = 145274, trials = 290585, probability = 0.5)
        assertEquals(0.499936335323571, result.statistic, 1e-10, "statistic for 145274/290585")
        assertP(0.946754474820775, result.pValue, tol = 1e-4, message = "145274/290585 two-sided")
        assertFalse(result.isSignificant(), "145274/290585 should not be significant at 5%")
    }

    @Test
    fun testVeryLargeTrialsAlternatives() {
        // scipy: binomtest(145274, 290585, 0.5, alternative='less') pvalue = 0.473377237410387
        val less = binomialTest(
            successes = 145274, trials = 290585, probability = 0.5,
            alternative = Alternative.LESS
        )
        assertP(0.473377237410387, less.pValue, tol = 1e-4, message = "145274/290585 less")

        // scipy: binomtest(145274, 290585, 0.5, alternative='greater') pvalue = 0.528099421105053
        val greater = binomialTest(
            successes = 145274, trials = 290585, probability = 0.5,
            alternative = Alternative.GREATER
        )
        assertP(0.528099421105053, greater.pValue, tol = 1e-4, message = "145274/290585 greater")

        // Same statistic regardless of alternative
        assertEquals(less.statistic, greater.statistic, 1e-14, "statistic unchanged by alternative")
    }

    @Test
    fun testVeryLargeTrialsClopperPearsonCI() {
        // scipy: binomtest(145274, 290585, 0.5).proportion_ci(0.95, method='exact')
        // CI: (0.498116674717248, 0.501755997199567)
        val result = binomialTest(successes = 145274, trials = 290585, probability = 0.5)
        assertCI(
            0.498116674717248, 0.501755997199567,
            result.confidenceInterval, tol = 1e-4, message = "CP CI 145274/290585"
        )
    }

    @Test
    fun testVeryLargeTrialsPValueRange() {
        // Property: p-value must be in [0, 1] for very large trials
        val result = binomialTest(successes = 145274, trials = 290585, probability = 0.5)
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1], got ${result.pValue}")
    }
}
