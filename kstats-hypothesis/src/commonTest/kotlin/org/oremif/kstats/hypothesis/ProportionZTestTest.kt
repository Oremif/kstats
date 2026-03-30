package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.*

class ProportionZTestTest {

    private val tol = 1e-10

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-10, message: String = "") {
        assertEquals(expected, actual, tol, "p-value $message")
    }

    private fun assertCI(
        expectedLow: Double,
        expectedHigh: Double,
        ci: Pair<Double, Double>?,
        tol: Double = 1e-10,
        message: String = ""
    ) {
        requireNotNull(ci) { "CI should not be null $message" }
        assertEquals(expectedLow, ci.first, tol, "CI lower $message")
        assertEquals(expectedHigh, ci.second, tol, "CI upper $message")
    }

    // =========================================================================
    // ONE-SAMPLE: Basic correctness
    // =========================================================================

    @Test
    fun testOneSampleKnownValues60of100() {
        // scipy: z = (0.6 - 0.5) / sqrt(0.5*0.5/100) = 2.0
        // scipy: 2 * norm.sf(2.0) = 0.0455002638963584
        val two = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        assertEquals(2.0, two.statistic, tol)
        assertP(0.0455002638963584, two.pValue, message = "60/100 p0=0.5 two-sided")
        assertCI(0.503981766472894, 0.696018233527106, two.confidenceInterval)
        assertTrue(two.isSignificant(), "p < 0.05")
    }

    @Test
    fun testOneSampleKnownValuesAllAlternatives60of100() {
        // scipy: z = 2.0 for all alternatives
        val two = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        val less = proportionZTest(successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.LESS)
        val greater = proportionZTest(successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.GREATER)

        assertEquals(2.0, two.statistic, tol)
        assertEquals(2.0, less.statistic, tol)
        assertEquals(2.0, greater.statistic, tol)

        // scipy: norm.cdf(2.0) = 0.977249868051821
        assertP(0.977249868051821, less.pValue, message = "60/100 less")
        // scipy: norm.sf(2.0) = 0.0227501319481792
        assertP(0.0227501319481792, greater.pValue, message = "60/100 greater")
    }

    @Test
    fun testOneSampleKnownValues30of200() {
        // scipy: z = (0.15 - 0.2) / sqrt(0.2*0.8/200) = -1.76776695296637
        // scipy: 2 * norm.sf(1.76776...) = 0.0770998717435417
        val two = proportionZTest(successes = 30, trials = 200, p0 = 0.2)
        assertEquals(-1.76776695296637, two.statistic, tol)
        assertP(0.0770998717435417, two.pValue, message = "30/200 p0=0.2 two-sided")
        assertCI(0.100513335147815, 0.199486664852185, two.confidenceInterval, tol = 1e-10)
        assertFalse(two.isSignificant(), "p > 0.05")
    }

    @Test
    fun testOneSampleKnownValues85of100() {
        // scipy: z = (0.85 - 0.75) / sqrt(0.75*0.25/100) = 2.3094010767585
        // scipy: 2 * norm.sf(2.3094...) = 0.020921335337794
        val two = proportionZTest(successes = 85, trials = 100, p0 = 0.75)
        assertEquals(2.3094010767585, two.statistic, tol)
        assertP(0.020921335337794, two.pValue, message = "85/100 p0=0.75 two-sided")
        assertCI(0.780015287409428, 0.919984712590572, two.confidenceInterval, tol = 1e-10)
        assertTrue(two.isSignificant(), "p < 0.05")
    }

    @Test
    fun testOneSampleKnownValues10of50() {
        // scipy: z = (0.2 - 0.3) / sqrt(0.3*0.7/50) = -1.54303349962092
        // scipy: 2 * norm.sf(1.54303...) = 0.122822648101393
        val two = proportionZTest(successes = 10, trials = 50, p0 = 0.3)
        assertEquals(-1.54303349962092, two.statistic, tol)
        assertP(0.122822648101393, two.pValue, message = "10/50 p0=0.3 two-sided")
    }

    @Test
    fun testOneSampleExactlyAtNull() {
        // scipy: z = (0.5 - 0.5) / sqrt(0.5*0.5/150) = 0.0, p_two = 1.0
        val result = proportionZTest(successes = 75, trials = 150, p0 = 0.5)
        assertEquals(0.0, result.statistic, tol)
        assertP(1.0, result.pValue, message = "exact null")
        assertCI(0.419984805394078, 0.580015194605922, result.confidenceInterval, tol = 1e-10)
        assertFalse(result.isSignificant())
    }

    // =========================================================================
    // ONE-SAMPLE: Edge cases
    // =========================================================================

    @Test
    fun testOneSampleSingleTrial() {
        // scipy: z = (1.0 - 0.5) / sqrt(0.5*0.5/1) = 1.0
        // scipy: 2 * norm.sf(1.0) = 0.317310507862914
        val result = proportionZTest(successes = 1, trials = 1, p0 = 0.5)
        assertEquals(1.0, result.statistic, tol)
        assertP(0.317310507862914, result.pValue, message = "1/1 p0=0.5 two-sided")
    }

    @Test
    fun testOneSampleZeroSuccesses() {
        // k=0, n=100, p0=0.5 => pHat=0.0, z=-10.0
        val result = proportionZTest(successes = 0, trials = 100, p0 = 0.5)
        assertEquals(-10.0, result.statistic, tol)
        assertTrue(result.pValue < 1e-20, "p should be extremely small for 0/100")
    }

    @Test
    fun testOneSampleAllSuccesses() {
        // k=100, n=100, p0=0.5 => pHat=1.0, z=10.0
        val result = proportionZTest(successes = 100, trials = 100, p0 = 0.5)
        assertEquals(10.0, result.statistic, tol)
        assertTrue(result.pValue < 1e-20, "p should be extremely small for 100/100")
    }

    @Test
    fun testOneSampleExtremePNearZero() {
        // scipy: k=5, n=100, p0=0.01 => z = 4.02015126103685
        // scipy: pval_two = 5.81607864165042e-05
        val result = proportionZTest(successes = 5, trials = 100, p0 = 0.01)
        assertEquals(4.02015126103685, result.statistic, tol)
        assertP(5.81607864165042e-05, result.pValue, tol = 1e-10, message = "extreme low p0")
    }

    @Test
    fun testOneSampleExtremePNearOne() {
        // scipy: k=95, n=100, p0=0.99 => z = -4.02015126103685
        // scipy: pval_two = 5.81607864165038e-05
        val result = proportionZTest(successes = 95, trials = 100, p0 = 0.99)
        assertEquals(-4.02015126103685, result.statistic, tol)
        assertP(5.81607864165038e-05, result.pValue, tol = 1e-10, message = "extreme high p0")
    }

    @Test
    fun testOneSampleOneSidedCILess() {
        // scipy: CI LESS(0.95) for 60/100 => [0, pHat + z_crit_95 * se_wald]
        // z_crit_95 = norm.ppf(0.95) = 1.64485362695147
        // se_wald = sqrt(0.6*0.4/100) = 0.0489897948556636
        // upper = 0.6 + 1.64485... * 0.04898... = 0.680581041751947
        val result = proportionZTest(
            successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.LESS
        )
        assertCI(0.0, 0.680581041751947, result.confidenceInterval, tol = 1e-10, message = "one-sided LESS")
    }

    @Test
    fun testOneSampleOneSidedCIGreater() {
        // scipy: CI GREATER(0.95) for 60/100 => [pHat - z_crit_95 * se_wald, 1]
        // lower = 0.6 - 1.64485... * 0.04898... = 0.519418958248053
        val result = proportionZTest(
            successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.GREATER
        )
        assertCI(0.519418958248053, 1.0, result.confidenceInterval, tol = 1e-10, message = "one-sided GREATER")
    }

    // =========================================================================
    // ONE-SAMPLE: Degenerate input / validation
    // =========================================================================

    @Test
    fun testOneSampleZeroTrials() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes = 0, trials = 0, p0 = 0.5)
        }
    }

    @Test
    fun testOneSampleNegativeTrials() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes = 0, trials = -5, p0 = 0.5)
        }
    }

    @Test
    fun testOneSampleNegativeSuccesses() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = -1, trials = 10, p0 = 0.5)
        }
    }

    @Test
    fun testOneSampleSuccessesExceedTrials() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 11, trials = 10, p0 = 0.5)
        }
    }

    @Test
    fun testOneSampleP0Zero() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 0.0)
        }
    }

    @Test
    fun testOneSampleP0One() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 1.0)
        }
    }

    @Test
    fun testOneSampleP0Negative() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = -0.1)
        }
    }

    @Test
    fun testOneSampleP0GreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 1.5)
        }
    }

    @Test
    fun testOneSampleConfidenceLevelZero() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, confidenceLevel = 0.0)
        }
    }

    @Test
    fun testOneSampleConfidenceLevelOne() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, confidenceLevel = 1.0)
        }
    }

    @Test
    fun testOneSampleConfidenceLevelNegative() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, confidenceLevel = -0.5)
        }
    }

    @Test
    fun testOneSampleConfidenceLevelAboveOne() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, confidenceLevel = 1.5)
        }
    }

    // =========================================================================
    // ONE-SAMPLE: Extreme parameters
    // =========================================================================

    @Test
    fun testOneSampleLargeN() {
        // scipy: k=500500, n=1000000, p0=0.5
        // z = 0.99999999999989, pval_two = 0.317310507862967
        val result = proportionZTest(successes = 500500, trials = 1000000, p0 = 0.5)
        assertEquals(0.99999999999989, result.statistic, 1e-6) // relaxed tol for large n float
        assertP(0.317310507862967, result.pValue, tol = 1e-6, message = "large n")
        assertTrue(result.statistic.isFinite())
        assertTrue(result.pValue.isFinite())
    }

    @Test
    fun testOneSampleLargeNClearSignal() {
        // scipy: k=510000, n=1000000, p0=0.5
        // z = 20, pval_two = 5.50724823721043e-89
        val result = proportionZTest(successes = 510000, trials = 1000000, p0 = 0.5)
        assertEquals(20.0, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-80, "p-value should be extremely small for z=20, got ${result.pValue}")
        assertTrue(result.isSignificant())
    }

    @Test
    fun testOneSampleZeroSuccessesWaldCI() {
        // When pHat=0, Wald SE=0, so CI should be (0, 0) for two-sided
        val result = proportionZTest(successes = 0, trials = 100, p0 = 0.5)
        val ci = result.confidenceInterval!!
        assertEquals(0.0, ci.first, tol)
        assertEquals(0.0, ci.second, tol)
    }

    @Test
    fun testOneSampleAllSuccessesWaldCI() {
        // When pHat=1.0, Wald SE=0, so CI should be (1, 1) for two-sided
        val result = proportionZTest(successes = 100, trials = 100, p0 = 0.5)
        val ci = result.confidenceInterval!!
        assertEquals(1.0, ci.first, tol)
        assertEquals(1.0, ci.second, tol)
    }

    // =========================================================================
    // ONE-SAMPLE: Non-finite
    // =========================================================================

    @Test
    fun testOneSampleNaNP0() {
        // On JVM, Double <= / >= follow IEEE 754: NaN comparisons return false.
        // So p0=NaN passes validation and NaN propagates through computation.
        val result = proportionZTest(successes = 50, trials = 100, p0 = Double.NaN)
        assertTrue(result.statistic.isNaN(), "statistic should be NaN when p0 is NaN")
    }

    @Test
    fun testOneSampleNaNConfidenceLevel() {
        // IEEE 754: NaN passes <= / >= validation. NaN propagates through alpha -> CI bounds.
        val result = proportionZTest(successes = 50, trials = 100, p0 = 0.5, confidenceLevel = Double.NaN)
        assertTrue(result.statistic.isFinite(), "statistic should be finite (NaN only affects CI)")
        val ci = result.confidenceInterval!!
        assertTrue(ci.first.isNaN(), "CI lower should be NaN when confidenceLevel is NaN")
        assertTrue(ci.second.isNaN(), "CI upper should be NaN when confidenceLevel is NaN")
    }

    @Test
    fun testOneSampleInfinityP0() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 50, trials = 100, p0 = Double.POSITIVE_INFINITY)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes = 50, trials = 100, p0 = Double.NEGATIVE_INFINITY)
        }
    }

    // =========================================================================
    // ONE-SAMPLE: Property-based
    // =========================================================================

    @Test
    fun testOneSamplePValueRange() {
        val cases = listOf(
            proportionZTest(successes = 60, trials = 100, p0 = 0.5),
            proportionZTest(successes = 30, trials = 200, p0 = 0.2),
            proportionZTest(successes = 1, trials = 1, p0 = 0.5),
            proportionZTest(successes = 0, trials = 100, p0 = 0.5),
            proportionZTest(successes = 100, trials = 100, p0 = 0.5),
        )
        for (result in cases) {
            assertTrue(result.pValue in 0.0..1.0,
                "p-value should be in [0, 1], got ${result.pValue}")
        }
    }

    @Test
    fun testOneSampleAlternativeConsistency() {
        val two = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        val less = proportionZTest(successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.LESS)
        val greater = proportionZTest(successes = 60, trials = 100, p0 = 0.5, alternative = Alternative.GREATER)

        // Same statistic regardless of alternative
        assertEquals(two.statistic, less.statistic, 1e-14, "statistic: two vs less")
        assertEquals(two.statistic, greater.statistic, 1e-14, "statistic: two vs greater")

        // One-sided p-values sum to 1
        assertEquals(1.0, less.pValue + greater.pValue, 1e-14, "less + greater = 1")

        // Two-sided = 2 * min(less, greater)
        assertEquals(two.pValue, 2.0 * minOf(less.pValue, greater.pValue), 1e-14,
            "two-sided = 2 * min(one-sided)")
    }

    @Test
    fun testOneSampleAlternativeConsistencyNegativeZ() {
        // When z is negative (pHat < p0), verify the same properties
        val two = proportionZTest(successes = 30, trials = 200, p0 = 0.2)
        val less = proportionZTest(successes = 30, trials = 200, p0 = 0.2, alternative = Alternative.LESS)
        val greater = proportionZTest(successes = 30, trials = 200, p0 = 0.2, alternative = Alternative.GREATER)

        assertEquals(two.statistic, less.statistic, 1e-14)
        assertEquals(two.statistic, greater.statistic, 1e-14)
        assertEquals(1.0, less.pValue + greater.pValue, 1e-14, "less + greater = 1")
        assertEquals(two.pValue, 2.0 * minOf(less.pValue, greater.pValue), 1e-14)
    }

    @Test
    fun testOneSampleSymmetryAroundNull() {
        // proportionZTest(k, n, p0) should give statistic = -proportionZTest(n-k, n, 1-p0)
        // i.e., testing 60/100 at p0=0.5 should mirror testing 40/100 at p0=0.5
        val r1 = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        val r2 = proportionZTest(successes = 40, trials = 100, p0 = 0.5)
        assertEquals(r1.statistic, -r2.statistic, 1e-14, "z-statistics should be symmetric")
        assertEquals(r1.pValue, r2.pValue, 1e-14, "p-values should be equal (two-sided)")
    }

    @Test
    fun testOneSampleIsSignificantConsistency() {
        val cases = listOf(
            proportionZTest(successes = 60, trials = 100, p0 = 0.5),  // p ≈ 0.046, significant
            proportionZTest(successes = 75, trials = 150, p0 = 0.5),  // p = 1.0, not significant
            proportionZTest(successes = 85, trials = 100, p0 = 0.75), // p ≈ 0.021, significant
            proportionZTest(successes = 10, trials = 50, p0 = 0.3),   // p ≈ 0.123, not significant
        )
        for (result in cases) {
            assertEquals(result.pValue < 0.05, result.isSignificant(),
                "isSignificant for p=${result.pValue}")
        }
    }

    @Test
    fun testOneSampleCINarrowerAtHigherConfidence() {
        val ci90 = proportionZTest(successes = 60, trials = 100, p0 = 0.5, confidenceLevel = 0.90).confidenceInterval!!
        val ci95 = proportionZTest(successes = 60, trials = 100, p0 = 0.5, confidenceLevel = 0.95).confidenceInterval!!
        val ci99 = proportionZTest(successes = 60, trials = 100, p0 = 0.5, confidenceLevel = 0.99).confidenceInterval!!

        // Higher confidence = wider CI
        assertTrue(ci90.first > ci95.first, "90% lower > 95% lower")
        assertTrue(ci90.second < ci95.second, "90% upper < 95% upper")
        assertTrue(ci95.first > ci99.first, "95% lower > 99% lower")
        assertTrue(ci95.second < ci99.second, "95% upper < 99% upper")
    }

    @Test
    fun testOneSampleCustomConfidenceLevels() {
        // scipy: Wald CI for 60/100 at different confidence levels
        val ci90 = proportionZTest(successes = 60, trials = 100, p0 = 0.5, confidenceLevel = 0.90).confidenceInterval!!
        val ci99 = proportionZTest(successes = 60, trials = 100, p0 = 0.5, confidenceLevel = 0.99).confidenceInterval!!

        // scipy: CI(0.9): [0.519418958248053, 0.680581041751947]
        assertCI(0.519418958248053, 0.680581041751947, ci90, tol = 1e-10, message = "90% CI")
        // scipy: CI(0.99): [0.473810650835933, 0.726189349164067]
        assertCI(0.473810650835933, 0.726189349164067, ci99, tol = 1e-10, message = "99% CI")
    }

    // =========================================================================
    // ONE-SAMPLE: Metadata
    // =========================================================================

    @Test
    fun testOneSampleTestName() {
        val result = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        assertEquals("One-Sample Proportion z-Test", result.testName)
    }

    @Test
    fun testOneSampleAdditionalInfo() {
        val result = proportionZTest(successes = 60, trials = 100, p0 = 0.5)
        assertEquals(0.6, result.additionalInfo["observedProportion"])
        assertEquals(0.5, result.additionalInfo["hypothesizedProportion"])
        assertEquals(0.05, result.additionalInfo["standardError"]!!, tol)
    }

    @Test
    fun testOneSampleAlternativeInResult() {
        assertEquals(Alternative.TWO_SIDED,
            proportionZTest(successes = 5, trials = 10, p0 = 0.5).alternative)
        assertEquals(Alternative.LESS,
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, alternative = Alternative.LESS).alternative)
        assertEquals(Alternative.GREATER,
            proportionZTest(successes = 5, trials = 10, p0 = 0.5, alternative = Alternative.GREATER).alternative)
    }

    // =========================================================================
    // TWO-SAMPLE: Basic correctness
    // =========================================================================

    @Test
    fun testTwoSampleKnownValues60vs40() {
        // scipy: 60/100 vs 40/100
        // z = 2.82842712474619, pval_two = 0.00467773498104728
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100
        )
        assertEquals(2.82842712474619, result.statistic, tol)
        assertP(0.00467773498104728, result.pValue, message = "60/100 vs 40/100 two-sided")
        assertCI(0.0642097119108593, 0.335790288089141, result.confidenceInterval, tol = 1e-10)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testTwoSampleKnownValuesAllAlternatives() {
        // scipy: 60/100 vs 40/100
        val two = proportionZTest(successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100)
        val less = proportionZTest(
            successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100,
            alternative = Alternative.LESS
        )
        val greater = proportionZTest(
            successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100,
            alternative = Alternative.GREATER
        )

        // scipy: pval_less = 0.997661132509476
        assertP(0.997661132509476, less.pValue, message = "60/100 vs 40/100 less")
        // scipy: pval_greater = 0.00233886749052364
        assertP(0.00233886749052364, greater.pValue, message = "60/100 vs 40/100 greater")
    }

    @Test
    fun testTwoSampleKnownValues30of200vs50of300() {
        // scipy: 30/200 vs 50/300
        // z = -0.498011920555997, pval_two = 0.61847564019994
        val result = proportionZTest(
            successes1 = 30, trials1 = 200,
            successes2 = 50, trials2 = 300
        )
        assertEquals(-0.498011920555997, result.statistic, tol)
        assertP(0.61847564019994, result.pValue, message = "30/200 vs 50/300 two-sided")
        assertCI(-0.0816849960509354, 0.0483516627176021, result.confidenceInterval, tol = 1e-10)
        assertFalse(result.isSignificant())
    }

    @Test
    fun testTwoSampleKnownValues90vs80() {
        // scipy: 90/100 vs 80/100
        // z = 1.98029508595335, pval_two = 0.0476703806561615
        val result = proportionZTest(
            successes1 = 90, trials1 = 100,
            successes2 = 80, trials2 = 100
        )
        assertEquals(1.98029508595335, result.statistic, tol)
        assertP(0.0476703806561615, result.pValue, message = "90/100 vs 80/100 two-sided")
        assertCI(0.00200180077299729, 0.197998199227003, result.confidenceInterval, tol = 1e-10)
    }

    @Test
    fun testTwoSampleEqualProportions() {
        // scipy: 150/500 vs 120/400 (both 0.3) => z = 0, pval = 1.0
        val result = proportionZTest(
            successes1 = 150, trials1 = 500,
            successes2 = 120, trials2 = 400
        )
        assertEquals(0.0, result.statistic, tol)
        assertP(1.0, result.pValue, message = "equal proportions two-sided")
        assertCI(-0.0602509633579078, 0.0602509633579078, result.confidenceInterval, tol = 1e-10)
        assertFalse(result.isSignificant())
    }

    // =========================================================================
    // TWO-SAMPLE: Edge cases
    // =========================================================================

    @Test
    fun testTwoSampleBothZeroSuccesses() {
        // Both pHat=0 => pPool=0, sePool=0, diff=0 => z=0, p=1
        val result = proportionZTest(
            successes1 = 0, trials1 = 100,
            successes2 = 0, trials2 = 100
        )
        assertEquals(0.0, result.statistic, tol)
        assertP(1.0, result.pValue, message = "both zero successes")
    }

    @Test
    fun testTwoSampleBothAllSuccesses() {
        // Both pHat=1 => pPool=1, sePool=0, diff=0 => z=0, p=1
        val result = proportionZTest(
            successes1 = 100, trials1 = 100,
            successes2 = 100, trials2 = 100
        )
        assertEquals(0.0, result.statistic, tol)
        assertP(1.0, result.pValue, message = "both all successes")
    }

    @Test
    fun testTwoSampleZeroVsAll() {
        // 0/100 vs 100/100: extreme case
        // scipy: ppool=0.5, se_pool=0.0707106781186548
        // z = -14.142135623731, pval ≈ 2e-45
        val result = proportionZTest(
            successes1 = 0, trials1 = 100,
            successes2 = 100, trials2 = 100
        )
        assertEquals(-14.142135623731, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-40, "p should be extremely small for 0/100 vs 100/100")
    }

    @Test
    fun testTwoSampleSingleTrialEach() {
        // Minimal sample sizes: 1/1 vs 0/1
        val result = proportionZTest(
            successes1 = 1, trials1 = 1,
            successes2 = 0, trials2 = 1
        )
        assertTrue(result.statistic.isFinite() || result.statistic.isInfinite(),
            "statistic should be computable for single trials")
        assertTrue(result.pValue in 0.0..1.0, "p should be in [0, 1]")
    }

    @Test
    fun testTwoSampleOneSidedCILess() {
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100,
            alternative = Alternative.LESS
        )
        val ci = result.confidenceInterval!!
        assertEquals(Double.NEGATIVE_INFINITY, ci.first, "LESS CI lower should be -Inf")
        assertTrue(ci.second.isFinite(), "LESS CI upper should be finite")
    }

    @Test
    fun testTwoSampleOneSidedCIGreater() {
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100,
            alternative = Alternative.GREATER
        )
        val ci = result.confidenceInterval!!
        assertTrue(ci.first.isFinite(), "GREATER CI lower should be finite")
        assertEquals(Double.POSITIVE_INFINITY, ci.second, "GREATER CI upper should be +Inf")
    }

    // =========================================================================
    // TWO-SAMPLE: Degenerate input / validation
    // =========================================================================

    @Test
    fun testTwoSampleZeroTrials1() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes1 = 0, trials1 = 0, successes2 = 5, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleZeroTrials2() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes1 = 5, trials1 = 10, successes2 = 0, trials2 = 0)
        }
    }

    @Test
    fun testTwoSampleNegativeTrials1() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes1 = 0, trials1 = -5, successes2 = 5, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleNegativeTrials2() {
        assertFailsWith<InsufficientDataException> {
            proportionZTest(successes1 = 5, trials1 = 10, successes2 = 0, trials2 = -5)
        }
    }

    @Test
    fun testTwoSampleNegativeSuccesses1() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes1 = -1, trials1 = 10, successes2 = 5, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleSuccesses1ExceedTrials1() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes1 = 11, trials1 = 10, successes2 = 5, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleNegativeSuccesses2() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes1 = 5, trials1 = 10, successes2 = -1, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleSuccesses2ExceedTrials2() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(successes1 = 5, trials1 = 10, successes2 = 11, trials2 = 10)
        }
    }

    @Test
    fun testTwoSampleConfidenceLevelZero() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(
                successes1 = 5, trials1 = 10,
                successes2 = 5, trials2 = 10,
                confidenceLevel = 0.0
            )
        }
    }

    @Test
    fun testTwoSampleConfidenceLevelOne() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(
                successes1 = 5, trials1 = 10,
                successes2 = 5, trials2 = 10,
                confidenceLevel = 1.0
            )
        }
    }

    @Test
    fun testTwoSampleConfidenceLevelNegative() {
        assertFailsWith<InvalidParameterException> {
            proportionZTest(
                successes1 = 5, trials1 = 10,
                successes2 = 5, trials2 = 10,
                confidenceLevel = -0.5
            )
        }
    }

    // =========================================================================
    // TWO-SAMPLE: Extreme parameters
    // =========================================================================

    @Test
    fun testTwoSampleLargeN() {
        // Large sample sizes should remain numerically stable
        val result = proportionZTest(
            successes1 = 50500, trials1 = 100000,
            successes2 = 49500, trials2 = 100000
        )
        assertTrue(result.statistic.isFinite(), "statistic should be finite for large n")
        assertTrue(result.pValue.isFinite(), "p-value should be finite for large n")
        assertTrue(result.pValue in 0.0..1.0)
    }

    @Test
    fun testTwoSampleAsymmetricSizes() {
        // Very different sample sizes: 10 vs 10000
        val result = proportionZTest(
            successes1 = 5, trials1 = 10,
            successes2 = 5000, trials2 = 10000
        )
        assertTrue(result.statistic.isFinite())
        assertTrue(result.pValue in 0.0..1.0)
    }

    // =========================================================================
    // TWO-SAMPLE: Non-finite
    // =========================================================================

    @Test
    fun testTwoSampleNaNConfidenceLevel() {
        // IEEE 754: NaN passes <= / >= validation. NaN propagates through alpha -> CI bounds.
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100,
            confidenceLevel = Double.NaN
        )
        assertTrue(result.statistic.isFinite(), "statistic should be finite (NaN only affects CI)")
        val ci = result.confidenceInterval!!
        assertTrue(ci.first.isNaN(), "CI lower should be NaN when confidenceLevel is NaN")
        assertTrue(ci.second.isNaN(), "CI upper should be NaN when confidenceLevel is NaN")
    }

    // =========================================================================
    // TWO-SAMPLE: Property-based
    // =========================================================================

    @Test
    fun testTwoSamplePValueRange() {
        val cases = listOf(
            proportionZTest(successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100),
            proportionZTest(successes1 = 30, trials1 = 200, successes2 = 50, trials2 = 300),
            proportionZTest(successes1 = 0, trials1 = 100, successes2 = 0, trials2 = 100),
            proportionZTest(successes1 = 100, trials1 = 100, successes2 = 100, trials2 = 100),
            proportionZTest(successes1 = 0, trials1 = 100, successes2 = 100, trials2 = 100),
            proportionZTest(successes1 = 1, trials1 = 1, successes2 = 0, trials2 = 1),
        )
        for (result in cases) {
            assertTrue(result.pValue in 0.0..1.0,
                "p-value should be in [0, 1], got ${result.pValue}")
        }
    }

    @Test
    fun testTwoSampleAlternativeConsistency() {
        val two = proportionZTest(successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100)
        val less = proportionZTest(
            successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100,
            alternative = Alternative.LESS
        )
        val greater = proportionZTest(
            successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100,
            alternative = Alternative.GREATER
        )

        // Same statistic regardless of alternative
        assertEquals(two.statistic, less.statistic, 1e-14, "statistic: two vs less")
        assertEquals(two.statistic, greater.statistic, 1e-14, "statistic: two vs greater")

        // One-sided p-values sum to 1
        assertEquals(1.0, less.pValue + greater.pValue, 1e-14, "less + greater = 1")

        // Two-sided = 2 * min(less, greater)
        assertEquals(two.pValue, 2.0 * minOf(less.pValue, greater.pValue), 1e-14,
            "two-sided = 2 * min(one-sided)")
    }

    @Test
    fun testTwoSampleAntiSymmetry() {
        // Swapping the two groups should negate the statistic, keep p-value (two-sided) equal
        val r1 = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100
        )
        val r2 = proportionZTest(
            successes1 = 40, trials1 = 100,
            successes2 = 60, trials2 = 100
        )
        assertEquals(r1.statistic, -r2.statistic, 1e-14, "swapping groups negates z")
        assertEquals(r1.pValue, r2.pValue, 1e-14, "two-sided p should be equal after swap")
    }

    @Test
    fun testTwoSampleEqualGroupsZeroStatistic() {
        // Same proportions => z = 0
        val result = proportionZTest(
            successes1 = 50, trials1 = 100,
            successes2 = 50, trials2 = 100
        )
        assertEquals(0.0, result.statistic, tol)
        assertP(1.0, result.pValue, message = "equal proportions => p = 1")
    }

    @Test
    fun testTwoSampleIsSignificantConsistency() {
        val cases = listOf(
            proportionZTest(successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100), // significant
            proportionZTest(successes1 = 150, trials1 = 500, successes2 = 120, trials2 = 400), // not significant
            proportionZTest(successes1 = 90, trials1 = 100, successes2 = 80, trials2 = 100), // borderline
        )
        for (result in cases) {
            assertEquals(result.pValue < 0.05, result.isSignificant(),
                "isSignificant for p=${result.pValue}")
        }
    }

    @Test
    fun testTwoSampleCIContainsDifference() {
        // The CI for the difference should contain the observed difference
        val cases = listOf(
            proportionZTest(successes1 = 60, trials1 = 100, successes2 = 40, trials2 = 100),
            proportionZTest(successes1 = 30, trials1 = 200, successes2 = 50, trials2 = 300),
            proportionZTest(successes1 = 150, trials1 = 500, successes2 = 120, trials2 = 400),
        )
        for (result in cases) {
            val diff = result.additionalInfo["proportionDifference"]!!
            val ci = result.confidenceInterval!!
            assertTrue(diff >= ci.first && diff <= ci.second,
                "CI should contain observed difference $diff, got [${ci.first}, ${ci.second}]")
        }
    }

    // =========================================================================
    // TWO-SAMPLE: Metadata
    // =========================================================================

    @Test
    fun testTwoSampleTestName() {
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100
        )
        assertEquals("Two-Sample Proportion z-Test", result.testName)
    }

    @Test
    fun testTwoSampleAdditionalInfo() {
        val result = proportionZTest(
            successes1 = 60, trials1 = 100,
            successes2 = 40, trials2 = 100
        )
        assertEquals(0.6, result.additionalInfo["proportion1"]!!, tol)
        assertEquals(0.4, result.additionalInfo["proportion2"]!!, tol)
        assertEquals(0.2, result.additionalInfo["proportionDifference"]!!, tol)
        assertEquals(0.5, result.additionalInfo["pooledProportion"]!!, tol)
        assertTrue(result.additionalInfo["standardError"]!! > 0.0)
    }

    @Test
    fun testTwoSampleAlternativeInResult() {
        assertEquals(Alternative.TWO_SIDED,
            proportionZTest(
                successes1 = 5, trials1 = 10, successes2 = 5, trials2 = 10
            ).alternative
        )
        assertEquals(Alternative.LESS,
            proportionZTest(
                successes1 = 5, trials1 = 10, successes2 = 5, trials2 = 10,
                alternative = Alternative.LESS
            ).alternative
        )
        assertEquals(Alternative.GREATER,
            proportionZTest(
                successes1 = 5, trials1 = 10, successes2 = 5, trials2 = 10,
                alternative = Alternative.GREATER
            ).alternative
        )
    }
}
