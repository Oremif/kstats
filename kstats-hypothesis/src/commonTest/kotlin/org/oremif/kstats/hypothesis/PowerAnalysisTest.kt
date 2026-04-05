package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class PowerAnalysisTest {

    private val tol = 1e-10

    // ===== tTestPower: Basic correctness =====

    @Test
    fun testTTestPowerKnownValuesTwoSampleTwoSided() {
        // Normal approximation: power = Phi(lambda - z_crit) + Phi(-lambda - z_crit)
        // lambda = d * sqrt(n/2), z_crit = Phi^-1(0.975)

        // scipy: d=0.5, n=64, TWO_SAMPLE, TWO_SIDED, alpha=0.05
        assertEquals(
            0.807430419432557,
            tTestPower(effectSize = 0.5, n = 64, type = TTestType.TWO_SAMPLE),
            tol,
        )
        // scipy: d=0.8, n=20, TWO_SAMPLE, TWO_SIDED, alpha=0.05
        assertEquals(
            0.715616606789121,
            tTestPower(effectSize = 0.8, n = 20, type = TTestType.TWO_SAMPLE),
            tol,
        )
        // scipy: d=0.2, n=100, TWO_SAMPLE, TWO_SIDED, alpha=0.05
        assertEquals(
            0.292988936447987,
            tTestPower(effectSize = 0.2, n = 100, type = TTestType.TWO_SAMPLE),
            tol,
        )
    }

    @Test
    fun testTTestPowerOneSampleAndPaired() {
        // ONE_SAMPLE uses factor = sqrt(n), giving higher power than TWO_SAMPLE
        // scipy: d=0.5, n=64, ONE_SAMPLE, TWO_SIDED, alpha=0.05
        assertEquals(
            0.979326631902576,
            tTestPower(effectSize = 0.5, n = 64, type = TTestType.ONE_SAMPLE),
            tol,
        )
        // PAIRED uses same factor as ONE_SAMPLE
        // scipy: d=0.5, n=20, PAIRED, TWO_SIDED, alpha=0.05
        assertEquals(
            0.608779484645457,
            tTestPower(effectSize = 0.5, n = 20, type = TTestType.PAIRED),
            tol,
        )
    }

    @Test
    fun testTTestPowerAlternatives() {
        // scipy: d=0.5, n=64, TWO_SAMPLE, GREATER, alpha=0.05
        assertEquals(
            0.881709031778347,
            tTestPower(effectSize = 0.5, n = 64, alternative = Alternative.GREATER),
            tol,
        )
        // LESS uses same formula as GREATER (effectSize is abs'd)
        assertEquals(
            0.881709031778347,
            tTestPower(effectSize = 0.5, n = 64, alternative = Alternative.LESS),
            tol,
        )
    }

    @Test
    fun testTTestPowerCustomAlpha() {
        // scipy: d=0.5, n=64, TWO_SAMPLE, TWO_SIDED, alpha=0.01
        assertEquals(
            0.599710525606104,
            tTestPower(effectSize = 0.5, n = 64, alpha = 0.01),
            tol,
        )
    }

    // ===== tTestPower: Edge cases =====

    @Test
    fun testTTestPowerZeroEffectSize() {
        // d=0 => lambda=0 => power = alpha (type I error rate)
        assertEquals(0.05, tTestPower(effectSize = 0.0, n = 64), tol)
    }

    @Test
    fun testTTestPowerZeroEffectSizeOneSided() {
        // d=0, one-sided => power = alpha
        // scipy: cdf(0 - z_0.95) = cdf(-1.6449) = 0.05
        assertEquals(
            0.05,
            tTestPower(effectSize = 0.0, n = 64, alternative = Alternative.GREATER),
            tol,
        )
    }

    @Test
    fun testTTestPowerMinimumN() {
        // n=2 is the minimum valid sample size
        // scipy: d=0.5, n=2, TWO_SAMPLE, TWO_SIDED
        assertEquals(
            0.0790975341605965,
            tTestPower(effectSize = 0.5, n = 2),
            tol,
        )
    }

    @Test
    fun testTTestPowerNegativeEffectSize() {
        // Negative effect size gives same power as positive (abs taken)
        val powerPos = tTestPower(effectSize = 0.5, n = 64)
        val powerNeg = tTestPower(effectSize = -0.5, n = 64)
        assertEquals(powerPos, powerNeg, tol, "Power should be symmetric in effectSize")
    }

    // ===== tTestPower: Degenerate input =====

    @Test
    fun testTTestPowerInvalidN() {
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 1)
        }
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = -5)
        }
    }

    @Test
    fun testTTestPowerInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 64, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 64, alpha = 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 64, alpha = -0.1)
        }
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 64, alpha = 1.5)
        }
    }

    // ===== tTestPower: Extreme parameters =====

    @Test
    fun testTTestPowerVeryLargeEffectSize() {
        // d=5.0, n=10 => lambda very large => power ~ 1.0
        // scipy: power = 1.0
        assertEquals(
            1.0,
            tTestPower(effectSize = 5.0, n = 10),
            tol,
        )
    }

    @Test
    fun testTTestPowerVerySmallEffectSize() {
        // d=0.01, n=1000 => power barely above alpha
        // scipy: power = 0.0557472499419697
        assertEquals(
            0.0557472499419697,
            tTestPower(effectSize = 0.01, n = 1000),
            tol,
        )
    }

    @Test
    fun testTTestPowerVeryLargeN() {
        // d=0.5, n=10000 => power ~ 1.0
        assertEquals(
            1.0,
            tTestPower(effectSize = 0.5, n = 10000),
            tol,
        )
    }

    // ===== tTestPower: Non-finite input =====

    @Test
    fun testTTestPowerNaNEffectSize() {
        // NaN passes validation (NaN <= 0.0 is false for alpha), propagates through computation
        val power = tTestPower(effectSize = Double.NaN, n = 64)
        assertTrue(power.isNaN(), "Power should be NaN when effectSize is NaN")
    }

    @Test
    fun testTTestPowerNaNAlpha() {
        // NaN alpha passes outer validation but causes InvalidParameterException
        // inside NormalDistribution.quantile(NaN) since NaN !in 0.0..1.0
        assertFailsWith<InvalidParameterException> {
            tTestPower(effectSize = 0.5, n = 64, alpha = Double.NaN)
        }
    }

    // ===== tTestPower: Property-based =====

    @Test
    fun testTTestPowerRange() {
        // Power should always be in [0, 1]
        val params = listOf(
            0.1 to 20, 0.2 to 50, 0.5 to 64, 0.8 to 30, 1.0 to 10, 2.0 to 5,
        )
        for ((d, n) in params) {
            val power = tTestPower(effectSize = d, n = n)
            assertTrue(power in 0.0..1.0, "power($d, $n) = $power should be in [0, 1]")
        }
    }

    @Test
    fun testTTestPowerMonotonicInN() {
        // For fixed d and alpha, power increases with n
        val nValues = listOf(5, 10, 20, 50, 100, 200, 500)
        var prevPower = 0.0
        for (n in nValues) {
            val power = tTestPower(effectSize = 0.5, n = n)
            assertTrue(power >= prevPower, "power(n=$n) = $power should be >= $prevPower")
            prevPower = power
        }
    }

    @Test
    fun testTTestPowerMonotonicInEffectSize() {
        // For fixed n and alpha, power increases with |d|
        val dValues = listOf(0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0)
        var prevPower = 0.0
        for (d in dValues) {
            val power = tTestPower(effectSize = d, n = 64)
            assertTrue(power >= prevPower, "power(d=$d) = $power should be >= $prevPower")
            prevPower = power
        }
    }

    @Test
    fun testTTestPowerOneSampleGreaterThanTwoSample() {
        // ONE_SAMPLE/PAIRED always has more power than TWO_SAMPLE for same n
        for (d in listOf(0.2, 0.5, 0.8)) {
            for (n in listOf(20, 50, 100)) {
                val powerOne = tTestPower(effectSize = d, n = n, type = TTestType.ONE_SAMPLE)
                val powerTwo = tTestPower(effectSize = d, n = n, type = TTestType.TWO_SAMPLE)
                assertTrue(
                    powerOne >= powerTwo,
                    "ONE_SAMPLE power ($powerOne) should >= TWO_SAMPLE power ($powerTwo) for d=$d, n=$n",
                )
            }
        }
    }

    @Test
    fun testTTestPowerOneSidedGreaterThanTwoSided() {
        // One-sided test has more power than two-sided
        for (d in listOf(0.2, 0.5, 0.8)) {
            val twoSided = tTestPower(effectSize = d, n = 64, alternative = Alternative.TWO_SIDED)
            val oneSided = tTestPower(effectSize = d, n = 64, alternative = Alternative.GREATER)
            assertTrue(
                oneSided >= twoSided,
                "one-sided power ($oneSided) should >= two-sided ($twoSided) for d=$d",
            )
        }
    }

    // ===== tTestRequiredN: Basic correctness =====

    @Test
    fun testTTestRequiredNKnownValues() {
        // Normal approximation: lambda = z_crit + z_beta, n = ceil(2 * (lambda/d)^2)
        // scipy: d=0.5, power=0.8, TWO_SAMPLE, TWO_SIDED, alpha=0.05 -> n=63
        assertEquals(63, tTestRequiredN(effectSize = 0.5))
        // scipy: d=0.8, power=0.8, TWO_SAMPLE, TWO_SIDED, alpha=0.05 -> n=25
        assertEquals(25, tTestRequiredN(effectSize = 0.8))
        // scipy: d=0.2, power=0.8, TWO_SAMPLE, TWO_SIDED, alpha=0.05 -> n=393
        assertEquals(393, tTestRequiredN(effectSize = 0.2))
    }

    @Test
    fun testTTestRequiredNOneSampleAndPaired() {
        // scipy: d=0.5, power=0.8, ONE_SAMPLE, TWO_SIDED, alpha=0.05 -> n=32
        assertEquals(32, tTestRequiredN(effectSize = 0.5, type = TTestType.ONE_SAMPLE))
        // PAIRED same as ONE_SAMPLE
        assertEquals(32, tTestRequiredN(effectSize = 0.5, type = TTestType.PAIRED))
    }

    @Test
    fun testTTestRequiredNCustomPowerAndAlpha() {
        // scipy: d=0.5, power=0.9, TWO_SAMPLE -> n=85
        assertEquals(85, tTestRequiredN(effectSize = 0.5, power = 0.9))
        // scipy: d=0.5, power=0.8, alpha=0.01, TWO_SAMPLE -> n=94
        assertEquals(94, tTestRequiredN(effectSize = 0.5, alpha = 0.01))
    }

    @Test
    fun testTTestRequiredNOneSided() {
        // scipy: d=0.5, power=0.8, TWO_SAMPLE, GREATER -> n=50
        assertEquals(50, tTestRequiredN(effectSize = 0.5, alternative = Alternative.GREATER))
    }

    // ===== tTestRequiredN: Edge cases =====

    @Test
    fun testTTestRequiredNNegativeEffectSize() {
        // Negative d gives same n as positive (abs taken)
        assertEquals(
            tTestRequiredN(effectSize = 0.5),
            tTestRequiredN(effectSize = -0.5),
        )
    }

    @Test
    fun testTTestRequiredNMinimumReturnsAtLeastTwo() {
        // Very large effect size could compute nRaw < 2, but result is clamped to 2
        val n = tTestRequiredN(effectSize = 100.0, power = 0.8)
        assertTrue(n >= 2, "n should be at least 2, got $n")
    }

    // ===== tTestRequiredN: Degenerate input =====

    @Test
    fun testTTestRequiredNZeroEffectSize() {
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.0)
        }
    }

    @Test
    fun testTTestRequiredNInvalidPower() {
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, power = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, power = 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, power = -0.1)
        }
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, power = 1.5)
        }
    }

    @Test
    fun testTTestRequiredNInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, alpha = 1.0)
        }
    }

    // ===== tTestRequiredN: Extreme parameters =====

    @Test
    fun testTTestRequiredNVerySmallEffectSize() {
        // d=0.01 needs very large n
        val n = tTestRequiredN(effectSize = 0.01)
        assertTrue(n > 100000, "n for d=0.01 should be very large, got $n")
    }

    @Test
    fun testTTestRequiredNVeryLargeEffectSize() {
        // d=10.0 needs very small n (clamped to 2)
        val n = tTestRequiredN(effectSize = 10.0)
        assertEquals(2, n, "n for d=10.0 should be clamped to minimum 2")
    }

    // ===== tTestRequiredN: Non-finite input =====

    @Test
    fun testTTestRequiredNNaNEffectSize() {
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = Double.NaN)
        }
    }

    @Test
    fun testTTestRequiredNInfinityEffectSize() {
        // Infinity effectSize: Inf/Inf = NaN -> ceil(NaN).toInt() = 0 -> maxOf(2, 0) = 2
        val n = tTestRequiredN(effectSize = Double.POSITIVE_INFINITY)
        assertEquals(2, n)
    }

    @Test
    fun testTTestRequiredNNaNPower() {
        // NaN power passes the range check but throws inside quantile()
        assertFailsWith<InvalidParameterException> {
            tTestRequiredN(effectSize = 0.5, power = Double.NaN)
        }
    }

    // ===== tTestRequiredN: Property-based =====

    @Test
    fun testTTestRequiredNPowerRoundTrip() {
        // If we compute n for target power, then compute power with that n,
        // the achieved power should be >= the target
        for (d in listOf(0.2, 0.5, 0.8)) {
            for (targetPower in listOf(0.8, 0.9)) {
                val n = tTestRequiredN(effectSize = d, power = targetPower)
                val achievedPower = tTestPower(effectSize = d, n = n)
                assertTrue(
                    achievedPower >= targetPower - 1e-10,
                    "achievedPower($achievedPower) should >= target($targetPower) for d=$d, n=$n",
                )
            }
        }
    }

    @Test
    fun testTTestRequiredNOneSampleLessThanTwoSample() {
        // ONE_SAMPLE needs fewer subjects than TWO_SAMPLE
        for (d in listOf(0.2, 0.5, 0.8)) {
            val nOne = tTestRequiredN(effectSize = d, type = TTestType.ONE_SAMPLE)
            val nTwo = tTestRequiredN(effectSize = d, type = TTestType.TWO_SAMPLE)
            assertTrue(
                nOne <= nTwo,
                "ONE_SAMPLE n ($nOne) should be <= TWO_SAMPLE n ($nTwo) for d=$d",
            )
        }
    }

    @Test
    fun testTTestRequiredNDecreasesWithEffectSize() {
        // Larger effect size => fewer subjects needed
        var prevN = Int.MAX_VALUE
        for (d in listOf(0.1, 0.2, 0.5, 0.8, 1.0, 2.0)) {
            val n = tTestRequiredN(effectSize = d)
            assertTrue(n <= prevN, "n(d=$d)=$n should be <= n(d_prev)=$prevN")
            prevN = n
        }
    }

    // ===== tTestMinimumEffect: Basic correctness =====

    @Test
    fun testTTestMinimumEffectKnownValues() {
        // lambda = z_crit + z_beta, effect = lambda / factor
        // scipy: n=64, power=0.8, TWO_SAMPLE, TWO_SIDED, alpha=0.05
        assertEquals(
            0.495254976449918,
            tTestMinimumEffect(n = 64),
            tol,
        )
        // scipy: n=20, power=0.8, TWO_SAMPLE, TWO_SIDED
        assertEquals(
            0.88593903482966,
            tTestMinimumEffect(n = 20),
            tol,
        )
        // scipy: n=100, power=0.8, TWO_SAMPLE, TWO_SIDED
        assertEquals(
            0.396203981159935,
            tTestMinimumEffect(n = 100),
            tol,
        )
    }

    @Test
    fun testTTestMinimumEffectOneSample() {
        // scipy: n=64, power=0.8, ONE_SAMPLE, TWO_SIDED
        assertEquals(
            0.350198152264121,
            tTestMinimumEffect(n = 64, type = TTestType.ONE_SAMPLE),
            tol,
        )
    }

    @Test
    fun testTTestMinimumEffectCustomParams() {
        // scipy: n=64, power=0.9, TWO_SAMPLE, TWO_SIDED
        assertEquals(
            0.573024406696625,
            tTestMinimumEffect(n = 64, power = 0.9),
            tol,
        )
        // scipy: n=64, power=0.8, TWO_SAMPLE, alpha=0.01
        assertEquals(
            0.604125612292111,
            tTestMinimumEffect(n = 64, alpha = 0.01),
            tol,
        )
        // scipy: n=64, power=0.8, TWO_SAMPLE, GREATER
        assertEquals(
            0.439550808781667,
            tTestMinimumEffect(n = 64, alternative = Alternative.GREATER),
            tol,
        )
    }

    // ===== tTestMinimumEffect: Edge cases =====

    @Test
    fun testTTestMinimumEffectMinimumN() {
        // n=2 gives very large effect size requirement
        // scipy: effect = 2.80158521811297
        assertEquals(
            2.80158521811297,
            tTestMinimumEffect(n = 2),
            tol,
        )
    }

    // ===== tTestMinimumEffect: Degenerate input =====

    @Test
    fun testTTestMinimumEffectInvalidN() {
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 1)
        }
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = -1)
        }
    }

    @Test
    fun testTTestMinimumEffectInvalidPower() {
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, power = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, power = 1.0)
        }
    }

    @Test
    fun testTTestMinimumEffectInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, alpha = 1.0)
        }
    }

    // ===== tTestMinimumEffect: Extreme parameters =====

    @Test
    fun testTTestMinimumEffectLargeN() {
        // n=10000 gives very small minimum effect
        // scipy: effect = 0.0396203981159935
        assertEquals(
            0.0396203981159935,
            tTestMinimumEffect(n = 10000),
            tol,
        )
    }

    @Test
    fun testTTestMinimumEffectHighPower() {
        // power=0.99 requires larger effect
        // scipy: effect = 0.757720045370716
        assertEquals(
            0.757720045370716,
            tTestMinimumEffect(n = 64, power = 0.99),
            tol,
        )
    }

    // ===== tTestMinimumEffect: Non-finite input =====

    @Test
    fun testTTestMinimumEffectNaNPower() {
        // NaN power passes outer validation but causes InvalidParameterException
        // inside NormalDistribution.quantile(NaN) since NaN !in 0.0..1.0
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, power = Double.NaN)
        }
    }

    @Test
    fun testTTestMinimumEffectNaNAlpha() {
        // NaN alpha passes outer validation but causes InvalidParameterException
        // inside NormalDistribution.quantile(NaN) since NaN !in 0.0..1.0
        assertFailsWith<InvalidParameterException> {
            tTestMinimumEffect(n = 64, alpha = Double.NaN)
        }
    }

    // ===== tTestMinimumEffect: Property-based =====

    @Test
    fun testTTestMinimumEffectPowerRoundTrip() {
        // minimumEffect(n) -> d, then power(d, n) should ~= target power
        for (n in listOf(10, 20, 50, 100, 200)) {
            val d = tTestMinimumEffect(n = n)
            val achievedPower = tTestPower(effectSize = d, n = n)
            assertEquals(0.8, achievedPower, 1e-6, "Round-trip power for n=$n, d=$d")
        }
    }

    @Test
    fun testTTestMinimumEffectDecreasesWithN() {
        // Larger n => smaller minimum detectable effect
        var prevEffect = Double.MAX_VALUE
        for (n in listOf(5, 10, 20, 50, 100, 500, 1000)) {
            val effect = tTestMinimumEffect(n = n)
            assertTrue(effect < prevEffect, "effect(n=$n) = $effect should be < $prevEffect")
            prevEffect = effect
        }
    }

    @Test
    fun testTTestMinimumEffectPositive() {
        // Minimum effect should be positive for reasonable parameters
        for (n in listOf(10, 50, 100)) {
            val effect = tTestMinimumEffect(n = n)
            assertTrue(effect > 0.0, "Minimum effect for n=$n should be positive, got $effect")
        }
    }

    @Test
    fun testTTestMinimumEffectOneSampleSmallerThanTwoSample() {
        // ONE_SAMPLE has more power, so minimum detectable effect is smaller
        for (n in listOf(20, 50, 100)) {
            val effectOne = tTestMinimumEffect(n = n, type = TTestType.ONE_SAMPLE)
            val effectTwo = tTestMinimumEffect(n = n, type = TTestType.TWO_SAMPLE)
            assertTrue(
                effectOne <= effectTwo,
                "ONE_SAMPLE effect ($effectOne) should <= TWO_SAMPLE ($effectTwo) for n=$n",
            )
        }
    }

    // ===== proportionZTestPower: Basic correctness =====

    @Test
    fun testProportionZTestPowerKnownValues() {
        // scipy: h=0.5, n=64, twoSample=true, TWO_SIDED, alpha=0.05
        assertEquals(
            0.807430419432557,
            proportionZTestPower(effectSize = 0.5, n = 64),
            tol,
        )
        // scipy: h=0.3, n=200, twoSample=true, TWO_SIDED
        assertEquals(
            0.850838768327056,
            proportionZTestPower(effectSize = 0.3, n = 200),
            tol,
        )
    }

    @Test
    fun testProportionZTestPowerOneSample() {
        // scipy: h=0.5, n=64, twoSample=false, TWO_SIDED
        assertEquals(
            0.979326631902576,
            proportionZTestPower(effectSize = 0.5, n = 64, twoSample = false),
            tol,
        )
    }

    @Test
    fun testProportionZTestPowerAlternatives() {
        // scipy: h=0.5, n=64, twoSample=true, GREATER
        assertEquals(
            0.881709031778347,
            proportionZTestPower(effectSize = 0.5, n = 64, alternative = Alternative.GREATER),
            tol,
        )
    }

    @Test
    fun testProportionZTestPowerCustomAlpha() {
        // scipy: h=0.5, n=64, twoSample=true, alpha=0.01
        assertEquals(
            0.599710525606104,
            proportionZTestPower(effectSize = 0.5, n = 64, alpha = 0.01),
            tol,
        )
    }

    // ===== proportionZTestPower: Edge cases =====

    @Test
    fun testProportionZTestPowerZeroEffectSize() {
        // h=0 => lambda=0 => power = alpha
        assertEquals(0.05, proportionZTestPower(effectSize = 0.0, n = 64), tol)
    }

    @Test
    fun testProportionZTestPowerNegativeEffectSize() {
        val powerPos = proportionZTestPower(effectSize = 0.5, n = 64)
        val powerNeg = proportionZTestPower(effectSize = -0.5, n = 64)
        assertEquals(powerPos, powerNeg, tol, "Power should be symmetric in effectSize")
    }

    @Test
    fun testProportionZTestPowerMinimumN() {
        // n=2 is the minimum valid sample size
        val power = proportionZTestPower(effectSize = 0.5, n = 2)
        assertTrue(power in 0.0..1.0, "Power should be in [0, 1]")
    }

    // ===== proportionZTestPower: Degenerate input =====

    @Test
    fun testProportionZTestPowerInvalidN() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestPower(effectSize = 0.5, n = 1)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestPower(effectSize = 0.5, n = 0)
        }
    }

    @Test
    fun testProportionZTestPowerInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestPower(effectSize = 0.5, n = 64, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestPower(effectSize = 0.5, n = 64, alpha = 1.0)
        }
    }

    // ===== proportionZTestPower: Extreme parameters =====

    @Test
    fun testProportionZTestPowerVeryLargeEffect() {
        // Very large h => power ~ 1.0
        val power = proportionZTestPower(effectSize = 5.0, n = 100)
        assertEquals(1.0, power, tol)
    }

    @Test
    fun testProportionZTestPowerVeryLargeN() {
        val power = proportionZTestPower(effectSize = 0.5, n = 10000)
        assertEquals(1.0, power, tol)
    }

    // ===== proportionZTestPower: Non-finite input =====

    @Test
    fun testProportionZTestPowerNaNEffectSize() {
        val power = proportionZTestPower(effectSize = Double.NaN, n = 64)
        assertTrue(power.isNaN(), "Power should be NaN when effectSize is NaN")
    }

    @Test
    fun testProportionZTestPowerNaNAlpha() {
        // NaN alpha passes outer validation but causes InvalidParameterException
        // inside NormalDistribution.quantile(NaN) since NaN !in 0.0..1.0
        assertFailsWith<InvalidParameterException> {
            proportionZTestPower(effectSize = 0.5, n = 64, alpha = Double.NaN)
        }
    }

    // ===== proportionZTestPower: Property-based =====

    @Test
    fun testProportionZTestPowerRange() {
        val params = listOf(0.1 to 50, 0.3 to 100, 0.5 to 64, 0.8 to 20)
        for ((h, n) in params) {
            val power = proportionZTestPower(effectSize = h, n = n)
            assertTrue(power in 0.0..1.0, "power(h=$h, n=$n) = $power should be in [0, 1]")
        }
    }

    @Test
    fun testProportionZTestPowerMonotonicInN() {
        val nValues = listOf(5, 10, 20, 50, 100, 200)
        var prevPower = 0.0
        for (n in nValues) {
            val power = proportionZTestPower(effectSize = 0.5, n = n)
            assertTrue(power >= prevPower, "power(n=$n) = $power should be >= $prevPower")
            prevPower = power
        }
    }

    @Test
    fun testProportionZTestPowerOneSampleGreaterThanTwoSample() {
        for (h in listOf(0.2, 0.5, 0.8)) {
            for (n in listOf(20, 50, 100)) {
                val powerOne = proportionZTestPower(effectSize = h, n = n, twoSample = false)
                val powerTwo = proportionZTestPower(effectSize = h, n = n, twoSample = true)
                assertTrue(
                    powerOne >= powerTwo,
                    "one-sample power ($powerOne) >= two-sample ($powerTwo) for h=$h, n=$n",
                )
            }
        }
    }

    @Test
    fun testProportionZTestPowerConsistentWithTTest() {
        // proportionZTest with twoSample=true should give same power as tTest with TWO_SAMPLE
        // because the noncentrality factor is the same: sqrt(n/2)
        for (d in listOf(0.2, 0.5, 0.8)) {
            for (n in listOf(20, 50, 100)) {
                val tPower = tTestPower(effectSize = d, n = n, type = TTestType.TWO_SAMPLE)
                val propPower = proportionZTestPower(effectSize = d, n = n, twoSample = true)
                assertEquals(
                    tPower, propPower, tol,
                    "tTest TWO_SAMPLE and proportionZTest twoSample should give same power for d=$d, n=$n",
                )
            }
        }
    }

    // ===== proportionZTestRequiredN: Basic correctness =====

    @Test
    fun testProportionZTestRequiredNKnownValues() {
        // scipy: h=0.5, power=0.8, twoSample=true, TWO_SIDED, alpha=0.05 -> n=63
        assertEquals(63, proportionZTestRequiredN(effectSize = 0.5))
        // scipy: h=0.3, power=0.8, twoSample=true -> n=175
        assertEquals(175, proportionZTestRequiredN(effectSize = 0.3))
    }

    @Test
    fun testProportionZTestRequiredNOneSample() {
        // scipy: h=0.5, power=0.8, twoSample=false -> n=32
        assertEquals(32, proportionZTestRequiredN(effectSize = 0.5, twoSample = false))
    }

    @Test
    fun testProportionZTestRequiredNCustomParams() {
        // scipy: h=0.5, power=0.9, twoSample=true -> n=85
        assertEquals(85, proportionZTestRequiredN(effectSize = 0.5, power = 0.9))
        // scipy: h=0.5, power=0.8, GREATER -> n=50
        assertEquals(
            50,
            proportionZTestRequiredN(effectSize = 0.5, alternative = Alternative.GREATER),
        )
    }

    // ===== proportionZTestRequiredN: Edge cases =====

    @Test
    fun testProportionZTestRequiredNNegativeEffectSize() {
        assertEquals(
            proportionZTestRequiredN(effectSize = 0.5),
            proportionZTestRequiredN(effectSize = -0.5),
        )
    }

    @Test
    fun testProportionZTestRequiredNMinimumResult() {
        val n = proportionZTestRequiredN(effectSize = 100.0)
        assertTrue(n >= 2, "n should be at least 2")
    }

    // ===== proportionZTestRequiredN: Degenerate input =====

    @Test
    fun testProportionZTestRequiredNZeroEffectSize() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = 0.0)
        }
    }

    @Test
    fun testProportionZTestRequiredNInvalidPower() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = 0.5, power = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = 0.5, power = 1.0)
        }
    }

    @Test
    fun testProportionZTestRequiredNInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = 0.5, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = 0.5, alpha = 1.0)
        }
    }

    // ===== proportionZTestRequiredN: Non-finite input =====

    @Test
    fun testProportionZTestRequiredNNaNEffectSize() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestRequiredN(effectSize = Double.NaN)
        }
    }

    @Test
    fun testProportionZTestRequiredNInfinityEffectSize() {
        val n = proportionZTestRequiredN(effectSize = Double.POSITIVE_INFINITY)
        assertEquals(2, n)
    }

    // ===== proportionZTestRequiredN: Extreme parameters =====

    @Test
    fun testProportionZTestRequiredNVerySmallEffectSize() {
        val n = proportionZTestRequiredN(effectSize = 0.01)
        assertTrue(n > 100000, "n for h=0.01 should be very large, got $n")
    }

    // ===== proportionZTestRequiredN: Property-based =====

    @Test
    fun testProportionZTestRequiredNPowerRoundTrip() {
        for (h in listOf(0.2, 0.5, 0.8)) {
            for (targetPower in listOf(0.8, 0.9)) {
                val n = proportionZTestRequiredN(effectSize = h, power = targetPower)
                val achievedPower = proportionZTestPower(effectSize = h, n = n)
                assertTrue(
                    achievedPower >= targetPower - 1e-10,
                    "achievedPower($achievedPower) >= target($targetPower) for h=$h, n=$n",
                )
            }
        }
    }

    @Test
    fun testProportionZTestRequiredNDecreasesWithEffectSize() {
        var prevN = Int.MAX_VALUE
        for (h in listOf(0.1, 0.2, 0.5, 0.8, 1.0)) {
            val n = proportionZTestRequiredN(effectSize = h)
            assertTrue(n <= prevN, "n(h=$h) = $n should be <= $prevN")
            prevN = n
        }
    }

    @Test
    fun testProportionZTestRequiredNConsistentWithTTest() {
        // proportionZTest twoSample=true should match tTest TWO_SAMPLE
        for (d in listOf(0.2, 0.5, 0.8)) {
            val nT = tTestRequiredN(effectSize = d, type = TTestType.TWO_SAMPLE)
            val nP = proportionZTestRequiredN(effectSize = d, twoSample = true)
            assertEquals(nT, nP, "tTest and proportionZTest should give same n for d=$d")
        }
    }

    // ===== proportionZTestMinimumEffect: Basic correctness =====

    @Test
    fun testProportionZTestMinimumEffectKnownValues() {
        // scipy: n=64, power=0.8, twoSample=true, TWO_SIDED, alpha=0.05
        assertEquals(
            0.495254976449918,
            proportionZTestMinimumEffect(n = 64),
            tol,
        )
        // scipy: n=200, power=0.8, twoSample=true, TWO_SIDED
        assertEquals(
            0.280158521811297,
            proportionZTestMinimumEffect(n = 200),
            tol,
        )
    }

    @Test
    fun testProportionZTestMinimumEffectOneSample() {
        // scipy: n=64, power=0.8, twoSample=false
        assertEquals(
            0.350198152264121,
            proportionZTestMinimumEffect(n = 64, twoSample = false),
            tol,
        )
    }

    @Test
    fun testProportionZTestMinimumEffectOneSided() {
        // scipy: n=64, power=0.8, GREATER
        assertEquals(
            0.439550808781667,
            proportionZTestMinimumEffect(n = 64, alternative = Alternative.GREATER),
            tol,
        )
    }

    // ===== proportionZTestMinimumEffect: Degenerate input =====

    @Test
    fun testProportionZTestMinimumEffectInvalidN() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 1)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 0)
        }
    }

    @Test
    fun testProportionZTestMinimumEffectInvalidPower() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 64, power = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 64, power = 1.0)
        }
    }

    @Test
    fun testProportionZTestMinimumEffectInvalidAlpha() {
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 64, alpha = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 64, alpha = 1.0)
        }
    }

    // ===== proportionZTestMinimumEffect: Non-finite input =====

    @Test
    fun testProportionZTestMinimumEffectNaNPower() {
        // NaN power passes outer validation but causes InvalidParameterException
        // inside NormalDistribution.quantile(NaN) since NaN !in 0.0..1.0
        assertFailsWith<InvalidParameterException> {
            proportionZTestMinimumEffect(n = 64, power = Double.NaN)
        }
    }

    // ===== proportionZTestMinimumEffect: Property-based =====

    @Test
    fun testProportionZTestMinimumEffectPowerRoundTrip() {
        for (n in listOf(10, 20, 50, 100, 200)) {
            val h = proportionZTestMinimumEffect(n = n)
            val achievedPower = proportionZTestPower(effectSize = h, n = n)
            assertEquals(0.8, achievedPower, 1e-6, "Round-trip power for n=$n, h=$h")
        }
    }

    @Test
    fun testProportionZTestMinimumEffectDecreasesWithN() {
        var prevEffect = Double.MAX_VALUE
        for (n in listOf(5, 10, 20, 50, 100, 500)) {
            val effect = proportionZTestMinimumEffect(n = n)
            assertTrue(effect < prevEffect, "effect(n=$n) = $effect should be < $prevEffect")
            prevEffect = effect
        }
    }

    @Test
    fun testProportionZTestMinimumEffectPositive() {
        for (n in listOf(10, 50, 100)) {
            val effect = proportionZTestMinimumEffect(n = n)
            assertTrue(effect > 0.0, "Minimum effect for n=$n should be positive, got $effect")
        }
    }

    @Test
    fun testProportionZTestMinimumEffectConsistentWithTTest() {
        // Same noncentrality factor => same minimum effect
        for (n in listOf(20, 50, 100)) {
            val tEffect = tTestMinimumEffect(n = n, type = TTestType.TWO_SAMPLE)
            val pEffect = proportionZTestMinimumEffect(n = n, twoSample = true)
            assertEquals(
                tEffect, pEffect, tol,
                "tTest and proportionZTest minimum effect should match for n=$n",
            )
        }
    }

    // ===== Cross-function consistency (all three t-test functions) =====

    @Test
    fun testTTestTriangleConsistency() {
        // The three functions form a triangle: any two should predict the third
        // Given d=0.5, power=0.8 -> n=63
        // Given d=0.5, n=63 -> power should be >= 0.8
        // Given n=63, power=0.8 -> d should be <= 0.5 (because n=63 achieves exactly 0.8 for d slightly < 0.5)
        val targetD = 0.5
        val targetPower = 0.8
        val n = tTestRequiredN(effectSize = targetD, power = targetPower)

        // Forward: power at n should be >= target
        val achievedPower = tTestPower(effectSize = targetD, n = n)
        assertTrue(
            achievedPower >= targetPower - 1e-10,
            "Power at n=$n should be >= $targetPower",
        )

        // Minimum effect at n should be <= targetD (since n achieves slightly more than target power)
        val minEffect = tTestMinimumEffect(n = n, power = targetPower)
        assertTrue(
            minEffect <= targetD + 1e-10,
            "Minimum effect at n=$n should be <= $targetD, got $minEffect",
        )
    }

    @Test
    fun testProportionZTestTriangleConsistency() {
        val targetH = 0.5
        val targetPower = 0.8
        val n = proportionZTestRequiredN(effectSize = targetH, power = targetPower)

        val achievedPower = proportionZTestPower(effectSize = targetH, n = n)
        assertTrue(
            achievedPower >= targetPower - 1e-10,
            "Power at n=$n should be >= $targetPower",
        )

        val minEffect = proportionZTestMinimumEffect(n = n, power = targetPower)
        assertTrue(
            minEffect <= targetH + 1e-10,
            "Minimum effect at n=$n should be <= $targetH, got $minEffect",
        )
    }
}
