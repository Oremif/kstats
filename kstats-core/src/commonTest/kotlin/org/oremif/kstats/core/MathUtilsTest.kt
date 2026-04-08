package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MathUtilsTest {

    private val PRECISE_TOLERANCE = 1e-10
    private val ITERATIVE_TOLERANCE = 1e-6

    // ── lnGamma / gamma ─────────────────────────────────────────────────

    @Test
    fun testLnGammaKnownValues() {
        // Gamma(1) = 1, ln(1) = 0
        assertEquals(0.0, lnGamma(1.0), PRECISE_TOLERANCE)
        // Gamma(2) = 1, ln(1) = 0
        assertEquals(0.0, lnGamma(2.0), PRECISE_TOLERANCE)
        // Gamma(0.5) = sqrt(pi)
        assertEquals(ln(sqrt(PI)), lnGamma(0.5), PRECISE_TOLERANCE)
        // Gamma(5) = 24
        assertEquals(ln(24.0), lnGamma(5.0), PRECISE_TOLERANCE)
        // Gamma(10) = 362880
        assertEquals(ln(362880.0), lnGamma(10.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testGammaKnownValues() {
        assertEquals(1.0, gamma(1.0), PRECISE_TOLERANCE)
        assertEquals(1.0, gamma(2.0), PRECISE_TOLERANCE)
        assertEquals(24.0, gamma(5.0), PRECISE_TOLERANCE)
        assertEquals(sqrt(PI), gamma(0.5), PRECISE_TOLERANCE)
    }

    @Test
    fun testLnGammaRequiresPositive() {
        assertFailsWith<InvalidParameterException> { lnGamma(0.0) }
        assertFailsWith<InvalidParameterException> { lnGamma(-1.0) }
    }

    @Test
    fun testLnGammaInfinity() {
        assertEquals(Double.POSITIVE_INFINITY, lnGamma(Double.POSITIVE_INFINITY))
        assertEquals(Double.POSITIVE_INFINITY, gamma(Double.POSITIVE_INFINITY))
    }

    @Test
    fun testLnGammaNaN() {
        assertTrue(lnGamma(Double.NaN).isNaN())
        assertTrue(gamma(Double.NaN).isNaN())
    }

    @Test
    fun testLnGammaNegativeInfinity() {
        assertFailsWith<InvalidParameterException> { lnGamma(Double.NEGATIVE_INFINITY) }
    }

    // ── Beta function ───────────────────────────────────────────────────

    @Test
    fun testBetaKnownValues() {
        // B(1,1) = 1
        assertEquals(1.0, beta(1.0, 1.0), PRECISE_TOLERANCE)
        // B(0.5, 0.5) = pi
        assertEquals(PI, beta(0.5, 0.5), PRECISE_TOLERANCE)
        // B(a, b) = Gamma(a)*Gamma(b)/Gamma(a+b)
        val a = 2.0
        val b = 3.0
        val expected = gamma(a) * gamma(b) / gamma(a + b)
        assertEquals(expected, beta(a, b), PRECISE_TOLERANCE)
    }

    // ── Regularized beta ────────────────────────────────────────────────

    @Test
    fun testRegularizedBetaBoundary() {
        assertEquals(0.0, regularizedBeta(0.0, 2.0, 3.0), PRECISE_TOLERANCE)
        assertEquals(1.0, regularizedBeta(1.0, 2.0, 3.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testRegularizedBetaKnownValues() {
        // I_{0.5}(1, 1) = 0.5
        assertEquals(0.5, regularizedBeta(0.5, 1.0, 1.0), PRECISE_TOLERANCE)
        // I_{0.5}(2, 3) from R: pbeta(0.5, 2, 3) = 0.6875
        assertEquals(0.6875, regularizedBeta(0.5, 2.0, 3.0), ITERATIVE_TOLERANCE)
    }

    @Test
    fun testRegularizedBetaLargeParametersRegression() {
        // Regression: regularizedBeta with a ~ b ~ 145K previously threw ConvergenceException
        // because the fixed 200-iteration limit was insufficient. The fix adds dynamic iteration
        // limits via betaMaxIterations(a, b) = max(200, floor(10 * sqrt(max(a, b)))).
        // scipy: special.betainc(145274.0, 145311.0, 0.4999) = 0.484375989852313
        val result = regularizedBeta(0.4999, 145274.0, 145311.0)
        assertTrue(result.isFinite(), "regularizedBeta should not throw for a=145274, b=145311")
        assertEquals(0.484375989852313, result, 1e-6, "betainc(145274, 145311, 0.4999)")

        // scipy: special.betainc(145274.0, 145311.0, 0.5) = 0.527361185858317
        val result2 = regularizedBeta(0.5, 145274.0, 145311.0)
        assertTrue(result2.isFinite(), "regularizedBeta(0.5, 145274, 145311) should be finite")
        assertEquals(0.527361185858317, result2, 1e-6, "betainc(145274, 145311, 0.5)")
    }

    @Test
    fun testRegularizedBetaLargeParametersKnownValues() {
        // scipy: special.betainc(1000, 1000, 0.5) = 0.499999999999999
        assertEquals(0.5, regularizedBeta(0.5, 1000.0, 1000.0), 1e-6)
        // scipy: special.betainc(1000, 1000, 0.49) = 0.185552659431512
        assertEquals(0.185552659431512, regularizedBeta(0.49, 1000.0, 1000.0), 1e-6)
        // scipy: special.betainc(1000, 1000, 0.51) = 0.814447340568488
        assertEquals(0.814447340568488, regularizedBeta(0.51, 1000.0, 1000.0), 1e-6)
        // scipy: special.betainc(10000, 10000, 0.5) = 0.5
        assertEquals(0.5, regularizedBeta(0.5, 10000.0, 10000.0), 1e-6)
        // scipy: special.betainc(10000, 10000, 0.499) = 0.388649952142536
        assertEquals(0.388649952142536, regularizedBeta(0.499, 10000.0, 10000.0), 1e-6)
    }

    @Test
    fun testRegularizedBetaVeryLargeParameters() {
        // Numerical stability with very large shape parameters
        // scipy: special.betainc(50000, 50000, 0.5) = 0.5
        assertEquals(0.5, regularizedBeta(0.5, 50000.0, 50000.0), 1e-6)
        // scipy: special.betainc(50000, 50000, 0.4999) = 0.474785548275951
        assertEquals(0.474785548275951, regularizedBeta(0.4999, 50000.0, 50000.0), 1e-6)
        // scipy: special.betainc(100000, 100000, 0.5) = 0.5
        assertEquals(0.5, regularizedBeta(0.5, 100000.0, 100000.0), 1e-6)
        // scipy: special.betainc(100000, 100000, 0.4999) = 0.464365081352024
        assertEquals(0.464365081352024, regularizedBeta(0.4999, 100000.0, 100000.0), 1e-6)
        // scipy: special.betainc(200000, 200000, 0.5) = 0.5
        assertEquals(0.5, regularizedBeta(0.5, 200000.0, 200000.0), 1e-6)
        // scipy: special.betainc(200000, 200000, 0.4999) = 0.44967162506792
        assertEquals(0.44967162506792, regularizedBeta(0.4999, 200000.0, 200000.0), 1e-6)
    }

    @Test
    fun testRegularizedBetaLargeParamsSymmetryProperty() {
        // Property: I_x(a, b) + I_{1-x}(b, a) = 1 for all valid x, a, b
        val cases = listOf(
            Triple(0.4999, 145274.0, 145311.0),
            Triple(0.499, 10000.0, 10000.0),
            Triple(0.4999, 50000.0, 50000.0),
            Triple(0.5, 1000.0, 1000.0),
        )
        for ((x, a, b) in cases) {
            val left = regularizedBeta(x, a, b)
            val right = regularizedBeta(1.0 - x, b, a)
            assertEquals(
                1.0, left + right, 1e-6,
                "I_$x($a,$b) + I_${1.0 - x}($b,$a) should equal 1"
            )
        }
    }

    @Test
    fun testRegularizedBetaLargeParamsMonotonicity() {
        // Property: regularizedBeta(x, a, b) is monotonically increasing in x
        val a = 10000.0
        val b = 10000.0
        val xs = listOf(0.48, 0.49, 0.495, 0.499, 0.5, 0.501, 0.505, 0.51, 0.52)
        var prev = regularizedBeta(xs[0], a, b)
        for (i in 1 until xs.size) {
            val curr = regularizedBeta(xs[i], a, b)
            assertTrue(
                curr >= prev,
                "regularizedBeta should be monotonic: I_${xs[i]}($a,$b) = $curr >= I_${xs[i - 1]}($a,$b) = $prev"
            )
            prev = curr
        }
    }

    @Test
    fun testRegularizedBetaLargeParamsRange() {
        // Property: result must be in [0, 1] for any valid inputs including large params
        val cases = listOf(
            Triple(0.4999, 145274.0, 145311.0),
            Triple(0.5, 145274.0, 145311.0),
            Triple(0.5, 100000.0, 100000.0),
            Triple(0.4999, 200000.0, 200000.0),
        )
        for ((x, a, b) in cases) {
            val result = regularizedBeta(x, a, b)
            assertTrue(result in 0.0..1.0, "regularizedBeta($x, $a, $b) = $result should be in [0, 1]")
        }
    }

    @Test
    fun testRegularizedBetaNaNPropagation() {
        assertTrue(regularizedBeta(Double.NaN, 2.0, 3.0).isNaN(), "NaN x")
        assertTrue(regularizedBeta(0.5, Double.NaN, 3.0).isNaN(), "NaN a")
        assertTrue(regularizedBeta(0.5, 2.0, Double.NaN).isNaN(), "NaN b")
    }

    @Test
    fun testRegularizedBetaInvalidParameters() {
        assertFailsWith<InvalidParameterException> {
            regularizedBeta(0.5, -1.0, 3.0)
        }
        assertFailsWith<InvalidParameterException> {
            regularizedBeta(0.5, 2.0, 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            regularizedBeta(0.5, 0.0, 3.0)
        }
    }

    // ── Regularized gamma ───────────────────────────────────────────────

    @Test
    fun testRegularizedGammaPKnownValues() {
        // P(1, 1) = 1 - e^(-1)
        assertEquals(1.0 - exp(-1.0), regularizedGammaP(1.0, 1.0), ITERATIVE_TOLERANCE)
        // P(1, 0) = 0
        assertEquals(0.0, regularizedGammaP(1.0, 0.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testRegularizedGammaPQComplementary() {
        val a = 3.0
        val x = 2.5
        val p = regularizedGammaP(a, x)
        val q = regularizedGammaQ(a, x)
        assertEquals(1.0, p + q, PRECISE_TOLERANCE)
    }

    @Test
    fun testRegularizedGammaInfinity() {
        // P(a, +∞) = 1.0 for any a > 0
        assertEquals(1.0, regularizedGammaP(1.0, Double.POSITIVE_INFINITY), PRECISE_TOLERANCE)
        assertEquals(1.0, regularizedGammaP(5.0, Double.POSITIVE_INFINITY), PRECISE_TOLERANCE)
        // Q(a, +∞) = 0.0 for any a > 0
        assertEquals(0.0, regularizedGammaQ(1.0, Double.POSITIVE_INFINITY), PRECISE_TOLERANCE)
        assertEquals(0.0, regularizedGammaQ(5.0, Double.POSITIVE_INFINITY), PRECISE_TOLERANCE)
    }

    @Test
    fun testRegularizedGammaNaNPropagation() {
        assertTrue(regularizedGammaP(Double.NaN, 1.0).isNaN())
        assertTrue(regularizedGammaP(1.0, Double.NaN).isNaN())
        assertTrue(regularizedGammaQ(Double.NaN, 1.0).isNaN())
        assertTrue(regularizedGammaQ(1.0, Double.NaN).isNaN())
    }

    @Test
    fun testRegularizedGammaInvalidA() {
        assertFailsWith<InvalidParameterException> { regularizedGammaP(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { regularizedGammaQ(-1.0, 1.0) }
    }

    @Test
    fun testRegularizedGammaLargeParameters() {
        // Large a values that previously exceeded the 200-iteration limit (Poisson lambda=500+)
        // scipy.special.gammaincc(501, 500) ≈ 0.5178
        val q501 = regularizedGammaQ(501.0, 500.0)
        assertTrue(q501 in 0.4..0.6, "Q(501, 500) should be near 0.5, got $q501")

        // scipy.special.gammaincc(1001, 1000) ≈ 0.5063
        val q1001 = regularizedGammaQ(1001.0, 1000.0)
        assertTrue(q1001 in 0.4..0.6, "Q(1001, 1000) should be near 0.5, got $q1001")
    }

    // ── Error function ──────────────────────────────────────────────────

    @Test
    fun testErfKnownValues() {
        assertEquals(0.0, erf(0.0), PRECISE_TOLERANCE)
        // erf(1) ≈ 0.8427007929
        assertEquals(0.8427007929, erf(1.0), ITERATIVE_TOLERANCE)
        // erf(-x) = -erf(x)
        assertEquals(-erf(1.5), erf(-1.5), PRECISE_TOLERANCE)
    }

    @Test
    fun testErfcComplement() {
        assertEquals(1.0, erf(2.0) + erfc(2.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testErfInvKnownValues() {
        assertEquals(0.0, erfInv(0.0), PRECISE_TOLERANCE)
        // erfInv(erf(x)) ≈ x for various x values
        for (x in listOf(0.1, 0.3, 0.5, 0.75, 1.0, 1.5)) {
            val erfX = erf(x)
            assertEquals(
                x,
                erfInv(erfX),
                ITERATIVE_TOLERANCE,
                "erfInv(erf($x)) should ≈ $x, erf($x)=$erfX, erfInv=${erfInv(erfX)}"
            )
        }
    }

    @Test
    fun testErfInvNaN() {
        assertTrue(erfInv(Double.NaN).isNaN())
    }

    @Test
    fun testErfInvExtremeValues() {
        // scipy.special.erfinv reference values for inputs near ±1
        // erfInv(0.999) = 2.3267537655135246
        assertEquals(2.3267537655135246, erfInv(0.999), 1e-6)
        // erfInv(0.9999) = 2.7510639057120607
        assertEquals(2.7510639057120607, erfInv(0.9999), 1e-5)
        // erfInv(0.99999) = 3.1234132743415708
        assertEquals(3.1234132743415708, erfInv(0.99999), 1e-4)
        // Symmetry
        assertEquals(-erfInv(0.999), erfInv(-0.999), 1e-6)
    }

    // ── erfcInv ──────────────────────────────────────────────────────────

    @Test
    fun testErfcInvBasicCorrectness() {
        // scipy.special.erfcinv reference values
        assertEquals(0.0, erfcInv(1.0), PRECISE_TOLERANCE)
        assertEquals(1.16308715367667, erfcInv(0.1), ITERATIVE_TOLERANCE)
        assertEquals(0.476936276204470, erfcInv(0.5), ITERATIVE_TOLERANCE)
        assertEquals(0.225312055012178, erfcInv(0.75), ITERATIVE_TOLERANCE)
        assertEquals(-0.225312055012178, erfcInv(1.25), ITERATIVE_TOLERANCE)
        assertEquals(-0.476936276204470, erfcInv(1.5), ITERATIVE_TOLERANCE)
        assertEquals(-0.813419847597618, erfcInv(1.75), ITERATIVE_TOLERANCE)
        assertEquals(-1.16308715367667, erfcInv(1.9), ITERATIVE_TOLERANCE)
    }

    @Test
    fun testErfcInvEdgeCases() {
        // Near domain boundaries
        assertTrue(erfcInv(1e-10) > 0.0)
        assertTrue(erfcInv(2.0 - 1e-10) < 0.0)
    }

    @Test
    fun testErfcInvBoundaryThrows() {
        assertFailsWith<InvalidParameterException> { erfcInv(0.0) }
        assertFailsWith<InvalidParameterException> { erfcInv(2.0) }
        assertFailsWith<InvalidParameterException> { erfcInv(-0.5) }
        assertFailsWith<InvalidParameterException> { erfcInv(2.5) }
    }

    @Test
    fun testErfcInvExtremeParameters() {
        // Very small y (close to 0) → large positive result
        val smallResult = erfcInv(1e-15)
        assertTrue(smallResult > 5.0, "erfcInv(1e-15) should be large positive, got $smallResult")
        // Very close to 2 → large negative result
        val largeResult = erfcInv(2.0 - 1e-15)
        assertTrue(largeResult < -5.0, "erfcInv(2-1e-15) should be large negative, got $largeResult")
    }

    @Test
    fun testErfcInvNonFiniteInput() {
        assertTrue(erfcInv(Double.NaN).isNaN())
    }

    @Test
    fun testErfcInvRoundTrip() {
        // erfcInv(erfc(x)) ≈ x
        for (x in listOf(0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)) {
            assertEquals(x, erfcInv(erfc(x)), ITERATIVE_TOLERANCE, "erfcInv(erfc($x)) should ≈ $x")
        }
        // erfc(erfcInv(y)) ≈ y
        for (y in listOf(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9)) {
            assertEquals(y, erfc(erfcInv(y)), ITERATIVE_TOLERANCE, "erfc(erfcInv($y)) should ≈ $y")
        }
    }

    @Test
    fun testErfcInvRelationshipToErfInv() {
        // erfcInv(y) == erfInv(1 - y) for y in (0, 2)
        for (y in listOf(0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.9)) {
            assertEquals(
                erfInv(1.0 - y),
                erfcInv(y),
                PRECISE_TOLERANCE,
                "erfcInv($y) should equal erfInv(${1.0 - y})"
            )
        }
    }

    // ── Digamma ──────────────────────────────────────────────────────────

    @Test
    fun testDigammaBasicCorrectness() {
        // scipy.special.digamma reference values (15+ digits)
        // digamma(1) = -EULER_MASCHERONI
        assertEquals(-0.5772156649015329, digamma(1.0), PRECISE_TOLERANCE)
        // digamma(2) = 1 - gamma = 0.42278433509846714
        assertEquals(0.42278433509846714, digamma(2.0), PRECISE_TOLERANCE)
        // digamma(3) = 1.5 - gamma = 0.9227843350984672
        assertEquals(0.9227843350984672, digamma(3.0), PRECISE_TOLERANCE)
        // digamma(5)
        assertEquals(1.5061176684318004, digamma(5.0), PRECISE_TOLERANCE)
        // digamma(10)
        assertEquals(2.2517525890667213, digamma(10.0), PRECISE_TOLERANCE)
        // digamma(100)
        assertEquals(4.600161852738088, digamma(100.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testDigammaEdgeCases() {
        // digamma(1) = -gamma (Euler-Mascheroni constant)
        assertEquals(-EULER_MASCHERONI, digamma(1.0), PRECISE_TOLERANCE)
        // digamma(0.5) = -gamma - 2*ln(2)
        assertEquals(-EULER_MASCHERONI - 2.0 * ln(2.0), digamma(0.5), PRECISE_TOLERANCE)
        // digamma(0.25) from scipy: -4.227453533...
        assertEquals(-4.227453533340973, digamma(0.25), PRECISE_TOLERANCE)
    }

    @Test
    fun testDigammaNegativeNonInteger() {
        // digamma(-0.5) from scipy: 0.03648997397...
        assertEquals(0.03648997397857652, digamma(-0.5), PRECISE_TOLERANCE)
        // digamma(-1.5) from scipy: 0.70315664064...
        assertEquals(0.7031566406452432, digamma(-1.5), PRECISE_TOLERANCE)
        // digamma(-2.5) from scipy: 1.10315664064...
        assertEquals(1.1031566406452432, digamma(-2.5), PRECISE_TOLERANCE)
    }

    @Test
    fun testDigammaDegenerateInput() {
        // Poles at 0, -1, -2
        assertFailsWith<InvalidParameterException> { digamma(0.0) }
        assertFailsWith<InvalidParameterException> { digamma(-1.0) }
        assertFailsWith<InvalidParameterException> { digamma(-2.0) }
        assertFailsWith<InvalidParameterException> { digamma(-10.0) }
    }

    @Test
    fun testDigammaExtremeParameters() {
        // Very large x: digamma(x) ~ ln(x) for large x
        val large = 1e10
        assertEquals(ln(large), digamma(large), 1e-4)
        // Very small positive x: digamma(x) ~ -1/x for small x
        val small = 1e-10
        assertEquals(-1.0 / small, digamma(small), 1e2) // dominated by -1/x term
    }

    @Test
    fun testDigammaNonFiniteInput() {
        assertTrue(digamma(Double.NaN).isNaN())
        assertEquals(Double.POSITIVE_INFINITY, digamma(Double.POSITIVE_INFINITY))
        assertTrue(digamma(Double.NEGATIVE_INFINITY).isNaN())
    }

    @Test
    fun testDigammaRecurrenceProperty() {
        // psi(x+1) = psi(x) + 1/x
        for (x in listOf(0.5, 1.0, 1.5, 2.0, 3.7, 5.3, 10.0)) {
            assertEquals(
                digamma(x) + 1.0 / x,
                digamma(x + 1.0),
                PRECISE_TOLERANCE,
                "Recurrence psi(${x}+1) = psi($x) + 1/$x"
            )
        }
    }

    @Test
    fun testDigammaReflectionProperty() {
        // psi(1-x) - psi(x) = pi/tan(pi*x) for non-integer x
        for (x in listOf(0.25, 0.3, 0.5, 0.75)) {
            val expected = PI / tan(PI * x)
            assertEquals(
                expected,
                digamma(1.0 - x) - digamma(x),
                PRECISE_TOLERANCE,
                "Reflection psi(1-$x) - psi($x) = pi/tan(pi*$x)"
            )
        }
    }

    // ── Trigamma ─────────────────────────────────────────────────────────

    @Test
    fun testTrigammaBasicCorrectness() {
        // scipy.special.polygamma(1, x) reference values
        // trigamma(1) = pi^2/6
        assertEquals(1.6449340668482264, trigamma(1.0), PRECISE_TOLERANCE)
        // trigamma(2)
        assertEquals(0.6449340668482264, trigamma(2.0), PRECISE_TOLERANCE)
        // trigamma(5)
        assertEquals(0.22132295573711532, trigamma(5.0), PRECISE_TOLERANCE)
        // trigamma(10)
        assertEquals(0.10516633568168575, trigamma(10.0), PRECISE_TOLERANCE)
        // trigamma(100)
        assertEquals(0.010050166663333571, trigamma(100.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testTrigammaEdgeCases() {
        // trigamma(1) = pi^2/6
        assertEquals(PI * PI / 6.0, trigamma(1.0), PRECISE_TOLERANCE)
        // trigamma(0.5) = pi^2/2
        assertEquals(PI * PI / 2.0, trigamma(0.5), PRECISE_TOLERANCE)
        // trigamma(0.25) from scipy: 17.19732...
        assertEquals(17.19732915450711, trigamma(0.25), PRECISE_TOLERANCE)
    }

    @Test
    fun testTrigammaNegativeNonInteger() {
        // trigamma(-0.5) from scipy: 8.934802200...
        assertEquals(8.934802200544679, trigamma(-0.5), PRECISE_TOLERANCE)
        // trigamma(-1.5) from scipy: 9.379246644989124
        assertEquals(9.379246644989124, trigamma(-1.5), PRECISE_TOLERANCE)
    }

    @Test
    fun testTrigammaDegenerateInput() {
        assertFailsWith<InvalidParameterException> { trigamma(0.0) }
        assertFailsWith<InvalidParameterException> { trigamma(-1.0) }
        assertFailsWith<InvalidParameterException> { trigamma(-2.0) }
        assertFailsWith<InvalidParameterException> { trigamma(-10.0) }
    }

    @Test
    fun testTrigammaExtremeParameters() {
        // Very large x: trigamma(x) ~ 1/x for large x
        val large = 1e10
        assertEquals(1.0 / large, trigamma(large), 1e-15)
        // Very small positive x: trigamma(x) ~ 1/x^2 for small x
        val small = 1e-10
        assertEquals(1.0 / (small * small), trigamma(small), 1e12) // dominated by 1/x^2
    }

    @Test
    fun testTrigammaNonFiniteInput() {
        assertTrue(trigamma(Double.NaN).isNaN())
        assertEquals(0.0, trigamma(Double.POSITIVE_INFINITY))
        assertTrue(trigamma(Double.NEGATIVE_INFINITY).isNaN())
    }

    @Test
    fun testTrigammaRecurrenceProperty() {
        // psi'(x+1) = psi'(x) - 1/x^2
        for (x in listOf(0.5, 1.0, 1.5, 2.0, 3.7, 5.3, 10.0)) {
            assertEquals(
                trigamma(x) - 1.0 / (x * x),
                trigamma(x + 1.0),
                PRECISE_TOLERANCE,
                "Recurrence psi'(${x}+1) = psi'($x) - 1/$x^2"
            )
        }
    }

    @Test
    fun testTrigammaReflectionProperty() {
        // psi'(x) + psi'(1-x) = pi^2/sin^2(pi*x) for non-integer x
        for (x in listOf(0.25, 0.3, 0.5, 0.75)) {
            val sinPiX = sin(PI * x)
            val expected = (PI * PI) / (sinPiX * sinPiX)
            assertEquals(
                expected,
                trigamma(x) + trigamma(1.0 - x),
                PRECISE_TOLERANCE,
                "Reflection psi'($x) + psi'(1-$x) = pi^2/sin^2(pi*$x)"
            )
        }
    }

    // ── Combinatorics ───────────────────────────────────────────────────

    @Test
    fun testLnFactorial() {
        assertEquals(0.0, lnFactorial(0), PRECISE_TOLERANCE)
        assertEquals(0.0, lnFactorial(1), PRECISE_TOLERANCE)
        assertEquals(ln(2.0), lnFactorial(2), PRECISE_TOLERANCE)
        assertEquals(ln(120.0), lnFactorial(5), PRECISE_TOLERANCE)
    }

    @Test
    fun testLnCombination() {
        assertEquals(0.0, lnCombination(5, 0), PRECISE_TOLERANCE)
        assertEquals(0.0, lnCombination(5, 5), PRECISE_TOLERANCE)
        // C(10, 3) = 120
        assertEquals(ln(120.0), lnCombination(10, 3), PRECISE_TOLERANCE)
    }

    @Test
    fun testLnCombinationInvalid() {
        assertFailsWith<InvalidParameterException> { lnCombination(3, 5) }
        assertFailsWith<InvalidParameterException> { lnCombination(-1, 0) }
    }

    // ── lnPermutation ────────────────────────────────────────────────────

    @Test
    fun testLnPermutationBasicCorrectness() {
        // P(n,k) = n! / (n-k)!, lnPermutation = ln(P(n,k))
        assertEquals(0.0, lnPermutation(5, 0), PRECISE_TOLERANCE)
        // P(5,1) = 5, ln(5)
        assertEquals(1.6094379124341, lnPermutation(5, 1), PRECISE_TOLERANCE)
        // P(5,2) = 20, ln(20)
        assertEquals(2.99573227355399, lnPermutation(5, 2), PRECISE_TOLERANCE)
        // P(5,3) = 60, ln(60)
        assertEquals(4.0943445622221, lnPermutation(5, 3), PRECISE_TOLERANCE)
        // P(5,5) = 120, ln(120)
        assertEquals(4.78749174278205, lnPermutation(5, 5), PRECISE_TOLERANCE)
        // P(10,3) = 720
        assertEquals(6.5792512120101, lnPermutation(10, 3), PRECISE_TOLERANCE)
        // P(10,5) = 30240
        assertEquals(10.3169208302935, lnPermutation(10, 5), PRECISE_TOLERANCE)
    }

    @Test
    fun testLnPermutationEdgeCases() {
        // P(0,0) = 1, ln(1) = 0
        assertEquals(0.0, lnPermutation(0, 0), PRECISE_TOLERANCE)
        // P(1,0) = 1
        assertEquals(0.0, lnPermutation(1, 0), PRECISE_TOLERANCE)
        // P(1,1) = 1
        assertEquals(0.0, lnPermutation(1, 1), PRECISE_TOLERANCE)
        // P(n,n) = n!
        assertEquals(lnFactorial(10), lnPermutation(10, 10), PRECISE_TOLERANCE)
    }

    @Test
    fun testLnPermutationDegenerate() {
        assertFailsWith<InvalidParameterException> { lnPermutation(3, 5) }
        assertFailsWith<InvalidParameterException> { lnPermutation(-1, 0) }
        assertFailsWith<InvalidParameterException> { lnPermutation(5, -1) }
    }

    @Test
    fun testLnPermutationExtremeParameters() {
        // Large n — should not overflow thanks to log-space computation
        assertEquals(215.26160860379, lnPermutation(100, 50), 1e-6)
        val result = lnPermutation(1000, 500)
        assertTrue(result.isFinite(), "lnPermutation(1000, 500) should be finite")
    }

    @Test
    fun testLnPermutationRelationshipToCombination() {
        // P(n,k) = C(n,k) * k!, so lnPermutation(n,k) = lnCombination(n,k) + lnFactorial(k)
        for ((n, k) in listOf(10 to 3, 10 to 5, 20 to 10, 15 to 7)) {
            assertEquals(
                lnCombination(n, k) + lnFactorial(k),
                lnPermutation(n, k),
                PRECISE_TOLERANCE,
                "lnPermutation($n, $k) should equal lnCombination + lnFactorial($k)"
            )
        }
    }

    // ── GCD and LCM ─────────────────────────────────────────────────────

    @Test
    fun testGcdBasicCorrectness() {
        assertEquals(4L, gcd(12, 8))
        assertEquals(1L, gcd(7, 13))
        assertEquals(25L, gcd(100, 75))
        assertEquals(6L, gcd(48, 18))
        assertEquals(1L, gcd(1, 1000000))
    }

    @Test
    fun testGcdEdgeCases() {
        // gcd(a, 0) = |a|, gcd(0, b) = |b|
        assertEquals(5L, gcd(0, 5))
        assertEquals(5L, gcd(5, 0))
        assertEquals(0L, gcd(0, 0))
        // gcd(a, a) = |a|
        assertEquals(7L, gcd(7, 7))
        // Commutativity
        assertEquals(gcd(12, 8), gcd(8, 12))
    }

    @Test
    fun testGcdNegativeInputs() {
        assertEquals(4L, gcd(-12, 8))
        assertEquals(4L, gcd(12, -8))
        assertEquals(4L, gcd(-12, -8))
    }

    @Test
    fun testGcdExtremeParameters() {
        // Large values
        assertEquals(1L, gcd(Long.MAX_VALUE, Long.MAX_VALUE - 1))
        assertEquals(Long.MAX_VALUE, gcd(Long.MAX_VALUE, Long.MAX_VALUE))
    }

    @Test
    fun testLcmBasicCorrectness() {
        assertEquals(24L, lcm(12, 8))
        assertEquals(91L, lcm(7, 13))
        assertEquals(300L, lcm(100, 75))
        assertEquals(144L, lcm(48, 18))
        assertEquals(1000000L, lcm(1, 1000000))
    }

    @Test
    fun testLcmEdgeCases() {
        // lcm(a, 0) = 0
        assertEquals(0L, lcm(0, 5))
        assertEquals(0L, lcm(5, 0))
        assertEquals(0L, lcm(0, 0))
        // lcm(a, a) = |a|
        assertEquals(7L, lcm(7, 7))
        // Commutativity
        assertEquals(lcm(12, 8), lcm(8, 12))
    }

    @Test
    fun testLcmNegativeInputs() {
        assertEquals(24L, lcm(-12, 8))
        assertEquals(24L, lcm(12, -8))
        assertEquals(24L, lcm(-12, -8))
    }

    @Test
    fun testGcdLcmRelationship() {
        // |a * b| = gcd(a, b) * lcm(a, b) for non-zero a, b
        for ((a, b) in listOf(12L to 8L, 7L to 13L, 100L to 75L, 48L to 18L)) {
            assertEquals(a * b, gcd(a, b) * lcm(a, b), "gcd($a,$b) * lcm($a,$b) should equal $a * $b")
        }
    }

    // ── Generalized harmonic numbers ────────────────────────────────────

    @Test
    fun testGeneralizedHarmonicKnownValues() {
        // H(1, s) = 1 for any s
        assertEquals(1.0, generalizedHarmonic(1, 1.0), PRECISE_TOLERANCE)
        assertEquals(1.0, generalizedHarmonic(1, 2.5), PRECISE_TOLERANCE)
        // H(0, s) = 0 (empty sum)
        assertEquals(0.0, generalizedHarmonic(0, 1.0), PRECISE_TOLERANCE)
        // H(n, 0) = n
        assertEquals(5.0, generalizedHarmonic(5, 0.0), PRECISE_TOLERANCE)
        assertEquals(10.0, generalizedHarmonic(10, 0.0), PRECISE_TOLERANCE)
        // H(10, 1) = 10th harmonic number
        assertEquals(2.928968253968254e+00, generalizedHarmonic(10, 1.0), PRECISE_TOLERANCE)
        // H(100, 1)
        assertEquals(5.187377517639621e+00, generalizedHarmonic(100, 1.0), PRECISE_TOLERANCE)
        // H(10, 2)
        assertEquals(1.549767731166541e+00, generalizedHarmonic(10, 2.0), PRECISE_TOLERANCE)
        // H(100, 2) ≈ π²/6
        assertEquals(1.634983900184893e+00, generalizedHarmonic(100, 2.0), PRECISE_TOLERANCE)
        // H(5, 3)
        assertEquals(1.185662037037037e+00, generalizedHarmonic(5, 3.0), PRECISE_TOLERANCE)
    }

    @Test
    fun testGeneralizedHarmonicInvalid() {
        assertFailsWith<InvalidParameterException> { generalizedHarmonic(-1, 1.0) }
    }

    // ── Compensated sum (Neumaier) ───────────────────────────────────────

    @Test
    fun testCompensatedSumPathological() {
        // Naive sum gives 0.0 due to catastrophic cancellation; compensated gives 1.0
        val data = doubleArrayOf(1e16, 1.0, -1e16)
        assertEquals(1.0, data.compensatedSum(), 0.0)
    }

    @Test
    fun testCompensatedSumLargeOffsetSmallDiffs() {
        // 1000 values: 1e15 + 0, 1e15 + 1, ..., 1e15 + 999
        val data = DoubleArray(1000) { i -> 1e15 + i.toDouble() }
        // Exact sum = 1000 * 1e15 + (0 + 1 + ... + 999) = 1e18 + 499500
        val expected = 1e18 + 499500.0
        assertEquals(expected, data.compensatedSum(), 0.0)
    }

    @Test
    fun testCompensatedSumEmpty() {
        assertEquals(0.0, doubleArrayOf().compensatedSum(), 0.0)
    }

    @Test
    fun testCompensatedSumSingleElement() {
        assertEquals(42.0, doubleArrayOf(42.0).compensatedSum(), 0.0)
    }

    @Test
    fun testCompensatedSumNormalCase() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        assertEquals(15.0, data.compensatedSum(), 0.0)
    }
}
