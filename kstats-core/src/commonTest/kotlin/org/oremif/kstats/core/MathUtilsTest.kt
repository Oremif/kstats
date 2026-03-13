package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

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
