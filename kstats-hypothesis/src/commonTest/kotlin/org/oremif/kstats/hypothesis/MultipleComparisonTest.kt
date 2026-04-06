package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class MultipleComparisonTest {

    private val tol = 1e-10

    // ===== bonferroniCorrection: Basic correctness =====

    @Test
    fun testBonferroniKnownValues() {
        // statsmodels: multipletests([0.01, 0.04, 0.03, 0.005], method='bonferroni')
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val result = bonferroniCorrection(pValues)
        assertEquals(0.04, result[0], tol, "bonferroni p[0]")
        assertEquals(0.16, result[1], tol, "bonferroni p[1]")
        assertEquals(0.12, result[2], tol, "bonferroni p[2]")
        assertEquals(0.02, result[3], tol, "bonferroni p[3]")
    }

    @Test
    fun testBonferroniKnownValuesFiveElements() {
        // statsmodels: multipletests([0.001, 0.01, 0.05, 0.1, 0.5], method='bonferroni')
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = bonferroniCorrection(pValues)
        assertEquals(0.005, result[0], tol, "bonferroni p[0]")
        assertEquals(0.05, result[1], tol, "bonferroni p[1]")
        assertEquals(0.25, result[2], tol, "bonferroni p[2]")
        assertEquals(0.5, result[3], tol, "bonferroni p[3]")
        assertEquals(1.0, result[4], tol, "bonferroni p[4] clamped to 1.0")
    }

    @Test
    fun testBonferroniClampingToOne() {
        // statsmodels: multipletests([0.3, 0.4, 0.5], method='bonferroni')
        val pValues = doubleArrayOf(0.3, 0.4, 0.5)
        val result = bonferroniCorrection(pValues)
        assertEquals(0.9, result[0], tol, "bonferroni p[0]")
        assertEquals(1.0, result[1], tol, "bonferroni p[1] clamped")
        assertEquals(1.0, result[2], tol, "bonferroni p[2] clamped")
    }

    // ===== holmBonferroniCorrection: Basic correctness =====

    @Test
    fun testHolmKnownValues() {
        // statsmodels: multipletests([0.01, 0.04, 0.03, 0.005], method='holm')
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.03, result[0], tol, "holm p[0]")
        assertEquals(0.06, result[1], tol, "holm p[1]")
        assertEquals(0.06, result[2], tol, "holm p[2]")
        assertEquals(0.02, result[3], tol, "holm p[3]")
    }

    @Test
    fun testHolmKnownValuesFiveElements() {
        // statsmodels: multipletests([0.001, 0.01, 0.05, 0.1, 0.5], method='holm')
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.005, result[0], tol, "holm p[0]")
        assertEquals(0.04, result[1], tol, "holm p[1]")
        assertEquals(0.15, result[2], tol, "holm p[2]")
        assertEquals(0.2, result[3], tol, "holm p[3]")
        assertEquals(0.5, result[4], tol, "holm p[4]")
    }

    @Test
    fun testHolmClampingAndMonotonicity() {
        // statsmodels: multipletests([0.3, 0.4, 0.5], method='holm')
        val pValues = doubleArrayOf(0.3, 0.4, 0.5)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.9, result[0], tol, "holm p[0]")
        assertEquals(0.9, result[1], tol, "holm p[1] monotonicity-enforced")
        assertEquals(0.9, result[2], tol, "holm p[2] monotonicity-enforced")
    }

    // ===== benjaminiHochbergCorrection: Basic correctness =====

    @Test
    fun testBenjaminiHochbergKnownValues() {
        // statsmodels: multipletests([0.01, 0.04, 0.03, 0.005], method='fdr_bh')
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.02, result[0], tol, "bh p[0]")
        assertEquals(0.04, result[1], tol, "bh p[1]")
        assertEquals(0.04, result[2], tol, "bh p[2]")
        assertEquals(0.02, result[3], tol, "bh p[3]")
    }

    @Test
    fun testBenjaminiHochbergKnownValuesFiveElements() {
        // statsmodels: multipletests([0.001, 0.01, 0.05, 0.1, 0.5], method='fdr_bh')
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.005, result[0], tol, "bh p[0]")
        assertEquals(0.025, result[1], tol, "bh p[1]")
        assertEquals(0.0833333333333333, result[2], tol, "bh p[2]")
        assertEquals(0.125, result[3], tol, "bh p[3]")
        assertEquals(0.5, result[4], tol, "bh p[4]")
    }

    @Test
    fun testBenjaminiHochbergClampingValues() {
        // statsmodels: multipletests([0.3, 0.4, 0.5], method='fdr_bh')
        val pValues = doubleArrayOf(0.3, 0.4, 0.5)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.5, result[0], tol, "bh p[0]")
        assertEquals(0.5, result[1], tol, "bh p[1]")
        assertEquals(0.5, result[2], tol, "bh p[2]")
    }

    // ===== Edge cases: single element =====

    @Test
    fun testBonferroniSingleElement() {
        // statsmodels: multipletests([0.03], method='bonferroni') => [0.03]
        val result = bonferroniCorrection(doubleArrayOf(0.03))
        assertEquals(0.03, result[0], tol)
        assertEquals(1, result.size)
    }

    @Test
    fun testHolmSingleElement() {
        // statsmodels: multipletests([0.03], method='holm') => [0.03]
        val result = holmBonferroniCorrection(doubleArrayOf(0.03))
        assertEquals(0.03, result[0], tol)
        assertEquals(1, result.size)
    }

    @Test
    fun testBenjaminiHochbergSingleElement() {
        // statsmodels: multipletests([0.03], method='fdr_bh') => [0.03]
        val result = benjaminiHochbergCorrection(doubleArrayOf(0.03))
        assertEquals(0.03, result[0], tol)
        assertEquals(1, result.size)
    }

    // ===== Edge cases: two elements =====

    @Test
    fun testBonferroniTwoElements() {
        // statsmodels: multipletests([0.02, 0.08], method='bonferroni')
        val pValues = doubleArrayOf(0.02, 0.08)
        val result = bonferroniCorrection(pValues)
        assertEquals(0.04, result[0], tol)
        assertEquals(0.16, result[1], tol)
    }

    @Test
    fun testHolmTwoElements() {
        // statsmodels: multipletests([0.02, 0.08], method='holm')
        val pValues = doubleArrayOf(0.02, 0.08)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.04, result[0], tol)
        assertEquals(0.08, result[1], tol)
    }

    @Test
    fun testBenjaminiHochbergTwoElements() {
        // statsmodels: multipletests([0.02, 0.08], method='fdr_bh')
        val pValues = doubleArrayOf(0.02, 0.08)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.04, result[0], tol)
        assertEquals(0.08, result[1], tol)
    }

    // ===== Edge cases: boundary p-values (0.0 and 1.0) =====

    @Test
    fun testBonferroniBoundaryPValues() {
        // statsmodels: multipletests([0.0, 0.5, 1.0], method='bonferroni')
        val pValues = doubleArrayOf(0.0, 0.5, 1.0)
        val result = bonferroniCorrection(pValues)
        assertEquals(0.0, result[0], tol, "bonferroni of 0.0 is 0.0")
        assertEquals(1.0, result[1], tol, "bonferroni of 0.5 * 3 = 1.5, clamped to 1.0")
        assertEquals(1.0, result[2], tol, "bonferroni of 1.0 * 3 = 3.0, clamped to 1.0")
    }

    @Test
    fun testHolmBoundaryPValues() {
        // statsmodels: multipletests([0.0, 0.5, 1.0], method='holm')
        val pValues = doubleArrayOf(0.0, 0.5, 1.0)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.0, result[0], tol, "holm of 0.0 is 0.0")
        assertEquals(1.0, result[1], tol, "holm p[1]")
        assertEquals(1.0, result[2], tol, "holm p[2]")
    }

    @Test
    fun testBenjaminiHochbergBoundaryPValues() {
        // statsmodels: multipletests([0.0, 0.5, 1.0], method='fdr_bh')
        val pValues = doubleArrayOf(0.0, 0.5, 1.0)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.0, result[0], tol, "bh of 0.0 is 0.0")
        assertEquals(0.75, result[1], tol, "bh p[1]")
        assertEquals(1.0, result[2], tol, "bh p[2]")
    }

    // ===== Edge cases: identical p-values =====

    @Test
    fun testBonferroniIdenticalPValues() {
        // statsmodels: multipletests([0.05, 0.05, 0.05], method='bonferroni')
        val pValues = doubleArrayOf(0.05, 0.05, 0.05)
        val result = bonferroniCorrection(pValues)
        for (i in result.indices) {
            assertEquals(0.15, result[i], tol, "bonferroni identical p[$i]")
        }
    }

    @Test
    fun testHolmIdenticalPValues() {
        // statsmodels: multipletests([0.05, 0.05, 0.05], method='holm')
        val pValues = doubleArrayOf(0.05, 0.05, 0.05)
        val result = holmBonferroniCorrection(pValues)
        for (i in result.indices) {
            assertEquals(0.15, result[i], tol, "holm identical p[$i]")
        }
    }

    @Test
    fun testBenjaminiHochbergIdenticalPValues() {
        // statsmodels: multipletests([0.05, 0.05, 0.05], method='fdr_bh')
        val pValues = doubleArrayOf(0.05, 0.05, 0.05)
        val result = benjaminiHochbergCorrection(pValues)
        for (i in result.indices) {
            assertEquals(0.05, result[i], tol, "bh identical p[$i]")
        }
    }

    // ===== Degenerate input: empty array =====

    @Test
    fun testBonferroniEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            bonferroniCorrection(doubleArrayOf())
        }
    }

    @Test
    fun testHolmEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            holmBonferroniCorrection(doubleArrayOf())
        }
    }

    @Test
    fun testBenjaminiHochbergEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            benjaminiHochbergCorrection(doubleArrayOf())
        }
    }

    // ===== Degenerate input: invalid p-values =====

    @Test
    fun testBonferroniNegativePValue() {
        assertFailsWith<InvalidParameterException> {
            bonferroniCorrection(doubleArrayOf(0.01, -0.05, 0.03))
        }
    }

    @Test
    fun testHolmNegativePValue() {
        assertFailsWith<InvalidParameterException> {
            holmBonferroniCorrection(doubleArrayOf(0.01, -0.05, 0.03))
        }
    }

    @Test
    fun testBenjaminiHochbergNegativePValue() {
        assertFailsWith<InvalidParameterException> {
            benjaminiHochbergCorrection(doubleArrayOf(0.01, -0.05, 0.03))
        }
    }

    @Test
    fun testBonferroniPValueGreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            bonferroniCorrection(doubleArrayOf(0.01, 1.5, 0.03))
        }
    }

    @Test
    fun testHolmPValueGreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            holmBonferroniCorrection(doubleArrayOf(0.01, 1.5, 0.03))
        }
    }

    @Test
    fun testBenjaminiHochbergPValueGreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            benjaminiHochbergCorrection(doubleArrayOf(0.01, 1.5, 0.03))
        }
    }

    // ===== Extreme parameters: very small p-values =====

    @Test
    fun testBonferroniVerySmallPValues() {
        // statsmodels: multipletests([1e-10, 1e-8, 1e-5, 1e-3], method='bonferroni')
        val pValues = doubleArrayOf(1e-10, 1e-8, 1e-5, 1e-3)
        val result = bonferroniCorrection(pValues)
        assertEquals(4e-10, result[0], 1e-20, "bonferroni 1e-10 * 4")
        assertEquals(4e-8, result[1], 1e-18, "bonferroni 1e-8 * 4")
        assertEquals(4e-5, result[2], 1e-15, "bonferroni 1e-5 * 4")
        assertEquals(0.004, result[3], tol, "bonferroni 1e-3 * 4")
    }

    @Test
    fun testHolmVerySmallPValues() {
        // statsmodels: multipletests([1e-10, 1e-8, 1e-5, 1e-3], method='holm')
        val pValues = doubleArrayOf(1e-10, 1e-8, 1e-5, 1e-3)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(4e-10, result[0], 1e-20, "holm 1e-10 * 4")
        assertEquals(3e-8, result[1], 1e-18, "holm 1e-8 * 3")
        assertEquals(2e-5, result[2], 1e-15, "holm 1e-5 * 2")
        assertEquals(0.001, result[3], tol, "holm 1e-3 * 1")
    }

    @Test
    fun testBenjaminiHochbergVerySmallPValues() {
        // statsmodels: multipletests([1e-10, 1e-8, 1e-5, 1e-3], method='fdr_bh')
        val pValues = doubleArrayOf(1e-10, 1e-8, 1e-5, 1e-3)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(4e-10, result[0], 1e-20, "bh 1e-10 * 4/1")
        assertEquals(2e-8, result[1], 1e-18, "bh 1e-8 * 4/2")
        assertEquals(1.33333333333333e-5, result[2], 1e-15, "bh 1e-5 * 4/3")
        assertEquals(0.001, result[3], tol, "bh 1e-3 * 4/4")
    }

    @Test
    fun testBonferroniExtremelySmallPValue() {
        // statsmodels: multipletests([1e-15, 0.5, 0.999999999], method='bonferroni')
        val pValues = doubleArrayOf(1e-15, 0.5, 0.999999999)
        val result = bonferroniCorrection(pValues)
        assertEquals(3e-15, result[0], 1e-25, "bonferroni 1e-15 * 3")
        assertEquals(1.0, result[1], tol, "bonferroni 0.5 * 3 clamped")
        assertEquals(1.0, result[2], tol, "bonferroni ~1.0 * 3 clamped")
    }

    @Test
    fun testHolmExtremelySmallPValue() {
        // statsmodels: multipletests([1e-15, 0.5, 0.999999999], method='holm')
        val pValues = doubleArrayOf(1e-15, 0.5, 0.999999999)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(3e-15, result[0], 1e-25, "holm 1e-15 * 3")
        assertEquals(1.0, result[1], tol, "holm p[1]")
        assertEquals(1.0, result[2], tol, "holm p[2]")
    }

    @Test
    fun testBenjaminiHochbergExtremelySmallPValue() {
        // statsmodels: multipletests([1e-15, 0.5, 0.999999999], method='fdr_bh')
        val pValues = doubleArrayOf(1e-15, 0.5, 0.999999999)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(3e-15, result[0], 1e-25, "bh 1e-15 * 3/1")
        assertEquals(0.75, result[1], tol, "bh 0.5 * 3/2")
        assertEquals(0.999999999, result[2], tol, "bh ~1.0 * 3/3")
    }

    // ===== Non-finite input: NaN propagation =====

    @Test
    fun testBonferroniNaNPropagation() {
        val pValues = doubleArrayOf(0.01, Double.NaN, 0.03)
        val result = bonferroniCorrection(pValues)
        // Non-NaN values should be corrected normally (m=3)
        assertEquals(0.03, result[0], tol, "non-NaN corrected normally")
        assertTrue(result[1].isNaN(), "NaN should propagate")
        assertEquals(0.09, result[2], tol, "non-NaN corrected normally")
    }

    @Test
    fun testHolmNaNPropagation() {
        val pValues = doubleArrayOf(0.01, Double.NaN, 0.03)
        val result = holmBonferroniCorrection(pValues)
        // NaN should propagate; non-NaN values corrected with m=3
        assertTrue(result[1].isNaN(), "NaN should propagate in Holm")
        // Non-NaN positions should have finite adjusted p-values
        assertTrue(result[0].isFinite(), "non-NaN should be finite")
        assertTrue(result[2].isFinite(), "non-NaN should be finite")
    }

    @Test
    fun testBenjaminiHochbergNaNPropagation() {
        val pValues = doubleArrayOf(0.01, Double.NaN, 0.03)
        val result = benjaminiHochbergCorrection(pValues)
        // NaN should propagate; non-NaN values corrected with m=3
        assertTrue(result[1].isNaN(), "NaN should propagate in BH")
        assertTrue(result[0].isFinite(), "non-NaN should be finite")
        assertTrue(result[2].isFinite(), "non-NaN should be finite")
    }

    @Test
    fun testBonferroniAllNaN() {
        val pValues = doubleArrayOf(Double.NaN, Double.NaN)
        val result = bonferroniCorrection(pValues)
        assertTrue(result[0].isNaN(), "all NaN input: result[0] should be NaN")
        assertTrue(result[1].isNaN(), "all NaN input: result[1] should be NaN")
    }

    @Test
    fun testHolmAllNaN() {
        val pValues = doubleArrayOf(Double.NaN, Double.NaN)
        val result = holmBonferroniCorrection(pValues)
        assertTrue(result[0].isNaN(), "all NaN input: result[0] should be NaN")
        assertTrue(result[1].isNaN(), "all NaN input: result[1] should be NaN")
    }

    @Test
    fun testBenjaminiHochbergAllNaN() {
        val pValues = doubleArrayOf(Double.NaN, Double.NaN)
        val result = benjaminiHochbergCorrection(pValues)
        assertTrue(result[0].isNaN(), "all NaN input: result[0] should be NaN")
        assertTrue(result[1].isNaN(), "all NaN input: result[1] should be NaN")
    }

    @Test
    fun testBonferroniSingleNaN() {
        val result = bonferroniCorrection(doubleArrayOf(Double.NaN))
        assertTrue(result[0].isNaN(), "single NaN should propagate")
        assertEquals(1, result.size)
    }

    // ===== Property: adjusted p-values are >= original p-values =====

    @Test
    fun testBonferroniAdjustedGreaterThanOrEqualOriginal() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = bonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertTrue(
                result[i] >= pValues[i],
                "Bonferroni adjusted p[$i]=${result[i]} should be >= original ${pValues[i]}"
            )
        }
    }

    @Test
    fun testHolmAdjustedGreaterThanOrEqualOriginal() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = holmBonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertTrue(
                result[i] >= pValues[i],
                "Holm adjusted p[$i]=${result[i]} should be >= original ${pValues[i]}"
            )
        }
    }

    @Test
    fun testBenjaminiHochbergAdjustedGreaterThanOrEqualOriginal() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val result = benjaminiHochbergCorrection(pValues)
        for (i in pValues.indices) {
            assertTrue(
                result[i] >= pValues[i],
                "BH adjusted p[$i]=${result[i]} should be >= original ${pValues[i]}"
            )
        }
    }

    // ===== Property: adjusted p-values in [0, 1] =====

    @Test
    fun testBonferroniResultInRange() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5, 0.99)
        val result = bonferroniCorrection(pValues)
        for (i in result.indices) {
            assertTrue(result[i] in 0.0..1.0, "Bonferroni result[$i]=${result[i]} should be in [0,1]")
        }
    }

    @Test
    fun testHolmResultInRange() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5, 0.99)
        val result = holmBonferroniCorrection(pValues)
        for (i in result.indices) {
            assertTrue(result[i] in 0.0..1.0, "Holm result[$i]=${result[i]} should be in [0,1]")
        }
    }

    @Test
    fun testBenjaminiHochbergResultInRange() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5, 0.99)
        val result = benjaminiHochbergCorrection(pValues)
        for (i in result.indices) {
            assertTrue(result[i] in 0.0..1.0, "BH result[$i]=${result[i]} should be in [0,1]")
        }
    }

    // ===== Property: output array has same size as input =====

    @Test
    fun testOutputSizeMatchesInput() {
        for (n in listOf(1, 2, 5, 10)) {
            val pValues = DoubleArray(n) { (it + 1).toDouble() / (n + 1) }
            assertEquals(n, bonferroniCorrection(pValues).size, "Bonferroni size for n=$n")
            assertEquals(n, holmBonferroniCorrection(pValues).size, "Holm size for n=$n")
            assertEquals(n, benjaminiHochbergCorrection(pValues).size, "BH size for n=$n")
        }
    }

    // ===== Property: Holm <= Bonferroni (Holm is uniformly more powerful) =====

    @Test
    fun testHolmLessThanOrEqualBonferroni() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val bonf = bonferroniCorrection(pValues)
        val holm = holmBonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertTrue(
                holm[i] <= bonf[i] + 1e-14,
                "Holm p[$i]=${holm[i]} should be <= Bonferroni p[$i]=${bonf[i]}"
            )
        }
    }

    // ===== Property: BH <= Holm (BH is uniformly more powerful than Holm) =====

    @Test
    fun testBenjaminiHochbergLessThanOrEqualHolm() {
        val pValues = doubleArrayOf(0.001, 0.01, 0.05, 0.1, 0.5)
        val holm = holmBonferroniCorrection(pValues)
        val bh = benjaminiHochbergCorrection(pValues)
        for (i in pValues.indices) {
            assertTrue(
                bh[i] <= holm[i] + 1e-14,
                "BH p[$i]=${bh[i]} should be <= Holm p[$i]=${holm[i]}"
            )
        }
    }

    // ===== Property: order invariance — corrections preserve original index mapping =====

    @Test
    fun testBonferroniPreservesIndexMapping() {
        // Bonferroni is order-independent: result[i] = p[i] * m regardless of order
        val pValues = doubleArrayOf(0.04, 0.01, 0.03, 0.005)
        val result = bonferroniCorrection(pValues)
        val m = pValues.size
        for (i in pValues.indices) {
            assertEquals(
                (pValues[i] * m).coerceAtMost(1.0), result[i], tol,
                "Bonferroni result[$i] should be p[$i]*m"
            )
        }
    }

    @Test
    fun testCorrectionResultsInOriginalOrder() {
        // Holm and BH: verify the result maps back to original positions
        // For reverse-sorted input, the results should be in the original order
        // statsmodels: multipletests([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001], method='holm')
        val pValues = doubleArrayOf(0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
        val holm = holmBonferroniCorrection(pValues)
        assertEquals(0.5, holm[0], tol, "holm reverse[0]")
        assertEquals(0.4, holm[1], tol, "holm reverse[1]")
        assertEquals(0.3, holm[2], tol, "holm reverse[2]")
        assertEquals(0.2, holm[3], tol, "holm reverse[3]")
        assertEquals(0.1, holm[4], tol, "holm reverse[4]")
        assertEquals(0.06, holm[5], tol, "holm reverse[5]")
        assertEquals(0.035, holm[6], tol, "holm reverse[6]")
        assertEquals(0.008, holm[7], tol, "holm reverse[7]")
    }

    @Test
    fun testBenjaminiHochbergReverseSorted() {
        // statsmodels: multipletests([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001], method='fdr_bh')
        val pValues = doubleArrayOf(0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001)
        val bh = benjaminiHochbergCorrection(pValues)
        assertEquals(0.5, bh[0], tol, "bh reverse[0]")
        assertEquals(0.228571428571429, bh[1], tol, "bh reverse[1]")
        assertEquals(0.133333333333333, bh[2], tol, "bh reverse[2]")
        assertEquals(0.08, bh[3], tol, "bh reverse[3]")
        assertEquals(0.04, bh[4], tol, "bh reverse[4]")
        assertEquals(0.0266666666666667, bh[5], tol, "bh reverse[5]")
        assertEquals(0.02, bh[6], tol, "bh reverse[6]")
        assertEquals(0.008, bh[7], tol, "bh reverse[7]")
    }

    // ===== Property: input array is not mutated =====

    @Test
    fun testBonferroniDoesNotMutateInput() {
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val copy = pValues.copyOf()
        bonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(copy[i], pValues[i], 0.0, "input[$i] should not be mutated")
        }
    }

    @Test
    fun testHolmDoesNotMutateInput() {
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val copy = pValues.copyOf()
        holmBonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(copy[i], pValues[i], 0.0, "input[$i] should not be mutated")
        }
    }

    @Test
    fun testBenjaminiHochbergDoesNotMutateInput() {
        val pValues = doubleArrayOf(0.01, 0.04, 0.03, 0.005)
        val copy = pValues.copyOf()
        benjaminiHochbergCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(copy[i], pValues[i], 0.0, "input[$i] should not be mutated")
        }
    }

    // ===== Property: Holm monotonicity — sorted adjusted p-values are non-decreasing =====

    @Test
    fun testHolmAdjustedPValuesMonotonicity() {
        // After Holm correction, if we sort by original p-values, adjusted should be non-decreasing
        val pValues = doubleArrayOf(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
        val result = holmBonferroniCorrection(pValues)
        // Since input is already sorted ascending, result should be non-decreasing
        val sortedIndices = pValues.indices.sortedBy { pValues[it] }
        for (i in 0 until sortedIndices.size - 1) {
            val curr = result[sortedIndices[i]]
            val next = result[sortedIndices[i + 1]]
            assertTrue(
                curr <= next + 1e-14,
                "Holm monotonicity: adjusted[rank $i]=$curr should be <= adjusted[rank ${i + 1}]=$next"
            )
        }
    }

    // ===== Property: BH monotonicity — sorted adjusted p-values are non-decreasing =====

    @Test
    fun testBenjaminiHochbergAdjustedPValuesMonotonicity() {
        val pValues = doubleArrayOf(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
        val result = benjaminiHochbergCorrection(pValues)
        val sortedIndices = pValues.indices.sortedBy { pValues[it] }
        for (i in 0 until sortedIndices.size - 1) {
            val curr = result[sortedIndices[i]]
            val next = result[sortedIndices[i + 1]]
            assertTrue(
                curr <= next + 1e-14,
                "BH monotonicity: adjusted[rank $i]=$curr should be <= adjusted[rank ${i + 1}]=$next"
            )
        }
    }

    // ===== Golden values: 10-element test with sorted random p-values =====

    @Test
    fun testBonferroniGoldenValues() {
        // statsmodels: sorted random p-values (seed=42), method='bonferroni'
        val pValues = doubleArrayOf(
            0.0580836121681995, 0.155994520336203, 0.156018640442437,
            0.374540118847362, 0.598658484197037, 0.601115011743209,
            0.708072577796045, 0.731993941811405, 0.866176145774935,
            0.950714306409916
        )
        val expected = doubleArrayOf(
            0.580836121681995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        val result = bonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(expected[i], result[i], tol, "bonferroni golden[$i]")
        }
    }

    @Test
    fun testHolmGoldenValues() {
        // statsmodels: sorted random p-values (seed=42), method='holm'
        val pValues = doubleArrayOf(
            0.0580836121681995, 0.155994520336203, 0.156018640442437,
            0.374540118847362, 0.598658484197037, 0.601115011743209,
            0.708072577796045, 0.731993941811405, 0.866176145774935,
            0.950714306409916
        )
        val expected = doubleArrayOf(
            0.580836121681995, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        )
        val result = holmBonferroniCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(expected[i], result[i], tol, "holm golden[$i]")
        }
    }

    @Test
    fun testBenjaminiHochbergGoldenValues() {
        // statsmodels: sorted random p-values (seed=42), method='fdr_bh'
        val pValues = doubleArrayOf(
            0.0580836121681995, 0.155994520336203, 0.156018640442437,
            0.374540118847362, 0.598658484197037, 0.601115011743209,
            0.708072577796045, 0.731993941811405, 0.866176145774935,
            0.950714306409916
        )
        val expected = doubleArrayOf(
            0.520062134808122, 0.520062134808122, 0.520062134808122,
            0.914992427264256, 0.914992427264256, 0.914992427264256,
            0.914992427264256, 0.914992427264256, 0.950714306409916,
            0.950714306409916
        )
        val result = benjaminiHochbergCorrection(pValues)
        for (i in pValues.indices) {
            assertEquals(expected[i], result[i], tol, "bh golden[$i]")
        }
    }

    // ===== Property: sorted ascending Holm known values =====

    @Test
    fun testHolmSortedAscendingKnownValues() {
        // statsmodels: multipletests([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], method='holm')
        val pValues = doubleArrayOf(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
        val result = holmBonferroniCorrection(pValues)
        assertEquals(0.008, result[0], tol, "holm sorted[0]")
        assertEquals(0.035, result[1], tol, "holm sorted[1]")
        assertEquals(0.06, result[2], tol, "holm sorted[2]")
        assertEquals(0.1, result[3], tol, "holm sorted[3]")
        assertEquals(0.2, result[4], tol, "holm sorted[4]")
        assertEquals(0.3, result[5], tol, "holm sorted[5]")
        assertEquals(0.4, result[6], tol, "holm sorted[6]")
        assertEquals(0.5, result[7], tol, "holm sorted[7]")
    }

    @Test
    fun testBenjaminiHochbergSortedAscendingKnownValues() {
        // statsmodels: multipletests([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5], method='fdr_bh')
        val pValues = doubleArrayOf(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
        val result = benjaminiHochbergCorrection(pValues)
        assertEquals(0.008, result[0], tol, "bh sorted[0]")
        assertEquals(0.02, result[1], tol, "bh sorted[1]")
        assertEquals(0.0266666666666667, result[2], tol, "bh sorted[2]")
        assertEquals(0.04, result[3], tol, "bh sorted[3]")
        assertEquals(0.08, result[4], tol, "bh sorted[4]")
        assertEquals(0.133333333333333, result[5], tol, "bh sorted[5]")
        assertEquals(0.228571428571429, result[6], tol, "bh sorted[6]")
        assertEquals(0.5, result[7], tol, "bh sorted[7]")
    }
}
