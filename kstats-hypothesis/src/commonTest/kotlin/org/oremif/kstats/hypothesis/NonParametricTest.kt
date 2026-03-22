package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.*

class NonParametricTest {

    private fun assertP(expected: Double, actual: Double, tol: Double = 1e-3, message: String = "") {
        assertTrue(abs(expected - actual) < tol, "p-value $message: expected=$expected, actual=$actual")
    }

    // ===== Mann-Whitney U =====

    @Test
    fun testMannWhitneyUSignificant() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testMannWhitneyUNotSignificant() {
        val s1 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)
        val s2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = mannWhitneyUTest(s1, s2)
        assertFalse(result.isSignificant(), "Interleaved data should not be significant")
    }

    @Test
    fun testMannWhitneyUWithTies() {
        // Data with ties: tie correction should reduce sigma
        val s1 = doubleArrayOf(1.0, 2.0, 2.0, 3.0, 4.0)
        val s2 = doubleArrayOf(2.0, 3.0, 3.0, 4.0, 5.0)
        val result = mannWhitneyUTest(s1, s2)
        assertTrue(result.pValue in 0.0..1.0)
        assertTrue(result.additionalInfo["z"]!!.isFinite())
    }

    @Test
    fun testMannWhitneyUAllTied() {
        // All values identical: sigma = 0, z = 0, p = 1
        val s1 = doubleArrayOf(5.0, 5.0, 5.0)
        val s2 = doubleArrayOf(5.0, 5.0, 5.0)
        val result = mannWhitneyUTest(s1, s2)
        assertEquals(0.0, result.additionalInfo["z"]!!, 1e-15, "z should be 0 when all tied")
        assertEquals(1.0, result.pValue, 1e-6, "p should be 1 when all tied")
    }

    @Test
    fun testMannWhitneyUValidation() {
        assertFailsWith<InsufficientDataException> {
            mannWhitneyUTest(doubleArrayOf(), doubleArrayOf(1.0))
        }
        assertFailsWith<InsufficientDataException> {
            mannWhitneyUTest(doubleArrayOf(1.0), doubleArrayOf())
        }
    }

    @Test
    fun testMannWhitneyUAlternatives() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val two = mannWhitneyUTest(s1, s2)
        val less = mannWhitneyUTest(s1, s2, Alternative.LESS)
        val greater = mannWhitneyUTest(s1, s2, Alternative.GREATER)
        assertEquals(Alternative.TWO_SIDED, two.alternative)
        assertEquals(Alternative.LESS, less.alternative)
        assertEquals(Alternative.GREATER, greater.alternative)
    }

    // ===== Wilcoxon Signed-Rank =====

    @Test
    fun testWilcoxonSignedRank() {
        val s1 = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val s2 = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)
        val result = wilcoxonSignedRankTest(s1, s2)
        assertTrue(result.statistic > 0)
    }

    @Test
    fun testWilcoxonOneSample() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = wilcoxonSignedRankTest(data)
        assertTrue(result.statistic > 0)
        assertEquals("Wilcoxon Signed-Rank Test", result.testName)
    }

    @Test
    fun testWilcoxonAllZeros() {
        assertFailsWith<DegenerateDataException> {
            wilcoxonSignedRankTest(doubleArrayOf(0.0, 0.0, 0.0))
        }
    }

    @Test
    fun testWilcoxonPairedAllEqual() {
        assertFailsWith<DegenerateDataException> {
            wilcoxonSignedRankTest(
                doubleArrayOf(5.0, 5.0, 5.0),
                doubleArrayOf(5.0, 5.0, 5.0)
            )
        }
    }

    @Test
    fun testWilcoxonMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            wilcoxonSignedRankTest(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        }
    }

    // ===== Kolmogorov-Smirnov =====

    @Test
    fun testKolmogorovSmirnovOneSample() {
        val sample = doubleArrayOf(-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, -0.3, 0.3, 0.8)
        val result = kolmogorovSmirnovTest(sample, org.oremif.kstats.distributions.NormalDistribution.STANDARD)
        assertFalse(result.isSignificant(), "Normal-looking data should not be significant against normal")
    }

    @Test
    fun testKolmogorovSmirnovTwoSample() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = kolmogorovSmirnovTest(s1, s2)
        assertTrue(result.statistic > 0.5)
    }

    @Test
    fun testKolmogorovSmirnovTwoSampleIdentical() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kolmogorovSmirnovTest(s1, s1.copyOf())
        assertEquals(0.0, result.statistic, 1e-15, "D should be 0 for identical samples")
    }

    @Test
    fun testKolmogorovSmirnovValidation() {
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(), org.oremif.kstats.distributions.NormalDistribution.STANDARD)
        }
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(), doubleArrayOf(1.0))
        }
        assertFailsWith<InsufficientDataException> {
            kolmogorovSmirnovTest(doubleArrayOf(1.0), doubleArrayOf())
        }
    }

    @Test
    fun testShapiroWilk() {
        val data = doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5)
        val result = shapiroWilkTest(data)
        assertTrue(result.statistic > 0.8)
    }
}
