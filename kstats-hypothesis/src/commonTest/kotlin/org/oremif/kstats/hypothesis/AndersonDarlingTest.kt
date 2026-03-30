package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AndersonDarlingTest {

    private val tolA2 = 1e-10
    private val tolP = 1e-3

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN3Symmetric() {
        // scipy: stats.anderson([-0.5, 0.0, 0.5], dist='norm') → A²=0.189488054537566
        val result = andersonDarlingTest(doubleArrayOf(-0.5, 0.0, 0.5))
        assertEquals(0.189488054537566, result.statistic, tolA2, "A² n=3 symmetric")
        assertEquals("Anderson-Darling Test", result.testName)
        assertTrue(result.pValue > 0.05, "Symmetric n=3 should not reject normality")
    }

    @Test
    fun testN7() {
        val result = andersonDarlingTest(doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5))
        assertEquals(0.138669788736141, result.statistic, tolA2, "A² n=7")
        assertTrue(result.pValue > 0.05)
    }

    @Test
    fun testN10Normal() {
        val result = andersonDarlingTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.5)
        )
        assertEquals(0.104221649214109, result.statistic, tolA2, "A² n=10")
        assertTrue(result.pValue > 0.15, "Normal data should not reject")
    }

    @Test
    fun testN20Normal() {
        // scipy: stats.anderson(data) → A²=0.490035999877421
        val result = andersonDarlingTest(TestData.NORMAL_N20)
        assertEquals(0.490035999877421, result.statistic, tolA2, "A² n=20")
        assertEquals(0.196, result.pValue, tolP, "p-value n=20")
    }

    @Test
    fun testN100Normal() {
        val result = andersonDarlingTest(TestData.NORMAL_N100)
        assertEquals(0.366112469096564, result.statistic, tolA2, "A² n=100")
        assertEquals(0.428, result.pValue, tolP, "p-value n=100")
    }

    // ===== Non-normal data: should reject =====

    @Test
    fun testUniformDataRejects() {
        val result = andersonDarlingTest(TestData.UNIFORM_N30)
        assertEquals(0.971334561238429, result.statistic, tolA2, "A² n=30 uniform")
        assertTrue(result.pValue < 0.05, "Uniform data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testExponentialDataRejects() {
        val result = andersonDarlingTest(TestData.EXPONENTIAL_N30)
        assertEquals(2.26816288595751, result.statistic, tolA2, "A² n=30 exponential")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject normality, p=${result.pValue}")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        val result = andersonDarlingTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertEquals(0.189488054537566, result.statistic, tolA2, "A² n=3 minimum")
        assertTrue(result.pValue > 0.0 && result.pValue <= 1.0)
    }

    @Test
    fun testModifiedStatisticInAdditionalInfo() {
        val result = andersonDarlingTest(doubleArrayOf(-1.0, 0.0, 1.0, 2.0, 3.0))
        val a2Star = result.additionalInfo["modifiedStatistic"]
        assertTrue(a2Star != null && a2Star > result.statistic, "Modified A²* should be larger than raw A²")
    }

    // ===== Degenerate input =====

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> { andersonDarlingTest(doubleArrayOf(1.0, 2.0)) }
        assertFailsWith<InsufficientDataException> { andersonDarlingTest(doubleArrayOf(1.0)) }
        assertFailsWith<InsufficientDataException> { andersonDarlingTest(doubleArrayOf()) }
    }

    @Test
    fun testConstantData() {
        val result = andersonDarlingTest(doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0))
        assertEquals(0.0, result.statistic, 1e-15)
        assertEquals(1.0, result.pValue, 1e-15)
    }

    @Test
    fun testNearConstantData() {
        val data = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0 + 1e-10)
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic >= 0.0, "A² should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSample() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(1000) { i -> normal.quantile((i + 0.5) / 1000) }
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic < 0.01, "Perfectly normal data should have very small A²")
        assertTrue(result.pValue > 0.05, "Perfectly normal data should not reject normality")
    }

    @Test
    fun testHighlySignificant() {
        val result = andersonDarlingTest(TestData.bimodal())
        assertTrue(result.pValue < 0.001, "Bimodal data should have very small p-value")
    }

    @Test
    fun testVeryLargeN() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(5000) { i -> normal.quantile((i + 0.5) / 5000) }
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic < 0.01, "n=5000 perfect normal should have tiny A²")
        assertTrue(result.pValue > 0.05, "n=5000 perfect normal should not reject")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val result = andersonDarlingTest(TestData.WITH_NAN)
        TestAssertions.assertNaNResult(result, "when input contains NaN")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        val normalResult = andersonDarlingTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.5)
        )
        TestAssertions.assertIsSignificantConsistency(normalResult)

        val expResult = andersonDarlingTest(TestData.EXPONENTIAL_N30)
        TestAssertions.assertIsSignificantConsistency(expResult)
    }
}
