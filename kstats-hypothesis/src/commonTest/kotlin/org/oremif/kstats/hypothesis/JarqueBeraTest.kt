package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JarqueBeraTest {

    private val tolJB = 1e-10
    private val tolP = 1e-3

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN10NormalData() {
        // scipy: jarque_bera(data) → JB=0.539772352201391, p=0.763466390112862
        val data = TestData.NORMAL_N20.sliceArray(0 until 10)
        val result = jarqueBeraTest(data)
        assertEquals(0.539772352201391, result.statistic, tolJB, "JB n=10 normal")
        assertEquals(0.763466390112862, result.pValue, tolP, "p-value n=10 normal")
        assertEquals("Jarque-Bera Test", result.testName)
        assertEquals(2.0, result.degreesOfFreedom)
        assertTrue(result.additionalInfo.containsKey("skewness"))
        assertTrue(result.additionalInfo.containsKey("kurtosis"))
    }

    @Test
    fun testN30NormalData() {
        // scipy: jarque_bera(data) → JB=0.274192655780974, p=0.871886235125062
        val result = jarqueBeraTest(TestData.NORMAL_N30)
        assertEquals(0.274192655780974, result.statistic, tolJB, "JB n=30 normal")
        assertEquals(0.871886235125062, result.pValue, tolP, "p-value n=30 normal")
    }

    @Test
    fun testN100NormalData() {
        // scipy: jarque_bera(data) → JB=4.9688973327694, p=0.0833715073578094
        val result = jarqueBeraTest(TestData.NORMAL_N100)
        assertEquals(4.9688973327694, result.statistic, tolJB, "JB n=100 normal")
        assertEquals(0.0833715073578094, result.pValue, tolP, "p-value n=100 normal")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        // scipy: jarque_bera([1,2,3]) → JB=0.28125, p=0.868815056262843
        val result = jarqueBeraTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertEquals(0.28125, result.statistic, tolJB, "JB n=3 arange")
        assertEquals(0.868815056262843, result.pValue, tolP, "p-value n=3 arange")
    }

    @Test
    fun testSymmetricData() {
        // scipy: jarque_bera(linspace(-5,5,30)) → JB=1.80801780745136, p=0.404943023808853
        val data = DoubleArray(30) { i -> -5.0 + 10.0 * i / 29.0 }
        val result = jarqueBeraTest(data)
        assertEquals(1.80801780745136, result.statistic, tolJB, "JB symmetric n=30")
        assertEquals(0.404943023808853, result.pValue, tolP, "p-value symmetric n=30")
        assertEquals(0.0, result.additionalInfo["skewness"]!!, 1e-15, "Skewness should be 0 for symmetric data")
    }

    // ===== Degenerate input =====

    @Test
    fun testConstantData() {
        val result = jarqueBeraTest(DoubleArray(20) { 5.0 })
        assertEquals(0.0, result.statistic, 1e-15)
        assertEquals(1.0, result.pValue, 1e-15)
        assertEquals(0.0, result.additionalInfo["skewness"]!!, 1e-15)
        assertEquals(0.0, result.additionalInfo["kurtosis"]!!, 1e-15)
    }

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> { jarqueBeraTest(doubleArrayOf(1.0, 2.0)) }
        assertFailsWith<InsufficientDataException> { jarqueBeraTest(doubleArrayOf(1.0)) }
        assertFailsWith<InsufficientDataException> { jarqueBeraTest(doubleArrayOf()) }
    }

    // ===== Non-normal rejection =====

    @Test
    fun testExponentialDataRejects() {
        // scipy: jarque_bera(data) → JB=16.0920762731418, p=0.000320368668398256
        val result = jarqueBeraTest(TestData.EXPONENTIAL_N30)
        assertEquals(16.0920762731418, result.statistic, tolJB, "JB exponential n=30")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject, p=${result.pValue}")
    }

    @Test
    fun testUniformDataRejects() {
        // scipy: jarque_bera(data) → JB=2.60053766842867, p=0.272458537010854
        val result = jarqueBeraTest(TestData.UNIFORM_N30)
        assertEquals(2.60053766842867, result.statistic, tolJB, "JB uniform n=30")
        assertEquals(0.272458537010854, result.pValue, tolP, "p-value uniform n=30")
    }

    @Test
    fun testBimodalDataRejects() {
        // scipy: jarque_bera(bimodal) → JB=8.33160025243092, p=0.0155172941378467
        val result = jarqueBeraTest(TestData.bimodal())
        assertEquals(8.33160025243092, result.statistic, tolJB, "JB bimodal n=50")
        assertTrue(result.pValue < 0.05, "Bimodal data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testHeavyTailedDataRejects() {
        // scipy: jarque_bera(data) → JB=884.694920553516, p≈0
        val result = jarqueBeraTest(TestData.HEAVY_TAILED_N50)
        assertEquals(884.694920553516, result.statistic, tolJB, "JB heavy-tailed n=50")
        assertTrue(result.pValue < 1e-10, "Heavy-tailed should strongly reject, p=${result.pValue}")
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSampleN1000() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(1000) { i -> normal.quantile((i + 0.5) / 1000) }
        val result = jarqueBeraTest(data)
        assertTrue(result.statistic < 0.1, "Perfect normal n=1000 should have small JB")
        assertTrue(result.pValue > 0.9, "Perfect normal n=1000 should have large p-value")
    }

    @Test
    fun testLargeSampleN5000() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(5000) { i -> normal.quantile((i + 0.5) / 5000) }
        val result = jarqueBeraTest(data)
        assertTrue(result.statistic < 0.1, "Perfect normal n=5000 should have small JB")
        assertTrue(result.pValue > 0.9, "Perfect normal n=5000 should have large p-value")
    }

    @Test
    fun testExtremeOutliers() {
        // scipy: JB=4612.4836358838, p≈0
        val data = DoubleArray(50) { i -> if (i < 49) i.toDouble() / 10.0 else 1000.0 }
        val result = jarqueBeraTest(data)
        assertEquals(4612.4836358838, result.statistic, tolJB, "JB extreme outlier n=50")
        assertTrue(result.pValue < 1e-10, "Data with extreme outlier should reject normality")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.NaN }
        val result = jarqueBeraTest(data)
        TestAssertions.assertNaNResult(result, "when input contains NaN")
    }

    @Test
    fun testPositiveInfinity() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.POSITIVE_INFINITY }
        val result = jarqueBeraTest(data)
        TestAssertions.assertNaNResult(result, "when input contains Infinity")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        val normalResult = jarqueBeraTest(TestData.NORMAL_N30)
        TestAssertions.assertIsSignificantConsistency(normalResult)

        val expResult = jarqueBeraTest(TestData.EXPONENTIAL_N30)
        TestAssertions.assertIsSignificantConsistency(expResult)
    }

    @Test
    fun testTestName() {
        val result = jarqueBeraTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertEquals("Jarque-Bera Test", result.testName)
    }
}
