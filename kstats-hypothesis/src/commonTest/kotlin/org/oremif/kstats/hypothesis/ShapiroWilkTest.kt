package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class ShapiroWilkTest {

    private fun assertW(expected: Double, actual: Double, message: String = "") {
        assertEquals(expected, actual, 1e-6, "W $message")
    }

    private fun assertP(expected: Double, actual: Double, message: String = "") {
        assertEquals(expected, actual, 1e-4, "p-value $message")
    }

    // ===== Basic correctness: various n sizes with scipy reference values =====

    @Test
    fun testN3Symmetric() {
        // scipy: W=1.0, p=1.0
        val result = shapiroWilkTest(doubleArrayOf(-0.5, 0.0, 0.5))
        assertW(1.0, result.statistic, "n=3 symmetric")
        assertTrue(result.pValue > 0.99, "p-value should be ~1.0 for perfectly normal n=3")
    }

    @Test
    fun testN3Linear() {
        val result = shapiroWilkTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertW(1.0, result.statistic, "n=3 linear")
        assertTrue(result.pValue > 0.99)
    }

    @Test
    fun testN3Skewed() {
        val result = shapiroWilkTest(doubleArrayOf(1.0, 1.0, 10.0))
        assertW(0.75, result.statistic, "n=3 skewed")
        assertTrue(result.pValue < 0.01, "Skewed n=3 should have very low p-value")
    }

    @Test
    fun testN4() {
        val result = shapiroWilkTest(doubleArrayOf(-1.0, -0.3, 0.3, 1.0))
        assertW(0.9970872391, result.statistic, "n=4")
        assertP(0.9901843527, result.pValue, "n=4")
    }

    @Test
    fun testN5() {
        val result = shapiroWilkTest(doubleArrayOf(-1.2, -0.5, 0.0, 0.5, 1.2))
        assertW(0.9978435112, result.statistic, "n=5")
        assertP(0.9986227625, result.pValue, "n=5")
    }

    @Test
    fun testN7() {
        val result = shapiroWilkTest(doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5))
        assertW(0.9842845208, result.statistic, "n=7")
        assertP(0.9776759576, result.pValue, "n=7")
    }

    @Test
    fun testN10() {
        val result = shapiroWilkTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.5)
        )
        assertW(0.9853627058, result.statistic, "n=10")
        assertP(0.9873787719, result.pValue, "n=10")
    }

    // ===== Boundary n: p-value path transitions =====

    @Test
    fun testN11_UpperBoundSmallNPath() {
        val result = shapiroWilkTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.3, 1.5)
        )
        assertW(0.9656642577, result.statistic, "n=11")
        assertP(0.8397238700, result.pValue, "n=11")
    }

    @Test
    fun testN12_LowerBoundLargeNPath() {
        val result = shapiroWilkTest(
            doubleArrayOf(-1.5, -1.1, -0.7, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5)
        )
        assertW(0.9798720563, result.statistic, "n=12")
        assertP(0.9831367877, result.pValue, "n=12")
    }

    // ===== Larger n: normal data =====

    @Test
    fun testN20Normal() {
        val result = shapiroWilkTest(TestData.NORMAL_N20)
        assertW(0.9343037786, result.statistic, "n=20")
        assertP(0.1867887050, result.pValue, "n=20")
    }

    @Test
    fun testN50Normal() {
        // scipy: W=0.9863243031, p=0.8267150136
        val data = doubleArrayOf(
            -0.1848623635452606, -0.6809295444039414, 1.22254133867403, -0.1545294820688022,
            -0.4283278221631072, -0.3521335504882296, 0.5323091855533487, 0.3654440643640783,
            0.4127326115959884, 0.4308210030078827, 2.141647600870461, -0.4064150163846156,
            -0.5122427290715373, -0.8137727282478777, 0.6159794225754956, 1.128972292720892,
            -0.1139474576548751, -0.840156476962528, -0.8244812156912396, 0.6505927878247011,
            0.7432541712034423, 0.543154268305195, -0.6655097072886943, 0.2321613230667198,
            0.1166858091407282, 0.2186885967290129, 0.8714287779481898, 0.2235955487746823,
            0.6789135630718949, 0.06757906948889146, 0.2891193986899842, 0.6312882258385404,
            -1.457155819855666, -0.3196712163573013, -0.4703726542927955, -0.6388778482433419,
            -0.2751422512266837, 1.494941311234396, -0.8658311156932432, 0.9682783545914808,
            -1.682869771615805, -0.3348850299857749, 0.1627530651050056, 0.5862223313592781,
            0.711226579792855, 0.7933472351999252, -0.3487250722484376, -0.4623517926645672,
            0.8579758812571538, -0.1913043248816149
        )
        val result = shapiroWilkTest(data)
        assertW(0.9863243031, result.statistic, "n=50")
        assertP(0.8267150136, result.pValue, "n=50")
    }

    @Test
    fun testN100Normal() {
        val result = shapiroWilkTest(TestData.NORMAL_N100)
        assertW(0.9821226162, result.statistic, "n=100")
        assertP(0.1938665676, result.pValue, "n=100")
    }

    // ===== Non-normal data: should reject =====

    @Test
    fun testUniformDataRejects() {
        val result = shapiroWilkTest(TestData.UNIFORM_N30)
        assertW(0.9114344184, result.statistic, "n=30 uniform")
        assertP(0.0161757787, result.pValue, "n=30 uniform")
        assertTrue(result.pValue < 0.05, "Uniform data should reject normality")
    }

    @Test
    fun testExponentialDataRejects() {
        val result = shapiroWilkTest(TestData.EXPONENTIAL_N30)
        assertW(0.7808256107, result.statistic, "n=30 exponential")
        assertP(0.0000298940, result.pValue, "n=30 exponential")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject normality")
    }

    // ===== Edge cases =====

    @Test
    fun testTooFewElements() {
        assertFailsWith<InsufficientDataException> { shapiroWilkTest(doubleArrayOf(1.0, 2.0)) }
        assertFailsWith<InsufficientDataException> { shapiroWilkTest(doubleArrayOf(1.0)) }
        assertFailsWith<InsufficientDataException> { shapiroWilkTest(doubleArrayOf()) }
    }

    @Test
    fun testTooManyElements() {
        assertFailsWith<InvalidParameterException> { shapiroWilkTest(DoubleArray(5001) { it.toDouble() }) }
    }

    @Test
    fun testConstantArray() {
        val result = shapiroWilkTest(doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0))
        assertEquals(1.0, result.statistic, 1e-10)
        assertEquals(1.0, result.pValue, 1e-10)
    }

    @Test
    fun testNearConstantArray() {
        val data = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0 + 1e-10)
        val result = shapiroWilkTest(data)
        assertTrue(result.statistic > 0.0 && result.statistic <= 1.0)
        assertTrue(result.pValue in 0.0..1.0)
    }

    // ===== Large n: numerical stability =====

    @Test
    fun testLargeNStability() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 1000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = shapiroWilkTest(data)
        assertTrue(result.statistic > 0.99, "Perfectly normal data should have W close to 1, got ${result.statistic}")
        assertTrue(result.pValue > 0.05, "Perfectly normal data should not reject, p=${result.pValue}")
    }

    @Test
    fun testMaxNStability() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 5000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = shapiroWilkTest(data)
        assertTrue(result.statistic > 0.99, "n=5000 normal data should have W close to 1, got ${result.statistic}")
        assertTrue(result.pValue > 0.05, "n=5000 normal data should not reject, p=${result.pValue}")
    }

    @Test
    fun testTestName() {
        val result = shapiroWilkTest(doubleArrayOf(-1.0, 0.0, 1.0))
        assertEquals("Shapiro-Wilk Test", result.testName)
    }
}
