package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class DagostinoPearsonTest {

    private val tolK2 = 1e-10
    private val tolP = 1e-3

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN20NormalData() {
        // scipy: normaltest(data) → K²=1.68648681521306, p=0.430312578511656
        val result = dagostinoPearsonTest(TestData.NORMAL_N20)
        assertEquals(1.68648681521306, result.statistic, tolK2, "K² n=20 normal")
        assertEquals(0.430312578511656, result.pValue, tolP, "p-value n=20 normal")
        assertEquals("D'Agostino-Pearson Test", result.testName)
        assertEquals(2.0, result.degreesOfFreedom)
        assertTrue(result.additionalInfo.containsKey("z1"))
        assertTrue(result.additionalInfo.containsKey("z2"))
        assertTrue(result.additionalInfo.containsKey("skewness"))
        assertTrue(result.additionalInfo.containsKey("kurtosis"))
    }

    @Test
    fun testN20ZScores() {
        // scipy: skewtest → z1=-1.21774543808947, kurtosistest → z2=-0.451201577152995
        val result = dagostinoPearsonTest(TestData.NORMAL_N20)
        assertEquals(-1.21774543808947, result.additionalInfo["z1"]!!, 1e-8, "z1 n=20")
        assertEquals(-0.451201577152995, result.additionalInfo["z2"]!!, 1e-8, "z2 n=20")
        assertEquals(-0.557894825215587, result.additionalInfo["skewness"]!!, 1e-10, "skewness n=20")
        assertEquals(2.30196781519457, result.additionalInfo["kurtosis"]!!, 1e-10, "kurtosis n=20")
    }

    @Test
    fun testN50NormalData() {
        // scipy: normaltest(data) → K²=0.419895247354848, p=0.810626702503978
        val data = doubleArrayOf(
            0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254,
            -0.23415337472333597, -0.23413695694918055, 1.5792128155073915, 0.7674347291529088,
            -0.4694743859349521, 0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
            0.24196227156603412, -1.913280244657798, -1.7249178325130328, -0.5622875292409727,
            -1.0128311203344238, 0.3142473325952739, -0.9080240755212111, -1.4123037013352917,
            1.465648768921554, -0.22577630048653566, 0.06752820468792384, -1.4247481862134568,
            -0.5443827245251827, 0.11092258970986608, -1.1509935774223028, 0.37569801834567196,
            -0.600638689918805, -0.2916937497932768, -0.6017066122293969, 1.8522781845089378,
            -0.013497224737933914, -1.0577109289559, 0.822544912103189, -1.2208436499710222,
            0.2088635950047554, -1.9596701238797756, -1.3281860488984305, 0.19686123586912352,
            0.7384665799954104, 0.1713682811899705, -0.11564828238824053, -0.3011036955892888,
            -1.4785219903674274, -0.7198442083947086, -0.4606387709597875, 1.0571222262189157,
            0.3436182895684614, -1.763040155362734
        )
        val result = dagostinoPearsonTest(data)
        assertEquals(0.419895247354848, result.statistic, tolK2, "K² n=50 normal")
        assertEquals(0.810626702503978, result.pValue, tolP, "p-value n=50 normal")
    }

    @Test
    fun testN100NormalData() {
        // scipy: normaltest(data) → K²=5.6000294516926, p=0.0608091671521755
        val result = dagostinoPearsonTest(TestData.NORMAL_N100)
        assertEquals(5.6000294516926, result.statistic, tolK2, "K² n=100")
        assertEquals(0.0608091671521755, result.pValue, tolP, "p-value n=100")
        assertEquals(2.11475142289713, result.additionalInfo["z1"]!!, 1e-8, "z1 n=100")
        assertEquals(1.06200558899055, result.additionalInfo["z2"]!!, 1e-8, "z2 n=100")
    }

    @Test
    fun testN20Arange() {
        val data = DoubleArray(20) { (it + 1).toDouble() }
        val result = dagostinoPearsonTest(data)
        assertEquals(0.0, result.additionalInfo["z1"]!!, 1e-15, "z1 should be 0 for perfectly symmetric data")
        assertTrue(result.statistic > 0.0, "K² should be positive (kurtosis term)")
        assertTrue(result.pValue in 0.0..1.0)
    }

    // ===== Non-normal rejection =====

    @Test
    fun testUniformDataRejects() {
        // scipy: normaltest(data) → K²=12.936732957711, p=0.00155175849401569
        val result = dagostinoPearsonTest(TestData.UNIFORM_N30)
        assertEquals(12.936732957711, result.statistic, tolK2, "K² uniform n=30")
        assertTrue(result.pValue < 0.01, "Uniform data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testExponentialDataRejects() {
        // scipy: normaltest(data) → K²=15.4528145680744, p=0.000441025754481787
        val result = dagostinoPearsonTest(TestData.EXPONENTIAL_N30)
        assertEquals(15.4528145680744, result.statistic, tolK2, "K² exponential n=30")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject, p=${result.pValue}")
    }

    @Test
    fun testBimodalDataRejects() {
        val result = dagostinoPearsonTest(TestData.bimodal())
        assertTrue(result.statistic > 100.0, "Bimodal should have very large K², got ${result.statistic}")
        assertTrue(result.pValue < 1e-10, "Bimodal should have very small p-value, p=${result.pValue}")
    }

    @Test
    fun testHeavyTailedDataRejects() {
        // scipy: K²=68.0227645925203
        val result = dagostinoPearsonTest(TestData.HEAVY_TAILED_N50)
        assertEquals(68.0227645925203, result.statistic, tolK2, "K² heavy-tailed n=50")
        assertTrue(result.pValue < 1e-10, "Heavy-tailed should strongly reject, p=${result.pValue}")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        val data = DoubleArray(20) { (it + 1).toDouble() }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic >= 0.0, "K² should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> { dagostinoPearsonTest(DoubleArray(19) { it.toDouble() }) }
        assertFailsWith<InsufficientDataException> { dagostinoPearsonTest(DoubleArray(10) { it.toDouble() }) }
        assertFailsWith<InsufficientDataException> { dagostinoPearsonTest(doubleArrayOf()) }
    }

    @Test
    fun testConstantData() {
        val result = dagostinoPearsonTest(DoubleArray(20) { 5.0 })
        assertEquals(0.0, result.statistic, 1e-15)
        assertEquals(1.0, result.pValue, 1e-15)
        assertEquals(0.0, result.additionalInfo["z1"]!!, 1e-15)
        assertEquals(0.0, result.additionalInfo["z2"]!!, 1e-15)
        assertEquals(0.0, result.additionalInfo["skewness"]!!, 1e-15)
        assertEquals(0.0, result.additionalInfo["kurtosis"]!!, 1e-15)
    }

    @Test
    fun testNearConstantData() {
        val data = DoubleArray(20) { 1.0 }.also { it[19] = 1.0 + 1e-10 }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic >= 0.0, "K² should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    @Test
    fun testSymmetricData() {
        // scipy: normaltest(linspace(-5,5,30)) → K²=5.41918814715635, z1≈0
        val data = DoubleArray(30) { i -> -5.0 + 10.0 * i / 29.0 }
        val result = dagostinoPearsonTest(data)
        assertEquals(5.41918814715635, result.statistic, tolK2, "K² symmetric n=30")
        assertTrue(
            abs(result.additionalInfo["z1"]!!) < 1e-10,
            "z1 should be ~0 for symmetric data, got ${result.additionalInfo["z1"]}"
        )
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSampleN1000() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(1000) { i -> normal.quantile((i + 0.5) / 1000) }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic < 0.1, "Perfect normal n=1000 should have small K²")
        assertTrue(result.pValue > 0.9, "Perfect normal n=1000 should have large p-value")
    }

    @Test
    fun testLargeSampleN5000() {
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val data = DoubleArray(5000) { i -> normal.quantile((i + 0.5) / 5000) }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic < 0.1, "Perfect normal n=5000 should have small K²")
        assertTrue(result.pValue > 0.9, "Perfect normal n=5000 should have large p-value")
    }

    @Test
    fun testExtremeOutliers() {
        val data = DoubleArray(50) { i -> if (i < 49) i.toDouble() / 10.0 else 1000.0 }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.pValue < 0.001, "Data with extreme outlier should reject normality")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.NaN }
        val result = dagostinoPearsonTest(data)
        TestAssertions.assertNaNResult(result, "when input contains NaN")
    }

    @Test
    fun testPositiveInfinity() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.POSITIVE_INFINITY }
        val result = dagostinoPearsonTest(data)
        TestAssertions.assertNaNResult(result, "when input contains Infinity")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        val normalResult = dagostinoPearsonTest(TestData.NORMAL_N20)
        TestAssertions.assertIsSignificantConsistency(normalResult)

        val expResult = dagostinoPearsonTest(TestData.EXPONENTIAL_N30)
        TestAssertions.assertIsSignificantConsistency(expResult)
    }
}
