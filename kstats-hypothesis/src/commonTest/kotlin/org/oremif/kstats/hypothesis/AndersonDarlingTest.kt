package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class AndersonDarlingTest {

    private val tolA2 = 1e-10
    private val tolP = 1e-3

    private fun assertA2(expected: Double, actual: Double, message: String = "") {
        assertEquals(expected, actual, tolA2, "A² $message")
    }

    private fun assertP(expected: Double, actual: Double, message: String = "") {
        assertTrue(abs(expected - actual) < tolP, "p-value $message: expected=$expected, actual=$actual")
    }

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN3Symmetric() {
        // scipy: stats.anderson([-0.5, 0.0, 0.5], dist='norm') → A²=0.189488054537566
        val result = andersonDarlingTest(doubleArrayOf(-0.5, 0.0, 0.5))
        assertA2(0.189488054537566, result.statistic, "n=3 symmetric")
        assertEquals("Anderson-Darling Test", result.testName)
        assertTrue(result.pValue > 0.05, "Symmetric n=3 should not reject normality")
    }

    @Test
    fun testN7() {
        // scipy: stats.anderson([-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5]) → A²=0.138669788736141
        val result = andersonDarlingTest(doubleArrayOf(-1.2, -0.5, 0.1, 0.3, 0.7, 1.0, 1.5))
        assertA2(0.138669788736141, result.statistic, "n=7")
        assertTrue(result.pValue > 0.05)
    }

    @Test
    fun testN10Normal() {
        // scipy: stats.anderson(data, dist='norm') → A²=0.104221649214109
        val result = andersonDarlingTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.5)
        )
        assertA2(0.104221649214109, result.statistic, "n=10")
        // A²* = 0.114383260012485, p ≈ 0.992
        assertTrue(result.pValue > 0.15, "Normal data should not reject")
    }

    @Test
    fun testN20Normal() {
        // scipy: stats.anderson(data) → A²=0.490035999877421
        val data = doubleArrayOf(
            0.3047170797544314, -1.039984106240496, 0.7504511958064572, 0.9405647163912139,
            -1.951035188653836, -1.302179506862318, 0.1278404031672854, -0.3162425923435822,
            -0.0168011575042888, -0.8530439275735801, 0.8793979748628286, 0.7777919354289483,
            0.06603069756121605, 1.127241206968033, 0.4675093422520456, -0.8592924628832382,
            0.3687507840824988, -0.9588826008289989, 0.8784503013072725, -0.0499259109862529
        )
        val result = andersonDarlingTest(data)
        assertA2(0.490035999877421, result.statistic, "n=20")
        // A²* = 0.511168802372134, p ≈ 0.196
        assertP(0.196, result.pValue, "n=20")
    }

    @Test
    fun testN100Normal() {
        // scipy: stats.anderson(data) → A²=0.366112469096564
        val data = doubleArrayOf(
            -1.275686323337922, -1.133287214003481, -0.9194522860016113, 0.497160744053764,
            0.1424257360705652, 0.6904853540677682, -0.4272526463365343, 0.1585396910767142,
            0.6255903939673367, -0.3093465397202384, 0.4567752375574115, -0.6619259410666513,
            -0.3630538465650718, -0.3817378939983291, -1.19583964558904, 0.4869724807855818,
            -0.4694023402027239, 0.01249411872768743, 0.480746658905909, 0.4465311760299441,
            0.6653851089727862, -0.09848548450942361, -0.4232983120441537, -0.07971821090639905,
            -1.68733443395803, -1.447112472423087, -1.322699612354402, -0.9972468276014818,
            0.3997742267234366, -0.9054790553600608, -0.3781625540393897, 1.299228297786065,
            -0.3562639710614259, 0.7375155684670865, -0.933617680009877, -0.20543755786763,
            -0.9500220549105812, -0.3390330759005625, 0.8403081374573955, -1.727320423192349,
            0.4344236435458573, 0.2377356023322779, -0.5941499556967944, -1.446057854388455,
            0.07212950771386951, -0.5294927090638024, 0.232676211354704, 0.02185214552344288,
            1.601778891320915, -0.2393556274730243, -1.023497492621865, 0.1792756349563162,
            0.2199966839717652, 1.359187575240437, 0.8351112459145785, 0.3568710591495093,
            1.463302891219562, -1.188763054322851, -0.6397515327497477, -0.9265759414055249,
            -0.389809803155768, -1.376686147556309, 0.6351509468144043, -0.2222226970987734,
            -1.470806294502658, -1.015579081207542, 0.3135138474501953, 0.8381265678943811,
            1.996730891691787, 2.91386246600733, 0.4144094332759964, -0.9895381200318641,
            -2.132046280731309, 0.2677114623438358, -0.812941095310326, -0.4153572601796853,
            -0.6120967990598081, -0.1407908864163853, 1.065980230787644, 0.1570485674453446,
            -0.1586348370386883, -1.035653752825812, -1.674682944704357, -0.4863079090733309,
            -0.05378255081832049, 1.767929913579883, 0.1302745214728858, 0.9827395110230576,
            -0.4992955985391521, -1.184943766417025, -0.9651167622323719, -0.7252260645357532,
            2.128469732435164, -0.8213866792243861, 0.8384892037363449, -0.9029271780870264,
            0.9315730128742441, 0.3849509661058632, -0.1566378976580904, -0.04076252613543403
        )
        val result = andersonDarlingTest(data)
        assertA2(0.366112469096564, result.statistic, "n=100")
        // A²* = 0.368940687920335, p ≈ 0.428
        assertP(0.428, result.pValue, "n=100")
    }

    // ===== Non-normal data: should reject =====

    @Test
    fun testUniformDataRejects() {
        // scipy: stats.anderson(data) → A²=0.971334561238429
        val data = doubleArrayOf(
            0.6823518632481435, 0.05382101880222268, 0.2203598727726114, 0.1843718106986697,
            0.1759059010850303, 0.8120945066557737, 0.9233449980270564, 0.2765743977971062,
            0.8197545615930021, 0.8898926931111859, 0.5129704552295319, 0.2449646010687965,
            0.8242415960974113, 0.2137629633750955, 0.7414670522347097, 0.6299402045896808,
            0.927407258525167, 0.2319081886064188, 0.7991251286200829, 0.5181650368527142,
            0.2315556248170675, 0.1659039932407446, 0.4977889684977939, 0.5827246406153199,
            0.1843379874284797, 0.01489491676023225, 0.4711332288904608, 0.7282433281832617,
            0.9186004917735433, 0.625534005735464
        )
        val result = andersonDarlingTest(data)
        assertA2(0.971334561238429, result.statistic, "n=30 uniform")
        assertTrue(result.pValue < 0.05, "Uniform data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testExponentialDataRejects() {
        // scipy: stats.anderson(data) → A²=2.26816288595751
        val data = doubleArrayOf(
            3.597929172389289, 0.63378567826403, 0.09064145483270467, 2.238734846667239,
            2.636866342451776, 0.03815228838838569, 0.4259543269460497, 1.026087708008696,
            0.1637386675711129, 0.3863021750023226, 0.2105797153156188, 0.9029134845795763,
            0.09197889116405482, 1.751339800824722, 0.1406342848609949, 0.2257405162277304,
            0.03079137602427886, 0.04770886720922435, 0.6161317404952943, 1.025810024012882,
            0.3895691313310642, 0.2446625326364912, 0.0008069149770931415, 0.6754925711777729,
            0.7653102097560376, 0.9929132465597339, 3.459278580268261, 1.476058283448357,
            0.1105616357433793, 0.9706210002233822
        )
        val result = andersonDarlingTest(data)
        assertA2(2.26816288595751, result.statistic, "n=30 exponential")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject normality, p=${result.pValue}")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        // n=3: minimum valid input
        // scipy: stats.anderson([1.0, 2.0, 3.0]) → A²=0.189488054537566
        val result = andersonDarlingTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertA2(0.189488054537566, result.statistic, "n=3 minimum")
        assertTrue(result.pValue > 0.0 && result.pValue <= 1.0)
    }

    @Test
    fun testModifiedStatisticInAdditionalInfo() {
        val result = andersonDarlingTest(doubleArrayOf(-1.0, 0.0, 1.0, 2.0, 3.0))
        val a2Star = result.additionalInfo["modifiedStatistic"]
        assertTrue(a2Star != null && a2Star > result.statistic,
            "Modified A²* should be larger than raw A²")
    }

    // ===== Degenerate input =====

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> {
            andersonDarlingTest(doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<InsufficientDataException> {
            andersonDarlingTest(doubleArrayOf(1.0))
        }
        assertFailsWith<InsufficientDataException> {
            andersonDarlingTest(doubleArrayOf())
        }
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
        assertTrue(result.pValue >= 0.0 && result.pValue <= 1.0,
            "p-value should be in [0, 1]")
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSample() {
        // n=1000: perfectly normal data by construction
        // scipy: A²=0.00153939858319063
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 1000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic < 0.01,
            "Perfectly normal data should have very small A², got ${result.statistic}")
        assertTrue(result.pValue > 0.05,
            "Perfectly normal data should not reject normality, p=${result.pValue}")
    }

    @Test
    fun testHighlySignificant() {
        // Bimodal data: clearly non-normal
        val data = DoubleArray(50) { i -> if (i < 25) -10.0 + i * 0.01 else 10.0 + (i - 25) * 0.01 }
        val result = andersonDarlingTest(data)
        assertTrue(result.pValue < 0.001,
            "Bimodal data should have very small p-value, got ${result.pValue}")
    }

    @Test
    fun testVeryLargeN() {
        // n=5000: stress test for numerical stability
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 5000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic < 0.01,
            "n=5000 perfect normal should have tiny A², got ${result.statistic}")
        assertTrue(result.pValue > 0.05,
            "n=5000 perfect normal should not reject, p=${result.pValue}")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val data = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val result = andersonDarlingTest(data)
        assertTrue(result.statistic.isNaN(), "A² should be NaN when input contains NaN")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN when input contains NaN")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        // Normal data: should not be significant
        val normalResult = andersonDarlingTest(
            doubleArrayOf(-1.5, -1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.5)
        )
        assertEquals(normalResult.pValue < 0.05, normalResult.isSignificant())

        // Exponential data: should be significant
        val expResult = andersonDarlingTest(
            doubleArrayOf(
                3.597929172389289, 0.63378567826403, 0.09064145483270467, 2.238734846667239,
                2.636866342451776, 0.03815228838838569, 0.4259543269460497, 1.026087708008696,
                0.1637386675711129, 0.3863021750023226, 0.2105797153156188, 0.9029134845795763,
                0.09197889116405482, 1.751339800824722, 0.1406342848609949, 0.2257405162277304,
                0.03079137602427886, 0.04770886720922435, 0.6161317404952943, 1.025810024012882,
                0.3895691313310642, 0.2446625326364912, 0.0008069149770931415, 0.6754925711777729,
                0.7653102097560376, 0.9929132465597339, 3.459278580268261, 1.476058283448357,
                0.1105616357433793, 0.9706210002233822
            )
        )
        assertEquals(expResult.pValue < 0.05, expResult.isSignificant())
    }
}
