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

    private fun assertK2(expected: Double, actual: Double, message: String = "") {
        assertEquals(expected, actual, tolK2, "K² $message")
    }

    private fun assertP(expected: Double, actual: Double, message: String = "") {
        assertTrue(abs(expected - actual) < tolP, "p-value $message: expected=$expected, actual=$actual")
    }

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN20NormalData() {
        // scipy: normaltest(data) → K²=1.68648681521306, p=0.430312578511656
        val data = doubleArrayOf(
            0.3047170797544314, -1.039984106240496, 0.7504511958064572, 0.9405647163912139,
            -1.951035188653836, -1.302179506862318, 0.1278404031672854, -0.3162425923435822,
            -0.0168011575042888, -0.8530439275735801, 0.8793979748628286, 0.7777919354289483,
            0.06603069756121605, 1.127241206968033, 0.4675093422520456, -0.8592924628832382,
            0.3687507840824988, -0.9588826008289989, 0.8784503013072725, -0.0499259109862529
        )
        val result = dagostinoPearsonTest(data)
        assertK2(1.68648681521306, result.statistic, "n=20 normal")
        assertP(0.430312578511656, result.pValue, "n=20 normal")
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
        val data = doubleArrayOf(
            0.3047170797544314, -1.039984106240496, 0.7504511958064572, 0.9405647163912139,
            -1.951035188653836, -1.302179506862318, 0.1278404031672854, -0.3162425923435822,
            -0.0168011575042888, -0.8530439275735801, 0.8793979748628286, 0.7777919354289483,
            0.06603069756121605, 1.127241206968033, 0.4675093422520456, -0.8592924628832382,
            0.3687507840824988, -0.9588826008289989, 0.8784503013072725, -0.0499259109862529
        )
        val result = dagostinoPearsonTest(data)
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
        assertK2(0.419895247354848, result.statistic, "n=50 normal")
        assertP(0.810626702503978, result.pValue, "n=50 normal")
    }

    @Test
    fun testN100NormalData() {
        // scipy: normaltest(data) → K²=5.6000294516926, p=0.0608091671521755
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
        val result = dagostinoPearsonTest(data)
        assertK2(5.6000294516926, result.statistic, "n=100")
        assertP(0.0608091671521755, result.pValue, "n=100")
        assertEquals(2.11475142289713, result.additionalInfo["z1"]!!, 1e-8, "z1 n=100")
        assertEquals(1.06200558899055, result.additionalInfo["z2"]!!, 1e-8, "z2 n=100")
    }

    @Test
    fun testN20Arange() {
        // Perfectly symmetric data: z1 should be exactly 0, K² = z2² only.
        // Note: scipy's skewtest replaces y=0 with y=1 (a workaround),
        // giving a different K². Our implementation is mathematically correct.
        val data = DoubleArray(20) { (it + 1).toDouble() }
        val result = dagostinoPearsonTest(data)
        assertEquals(0.0, result.additionalInfo["z1"]!!, 1e-15,
            "z1 should be 0 for perfectly symmetric data")
        assertTrue(result.statistic > 0.0, "K² should be positive (kurtosis term)")
        assertTrue(result.pValue in 0.0..1.0)
    }

    // ===== Non-normal rejection =====

    @Test
    fun testUniformDataRejects() {
        // scipy: normaltest(data) → K²=12.936732957711, p=0.00155175849401569
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
        val result = dagostinoPearsonTest(data)
        assertK2(12.936732957711, result.statistic, "uniform n=30")
        assertTrue(result.pValue < 0.01, "Uniform data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testExponentialDataRejects() {
        // scipy: normaltest(data) → K²=15.4528145680744, p=0.000441025754481787
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
        val result = dagostinoPearsonTest(data)
        assertK2(15.4528145680744, result.statistic, "exponential n=30")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject, p=${result.pValue}")
    }

    @Test
    fun testBimodalDataRejects() {
        // Bimodal: two clusters far apart
        val data = DoubleArray(50) { i ->
            if (i < 25) -10.0 + i * 0.01 else 10.0 + (i - 25) * 0.01
        }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic > 100.0, "Bimodal should have very large K², got ${result.statistic}")
        assertTrue(result.pValue < 1e-10, "Bimodal should have very small p-value, p=${result.pValue}")
    }

    @Test
    fun testHeavyTailedDataRejects() {
        // Cauchy-like heavy tails (scipy seed=123)
        val data = doubleArrayOf(
            -1.0885201381521286, -0.18786396542275827, -0.3503617844047363, 5.657747242986373,
            -1.4605714183822414, 7.168129473296525, -2.334301088215893, 1.0221725699178252,
            1.0087543970798958, 2.599920401195345, 0.49463522750693306, -0.7958927979839227,
            1.9660959839771013, -0.6349250715410574, 0.16253893173294215, 0.0913386570510911,
            2.5312054985294687, -5.341427619339797, 0.00413516707731555, -3.1010282479224323,
            0.4661577464939563, -0.6812405019995409, -28.620025275019614, 5.793884258939474,
            0.4372981788530975, 1.2457643535499165, -2.1849312693396534, 0.027759082583709824,
            0.5075579262283797, 1.3985920975547956, -0.9721285771194503, -0.236960098863599,
            1.755741948940382, -0.19515928787527004, -6.00579731221582, -0.5636002049270417,
            -1.059242455464236, -2.041978485161492, 3.2005662567267983, 0.5215044478683457,
            -1.8221714203395658, 2.0573958204783858, 0.28985859722828494, 0.604080836015793,
            12.69215364462557, -0.9075659312224382, 0.1536980894213486, -0.32490162370852954,
            0.7954459455251501, -1.000591222146589
        )
        val result = dagostinoPearsonTest(data)
        // scipy: K²=68.0227645925203
        assertK2(68.0227645925203, result.statistic, "heavy-tailed n=50")
        assertTrue(result.pValue < 1e-10, "Heavy-tailed should strongly reject, p=${result.pValue}")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        // n=20: minimum valid input
        val data = DoubleArray(20) { (it + 1).toDouble() }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic >= 0.0, "K² should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    @Test
    fun testInsufficientData() {
        assertFailsWith<InsufficientDataException> {
            dagostinoPearsonTest(DoubleArray(19) { it.toDouble() })
        }
        assertFailsWith<InsufficientDataException> {
            dagostinoPearsonTest(DoubleArray(10) { it.toDouble() })
        }
        assertFailsWith<InsufficientDataException> {
            dagostinoPearsonTest(doubleArrayOf())
        }
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
        val data = DoubleArray(20) { 1.0 } .also { it[19] = 1.0 + 1e-10 }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic >= 0.0, "K² should be non-negative")
        assertTrue(result.pValue in 0.0..1.0, "p-value should be in [0, 1]")
    }

    @Test
    fun testSymmetricData() {
        // scipy: normaltest(linspace(-5,5,30)) → K²=5.41918814715635, z1≈0
        val data = DoubleArray(30) { i -> -5.0 + 10.0 * i / 29.0 }
        val result = dagostinoPearsonTest(data)
        assertK2(5.41918814715635, result.statistic, "symmetric n=30")
        // z1 should be nearly zero for symmetric data
        assertTrue(abs(result.additionalInfo["z1"]!!) < 1e-10,
            "z1 should be ~0 for symmetric data, got ${result.additionalInfo["z1"]}")
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSampleN1000() {
        // n=1000: perfectly normal data by construction
        // scipy: K²=0.00476512819487289, p=0.997620271955596
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 1000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic < 0.1,
            "Perfect normal n=1000 should have small K², got ${result.statistic}")
        assertTrue(result.pValue > 0.9,
            "Perfect normal n=1000 should have large p-value, p=${result.pValue}")
    }

    @Test
    fun testLargeSampleN5000() {
        // n=5000: perfectly normal data
        // scipy: K²=0.00275709007819252, p=0.998622404717639
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 5000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic < 0.1,
            "Perfect normal n=5000 should have small K², got ${result.statistic}")
        assertTrue(result.pValue > 0.9,
            "Perfect normal n=5000 should have large p-value, p=${result.pValue}")
    }

    @Test
    fun testExtremeOutliers() {
        // Normal data with extreme outlier → should detect non-normality
        val data = DoubleArray(50) { i -> if (i < 49) i.toDouble() / 10.0 else 1000.0 }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.pValue < 0.001,
            "Data with extreme outlier should reject normality, p=${result.pValue}")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.NaN }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic.isNaN(), "K² should be NaN when input contains NaN")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN when input contains NaN")
    }

    @Test
    fun testPositiveInfinity() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.POSITIVE_INFINITY }
        val result = dagostinoPearsonTest(data)
        assertTrue(result.statistic.isNaN(), "K² should be NaN when input contains Infinity")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN when input contains Infinity")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        // Normal data: should not be significant
        val normalResult = dagostinoPearsonTest(
            doubleArrayOf(
                0.3047170797544314, -1.039984106240496, 0.7504511958064572, 0.9405647163912139,
                -1.951035188653836, -1.302179506862318, 0.1278404031672854, -0.3162425923435822,
                -0.0168011575042888, -0.8530439275735801, 0.8793979748628286, 0.7777919354289483,
                0.06603069756121605, 1.127241206968033, 0.4675093422520456, -0.8592924628832382,
                0.3687507840824988, -0.9588826008289989, 0.8784503013072725, -0.0499259109862529
            )
        )
        assertEquals(normalResult.pValue < 0.05, normalResult.isSignificant())

        // Exponential data: should be significant
        val expResult = dagostinoPearsonTest(
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
