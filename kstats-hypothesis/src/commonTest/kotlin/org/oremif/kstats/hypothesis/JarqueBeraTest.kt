package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class JarqueBeraTest {

    private val tolJB = 1e-10
    private val tolP = 1e-3

    private fun assertJB(expected: Double, actual: Double, message: String = "") {
        assertEquals(expected, actual, tolJB, "JB $message")
    }

    private fun assertP(expected: Double, actual: Double, message: String = "") {
        assertTrue(abs(expected - actual) < tolP, "p-value $message: expected=$expected, actual=$actual")
    }

    // ===== Basic correctness: scipy reference values =====

    @Test
    fun testN10NormalData() {
        // scipy: jarque_bera(data) → JB=0.539772352201391, p=0.763466390112862
        val data = doubleArrayOf(
            0.3047170797544314, -1.039984106240496, 0.7504511958064572, 0.9405647163912139,
            -1.951035188653836, -1.302179506862318, 0.1278404031672854, -0.3162425923435822,
            -0.0168011575042888, -0.8530439275735801
        )
        val result = jarqueBeraTest(data)
        assertJB(0.539772352201391, result.statistic, "n=10 normal")
        assertP(0.763466390112862, result.pValue, "n=10 normal")
        assertEquals("Jarque-Bera Test", result.testName)
        assertEquals(2.0, result.degreesOfFreedom)
        assertTrue(result.additionalInfo.containsKey("skewness"))
        assertTrue(result.additionalInfo.containsKey("kurtosis"))
    }

    @Test
    fun testN30NormalData() {
        // scipy: jarque_bera(data) → JB=0.274192655780974, p=0.871886235125062
        val data = doubleArrayOf(
            0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254,
            -0.23415337472333597, -0.23413695694918055, 1.5792128155073915, 0.7674347291529088,
            -0.4694743859349521, 0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
            0.24196227156603412, -1.913280244657798, -1.7249178325130328, -0.5622875292409727,
            -1.0128311203344238, 0.3142473325952739, -0.9080240755212111, -1.4123037013352917,
            1.465648768921554, -0.22577630048653566, 0.06752820468792384, -1.4247481862134568,
            -0.5443827245251827, 0.11092258970986608, -1.1509935774223028, 0.37569801834567196,
            -0.600638689918805, -0.2916937497932768
        )
        val result = jarqueBeraTest(data)
        assertJB(0.274192655780974, result.statistic, "n=30 normal")
        assertP(0.871886235125062, result.pValue, "n=30 normal")
    }

    @Test
    fun testN100NormalData() {
        // scipy: jarque_bera(data) → JB=4.9688973327694, p=0.0833715073578094
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
        val result = jarqueBeraTest(data)
        assertJB(4.9688973327694, result.statistic, "n=100 normal")
        assertP(0.0833715073578094, result.pValue, "n=100 normal")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        // scipy: jarque_bera([1,2,3]) → JB=0.28125, p=0.868815056262843
        val data = doubleArrayOf(1.0, 2.0, 3.0)
        val result = jarqueBeraTest(data)
        assertJB(0.28125, result.statistic, "n=3 arange")
        assertP(0.868815056262843, result.pValue, "n=3 arange")
    }

    @Test
    fun testSymmetricData() {
        // scipy: jarque_bera(linspace(-5,5,30)) → JB=1.80801780745136, p=0.404943023808853
        val data = DoubleArray(30) { i -> -5.0 + 10.0 * i / 29.0 }
        val result = jarqueBeraTest(data)
        assertJB(1.80801780745136, result.statistic, "symmetric n=30")
        assertP(0.404943023808853, result.pValue, "symmetric n=30")
        // Skewness should be zero for perfectly symmetric data
        assertEquals(
            0.0, result.additionalInfo["skewness"]!!, 1e-15,
            "Skewness should be 0 for symmetric data"
        )
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
        assertFailsWith<InsufficientDataException> {
            jarqueBeraTest(doubleArrayOf(1.0, 2.0))
        }
        assertFailsWith<InsufficientDataException> {
            jarqueBeraTest(doubleArrayOf(1.0))
        }
        assertFailsWith<InsufficientDataException> {
            jarqueBeraTest(doubleArrayOf())
        }
    }

    // ===== Non-normal rejection =====

    @Test
    fun testExponentialDataRejects() {
        // scipy: jarque_bera(data) → JB=16.0920762731418, p=0.000320368668398256
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
        val result = jarqueBeraTest(data)
        assertJB(16.0920762731418, result.statistic, "exponential n=30")
        assertTrue(result.pValue < 0.001, "Exponential data should strongly reject, p=${result.pValue}")
    }

    @Test
    fun testUniformDataRejects() {
        // scipy: jarque_bera(data) → JB=2.60053766842867, p=0.272458537010854
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
        val result = jarqueBeraTest(data)
        assertJB(2.60053766842867, result.statistic, "uniform n=30")
        assertP(0.272458537010854, result.pValue, "uniform n=30")
    }

    @Test
    fun testBimodalDataRejects() {
        // scipy: jarque_bera(bimodal) → JB=8.33160025243092, p=0.0155172941378467
        val data = DoubleArray(50) { i ->
            if (i < 25) -10.0 + i * 0.01 else 10.0 + (i - 25) * 0.01
        }
        val result = jarqueBeraTest(data)
        assertJB(8.33160025243092, result.statistic, "bimodal n=50")
        assertTrue(result.pValue < 0.05, "Bimodal data should reject normality, p=${result.pValue}")
    }

    @Test
    fun testHeavyTailedDataRejects() {
        // scipy: jarque_bera(data) → JB=884.694920553516, p≈0
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
        val result = jarqueBeraTest(data)
        assertJB(884.694920553516, result.statistic, "heavy-tailed n=50")
        assertTrue(result.pValue < 1e-10, "Heavy-tailed should strongly reject, p=${result.pValue}")
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeSampleN1000() {
        // scipy: jarque_bera → JB=0.0319800907116113, p=0.984137116741359
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 1000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = jarqueBeraTest(data)
        assertTrue(
            result.statistic < 0.1,
            "Perfect normal n=1000 should have small JB, got ${result.statistic}"
        )
        assertTrue(
            result.pValue > 0.9,
            "Perfect normal n=1000 should have large p-value, p=${result.pValue}"
        )
    }

    @Test
    fun testLargeSampleN5000() {
        // scipy: jarque_bera → JB=0.0107506081455057, p=0.994639117023355
        val normal = org.oremif.kstats.distributions.NormalDistribution.STANDARD
        val n = 5000
        val data = DoubleArray(n) { i -> normal.quantile((i + 0.5) / n) }
        val result = jarqueBeraTest(data)
        assertTrue(
            result.statistic < 0.1,
            "Perfect normal n=5000 should have small JB, got ${result.statistic}"
        )
        assertTrue(
            result.pValue > 0.9,
            "Perfect normal n=5000 should have large p-value, p=${result.pValue}"
        )
    }

    @Test
    fun testExtremeOutliers() {
        // scipy: JB=4612.4836358838, p≈0
        val data = DoubleArray(50) { i -> if (i < 49) i.toDouble() / 10.0 else 1000.0 }
        val result = jarqueBeraTest(data)
        assertJB(4612.4836358838, result.statistic, "extreme outlier n=50")
        assertTrue(
            result.pValue < 1e-10,
            "Data with extreme outlier should reject normality, p=${result.pValue}"
        )
    }

    // ===== Non-finite input =====

    @Test
    fun testNaN() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.NaN }
        val result = jarqueBeraTest(data)
        assertTrue(result.statistic.isNaN(), "JB should be NaN when input contains NaN")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN when input contains NaN")
    }

    @Test
    fun testPositiveInfinity() {
        val data = DoubleArray(20) { it.toDouble() }.also { it[10] = Double.POSITIVE_INFINITY }
        val result = jarqueBeraTest(data)
        assertTrue(result.statistic.isNaN(), "JB should be NaN when input contains Infinity")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN when input contains Infinity")
    }

    // ===== isSignificant =====

    @Test
    fun testIsSignificant() {
        // Normal data: should not be significant
        val normalResult = jarqueBeraTest(
            doubleArrayOf(
                0.4967141530112327, -0.13826430117118466, 0.6476885381006925, 1.5230298564080254,
                -0.23415337472333597, -0.23413695694918055, 1.5792128155073915, 0.7674347291529088,
                -0.4694743859349521, 0.5425600435859647, -0.46341769281246226, -0.46572975357025687,
                0.24196227156603412, -1.913280244657798, -1.7249178325130328, -0.5622875292409727,
                -1.0128311203344238, 0.3142473325952739, -0.9080240755212111, -1.4123037013352917,
                1.465648768921554, -0.22577630048653566, 0.06752820468792384, -1.4247481862134568,
                -0.5443827245251827, 0.11092258970986608, -1.1509935774223028, 0.37569801834567196,
                -0.600638689918805, -0.2916937497932768
            )
        )
        assertEquals(normalResult.pValue < 0.05, normalResult.isSignificant())

        // Exponential data: should be significant
        val expResult = jarqueBeraTest(
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

    @Test
    fun testTestName() {
        val result = jarqueBeraTest(doubleArrayOf(1.0, 2.0, 3.0))
        assertEquals("Jarque-Bera Test", result.testName)
    }
}
