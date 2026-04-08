package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ProcessCapabilityTest {

    private val tol = 1e-10

    // ===== Basic correctness: known values =====

    @Test
    fun testProcessCapabilityKnownValues() {
        // numpy: d=np.array([2,4,4,4,5,5,7,9]); lsl=1; usl=10
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = data.processCapability(lsl = 1.0, usl = 10.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(0.701560760020114, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(0.623609564462324, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(0.75, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(0.666666666666667, result.ppk, tol, "ppk")
    }

    @Test
    fun testProcessCapabilityCenteredProcess() {
        // numpy: d=np.array([50.0,50.5,49.5,50.2,49.8,50.1,49.9,50.3,49.7,50.0]); lsl=48; usl=52
        val data = doubleArrayOf(50.0, 50.5, 49.5, 50.2, 49.8, 50.1, 49.9, 50.3, 49.7, 50.0)
        val result = data.processCapability(lsl = 48.0, usl = 52.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(2.26455406828919, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(2.26455406828918, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(2.38704958013144, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(2.38704958013144, result.ppk, tol, "ppk")
    }

    @Test
    fun testProcessCapabilityOffCenter() {
        // numpy: d=np.array([7.0,7.5,8.0,7.2,7.8,8.1,7.6,7.3,7.9,8.2]); lsl=5; usl=10
        val data = doubleArrayOf(7.0, 7.5, 8.0, 7.2, 7.8, 8.1, 7.6, 7.3, 7.9, 8.2)
        val result = data.processCapability(lsl = 5.0, usl = 10.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(2.05221594782225, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(1.92087412716162, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(2.16322554854652, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(2.02477911343954, result.ppk, tol, "ppk")
    }

    @Test
    fun testProcessCapabilityNegativeValues() {
        // numpy: d=np.array([-5,-3,-4,-2,-6]); lsl=-10; usl=0
        val data = doubleArrayOf(-5.0, -3.0, -4.0, -2.0, -6.0)
        val result = data.processCapability(lsl = -10.0, usl = 0.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(1.05409255338946, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(0.843274042711568, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(1.17851130197758, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(0.942809041582063, result.ppk, tol, "ppk")
    }

    @Test
    fun testProcessCapabilityMeanAtLsl() {
        // numpy: d=np.array([0.0,1.0,-1.0,0.5,-0.5]); lsl=0; usl=10
        // mean = 0 which equals lsl => cpk and ppk should be 0
        val data = doubleArrayOf(0.0, 1.0, -1.0, 0.5, -0.5)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(2.10818510677892, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(0.0, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(2.35702260395516, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(0.0, result.ppk, tol, "ppk")
    }

    // ===== Edge cases =====

    @Test
    fun testTwoElements() {
        // numpy: d=np.array([3,5]); lsl=0; usl=10
        val data = doubleArrayOf(3.0, 5.0)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(1.17851130197758, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(0.942809041582063, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(1.66666666666667, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(1.33333333333333, result.ppk, tol, "ppk")
    }

    @Test
    fun testMeanExactlyAtMidpoint() {
        // When mean = (lsl + usl) / 2, Cp should equal Cpk and Pp should equal Ppk
        // numpy: d=np.array([4.0,5.0,6.0,4.5,5.5,5.0]); lsl=0; usl=10
        val data = doubleArrayOf(4.0, 5.0, 6.0, 4.5, 5.5, 5.0)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(2.35702260395516, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(2.35702260395516, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(2.58198889747161, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(2.58198889747161, result.ppk, tol, "ppk")
    }

    // ===== Degenerate input =====

    @Test
    fun testEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf().processCapability(lsl = 0.0, usl = 10.0)
        }
    }

    @Test
    fun testSingleElement() {
        assertFailsWith<InsufficientDataException> {
            doubleArrayOf(5.0).processCapability(lsl = 0.0, usl = 10.0)
        }
    }

    @Test
    fun testConstantData() {
        // All values identical => standard deviation = 0 => DegenerateDataException
        assertFailsWith<DegenerateDataException> {
            doubleArrayOf(5.0, 5.0, 5.0, 5.0).processCapability(lsl = 0.0, usl = 10.0)
        }
    }

    @Test
    fun testLslEqualsUsl() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0, 3.0).processCapability(lsl = 5.0, usl = 5.0)
        }
    }

    @Test
    fun testLslGreaterThanUsl() {
        assertFailsWith<InvalidParameterException> {
            doubleArrayOf(1.0, 2.0, 3.0).processCapability(lsl = 10.0, usl = 5.0)
        }
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeOffsetData() {
        // Tests Welford algorithm numerical stability with large-offset data
        // numpy: d=np.array([1e15+1,1e15+2,1e15+3,1e15+4,1e15+5]); lsl=1e15; usl=1e15+6
        val data = doubleArrayOf(1e15 + 1.0, 1e15 + 2.0, 1e15 + 3.0, 1e15 + 4.0, 1e15 + 5.0)
        val result = data.processCapability(lsl = 1e15, usl = 1e15 + 6.0)

        // numpy: (usl-lsl)/(6*np.std(d,ddof=1))
        assertEquals(0.632455532033676, result.cp, tol, "cp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=1)), (np.mean(d)-lsl)/(3*np.std(d,ddof=1)))
        assertEquals(0.632455532033676, result.cpk, tol, "cpk")
        // numpy: (usl-lsl)/(6*np.std(d,ddof=0))
        assertEquals(0.707106781186547, result.pp, tol, "pp")
        // numpy: min((usl-np.mean(d))/(3*np.std(d,ddof=0)), (np.mean(d)-lsl)/(3*np.std(d,ddof=0)))
        assertEquals(0.707106781186547, result.ppk, tol, "ppk")
    }

    @Test
    fun testVerySmallVariance() {
        // Data clustered extremely tightly => very large capability indices
        val data = doubleArrayOf(1.0, 1.0 + 1e-10, 1.0 - 1e-10)
        val result = data.processCapability(lsl = 0.0, usl = 2.0)

        assertTrue(result.cp.isFinite(), "cp should be finite for very small variance")
        assertTrue(result.cpk.isFinite(), "cpk should be finite for very small variance")
        assertTrue(result.pp.isFinite(), "pp should be finite for very small variance")
        assertTrue(result.ppk.isFinite(), "ppk should be finite for very small variance")

        // All should be very large positive numbers
        assertTrue(result.cp > 1e6, "cp should be very large for tiny variance")
        assertTrue(result.cpk > 1e6, "cpk should be very large for tiny variance")
    }

    @Test
    fun testWideToleranceBand() {
        // Very wide tolerance compared to data spread
        val data = doubleArrayOf(5.0, 5.1, 4.9, 5.05, 4.95)
        val result = data.processCapability(lsl = -1000.0, usl = 1000.0)

        assertTrue(result.cp > 100.0, "cp should be very large for wide tolerance")
        assertTrue(result.cpk > 100.0, "cpk should be very large for wide tolerance")
    }

    // ===== Non-finite input =====

    @Test
    fun testNaNPropagation() {
        // NaN in data should propagate through Welford to produce NaN results
        val data = doubleArrayOf(1.0, Double.NaN, 3.0)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        assertTrue(result.cp.isNaN(), "cp should be NaN when data contains NaN")
        assertTrue(result.cpk.isNaN(), "cpk should be NaN when data contains NaN")
        assertTrue(result.pp.isNaN(), "pp should be NaN when data contains NaN")
        assertTrue(result.ppk.isNaN(), "ppk should be NaN when data contains NaN")
    }

    @Test
    fun testNaNLsl() {
        // NaN for lsl: NaN >= usl is false so validation passes, results should be NaN
        val data = doubleArrayOf(1.0, 2.0, 3.0)
        val result = data.processCapability(lsl = Double.NaN, usl = 10.0)

        assertTrue(result.cp.isNaN(), "cp should be NaN when lsl is NaN")
        assertTrue(result.cpk.isNaN(), "cpk should be NaN when lsl is NaN")
    }

    @Test
    fun testNaNUsl() {
        // NaN for usl: lsl >= NaN is false so validation passes, results should be NaN
        val data = doubleArrayOf(1.0, 2.0, 3.0)
        val result = data.processCapability(lsl = 0.0, usl = Double.NaN)

        assertTrue(result.cp.isNaN(), "cp should be NaN when usl is NaN")
        assertTrue(result.cpk.isNaN(), "cpk should be NaN when usl is NaN")
    }

    @Test
    fun testInfinityInData() {
        // Infinity in data: mean becomes Infinity, sigma becomes NaN
        val data = doubleArrayOf(1.0, Double.POSITIVE_INFINITY, 3.0)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        // When mean is Infinity and sigma is NaN, results should be NaN
        assertTrue(result.cp.isNaN(), "cp should be NaN when data contains Infinity")
        assertTrue(result.cpk.isNaN(), "cpk should be NaN when data contains Infinity")
    }

    @Test
    fun testInfinityLslUsl() {
        // Infinite tolerance band
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = data.processCapability(lsl = Double.NEGATIVE_INFINITY, usl = Double.POSITIVE_INFINITY)

        // tolerance = Inf - (-Inf) = Inf, cp = Inf / (6*sigma) = Inf
        assertEquals(Double.POSITIVE_INFINITY, result.cp, "cp should be +Inf for infinite tolerance")
    }

    // ===== Property-based tests =====

    @Test
    fun testCpkLeqCp() {
        // Cpk <= Cp always (Cpk penalizes off-center processes)
        val datasets = listOf(
            doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0),
            doubleArrayOf(7.0, 7.5, 8.0, 7.2, 7.8, 8.1, 7.6, 7.3, 7.9, 8.2),
            doubleArrayOf(-5.0, -3.0, -4.0, -2.0, -6.0),
            doubleArrayOf(3.0, 5.0),
        )
        val specs = listOf(
            Pair(1.0, 10.0),
            Pair(5.0, 10.0),
            Pair(-10.0, 0.0),
            Pair(0.0, 10.0),
        )
        for (i in datasets.indices) {
            val result = datasets[i].processCapability(lsl = specs[i].first, usl = specs[i].second)
            assertTrue(
                result.cpk <= result.cp + 1e-14,
                "Cpk (${result.cpk}) should be <= Cp (${result.cp}) for dataset $i"
            )
        }
    }

    @Test
    fun testPpkLeqPp() {
        // Ppk <= Pp always
        val data = doubleArrayOf(7.0, 7.5, 8.0, 7.2, 7.8, 8.1, 7.6, 7.3, 7.9, 8.2)
        val result = data.processCapability(lsl = 5.0, usl = 10.0)
        assertTrue(
            result.ppk <= result.pp + 1e-14,
            "Ppk (${result.ppk}) should be <= Pp (${result.pp})"
        )
    }

    @Test
    fun testCpEqualsCpkWhenCentered() {
        // When mean = (lsl + usl) / 2, Cp should equal Cpk
        val data = doubleArrayOf(4.0, 5.0, 6.0, 4.5, 5.5, 5.0)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)
        assertEquals(result.cp, result.cpk, 1e-10, "Cp should equal Cpk when process is centered")
        assertEquals(result.pp, result.ppk, 1e-10, "Pp should equal Ppk when process is centered")
    }

    @Test
    fun testAllIndicesPositive() {
        // For data within spec limits with non-negative min(mean-lsl, usl-mean),
        // all indices should be positive
        val data = doubleArrayOf(4.0, 5.0, 6.0, 5.5, 4.5)
        val result = data.processCapability(lsl = 0.0, usl = 10.0)

        assertTrue(result.cp > 0.0, "cp should be positive")
        assertTrue(result.cpk > 0.0, "cpk should be positive")
        assertTrue(result.pp > 0.0, "pp should be positive")
        assertTrue(result.ppk > 0.0, "ppk should be positive")
    }

    @Test
    fun testPpUsesPopulationSigma() {
        // Pp/Ppk use population sigma (N divisor), Cp/Cpk use sample sigma (N-1 divisor)
        // For n>2: sample_sigma > pop_sigma => Cp < Pp and Cpk < Ppk
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val result = data.processCapability(lsl = 1.0, usl = 10.0)

        assertTrue(
            result.cp < result.pp,
            "Cp (${result.cp}) should be < Pp (${result.pp}) because sample sigma > pop sigma"
        )
        assertTrue(
            result.cpk < result.ppk,
            "Cpk (${result.cpk}) should be < Ppk (${result.ppk}) because sample sigma > pop sigma"
        )
    }

    @Test
    fun testCpIndependentOfMean() {
        // Cp depends only on tolerance and sigma, not on process centering
        // Shifting the data should not change Cp (if the shift doesn't change variance)
        val data1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val data2 = doubleArrayOf(101.0, 102.0, 103.0, 104.0, 105.0)

        val result1 = data1.processCapability(lsl = -10.0, usl = 20.0)
        val result2 = data2.processCapability(lsl = 90.0, usl = 120.0)

        // Same tolerance (30) and same variance => same Cp
        assertEquals(result1.cp, result2.cp, 1e-10, "Cp should be the same for shifted data with same tolerance")
    }

    @Test
    fun testIterableConsistency() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val fromArray = data.processCapability(lsl = 1.0, usl = 10.0)
        val fromIterable = data.toList().processCapability(lsl = 1.0, usl = 10.0)

        assertEquals(fromArray.cp, fromIterable.cp, 1e-14, "Iterable cp should match DoubleArray cp")
        assertEquals(fromArray.cpk, fromIterable.cpk, 1e-14, "Iterable cpk should match DoubleArray cpk")
        assertEquals(fromArray.pp, fromIterable.pp, 1e-14, "Iterable pp should match DoubleArray pp")
        assertEquals(fromArray.ppk, fromIterable.ppk, 1e-14, "Iterable ppk should match DoubleArray ppk")
    }

    @Test
    fun testSequenceConsistency() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val fromArray = data.processCapability(lsl = 1.0, usl = 10.0)
        val fromSequence = data.asSequence().processCapability(lsl = 1.0, usl = 10.0)

        assertEquals(fromArray.cp, fromSequence.cp, 1e-14, "Sequence cp should match DoubleArray cp")
        assertEquals(fromArray.cpk, fromSequence.cpk, 1e-14, "Sequence cpk should match DoubleArray cpk")
        assertEquals(fromArray.pp, fromSequence.pp, 1e-14, "Sequence pp should match DoubleArray pp")
        assertEquals(fromArray.ppk, fromSequence.ppk, 1e-14, "Sequence ppk should match DoubleArray ppk")
    }

    @Test
    fun testDataClassEquality() {
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val r1 = data.processCapability(lsl = 1.0, usl = 10.0)
        val r2 = data.processCapability(lsl = 1.0, usl = 10.0)

        assertEquals(r1, r2, "Same input should produce equal ProcessCapabilityResult")
    }
}
