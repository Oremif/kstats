package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class GrubbsTestTest {

    private val tol = 1e-10

    // ===== Basic correctness: grubbsTest two-sided =====

    @Test
    fun testTwoSidedKnownValues() {
        // Dataset with a clear outlier (245.57)
        val data = doubleArrayOf(199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57)
        val result = grubbsTest(data)
        // scipy-equivalent: G = max|x_i - mean| / sd = 2.46876461121245
        assertEquals(2.46876461121245, result.statistic, tol, "G statistic")
        // p-value with Bonferroni correction via Student t(n-2)
        assertEquals(3.00263868207136e-07, result.pValue, 1e-12, "p-value")
        assertEquals("Grubbs' Test", result.testName)
        assertEquals(6.0, result.degreesOfFreedom, tol, "df = n - 2 = 6")
        assertEquals(7.0, result.additionalInfo["outlierIndex"]!!, tol, "outlier at index 7")
        assertEquals(245.57, result.additionalInfo["outlierValue"]!!, tol, "outlier value = 245.57")
    }

    @Test
    fun testTwoSidedSpringConstantData() {
        // Classic Grubbs 1969 spring constant data: suspected outlier 7.019
        val data = doubleArrayOf(
            7.006, 7.003, 7.009, 7.012, 7.019, 7.008, 6.996, 6.997,
            7.006, 7.005, 7.001, 7.003, 6.998, 7.005, 7.007
        )
        val result = grubbsTest(data)
        // G = 2.35683174419494, p = 0.126596314787549 (not significant at 5%)
        assertEquals(2.35683174419494, result.statistic, tol, "G statistic")
        assertEquals(0.126596314787549, result.pValue, 1e-8, "p-value")
        assertFalse(result.isSignificant(0.05), "Spring data outlier not significant at 5%")
    }

    @Test
    fun testTwoSidedClearMaxOutlier() {
        val data = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 100.0)
        val result = grubbsTest(data)
        // G = 2.03988574609118, p = 3.96907016358278e-06
        assertEquals(2.03988574609118, result.statistic, tol, "G statistic")
        assertEquals(3.96907016358278e-06, result.pValue, 1e-12, "p-value")
        assertTrue(result.isSignificant(0.05), "100.0 is clearly an outlier")
        assertEquals(5.0, result.additionalInfo["outlierIndex"]!!, tol, "outlier at index 5")
    }

    @Test
    fun testTwoSidedClearMinOutlier() {
        val data = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, -50.0)
        val result = grubbsTest(data)
        // G = 2.03720471510762, p = 3.51745319825601e-05
        assertEquals(2.03720471510762, result.statistic, tol, "G statistic")
        assertEquals(3.51745319825601e-05, result.pValue, 1e-10, "p-value")
        assertTrue(result.isSignificant(0.05), "-50.0 is clearly an outlier")
        assertEquals(5.0, result.additionalInfo["outlierIndex"]!!, tol, "outlier at index 5")
    }

    @Test
    fun testTwoSidedNoOutlier() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        val result = grubbsTest(data)
        // G = 1.26491106406735, p = 0.908450569081046
        assertEquals(1.26491106406735, result.statistic, tol, "G statistic")
        assertEquals(0.908450569081046, result.pValue, 1e-8, "p-value")
        assertFalse(result.isSignificant(0.05), "Uniform spacing: no outlier")
    }

    @Test
    fun testTwoSidedFourElements() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 20.0)
        val result = grubbsTest(data)
        // G = 1.49386500496572, p = 0.0163599867580859
        assertEquals(1.49386500496572, result.statistic, tol, "G statistic")
        assertEquals(0.0163599867580859, result.pValue, 1e-8, "p-value")
        assertTrue(result.isSignificant(0.05), "20 is an outlier in [1,2,3,20]")
    }

    // ===== Basic correctness: grubbsTest one-sided =====

    @Test
    fun testGreaterKnownValues() {
        val data = doubleArrayOf(2.1, 2.5, 2.3, 2.8, 10.0, 2.4, 2.2)
        val result = grubbsTest(data, alternative = Alternative.GREATER)
        // G = (max - mean) / sd = 2.26078848837738, p = 3.5504798221851e-06
        assertEquals(2.26078848837738, result.statistic, tol, "G statistic")
        assertEquals(3.5504798221851e-06, result.pValue, 1e-12, "p-value")
        assertEquals(4.0, result.additionalInfo["outlierIndex"]!!, tol, "max outlier at index 4")
        assertEquals(10.0, result.additionalInfo["outlierValue"]!!, tol, "outlier = 10.0")
    }

    @Test
    fun testLessKnownValues() {
        val data = doubleArrayOf(2.1, 2.5, 2.3, 2.8, -5.0, 2.4, 2.2)
        val result = grubbsTest(data, alternative = Alternative.LESS)
        // G = (mean - min) / sd = 2.26034137573037, p = 4.14458650092169e-06
        assertEquals(2.26034137573037, result.statistic, tol, "G statistic")
        assertEquals(4.14458650092169e-06, result.pValue, 1e-12, "p-value")
        assertEquals(4.0, result.additionalInfo["outlierIndex"]!!, tol, "min outlier at index 4")
        assertEquals(-5.0, result.additionalInfo["outlierValue"]!!, tol, "outlier = -5.0")
    }

    @Test
    fun testGreaterNoOutlier() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        val result = grubbsTest(data, alternative = Alternative.GREATER)
        // G = 1.26491106406735, p = 0.454225284540523
        assertEquals(1.26491106406735, result.statistic, tol, "G statistic")
        assertEquals(0.454225284540523, result.pValue, 1e-8, "p-value")
        assertFalse(result.isSignificant(), "No outlier")
    }

    @Test
    fun testLessNoOutlier() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        val result = grubbsTest(data, alternative = Alternative.LESS)
        // G = 1.26491106406735, p = 0.454225284540523
        assertEquals(1.26491106406735, result.statistic, tol, "G statistic")
        assertEquals(0.454225284540523, result.pValue, 1e-8, "p-value")
        assertFalse(result.isSignificant(), "No outlier")
    }

    // ===== Edge cases =====

    @Test
    fun testMinimumSampleSize() {
        val data = doubleArrayOf(1.0, 2.0, 3.0)
        val result = grubbsTest(data)
        // G = 1.0, p = 1.0 for [1,2,3] — three equidistant points
        assertEquals(1.0, result.statistic, tol, "G statistic for 3 elements")
        assertEquals(1.0, result.pValue, tol, "p = 1 for 3 equidistant points")
        assertEquals(1.0, result.degreesOfFreedom, tol, "df = n - 2 = 1")
        assertTrue(result.statistic.isFinite())
        assertTrue(result.pValue.isFinite())
    }

    @Test
    fun testMinimumSampleSizeWithOutlier() {
        val data = doubleArrayOf(1.0, 2.0, 100.0)
        val result = grubbsTest(data)
        // G = 1.15465591083419, p = 0.0167913103513126
        assertEquals(1.15465591083419, result.statistic, tol, "G statistic")
        assertEquals(0.0167913103513126, result.pValue, 1e-8, "p-value")
        assertEquals(2.0, result.additionalInfo["outlierIndex"]!!, tol, "outlier at index 2")
    }

    @Test
    fun testMinimumSampleSizeAllAlternatives() {
        val data = doubleArrayOf(1.0, 2.0, 100.0)

        val twoSided = grubbsTest(data, alternative = Alternative.TWO_SIDED)
        // G = 1.15465591083419, p = 0.0167913103513126
        assertEquals(1.15465591083419, twoSided.statistic, tol)

        val greater = grubbsTest(data, alternative = Alternative.GREATER)
        // G = 1.15465591083419, p = 0.00839565517565631
        assertEquals(1.15465591083419, greater.statistic, tol)
        assertEquals(0.00839565517565631, greater.pValue, 1e-8)

        val less = grubbsTest(data, alternative = Alternative.LESS)
        // G = 0.586119751692483, p = 0.991604344824341
        assertEquals(0.586119751692483, less.statistic, tol)
        assertEquals(0.991604344824341, less.pValue, 1e-8)
    }

    @Test
    fun testSymmetricData() {
        val data = doubleArrayOf(-2.0, -1.0, 0.0, 1.0, 2.0)
        val result = grubbsTest(data)
        // G = 1.26491106406735, p = 0.908450569081046
        assertEquals(1.26491106406735, result.statistic, tol, "G statistic for symmetric data")
        assertEquals(0.908450569081046, result.pValue, 1e-8, "p-value for symmetric data")
        assertFalse(result.isSignificant(), "Symmetric data should have no outlier")
    }

    @Test
    fun testEquidistantExtremes() {
        // Both extremes are equidistant from the mean; TWO_SIDED picks the first one found
        val data = doubleArrayOf(-10.0, 0.0, 1.0, 2.0, 3.0, 13.0)
        val result = grubbsTest(data)
        // G = 1.56640282518919, p = 0.449255071261881
        assertEquals(1.56640282518919, result.statistic, tol, "G statistic")
        assertEquals(0.449255071261881, result.pValue, 1e-8, "p-value")
    }

    // ===== Degenerate input =====

    @Test
    fun testEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            grubbsTest(doubleArrayOf())
        }
    }

    @Test
    fun testSingleElement() {
        assertFailsWith<InsufficientDataException> {
            grubbsTest(doubleArrayOf(5.0))
        }
    }

    @Test
    fun testTwoElements() {
        assertFailsWith<InsufficientDataException> {
            grubbsTest(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun testConstantData() {
        val data = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = grubbsTest(data)
        assertEquals(0.0, result.statistic, 0.0, "G = 0 for constant data")
        assertEquals(1.0, result.pValue, 0.0, "p = 1 for constant data")
        assertEquals(3.0, result.degreesOfFreedom, tol, "df = n - 2 = 3")
    }

    @Test
    fun testConstantDataAllAlternatives() {
        val data = doubleArrayOf(3.0, 3.0, 3.0, 3.0)
        for (alt in Alternative.entries) {
            val result = grubbsTest(data, alternative = alt)
            assertEquals(0.0, result.statistic, 0.0, "G = 0 for constant data with $alt")
            assertEquals(1.0, result.pValue, 0.0, "p = 1 for constant data with $alt")
        }
    }

    // ===== Extreme parameters =====

    @Test
    fun testLargeOffsetData() {
        // Numerical stability: data with large offset and an outlier
        val data = doubleArrayOf(1e8, 1e8 + 1.0, 1e8 + 2.0, 1e8 + 3.0, 1e8 + 100.0)
        val result = grubbsTest(data)
        // G = 1.78827848762067, p = 3.46684690979142e-05
        assertEquals(1.78827848762067, result.statistic, 1e-6, "G for large offset")
        assertEquals(3.46684690979142e-05, result.pValue, 1e-8, "p-value for large offset")
        assertTrue(result.isSignificant(), "Outlier should be detected with large offset")
    }

    @Test
    fun testVerySmallValues() {
        val data = doubleArrayOf(1e-10, 2e-10, 3e-10, 4e-10, 5e-10)
        val result = grubbsTest(data)
        // G = 1.26491106406735, p = 0.908450569081046 (same as [1,2,3,4,5] scaled)
        assertEquals(1.26491106406735, result.statistic, tol, "G for tiny values")
        assertEquals(0.908450569081046, result.pValue, 1e-8, "p-value for tiny values")
        assertTrue(result.statistic.isFinite(), "Statistic should be finite")
        assertTrue(result.pValue.isFinite(), "p-value should be finite")
    }

    @Test
    fun testLargeSampleSize() {
        // Large N with injected outlier — verify statistic is finite and test detects it
        val data = DoubleArray(100) { it.toDouble() }
        data[50] = 1000.0 // inject outlier
        val result = grubbsTest(data)
        assertTrue(result.statistic.isFinite(), "Statistic should be finite for n=100")
        assertTrue(result.pValue.isFinite(), "p-value should be finite for n=100")
        assertTrue(result.isSignificant(0.01), "Injected outlier should be detected")
        assertEquals(50.0, result.additionalInfo["outlierIndex"]!!, tol, "Outlier at index 50")
    }

    @Test
    fun testVeryLargeSampleSize() {
        // n=200 with injected outlier
        val data = DoubleArray(200) { it.toDouble() }
        data[100] = 5000.0
        val result = grubbsTest(data)
        assertTrue(result.statistic.isFinite())
        assertTrue(result.pValue < 0.001, "Obvious outlier should have tiny p-value")
        assertEquals(100.0, result.additionalInfo["outlierIndex"]!!, tol)
    }

    // ===== Non-finite input =====

    @Test
    fun testNaNInData() {
        val data = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val result = grubbsTest(data)
        TestAssertions.assertNaNResult(result, "when input contains NaN")
        assertTrue(result.additionalInfo["outlierIndex"]!!.isNaN(), "outlierIndex should be NaN")
        assertTrue(result.additionalInfo["outlierValue"]!!.isNaN(), "outlierValue should be NaN")
    }

    @Test
    fun testPositiveInfinityInData() {
        val data = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val result = grubbsTest(data)
        TestAssertions.assertNaNResult(result, "when input contains +Infinity")
    }

    @Test
    fun testNegativeInfinityInData() {
        val data = doubleArrayOf(1.0, 2.0, Double.NEGATIVE_INFINITY, 4.0, 5.0)
        val result = grubbsTest(data)
        TestAssertions.assertNaNResult(result, "when input contains -Infinity")
    }

    @Test
    fun testAllNaN() {
        val data = doubleArrayOf(Double.NaN, Double.NaN, Double.NaN)
        val result = grubbsTest(data)
        TestAssertions.assertNaNResult(result, "when all values are NaN")
    }

    @Test
    fun testMixedNonFinite() {
        val data = doubleArrayOf(1.0, Double.NaN, Double.POSITIVE_INFINITY)
        val result = grubbsTest(data)
        TestAssertions.assertNaNResult(result, "when data contains both NaN and Infinity")
    }

    @Test
    fun testNonFiniteAllAlternatives() {
        val data = TestData.WITH_NAN
        for (alt in Alternative.entries) {
            val result = grubbsTest(data, alternative = alt)
            TestAssertions.assertNaNResult(result, "with $alt and NaN in data")
        }
    }

    // ===== Property-based tests =====

    @Test
    fun testPValueRange() {
        val datasets = listOf(
            doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0),
            doubleArrayOf(1.0, 2.0, 3.0, 100.0),
            doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 100.0),
            doubleArrayOf(199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57)
        )
        for (data in datasets) {
            for (alt in Alternative.entries) {
                val result = grubbsTest(data, alternative = alt)
                assertTrue(
                    result.pValue in 0.0..1.0,
                    "p-value should be in [0, 1] for $alt, got ${result.pValue}"
                )
            }
        }
    }

    @Test
    fun testStatisticNonNegative() {
        val datasets = listOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(1.0, 2.0, 3.0, 100.0),
            doubleArrayOf(-5.0, 1.0, 2.0, 3.0, 4.0),
            doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        )
        for (data in datasets) {
            for (alt in Alternative.entries) {
                val result = grubbsTest(data, alternative = alt)
                assertTrue(
                    result.statistic >= 0.0,
                    "G statistic should be >= 0 for $alt, got ${result.statistic}"
                )
            }
        }
    }

    @Test
    fun testAlternativeConsistencySameStatistic() {
        // For two-sided vs greater, when the max deviation comes from the maximum value,
        // the G statistic should be the same
        val data = doubleArrayOf(2.1, 2.5, 2.3, 2.8, 10.0, 2.4, 2.2)
        val twoSided = grubbsTest(data, alternative = Alternative.TWO_SIDED)
        val greater = grubbsTest(data, alternative = Alternative.GREATER)
        // When max deviation = max value deviation, both G statistics are equal
        assertEquals(
            twoSided.statistic, greater.statistic, 1e-14,
            "TWO_SIDED and GREATER should have same G when max value has max deviation"
        )
    }

    @Test
    fun testTwoSidedEqualsDoubleOneSidedForMaxOutlier() {
        // When the most extreme value is the maximum, two-sided p = 2 * greater p
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0)
        val twoSided = grubbsTest(data, alternative = Alternative.TWO_SIDED)
        val greater = grubbsTest(data, alternative = Alternative.GREATER)
        assertEquals(
            twoSided.pValue, 2.0 * greater.pValue, 1e-14,
            "two-sided p should equal 2 * greater p when max outlier has max deviation"
        )
    }

    @Test
    fun testDegreesOfFreedom() {
        // df = n - 2 for all sample sizes
        for (n in listOf(3, 5, 10, 20, 50)) {
            val data = DoubleArray(n) { it.toDouble() }
            val result = grubbsTest(data)
            assertEquals(
                (n - 2).toDouble(), result.degreesOfFreedom, tol,
                "df should be ${n - 2} for n = $n"
            )
        }
    }

    @Test
    fun testOutlierIndexValid() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 50.0)
        val result = grubbsTest(data)
        val idx = result.additionalInfo["outlierIndex"]!!.toInt()
        assertTrue(idx in data.indices, "Outlier index should be within array bounds")
        assertEquals(data[idx], result.additionalInfo["outlierValue"]!!, 0.0,
            "outlierValue should match data[outlierIndex]")
    }

    @Test
    fun testOutlierValueMatchesData() {
        // Verify that outlierIndex and outlierValue are consistent for all alternatives
        val data = doubleArrayOf(2.1, 2.5, 2.3, 2.8, 10.0, 2.4, 2.2)
        for (alt in Alternative.entries) {
            val result = grubbsTest(data, alternative = alt)
            val idx = result.additionalInfo["outlierIndex"]!!.toInt()
            assertEquals(
                data[idx], result.additionalInfo["outlierValue"]!!, 0.0,
                "outlierValue should match data[outlierIndex] for $alt"
            )
        }
    }

    @Test
    fun testGreaterDetectsMaximum() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 50.0)
        val result = grubbsTest(data, alternative = Alternative.GREATER)
        assertEquals(4.0, result.additionalInfo["outlierIndex"]!!, tol,
            "GREATER should detect the maximum value")
        assertEquals(50.0, result.additionalInfo["outlierValue"]!!, tol)
    }

    @Test
    fun testLessDetectsMinimum() {
        val data = doubleArrayOf(-50.0, 1.0, 2.0, 3.0, 4.0)
        val result = grubbsTest(data, alternative = Alternative.LESS)
        assertEquals(0.0, result.additionalInfo["outlierIndex"]!!, tol,
            "LESS should detect the minimum value")
        assertEquals(-50.0, result.additionalInfo["outlierValue"]!!, tol)
    }

    @Test
    fun testScaleInvariance() {
        // Grubbs G statistic is scale-invariant: G(c*x) = G(x) for c > 0
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 50.0)
        val scaled = DoubleArray(data.size) { data[it] * 1000.0 }
        val result1 = grubbsTest(data)
        val result2 = grubbsTest(scaled)
        assertEquals(result1.statistic, result2.statistic, 1e-8,
            "G should be scale-invariant")
        assertEquals(result1.pValue, result2.pValue, 1e-8,
            "p-value should be scale-invariant")
    }

    @Test
    fun testLocationInvariance() {
        // Grubbs G statistic is location-invariant: G(x + c) = G(x)
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 50.0)
        val shifted = DoubleArray(data.size) { data[it] + 1e6 }
        val result1 = grubbsTest(data)
        val result2 = grubbsTest(shifted)
        assertEquals(result1.statistic, result2.statistic, 1e-6,
            "G should be location-invariant")
    }

    @Test
    fun testIsSignificantConsistency() {
        val data = doubleArrayOf(199.31, 199.53, 200.19, 200.82, 201.92, 201.95, 202.18, 245.57)
        val result = grubbsTest(data)
        TestAssertions.assertIsSignificantConsistency(result, 0.05)
        TestAssertions.assertIsSignificantConsistency(result, 0.01)
        TestAssertions.assertIsSignificantConsistency(result, 0.001)
    }

    // ===== grubbsTestIterative =====

    @Test
    fun testIterativeBasic() {
        // Dataset with a single clear outlier
        val data = doubleArrayOf(3.0, 4.0, 5.0, 6.0, 7.0, 100.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        assertTrue(result.outlierIndices.contains(5), "Should detect outlier at index 5")
        assertTrue(100.0 !in result.cleanedData.toList(), "Cleaned data should not contain 100.0")
        assertTrue(result.iterations.isNotEmpty(), "Should have at least one iteration")
    }

    @Test
    fun testIterativeNoOutlier() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        assertTrue(result.outlierIndices.isEmpty(), "No outliers in uniform data")
        assertEquals(data.size, result.cleanedData.size, "Cleaned data should equal original")
        assertEquals(1, result.iterations.size, "Should have exactly one iteration (no rejection)")
    }

    @Test
    fun testIterativeMultipleOutliers() {
        // Larger dataset with two outliers — enough inliers so SD isn't dominated by outliers
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 50.0, 80.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        // Should detect both 80.0 and 50.0 as outliers
        assertTrue(result.outlierIndices.size >= 2, "Should find at least 2 outliers")
        assertTrue(result.outlierIndices.contains(11), "Should detect 80.0 at original index 11")
        assertTrue(result.outlierIndices.contains(10), "Should detect 50.0 at original index 10")
        for (idx in result.outlierIndices) {
            assertTrue(idx in data.indices, "Outlier index $idx should be valid")
        }
        // Cleaned data should be smaller
        assertTrue(result.cleanedData.size < data.size, "Cleaned data should be smaller")
    }

    @Test
    fun testIterativeCleanedDataConsistency() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 100.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        // Outlier indices + cleaned data should account for all original elements
        assertEquals(
            data.size, result.outlierIndices.size + result.cleanedData.size,
            "outliers + cleaned = original size"
        )
    }

    @Test
    fun testIterativeConstantData() {
        val data = doubleArrayOf(5.0, 5.0, 5.0, 5.0, 5.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        assertTrue(result.outlierIndices.isEmpty(), "No outliers in constant data")
        assertTrue(result.cleanedData.contentEquals(data), "Cleaned data should equal original")
    }

    @Test
    fun testIterativeWithAlternatives() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 100.0)
        for (alt in Alternative.entries) {
            val result = grubbsTestIterative(data, alpha = 0.05, alternative = alt)
            // Should not crash and should produce valid results
            assertTrue(result.iterations.isNotEmpty(), "Should have iterations for $alt")
        }
    }

    // ===== grubbsTestIterative: degenerate =====

    @Test
    fun testIterativeEmptyArray() {
        assertFailsWith<InsufficientDataException> {
            grubbsTestIterative(doubleArrayOf())
        }
    }

    @Test
    fun testIterativeTooFewElements() {
        assertFailsWith<InsufficientDataException> {
            grubbsTestIterative(doubleArrayOf(1.0, 2.0))
        }
    }

    @Test
    fun testIterativeInvalidAlphaZero() {
        assertFailsWith<InvalidParameterException> {
            grubbsTestIterative(doubleArrayOf(1.0, 2.0, 3.0, 4.0), alpha = 0.0)
        }
    }

    @Test
    fun testIterativeInvalidAlphaOne() {
        assertFailsWith<InvalidParameterException> {
            grubbsTestIterative(doubleArrayOf(1.0, 2.0, 3.0, 4.0), alpha = 1.0)
        }
    }

    @Test
    fun testIterativeInvalidAlphaNegative() {
        assertFailsWith<InvalidParameterException> {
            grubbsTestIterative(doubleArrayOf(1.0, 2.0, 3.0, 4.0), alpha = -0.1)
        }
    }

    @Test
    fun testIterativeInvalidAlphaGreaterThanOne() {
        assertFailsWith<InvalidParameterException> {
            grubbsTestIterative(doubleArrayOf(1.0, 2.0, 3.0, 4.0), alpha = 1.5)
        }
    }

    // ===== grubbsTestIterative: non-finite input =====

    @Test
    fun testIterativeNaNInData() {
        val data = doubleArrayOf(1.0, 2.0, Double.NaN, 4.0, 5.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        // Iterative should stop at first NaN result
        assertTrue(result.outlierIndices.isEmpty(), "No outliers should be detected when NaN present")
        assertTrue(result.iterations.isNotEmpty(), "Should have at least one iteration")
        val lastResult = result.iterations.last()
        assertTrue(lastResult.pValue.isNaN(), "Last iteration p-value should be NaN")
    }

    @Test
    fun testIterativeInfinityInData() {
        val data = doubleArrayOf(1.0, 2.0, Double.POSITIVE_INFINITY, 4.0, 5.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        assertTrue(result.outlierIndices.isEmpty(), "No outliers when Infinity present")
    }

    // ===== grubbsTestIterative: property-based =====

    @Test
    fun testIterativeOutlierIndicesAreUnique() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 50.0, 80.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        val uniqueIndices = result.outlierIndices.toSet()
        assertEquals(
            result.outlierIndices.size, uniqueIndices.size,
            "Outlier indices should be unique"
        )
    }

    @Test
    fun testIterativeOutlierIndicesInRange() {
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 50.0, 80.0)
        val result = grubbsTestIterative(data, alpha = 0.05)
        for (idx in result.outlierIndices) {
            assertTrue(idx in data.indices, "Outlier index $idx should be in [0, ${data.size})")
        }
    }

    @Test
    fun testIterativeStopsAtMinSize() {
        // With aggressive alpha, should stop before reducing below 3 elements
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = grubbsTestIterative(data, alpha = 0.99)
        assertTrue(result.cleanedData.size >= 3, "Should not reduce below 3 elements")
    }

    @Test
    fun testIterativeStrictAlphaRemovesLess() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 100.0)
        val strict = grubbsTestIterative(data, alpha = 0.001)
        val lenient = grubbsTestIterative(data, alpha = 0.1)
        assertTrue(
            strict.outlierIndices.size <= lenient.outlierIndices.size,
            "Stricter alpha should find fewer or equal outliers"
        )
    }

    // ===== GrubbsIterativeResult equality =====

    @Test
    fun testIterativeResultEquality() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 100.0)
        val result1 = grubbsTestIterative(data, alpha = 0.05)
        val result2 = grubbsTestIterative(data, alpha = 0.05)
        assertEquals(result1, result2, "Same input should produce equal results")
        assertEquals(result1.hashCode(), result2.hashCode(), "Equal results should have same hashCode")
    }

    @Test
    fun testIterativeResultInequality() {
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 100.0)
        val result1 = grubbsTestIterative(data, alpha = 0.05)
        val result2 = grubbsTestIterative(data, alpha = 0.001)
        // May or may not be equal depending on detection, but at least they should not crash
        assertTrue(result1.outlierIndices.size >= result2.outlierIndices.size)
    }
}
