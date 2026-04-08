package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation
import org.oremif.kstats.distributions.StudentTDistribution
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The result of an iterative Grubbs' test for multiple outlier detection.
 *
 * Contains the indices of all detected outliers (in the original array), the remaining
 * data after outlier removal, and the [TestResult] from each iteration. The iteration
 * list always has at least one entry — the final entry is the test that was not significant
 * (or that had too few observations to continue).
 *
 * ### Example:
 * ```kotlin
 * val result = grubbsTestIterative(doubleArrayOf(10.0, 11.0, 12.0, 80.0, 90.0), alpha = 0.05)
 * result.outlierIndices // indices of detected outliers in the original array
 * result.cleanedData    // data with outliers removed
 * result.iterations     // TestResult for each round
 * ```
 *
 * @property outlierIndices the zero-based indices of the detected outliers in the original
 * input array, in the order they were removed.
 * @property cleanedData the remaining observations after all detected outliers have been removed.
 * @property iterations the [TestResult] produced by each round of the iterative procedure.
 */
public data class GrubbsIterativeResult(
    val outlierIndices: List<Int>,
    val cleanedData: DoubleArray,
    val iterations: List<TestResult>
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is GrubbsIterativeResult) return false
        return outlierIndices == other.outlierIndices &&
            cleanedData.contentEquals(other.cleanedData) &&
            iterations == other.iterations
    }

    override fun hashCode(): Int {
        var result = outlierIndices.hashCode()
        result = 31 * result + cleanedData.contentHashCode()
        result = 31 * result + iterations.hashCode()
        return result
    }
}

/**
 * Performs Grubbs' test for detecting a single outlier in a univariate dataset.
 *
 * Grubbs' test (also called the extreme studentized deviate test) checks whether the
 * observation farthest from the sample mean is a statistically significant outlier,
 * assuming the data are normally distributed. The test statistic G is the ratio of the
 * maximum absolute deviation from the mean to the sample standard deviation. The p-value
 * is computed by converting G to a t-statistic and applying a Bonferroni correction for
 * testing all N observations.
 *
 * If all values are identical (zero standard deviation), returns G = 0 and p-value = 1.
 * If any value is non-finite (NaN or Infinity), returns NaN for both statistic and p-value.
 *
 * ### Example:
 * ```kotlin
 * val latencies = doubleArrayOf(12.0, 14.0, 11.0, 13.0, 15.0, 98.0, 12.0)
 * val result = grubbsTest(latencies)
 * result.statistic                       // G statistic
 * result.pValue                          // p-value (Bonferroni-corrected)
 * result.additionalInfo["outlierIndex"]  // index of the suspected outlier
 * result.additionalInfo["outlierValue"]  // value of the suspected outlier
 * result.isSignificant()                 // true if the outlier is significant at 5%
 * ```
 *
 * @param sample the observed values. Must have at least 3 elements.
 * @param alternative the direction of the alternative hypothesis. [Alternative.TWO_SIDED]
 * tests the value with the largest absolute deviation; [Alternative.GREATER] tests the maximum;
 * [Alternative.LESS] tests the minimum. Defaults to [Alternative.TWO_SIDED].
 * @return a [TestResult] containing the G statistic, Bonferroni-corrected p-value,
 * degrees of freedom (n − 2), and additional info with "outlierIndex" and "outlierValue".
 * @see grubbsTestIterative
 */
public fun grubbsTest(
    sample: DoubleArray,
    alternative: Alternative = Alternative.TWO_SIDED
): TestResult {
    if (sample.size < 3) throw InsufficientDataException(
        "Grubbs' test requires at least 3 elements, got ${sample.size}"
    )

    if (sample.any { !it.isFinite() }) {
        return TestResult(
            testName = "Grubbs' Test",
            statistic = Double.NaN,
            pValue = Double.NaN,
            alternative = alternative,
            additionalInfo = mapOf("outlierIndex" to Double.NaN, "outlierValue" to Double.NaN)
        )
    }

    val n = sample.size
    val mean = sample.mean()
    val sd = sample.standardDeviation()

    // Constant data: zero standard deviation means no outlier is possible
    if (sd == 0.0) {
        return TestResult(
            testName = "Grubbs' Test",
            statistic = 0.0,
            pValue = 1.0,
            degreesOfFreedom = (n - 2).toDouble(),
            alternative = alternative,
            additionalInfo = mapOf("outlierIndex" to 0.0, "outlierValue" to sample[0])
        )
    }

    // Compute G statistic and identify the suspected outlier
    val outlierIndex: Int
    val g: Double

    when (alternative) {
        Alternative.TWO_SIDED -> {
            var maxDev = 0.0
            var maxIdx = 0
            for (i in sample.indices) {
                val dev = abs(sample[i] - mean)
                if (dev > maxDev) {
                    maxDev = dev
                    maxIdx = i
                }
            }
            g = maxDev / sd
            outlierIndex = maxIdx
        }
        Alternative.GREATER -> {
            var maxVal = sample[0]
            var maxIdx = 0
            for (i in 1 until n) {
                if (sample[i] > maxVal) {
                    maxVal = sample[i]
                    maxIdx = i
                }
            }
            g = (maxVal - mean) / sd
            outlierIndex = maxIdx
        }
        Alternative.LESS -> {
            var minVal = sample[0]
            var minIdx = 0
            for (i in 1 until n) {
                if (sample[i] < minVal) {
                    minVal = sample[i]
                    minIdx = i
                }
            }
            g = (mean - minVal) / sd
            outlierIndex = minIdx
        }
    }

    // Convert G to t-statistic: t² = G² · N · (N−2) / ((N−1)² − G² · N)
    // Then p-value via StudentT(N−2) with Bonferroni correction
    val gSq = g * g
    val nDbl = n.toDouble()
    val denom = (nDbl - 1.0) * (nDbl - 1.0) - gSq * nDbl

    val pValue: Double = if (denom <= 0.0) {
        0.0
    } else {
        val tSq = gSq * nDbl * (nDbl - 2.0) / denom
        val tVal = sqrt(tSq)
        val tDist = StudentTDistribution((n - 2).toDouble())
        val pTail = tDist.sf(tVal)

        when (alternative) {
            Alternative.TWO_SIDED -> (2.0 * nDbl * pTail).coerceIn(0.0, 1.0)
            else -> (nDbl * pTail).coerceIn(0.0, 1.0)
        }
    }

    return TestResult(
        testName = "Grubbs' Test",
        statistic = g,
        pValue = pValue,
        degreesOfFreedom = (n - 2).toDouble(),
        alternative = alternative,
        additionalInfo = mapOf(
            "outlierIndex" to outlierIndex.toDouble(),
            "outlierValue" to sample[outlierIndex]
        )
    )
}

/**
 * Performs iterative Grubbs' test to detect multiple outliers.
 *
 * Repeatedly applies [grubbsTest] to the data: when an outlier is found significant at
 * level [alpha], it is removed and the test is rerun on the remaining data. The procedure
 * stops when no further outlier is significant or fewer than 3 observations remain.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 80.0, 90.0)
 * val result = grubbsTestIterative(data, alpha = 0.05)
 * result.outlierIndices // e.g. [6, 5] — indices in the original array
 * result.cleanedData    // observations with outliers removed
 * result.iterations     // TestResult for each round
 * ```
 *
 * @param sample the observed values. Must have at least 3 elements.
 * @param alpha the significance level for each iteration. Must be in (0, 1). Defaults to `0.05`.
 * @param alternative the direction of the alternative hypothesis, passed to each
 * [grubbsTest] call. Defaults to [Alternative.TWO_SIDED].
 * @return a [GrubbsIterativeResult] with outlier indices, cleaned data, and per-iteration results.
 * @see grubbsTest
 */
public fun grubbsTestIterative(
    sample: DoubleArray,
    alpha: Double = 0.05,
    alternative: Alternative = Alternative.TWO_SIDED
): GrubbsIterativeResult {
    if (sample.size < 3) throw InsufficientDataException(
        "Grubbs' test requires at least 3 elements, got ${sample.size}"
    )
    if (alpha <= 0.0 || alpha >= 1.0) throw InvalidParameterException(
        "Significance level alpha must be in (0, 1), got $alpha"
    )

    val outlierIndices = mutableListOf<Int>()
    val iterations = mutableListOf<TestResult>()

    // Track mapping from current indices back to original indices
    val currentToOriginal = sample.indices.toMutableList()
    var currentData = sample.copyOf()

    while (currentData.size >= 3) {
        val result = grubbsTest(currentData, alternative)
        iterations.add(result)

        if (result.pValue >= alpha || result.pValue.isNaN()) break

        val localIdx = result.additionalInfo["outlierIndex"]!!.toInt()
        outlierIndices.add(currentToOriginal[localIdx])

        currentToOriginal.removeAt(localIdx)
        val newData = DoubleArray(currentData.size - 1)
        for (i in newData.indices) {
            newData[i] = if (i < localIdx) currentData[i] else currentData[i + 1]
        }
        currentData = newData
    }

    return GrubbsIterativeResult(
        outlierIndices = outlierIndices,
        cleanedData = currentData,
        iterations = iterations
    )
}
