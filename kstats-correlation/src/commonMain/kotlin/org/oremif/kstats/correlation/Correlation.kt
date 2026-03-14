package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.PopulationKind
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.distributions.StudentTDistribution
import org.oremif.kstats.sampling.TieMethod
import org.oremif.kstats.sampling.rank
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The result of a correlation computation.
 *
 * @property coefficient the correlation coefficient, ranging from -1.0 (perfect negative
 * correlation) through 0.0 (no correlation) to 1.0 (perfect positive correlation). Returns
 * [Double.NaN] when the correlation is undefined (e.g. constant input).
 * @property pValue the two-sided p-value for testing the null hypothesis that the true
 * correlation is zero. Smaller values indicate stronger evidence of a real association.
 * Returns [Double.NaN] when the p-value cannot be computed.
 * @property n the number of observations used in the computation.
 */
public data class CorrelationResult(
    val coefficient: Double,
    val pValue: Double,
    val n: Int
)

/**
 * Computes the Pearson product-moment correlation coefficient between two arrays.
 *
 * The Pearson correlation measures the strength and direction of the linear relationship
 * between two variables. A value of 1.0 indicates a perfect positive linear relationship,
 * -1.0 indicates a perfect negative linear relationship, and 0.0 indicates no linear
 * relationship.
 *
 * The p-value is computed using a two-sided t-test for the null hypothesis that the true
 * correlation is zero. Uses the numerically stable form (1-r)(1+r) instead of (1-r²) to
 * avoid catastrophic cancellation when r is close to ±1.
 *
 * Returns [Double.NaN] for both coefficient and p-value when either array has zero variance
 * (all values identical).
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
 * val result = pearsonCorrelation(x, y)
 * result.coefficient // 1.0
 * result.pValue      // 0.0
 * result.n           // 5
 * ```
 *
 * @param x the first array of observations.
 * @param y the second array of observations, must have the same size as [x].
 * @return a [CorrelationResult] containing the Pearson r, two-sided p-value, and sample size.
 */
public fun pearsonCorrelation(x: DoubleArray, y: DoubleArray): CorrelationResult {
    if (x.size != y.size) throw InvalidParameterException("Arrays must have the same size")
    val n = x.size
    if (n < 3) throw InsufficientDataException("Need at least 3 observations")

    val mx = x.mean()
    val my = y.mean()

    var sxy = 0.0
    var sxx = 0.0
    var syy = 0.0
    for (i in 0 until n) {
        val dx = x[i] - mx
        val dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    }

    if (sxx == 0.0 || syy == 0.0) {
        return CorrelationResult(Double.NaN, Double.NaN, n)
    }

    val rawR = sxy / sqrt(sxx * syy)
    if (rawR.isNaN()) {
        return CorrelationResult(Double.NaN, Double.NaN, n)
    }
    val r = rawR.coerceIn(-1.0, 1.0)

    // t-test for correlation significance
    // Use (1-r)(1+r) instead of (1-r²) to avoid catastrophic cancellation when r → ±1
    val oneMinusR2 = (1.0 - r) * (1.0 + r)
    if (oneMinusR2 == 0.0) {
        return CorrelationResult(r, 0.0, n)
    }
    val t = r * sqrt((n - 2).toDouble() / oneMinusR2)
    val dist = StudentTDistribution((n - 2).toDouble())
    val pValue = 2.0 * dist.sf(abs(t))

    return CorrelationResult(r, pValue.coerceIn(0.0, 1.0), n)
}

/**
 * Computes the Spearman rank correlation coefficient between two arrays.
 *
 * The Spearman correlation is a non-parametric measure of the monotonic relationship between
 * two variables. Unlike Pearson, it does not assume linearity — it detects whether the
 * variables tend to increase or decrease together, regardless of the rate.
 *
 * Computed by applying the Pearson correlation to the average-method ranks of the input
 * arrays. Tied values receive the average of the ranks they would occupy.
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val y = doubleArrayOf(1.0, 4.0, 9.0, 16.0, 25.0)
 * val result = spearmanCorrelation(x, y)
 * result.coefficient // 1.0 (perfect monotonic relationship)
 * ```
 *
 * @param x the first array of observations.
 * @param y the second array of observations, must have the same size as [x].
 * @return a [CorrelationResult] containing the Spearman rho, two-sided p-value, and sample size.
 */
public fun spearmanCorrelation(x: DoubleArray, y: DoubleArray): CorrelationResult {
    if (x.size != y.size) throw InvalidParameterException("Arrays must have the same size")
    val n = x.size
    if (n < 3) throw InsufficientDataException("Need at least 3 observations")

    val xRanks = x.rank(TieMethod.AVERAGE)
    val yRanks = y.rank(TieMethod.AVERAGE)

    return pearsonCorrelation(xRanks, yRanks)
}

/**
 * Computes Kendall's tau-b rank correlation coefficient between two arrays.
 *
 * Kendall's tau-b measures the ordinal association between two variables. It counts the
 * number of concordant pairs (both values increase together) versus discordant pairs
 * (one increases while the other decreases), with an adjustment for ties. Values range
 * from -1.0 (all pairs discordant) to 1.0 (all pairs concordant).
 *
 * Uses an O(n log n) merge-sort algorithm (Knight, 1966) for counting discordant pairs,
 * rather than the naive O(n²) approach. The p-value is computed using a normal approximation
 * with the ties-adjusted variance formula (Kendall, 1970).
 *
 * Returns [Double.NaN] for both coefficient and p-value when all values in either array
 * are identical (denominator becomes zero).
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val y = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
 * val result = kendallTau(x, y)
 * result.coefficient // 0.6
 * result.pValue      // p-value from normal approximation
 * ```
 *
 * @param x the first array of observations.
 * @param y the second array of observations, must have the same size as [x].
 * @return a [CorrelationResult] containing tau-b, two-sided p-value, and sample size.
 */
public fun kendallTau(x: DoubleArray, y: DoubleArray): CorrelationResult {
    if (x.size != y.size) throw InvalidParameterException("Arrays must have the same size")
    val n = x.size
    if (n < 3) throw InsufficientDataException("Need at least 3 observations")

    // Step 1: Sort indices by (x, y) lexicographically
    val indices = (0 until n).sortedWith(compareBy<Int> { x[it] }.thenBy { y[it] })

    // Step 2: Scan sorted indices for x-tied groups and joint ties
    var n1 = 0L // x-tied pairs
    var n3 = 0L // jointly tied pairs (same x and y)
    var vt = 0L // Σ t(t-1)(2t+5)
    var v1x = 0L // Σ t(t-1)
    var v2x = 0L // Σ t(t-1)(t-2)

    var i = 0
    while (i < n) {
        // Find end of x-tied group
        var j = i + 1
        while (j < n && x[indices[j]] == x[indices[i]]) j++
        val t = (j - i).toLong()
        if (t > 1) {
            n1 += t * (t - 1) / 2
            vt += t * (t - 1) * (2 * t + 5)
            v1x += t * (t - 1)
            v2x += t * (t - 1) * (t - 2)
            // Within this x-tied group, scan for joint ties (same y too)
            var k = i
            while (k < j) {
                var l = k + 1
                while (l < j && y[indices[l]] == y[indices[k]]) l++
                val s = (l - k).toLong()
                if (s > 1) {
                    n3 += s * (s - 1) / 2
                }
                k = l
            }
        }
        i = j
    }

    // Step 3: Extract y-permutation and count y-tied groups
    val yPerm = DoubleArray(n) { y[indices[it]] }

    // Sort a copy of y to find y-tied groups
    val ySorted = y.copyOf()
    ySorted.sort()
    var n2 = 0L // y-tied pairs
    var vu = 0L // Σ u(u-1)(2u+5)
    var v1y = 0L // Σ u(u-1)
    var v2y = 0L // Σ u(u-1)(u-2)

    i = 0
    while (i < n) {
        var j = i + 1
        while (j < n && ySorted[j] == ySorted[i]) j++
        val u = (j - i).toLong()
        if (u > 1) {
            n2 += u * (u - 1) / 2
            vu += u * (u - 1) * (2 * u + 5)
            v1y += u * (u - 1)
            v2y += u * (u - 1) * (u - 2)
        }
        i = j
    }

    // Step 4: Count discordant pairs via merge sort on yPerm
    val temp = DoubleArray(n)
    val dis = countDiscordant(yPerm, temp, 0, n - 1)

    // Step 5: Compute tau-b
    val n0 = n.toLong() * (n - 1) / 2
    val nS = n0 - n1 - n2 + n3 - 2 * dis
    val denom1 = n0 - n1
    val denom2 = n0 - n2

    if (denom1 == 0L || denom2 == 0L) {
        return CorrelationResult(Double.NaN, Double.NaN, n)
    }

    val tau = nS.toDouble() / sqrt(denom1.toDouble() * denom2.toDouble())

    // Step 6: Ties-adjusted p-value (Kendall, 1970)
    val nL = n.toLong()
    val v0 = nL * (nL - 1) * (2 * nL + 5)
    var varS = (v0 - vt - vu).toDouble() / 18.0
    if (nL >= 2) {
        varS += v1x.toDouble() * v1y.toDouble() / (2.0 * nL * (nL - 1))
    }
    if (nL >= 3) {
        varS += v2x.toDouble() * v2y.toDouble() / (9.0 * nL * (nL - 1) * (nL - 2))
    }

    if (varS <= 0.0) {
        return CorrelationResult(tau, if (tau == 0.0) 1.0 else 0.0, n)
    }

    val z = nS.toDouble() / sqrt(varS)
    val pValue = 2.0 * NormalDistribution.STANDARD.sf(abs(z))

    return CorrelationResult(tau.coerceIn(-1.0, 1.0), pValue.coerceIn(0.0, 1.0), n)
}

/**
 * Counts discordant pairs (inversions) in [arr] via merge sort.
 *
 * Sorts [arr] in-place as a side effect. Uses [temp] as scratch space for merging.
 * Returns the count as Long to avoid overflow for large n (up to ~2 billion pairs).
 */
private fun countDiscordant(arr: DoubleArray, temp: DoubleArray, left: Int, right: Int): Long {
    if (left >= right) return 0L

    val mid = left + (right - left) / 2
    var count = countDiscordant(arr, temp, left, mid)
    count += countDiscordant(arr, temp, mid + 1, right)

    // Merge and count inversions
    var i = left
    var j = mid + 1
    var k = left
    while (i <= mid && j <= right) {
        if (arr[i].compareTo(arr[j]) <= 0) {
            temp[k++] = arr[i++]
        } else {
            // arr[j] < arr[i]: all remaining elements in left half (i..mid) are inversions
            count += (mid - i + 1).toLong()
            temp[k++] = arr[j++]
        }
    }
    while (i <= mid) temp[k++] = arr[i++]
    while (j <= right) temp[k++] = arr[j++]

    for (idx in left..right) {
        arr[idx] = temp[idx]
    }

    return count
}

/**
 * Computes the covariance between two arrays.
 *
 * Covariance measures how two variables change together. A positive value means they tend
 * to increase together, a negative value means one tends to increase when the other decreases,
 * and a value near zero means no linear association. Unlike correlation, the magnitude of
 * covariance depends on the scale of the variables.
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
 * covariance(x, y) // 5.0 (sample covariance)
 * ```
 *
 * @param x the first array of observations.
 * @param y the second array of observations, must have the same size as [x].
 * @param kind whether to compute sample or population covariance. Defaults to [PopulationKind.SAMPLE],
 * which divides by n-1 (Bessel's correction) to produce an unbiased estimate.
 * @return the covariance between [x] and [y].
 */
public fun covariance(
    x: DoubleArray,
    y: DoubleArray,
    kind: PopulationKind = PopulationKind.SAMPLE
): Double {
    if (x.size != y.size) throw InvalidParameterException("Arrays must have the same size")
    val n = x.size
    if (n < 2) throw InsufficientDataException("Need at least 2 observations")

    val mx = x.mean()
    val my = y.mean()

    var sum = 0.0
    for (i in 0 until n) {
        sum += (x[i] - mx) * (y[i] - my)
    }

    return sum / (if (kind == PopulationKind.SAMPLE) n - 1 else n)
}

/**
 * Computes the Pearson correlation matrix for multiple variables.
 *
 * Returns a symmetric k×k matrix where element (i, j) is the Pearson correlation between
 * variables i and j. Diagonal elements are always 1.0 (a variable is perfectly correlated
 * with itself).
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
 * val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
 * val matrix = correlationMatrix(x, y)
 * matrix[0][0] // 1.0 (x with x)
 * matrix[0][1] // Pearson r between x and y
 * matrix[1][0] // same as matrix[0][1] (symmetric)
 * ```
 *
 * @param variables two or more arrays of observations, all with the same size.
 * @return a k×k array of Pearson correlation coefficients.
 */
public fun correlationMatrix(vararg variables: DoubleArray): Array<DoubleArray> {
    val k = variables.size
    if (k < 2) throw InsufficientDataException("Need at least 2 variables")
    val n = variables[0].size
    if (!variables.all { it.size == n }) throw InvalidParameterException("All variables must have the same size")

    return Array(k) { i ->
        DoubleArray(k) { j ->
            if (i == j) 1.0 else pearsonCorrelation(variables[i], variables[j]).coefficient
        }
    }
}

/**
 * Computes the covariance matrix for multiple variables.
 *
 * Returns a symmetric k×k matrix where element (i, j) is the covariance between variables
 * i and j. Diagonal elements are the variances of each variable.
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
 * val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
 * val matrix = covarianceMatrix(x, y)
 * matrix[0][0] // variance of x
 * matrix[0][1] // covariance of x and y
 * ```
 *
 * @param variables two or more arrays of observations, all with the same size.
 * @param kind whether to compute sample or population covariance. Defaults to [PopulationKind.SAMPLE],
 * which divides by n-1 (Bessel's correction) to produce an unbiased estimate.
 * @return a k×k array of covariance values.
 */
public fun covarianceMatrix(
    vararg variables: DoubleArray,
    kind: PopulationKind = PopulationKind.SAMPLE
): Array<DoubleArray> {
    val k = variables.size
    if (k < 2) throw InsufficientDataException("Need at least 2 variables")
    val n = variables[0].size
    if (!variables.all { it.size == n }) throw InvalidParameterException("All variables must have the same size")

    return Array(k) { i ->
        DoubleArray(k) { j ->
            covariance(variables[i], variables[j], kind)
        }
    }
}
