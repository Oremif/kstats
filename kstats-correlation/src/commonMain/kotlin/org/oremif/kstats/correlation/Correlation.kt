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

public data class CorrelationResult(
    val coefficient: Double,
    val pValue: Double,
    val n: Int
)

/**
 * Pearson product-moment correlation coefficient with p-value.
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
 * Spearman rank correlation coefficient.
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
 * Kendall's tau-b correlation coefficient.
 *
 * Uses O(n log n) merge-sort based algorithm (Knight, 1966) for counting
 * discordant pairs, with ties-adjusted variance formula (Kendall, 1970)
 * for the p-value.
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
 * Counts the number of discordant pairs (inversions) in [arr] using merge sort.
 * Modifies [arr] in-place (sorts it). Uses [temp] as scratch space.
 * Returns the inversion count as Long to avoid overflow for large n.
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
 * Covariance between two arrays.
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
 * Correlation matrix for multiple variables.
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
 * Covariance matrix for multiple variables.
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
