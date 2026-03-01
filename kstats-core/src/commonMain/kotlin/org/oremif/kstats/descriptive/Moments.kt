package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.sqrt

private fun intPow(base: Double, exp: Int): Double = when (exp) {
    0 -> 1.0
    1 -> base
    2 -> base * base
    3 -> base * base * base
    4 -> {
        val b2 = base * base; b2 * b2
    }

    else -> {
        var result = 1.0
        var b = base
        var e = exp
        while (e > 0) {
            if (e % 2 == 1) result *= b
            b *= b
            e /= 2
        }
        result
    }
}

// ── centralMoment ──────────────────────────────────────────────────────────

/**
 * Computes the n-th central moment of the values.
 *
 * The r-th central moment is the average of the r-th power of deviations from the mean,
 * dividing by n (population-style). Order 0 returns 1.0 by convention, order 1 returns 0.0
 * by definition, and order 2 equals the population variance. Odd central moments measure
 * asymmetry, while even central moments measure tail weight.
 *
 * Uses a two-pass algorithm with z-normalization (dividing deviations by the standard
 * deviation) to prevent overflow with large-magnitude data.
 *
 * ### Example:
 * ```kotlin
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(2) // 4.0
 * listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(3) // 5.25
 * ```
 *
 * @param order the moment order. Must be non-negative.
 * @return the central moment of the given order.
 * @see kStatistic
 * @see variance
 */
public fun Iterable<Double>.centralMoment(order: Int): Double {
    if (order < 0) throw InvalidParameterException("order must be non-negative, got $order")

    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")

    val n = list.size
    if (order == 0) return 1.0
    if (order == 1) return 0.0

    return list.welford { _, mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return 0.0
        val sd = sqrt(variance)

        var sumZr = 0.0
        for (x in list) {
            val z = (x - mean) / sd
            sumZr += intPow(z, order)
        }

        val avgZr = sumZr / n
        if (avgZr == 0.0) return 0.0
        avgZr * intPow(sd, order)
    }
}

/**
 * Computes the n-th central moment of the values.
 *
 * The r-th central moment is the average of the r-th power of deviations from the mean,
 * dividing by n (population-style). Order 0 returns 1.0 by convention, order 1 returns 0.0
 * by definition, and order 2 equals the population variance. Odd central moments measure
 * asymmetry, while even central moments measure tail weight.
 *
 * Uses a two-pass algorithm with z-normalization (dividing deviations by the standard
 * deviation) to prevent overflow with large-magnitude data.
 *
 * ### Example:
 * ```kotlin
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(2) // 4.0
 * doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(3) // 5.25
 * ```
 *
 * @param order the moment order. Must be non-negative.
 * @return the central moment of the given order.
 * @see kStatistic
 * @see variance
 */
public fun DoubleArray.centralMoment(order: Int): Double {
    if (order < 0) throw InvalidParameterException("order must be non-negative, got $order")
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")

    val n = size
    if (order == 0) return 1.0
    if (order == 1) return 0.0

    return welford { mean, m2 ->
        val variance = m2 / n
        if (variance == 0.0) return 0.0
        val sd = sqrt(variance)

        var sumZr = 0.0
        for (x in this) {
            val z = (x - mean) / sd
            sumZr += intPow(z, order)
        }

        val avgZr = sumZr / n
        if (avgZr == 0.0) return 0.0
        avgZr * intPow(sd, order)
    }
}

/**
 * Computes the n-th central moment of the values in this sequence.
 *
 * The r-th central moment is the average of the r-th power of deviations from the mean,
 * dividing by n (population-style). Order 0 returns 1.0 by convention, order 1 returns 0.0
 * by definition, and order 2 equals the population variance. Odd central moments measure
 * asymmetry, while even central moments measure tail weight.
 *
 * Uses a two-pass algorithm with z-normalization (dividing deviations by the standard
 * deviation) to prevent overflow with large-magnitude data. The sequence is materialized
 * into a list internally.
 *
 * ### Example:
 * ```kotlin
 * sequenceOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(2) // 4.0
 * sequenceOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0).centralMoment(3) // 5.25
 * ```
 *
 * @param order the moment order. Must be non-negative.
 * @return the central moment of the given order.
 * @see kStatistic
 * @see variance
 */
public fun Sequence<Double>.centralMoment(order: Int): Double =
    toList().toDoubleArray().centralMoment(order)

// ── kStatistic ─────────────────────────────────────────────────────────────

/**
 * Computes the k-statistic of the given order, the unique symmetric unbiased estimator
 * of the corresponding cumulant.
 *
 * K-statistics generalize familiar estimators to higher orders: k1 equals the sample mean,
 * k2 equals the sample variance (with Bessel's correction, dividing by n-1), k3 and k4
 * estimate the third and fourth cumulants respectively. Only orders 1 through 4 are
 * supported, matching scipy's `kstat` function. Each order requires at least that many
 * data points (e.g. k4 needs n >= 4).
 *
 * Uses z-normalization for orders 3 and 4 to prevent overflow with large-magnitude data.
 *
 * ### Example:
 * ```kotlin
 * val data = listOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
 * data.kStatistic(1) // 5.0 (sample mean)
 * data.kStatistic(2) // 4.5714... (sample variance)
 * data.kStatistic(3) // 8.0
 * data.kStatistic(4) // 19.6571...
 * ```
 *
 * @param order the k-statistic order, must be 1, 2, 3, or 4.
 * @return the k-statistic of the given order.
 * @see centralMoment
 * @see mean
 * @see variance
 */
public fun Iterable<Double>.kStatistic(order: Int): Double {
    if (order !in 1..4) throw InvalidParameterException(
        "k-statistic order must be 1, 2, 3, or 4, got $order"
    )

    val list = toList()
    if (list.isEmpty()) throw InsufficientDataException("Collection must not be empty")

    val n = list.size
    if (n < order) throw InsufficientDataException(
        "k-statistic of order $order requires at least $order elements, got $n"
    )

    return list.welford { _, mean, m2 ->
        if (order == 1) return mean

        val variance = m2 / n
        if (variance == 0.0) return 0.0

        val nd = n.toDouble()
        if (order == 2) return m2 / (nd - 1.0)

        val sd = sqrt(variance)

        var sumZ3 = 0.0
        var sumZ4 = 0.0
        for (x in list) {
            val z = (x - mean) / sd
            val z2 = z * z
            sumZ3 += z2 * z
            if (order == 4) sumZ4 += z2 * z2
        }

        if (order == 3) {
            if (sumZ3 == 0.0) return 0.0
            val s3 = sd * sd * sd * sumZ3
            return nd * s3 / ((nd - 1.0) * (nd - 2.0))
        }

        // order == 4
        val sumZ2 = m2 / (sd * sd)
        val numerator = nd * (nd + 1.0) * sumZ4 - 3.0 * (nd - 1.0) * sumZ2 * sumZ2
        if (numerator == 0.0) return 0.0
        val sd4 = sd * sd * sd * sd
        sd4 * numerator / ((nd - 1.0) * (nd - 2.0) * (nd - 3.0))
    }
}

/**
 * Computes the k-statistic of the given order, the unique symmetric unbiased estimator
 * of the corresponding cumulant.
 *
 * K-statistics generalize familiar estimators to higher orders: k1 equals the sample mean,
 * k2 equals the sample variance (with Bessel's correction, dividing by n-1), k3 and k4
 * estimate the third and fourth cumulants respectively. Only orders 1 through 4 are
 * supported, matching scipy's `kstat` function. Each order requires at least that many
 * data points (e.g. k4 needs n >= 4).
 *
 * Uses z-normalization for orders 3 and 4 to prevent overflow with large-magnitude data.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
 * data.kStatistic(1) // 5.0 (sample mean)
 * data.kStatistic(2) // 4.5714... (sample variance)
 * data.kStatistic(3) // 8.0
 * data.kStatistic(4) // 19.6571...
 * ```
 *
 * @param order the k-statistic order, must be 1, 2, 3, or 4.
 * @return the k-statistic of the given order.
 * @see centralMoment
 * @see mean
 * @see variance
 */
public fun DoubleArray.kStatistic(order: Int): Double {
    if (order !in 1..4) throw InvalidParameterException(
        "k-statistic order must be 1, 2, 3, or 4, got $order"
    )
    if (isEmpty()) throw InsufficientDataException("Array must not be empty")

    val n = size
    if (n < order) throw InsufficientDataException(
        "k-statistic of order $order requires at least $order elements, got $n"
    )

    return welford { mean, m2 ->
        if (order == 1) return mean

        val variance = m2 / n
        if (variance == 0.0) return 0.0

        val nd = n.toDouble()
        if (order == 2) return m2 / (nd - 1.0)

        val sd = sqrt(variance)

        var sumZ3 = 0.0
        var sumZ4 = 0.0
        for (x in this) {
            val z = (x - mean) / sd
            val z2 = z * z
            sumZ3 += z2 * z
            if (order == 4) sumZ4 += z2 * z2
        }

        if (order == 3) {
            if (sumZ3 == 0.0) return 0.0
            val s3 = sd * sd * sd * sumZ3
            return nd * s3 / ((nd - 1.0) * (nd - 2.0))
        }

        // order == 4
        val sumZ2 = m2 / (sd * sd)
        val numerator = nd * (nd + 1.0) * sumZ4 - 3.0 * (nd - 1.0) * sumZ2 * sumZ2
        if (numerator == 0.0) return 0.0
        val sd4 = sd * sd * sd * sd
        sd4 * numerator / ((nd - 1.0) * (nd - 2.0) * (nd - 3.0))
    }
}

/**
 * Computes the k-statistic of the given order, the unique symmetric unbiased estimator
 * of the corresponding cumulant.
 *
 * K-statistics generalize familiar estimators to higher orders: k1 equals the sample mean,
 * k2 equals the sample variance (with Bessel's correction, dividing by n-1), k3 and k4
 * estimate the third and fourth cumulants respectively. Only orders 1 through 4 are
 * supported, matching scipy's `kstat` function. Each order requires at least that many
 * data points (e.g. k4 needs n >= 4).
 *
 * Uses z-normalization for orders 3 and 4 to prevent overflow with large-magnitude data.
 * The sequence is materialized into a list internally.
 *
 * ### Example:
 * ```kotlin
 * val data = sequenceOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
 * data.kStatistic(1) // 5.0 (sample mean)
 * data.kStatistic(2) // 4.5714... (sample variance)
 * ```
 *
 * @param order the k-statistic order, must be 1, 2, 3, or 4.
 * @return the k-statistic of the given order.
 * @see centralMoment
 * @see mean
 * @see variance
 */
public fun Sequence<Double>.kStatistic(order: Int): Double =
    toList().toDoubleArray().kStatistic(order)
