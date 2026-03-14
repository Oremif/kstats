package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.*

// ── Mathematical constants ──────────────────────────────────────────────────

/**
 * The Euler-Mascheroni constant, approximately 0.5772.
 *
 * This constant appears in many areas of mathematics and statistics, including the mean of
 * the Gumbel distribution and the digamma function. It is the limiting difference between
 * the harmonic series and the natural logarithm.
 *
 * ### Example:
 * ```kotlin
 * EULER_MASCHERONI // 0.5772156649015328606
 * ```
 */
public const val EULER_MASCHERONI: Double = 0.5772156649015328606

// ── Lanczos coefficients for ln(Gamma) ──────────────────────────────────────

private val LANCZOS_COEFFICIENTS = doubleArrayOf(
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7
)

private const val LANCZOS_G = 7.0

/**
 * Computes the natural logarithm of the gamma function at [x].
 *
 * The gamma function generalizes the factorial to real numbers: for positive integers,
 * gamma(n) equals (n-1)!. The logarithmic form is used to avoid overflow for large arguments.
 * Uses the Lanczos approximation with a reflection formula for values less than 0.5.
 *
 * ### Example:
 * ```kotlin
 * lnGamma(5.0) // 3.1780... (ln(24), since Gamma(5) = 4! = 24)
 * lnGamma(0.5) // 0.5723... (ln(sqrt(pi)))
 * ```
 *
 * @param x the point at which to evaluate. Must be positive.
 * @return the natural logarithm of gamma([x]).
 */
public fun lnGamma(x: Double): Double {
    if (x <= 0.0) throw InvalidParameterException("lnGamma requires x > 0, got $x")
    if (x < 0.5) {
        // Reflection formula: Gamma(x)*Gamma(1-x) = pi/sin(pi*x)
        return ln(PI / sin(PI * x)) - lnGamma(1.0 - x)
    }
    val xx = x - 1.0
    var sum = LANCZOS_COEFFICIENTS[0]
    for (i in 1 until LANCZOS_COEFFICIENTS.size) {
        sum += LANCZOS_COEFFICIENTS[i] / (xx + i)
    }
    val t = xx + LANCZOS_G + 0.5
    return 0.5 * ln(2.0 * PI) + (xx + 0.5) * ln(t) - t + ln(sum)
}

/**
 * Computes the gamma function at [x].
 *
 * The gamma function generalizes the factorial to real numbers: for positive integers,
 * gamma(n) equals (n-1)!. Computed as the exponential of [lnGamma].
 *
 * ### Example:
 * ```kotlin
 * gamma(5.0) // 24.0 (since Gamma(5) = 4! = 24)
 * gamma(0.5) // 1.7724... (sqrt(pi))
 * ```
 *
 * @param x the point at which to evaluate. Must be positive.
 * @return the value of the gamma function at [x].
 */
public fun gamma(x: Double): Double = exp(lnGamma(x))

// ── Beta function ───────────────────────────────────────────────────────────

/**
 * Computes the natural logarithm of the beta function for parameters [a] and [b].
 *
 * The beta function is defined as gamma(a) * gamma(b) / gamma(a + b). The logarithmic form
 * avoids overflow for large parameters. It is used internally by the regularized incomplete
 * beta function and by beta distributions.
 *
 * ### Example:
 * ```kotlin
 * lnBeta(2.0, 3.0) // -2.4849... (ln(1/12))
 * lnBeta(0.5, 0.5) // 1.1447... (ln(pi))
 * ```
 *
 * @param a the first shape parameter. Must be positive.
 * @param b the second shape parameter. Must be positive.
 * @return the natural logarithm of beta([a], [b]).
 */
public fun lnBeta(a: Double, b: Double): Double {
    if (a <= 0.0 || b <= 0.0) throw InvalidParameterException("lnBeta requires a > 0 and b > 0, got a=$a, b=$b")
    return lnGamma(a) + lnGamma(b) - lnGamma(a + b)
}

/**
 * Computes the beta function for parameters [a] and [b].
 *
 * The beta function is defined as gamma(a) * gamma(b) / gamma(a + b). Computed as the
 * exponential of [lnBeta].
 *
 * ### Example:
 * ```kotlin
 * beta(1.0, 1.0) // 1.0
 * beta(0.5, 0.5) // 3.1415... (pi)
 * ```
 *
 * @param a the first shape parameter. Must be positive.
 * @param b the second shape parameter. Must be positive.
 * @return the value of beta([a], [b]).
 */
public fun beta(a: Double, b: Double): Double = exp(lnBeta(a, b))

// ── Regularized incomplete beta function I_x(a, b) ─────────────────────────

private const val BETA_MAX_ITERATIONS = 200
private const val BETA_EPSILON = 1e-14

/**
 * Computes the regularized incomplete beta function I(x; a, b) at point [x].
 *
 * The regularized incomplete beta function gives the cumulative probability for beta-distributed
 * random variables. It is central to computing p-values for t-tests, F-tests, and other
 * hypothesis tests. Uses Lentz's continued fraction algorithm with a symmetry relation
 * for numerical stability.
 *
 * ### Example:
 * ```kotlin
 * regularizedBeta(0.5, 1.0, 1.0) // 0.5 (uniform on [0,1])
 * regularizedBeta(0.5, 2.0, 3.0) // 0.6875
 * ```
 *
 * @param x the point at which to evaluate, in the range [0, 1].
 * @param a the first shape parameter. Must be positive.
 * @param b the second shape parameter. Must be positive.
 * @return the regularized incomplete beta function value at [x], in the range [0, 1].
 * @throws org.oremif.kstats.core.exceptions.ConvergenceException if the continued fraction does not converge within 200 iterations.
 */
public fun regularizedBeta(x: Double, a: Double, b: Double): Double {
    if (a <= 0.0 || b <= 0.0) throw InvalidParameterException("regularizedBeta requires a > 0 and b > 0")
    if (x <= 0.0) return 0.0
    if (x >= 1.0) return 1.0

    // Use symmetry relation when x > (a+1)/(a+b+2)
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - regularizedBeta(1.0 - x, b, a)
    }

    // Lentz's continued fraction
    val lnPrefactor = a * ln(x) + b * ln(1.0 - x) - ln(a) - lnBeta(a, b)
    val prefactor = exp(lnPrefactor)

    var c = 1.0
    var d = 1.0 - (a + b) * x / (a + 1.0)
    if (abs(d) < 1e-30) d = 1e-30
    d = 1.0 / d
    var result = d

    var converged = false
    for (m in 1..BETA_MAX_ITERATIONS) {
        // even step
        val mDouble = m.toDouble()
        var numerator = mDouble * (b - mDouble) * x / ((a + 2.0 * mDouble - 1.0) * (a + 2.0 * mDouble))

        d = 1.0 + numerator * d
        if (abs(d) < 1e-30) d = 1e-30
        c = 1.0 + numerator / c
        if (abs(c) < 1e-30) c = 1e-30
        d = 1.0 / d
        result *= d * c

        // odd step
        numerator = -(a + mDouble) * (a + b + mDouble) * x / ((a + 2.0 * mDouble) * (a + 2.0 * mDouble + 1.0))

        d = 1.0 + numerator * d
        if (abs(d) < 1e-30) d = 1e-30
        c = 1.0 + numerator / c
        if (abs(c) < 1e-30) c = 1e-30
        d = 1.0 / d
        val delta = d * c
        result *= delta

        if (abs(delta - 1.0) < BETA_EPSILON) {
            converged = true
            break
        }
    }

    checkConvergence(converged, BETA_MAX_ITERATIONS, prefactor * result) {
        "regularizedBeta did not converge for x=$x, a=$a, b=$b after $BETA_MAX_ITERATIONS iterations"
    }

    return prefactor * result
}

// ── Regularized incomplete gamma functions ──────────────────────────────────

private const val GAMMA_MAX_ITERATIONS = 200
private const val GAMMA_EPSILON = 1e-14

/**
 * Computes the lower regularized incomplete gamma function P(a, x).
 *
 * This gives the probability that a gamma-distributed random variable with shape parameter [a]
 * is less than or equal to [x]. It is used internally to compute CDF values for chi-squared,
 * gamma, and Poisson distributions, as well as the error function. Uses a series expansion
 * when x is less than a + 1, and a continued fraction otherwise.
 *
 * ### Example:
 * ```kotlin
 * regularizedGammaP(1.0, 1.0) // 0.6321... (1 - e^(-1))
 * regularizedGammaP(1.0, 0.0) // 0.0
 * ```
 *
 * @param a the shape parameter. Must be positive.
 * @param x the upper integration limit. Returns 0.0 for non-positive values.
 * @return the value of P([a], [x]), in the range [0, 1].
 * @throws ConvergenceException if the iterative computation does not converge within 200 iterations.
 */
public fun regularizedGammaP(a: Double, x: Double): Double {
    if (a <= 0.0) throw InvalidParameterException("regularizedGammaP requires a > 0, got $a")
    if (x < 0.0) return 0.0
    if (x == 0.0) return 0.0

    if (x < a + 1.0) {
        // Series expansion
        return gammaSeriesP(a, x)
    } else {
        // Continued fraction
        return 1.0 - gammaContinuedFractionQ(a, x)
    }
}

/**
 * Computes the upper regularized incomplete gamma function Q(a, x), the complement of [regularizedGammaP].
 *
 * This gives the probability that a gamma-distributed random variable with shape parameter [a]
 * exceeds [x]. Equivalent to 1 - P(a, x), but computed directly for better numerical precision
 * in the upper tail.
 *
 * ### Example:
 * ```kotlin
 * regularizedGammaQ(1.0, 1.0) // 0.3678... (e^(-1))
 * regularizedGammaQ(1.0, 0.0) // 1.0
 * ```
 *
 * @param a the shape parameter. Must be positive.
 * @param x the lower integration limit. Returns 1.0 for non-positive values.
 * @return the value of Q([a], [x]), in the range [0, 1].
 * @throws ConvergenceException if the iterative computation does not converge within 200 iterations.
 */
public fun regularizedGammaQ(a: Double, x: Double): Double {
    if (a <= 0.0) throw InvalidParameterException("regularizedGammaQ requires a > 0, got $a")
    if (x < 0.0) return 1.0
    if (x == 0.0) return 1.0

    if (x < a + 1.0) {
        return 1.0 - gammaSeriesP(a, x)
    } else {
        return gammaContinuedFractionQ(a, x)
    }
}

private fun gammaSeriesP(a: Double, x: Double): Double {
    val lnPrefix = a * ln(x) - x - lnGamma(a)
    var sum = 1.0 / a
    var term = 1.0 / a
    var converged = false
    for (n in 1..GAMMA_MAX_ITERATIONS) {
        term *= x / (a + n)
        sum += term
        if (abs(term) < abs(sum) * GAMMA_EPSILON) {
            converged = true
            break
        }
    }
    checkConvergence(converged, GAMMA_MAX_ITERATIONS, sum * exp(lnPrefix)) {
        "gammaSeriesP did not converge for a=$a, x=$x after $GAMMA_MAX_ITERATIONS iterations"
    }
    return sum * exp(lnPrefix)
}

private fun gammaContinuedFractionQ(a: Double, x: Double): Double {
    val lnPrefix = a * ln(x) - x - lnGamma(a)
    // Modified Lentz's method
    val b0 = x + 1.0 - a
    var c = 1.0 / 1e-30
    var d = 1.0 / b0
    var h = d

    var converged = false
    for (i in 1..GAMMA_MAX_ITERATIONS) {
        val an = -i.toDouble() * (i.toDouble() - a)
        val bn = x + 2.0 * i + 1.0 - a

        d = an * d + bn
        if (abs(d) < 1e-30) d = 1e-30
        c = bn + an / c
        if (abs(c) < 1e-30) c = 1e-30
        d = 1.0 / d
        val delta = d * c
        h *= delta

        if (abs(delta - 1.0) < GAMMA_EPSILON) {
            converged = true
            break
        }
    }

    checkConvergence(converged, GAMMA_MAX_ITERATIONS, exp(lnPrefix) * h) {
        "gammaContinuedFractionQ did not converge for a=$a, x=$x after $GAMMA_MAX_ITERATIONS iterations"
    }

    return exp(lnPrefix) * h
}

// ── Error function ──────────────────────────────────────────────────────────

/**
 * Computes the error function at [x].
 *
 * The error function measures the probability that a standard normally distributed random
 * variable falls within the range [-x*sqrt(2), x*sqrt(2)]. It ranges from -1 to 1, with
 * erf(0) = 0. Computed via [regularizedGammaP] for high precision.
 *
 * ### Example:
 * ```kotlin
 * erf(0.0)  // 0.0
 * erf(1.0)  // 0.8427... (about 84% of the area under the standard normal curve)
 * erf(-1.5) // -erf(1.5), the function is odd
 * ```
 *
 * @param x the point at which to evaluate.
 * @return the error function value at [x], in the range [-1, 1].
 */
public fun erf(x: Double): Double {
    if (x.isNaN()) return Double.NaN
    if (x == Double.POSITIVE_INFINITY) return 1.0
    if (x == Double.NEGATIVE_INFINITY) return -1.0
    if (x == 0.0) return 0.0

    val sign = if (x >= 0) 1.0 else -1.0
    val ax = abs(x)

    return sign * regularizedGammaP(0.5, ax * ax)
}

/**
 * Computes the complementary error function at [x], equal to 1 - erf(x).
 *
 * The complementary form is useful when erf(x) is close to 1, since computing 1 - erf(x)
 * directly would lose precision. Computed via [regularizedGammaQ] for the positive branch.
 *
 * ### Example:
 * ```kotlin
 * erfc(0.0) // 1.0
 * erfc(2.0) // 0.0046... (the tail probability)
 * erf(2.0) + erfc(2.0) // 1.0
 * ```
 *
 * @param x the point at which to evaluate.
 * @return the complementary error function value at [x], in the range [0, 2].
 */
public fun erfc(x: Double): Double {
    if (x.isNaN()) return Double.NaN
    if (x == Double.POSITIVE_INFINITY) return 0.0
    if (x == Double.NEGATIVE_INFINITY) return 2.0
    if (x == 0.0) return 1.0

    return if (x >= 0) {
        regularizedGammaQ(0.5, x * x)
    } else {
        1.0 + regularizedGammaP(0.5, x * x)
    }
}

/**
 * Computes the inverse error function at [x].
 *
 * Returns the value y such that erf(y) = [x]. This is used internally to compute quantiles
 * of the normal distribution. Uses Winitzki's rational approximation as an initial guess,
 * refined with four iterations of Newton's method for high precision.
 *
 * ### Example:
 * ```kotlin
 * erfInv(0.0)                // 0.0
 * erfInv(erf(1.0))           // 1.0 (round-trip)
 * erfInv(0.8427007929497149) // 1.0 (since erf(1) ≈ 0.8427)
 * ```
 *
 * @param x the value at which to evaluate. Must be strictly between -1 and 1 (exclusive).
 * @return the inverse error function value at [x].
 */
public fun erfInv(x: Double): Double {
    if (x <= -1.0 || x >= 1.0) throw InvalidParameterException("erfInv requires -1 < x < 1, got $x")
    if (x == 0.0) return 0.0

    val a = abs(x)
    val sign = if (x > 0) 1.0 else -1.0

    // Winitzki's approximation as initial guess
    // erfInv(x) ≈ sign(x) * sqrt(sqrt((2/(pi*a) + ln(1-x^2)/2)^2 - ln(1-x^2)/a) - (2/(pi*a) + ln(1-x^2)/2))
    val lnOneMinusA2 = ln(1.0 - a * a)
    val c = 2.0 / (PI * 0.147) + lnOneMinusA2 / 2.0
    var p = sqrt(sqrt(c * c - lnOneMinusA2 / 0.147) - c)

    // Newton's method refinements for high precision
    val twoOverSqrtPi = 2.0 / sqrt(PI)
    (0..3).forEach { _ ->
        val err = erf(p) - a
        p -= err / (twoOverSqrtPi * exp(-p * p))
    }

    return sign * p
}

// ── Generalized harmonic numbers ─────────────────────────────────────────────

/**
 * Computes the generalized harmonic number H([n], [s]) = Σ_{i=1}^{n} 1/i^s.
 *
 * Uses Neumaier compensated summation (inline, no array allocation) for numerical accuracy.
 * The ordinary harmonic number is the special case s = 1. When s = 0, the result equals [n].
 *
 * ### Example:
 * ```kotlin
 * generalizedHarmonic(10, 1.0) // 2.92896... (10th harmonic number)
 * generalizedHarmonic(5, 0.0)  // 5.0
 * generalizedHarmonic(0, 2.0)  // 0.0 (empty sum)
 * ```
 *
 * @param n the upper summation limit. Must be non-negative.
 * @param s the exponent applied to each term.
 * @return the generalized harmonic number H([n], [s]).
 */
public fun generalizedHarmonic(n: Int, s: Double): Double {
    if (n < 0) throw InvalidParameterException("generalizedHarmonic requires n >= 0, got $n")
    if (n == 0) return 0.0
    var sum = 0.0
    var compensation = 0.0
    for (i in 1..n) {
        val x = 1.0 / i.toDouble().pow(s)
        val t = sum + x
        compensation += if (abs(sum) >= abs(x)) (sum - t) + x else (x - t) + sum
        sum = t
    }
    return sum + compensation
}

// ── Combinatorics ───────────────────────────────────────────────────────────

/**
 * Computes the natural logarithm of n factorial.
 *
 * The logarithmic form avoids overflow for large [n] by delegating to [lnGamma](n + 1).
 * Returns 0.0 for n = 0 and n = 1 (since 0! = 1! = 1).
 *
 * ### Example:
 * ```kotlin
 * lnFactorial(0)  // 0.0 (ln(1))
 * lnFactorial(5)  // 4.7874... (ln(120))
 * lnFactorial(20) // 42.3356... (ln(20!), no overflow)
 * ```
 *
 * @param n the non-negative integer whose factorial logarithm to compute.
 * @return the natural logarithm of [n]!.
 */
public fun lnFactorial(n: Int): Double {
    if (n < 0) throw InvalidParameterException("lnFactorial requires n >= 0, got $n")
    if (n <= 1) return 0.0
    return lnGamma(n.toDouble() + 1.0)
}

/**
 * Computes the natural logarithm of the binomial coefficient "n choose k".
 *
 * The logarithmic form avoids overflow for large [n] and [k]. Computed as
 * lnFactorial(n) - lnFactorial(k) - lnFactorial(n - k). Returns 0.0 when [k] is 0 or
 * equal to [n] (since C(n, 0) = C(n, n) = 1).
 *
 * ### Example:
 * ```kotlin
 * lnCombination(10, 3) // 4.7874... (ln(120), since C(10,3) = 120)
 * lnCombination(5, 0)  // 0.0 (ln(1))
 * ```
 *
 * @param n the total number of items. Must be non-negative.
 * @param k the number of items to choose. Must satisfy 0 <= k <= n.
 * @return the natural logarithm of C([n], [k]).
 */
public fun lnCombination(n: Int, k: Int): Double {
    if (n < 0 || k < 0 || k > n) throw InvalidParameterException("lnCombination requires 0 <= k <= n, got n=$n, k=$k")
    if (k == 0 || k == n) return 0.0
    return lnFactorial(n) - lnFactorial(k) - lnFactorial(n - k)
}

// ── Compensated summation (Neumaier) ────────────────────────────────────

/**
 * Neumaier compensated summation of the array elements.
 *
 * Reduces floating-point rounding error from O(n*epsilon) to O(epsilon) by tracking a running
 * compensation term. Improves on Kahan summation by handling the case where the next addend
 * is larger than the running sum.
 */
internal fun DoubleArray.compensatedSum(): Double {
    var sum = 0.0
    var compensation = 0.0
    for (x in this) {
        val t = sum + x
        compensation += if (abs(sum) >= abs(x)) (sum - t) + x else (x - t) + sum
        sum = t
    }
    return sum + compensation
}
