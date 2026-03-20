package org.oremif.kstats.core

import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.core.exceptions.ConvergenceException
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tan

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

    return if (x < a + 1.0) {
        // Series expansion
        gammaSeriesP(a, x)
    } else {
        // Continued fraction
        1.0 - gammaContinuedFractionQ(a, x)
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

    return if (x < a + 1.0) {
        1.0 - gammaSeriesP(a, x)
    } else {
        gammaContinuedFractionQ(a, x)
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

/**
 * Computes the inverse complementary error function at [y].
 *
 * Returns the value x such that erfc(x) = [y]. Equivalent to erfInv(1 - y) but accepts
 * the complementary probability directly, which is convenient when working with tail
 * probabilities. Used internally by quantile functions of distributions related to the
 * normal distribution.
 *
 * ### Example:
 * ```kotlin
 * erfcInv(1.0)              // 0.0 (since erfc(0) = 1)
 * erfcInv(0.5)              // 0.4769... (since erfc(0.4769) ≈ 0.5)
 * erfcInv(erfc(1.0))        // 1.0 (round-trip)
 * ```
 *
 * @param y the value at which to evaluate. Must be strictly between 0 and 2 (exclusive).
 * @return the inverse complementary error function value at [y].
 */
public fun erfcInv(y: Double): Double {
    if (y.isNaN()) return Double.NaN
    if (y <= 0.0 || y >= 2.0) throw InvalidParameterException("erfcInv requires 0 < y < 2, got $y")
    if (y == 1.0) return 0.0
    return erfInv(1.0 - y)
}

// ── Digamma and trigamma functions ───────────────────────────────────────────

// Bernoulli-number based coefficients for digamma asymptotic expansion:
// B_{2k}/(2k) for k=1..6: 1/12, 1/120, 1/252, 1/240, 1/132, 691/32760
private val DIGAMMA_ASYMPTOTIC_COEFFICIENTS = doubleArrayOf(
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
)

// Coefficients for trigamma asymptotic expansion (derived from Bernoulli numbers):
// B_{2k}/(x^{2k+1}) terms
private val TRIGAMMA_ASYMPTOTIC_COEFFICIENTS = doubleArrayOf(
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
)

/**
 * Computes the digamma (psi) function at [x].
 *
 * The digamma function is the logarithmic derivative of the gamma function:
 * psi(x) = d/dx [ln(Gamma(x))] = Gamma'(x) / Gamma(x).
 * It appears in the entropy of gamma, beta, and related distributions, and in
 * maximum-likelihood parameter estimation for exponential-family distributions.
 *
 * Uses the asymptotic expansion for x >= 6, with recurrence relation to shift
 * smaller arguments upward, and a reflection formula for negative non-integer arguments.
 *
 * ### Example:
 * ```kotlin
 * digamma(1.0)  // -0.5772... (negative Euler-Mascheroni constant)
 * digamma(2.0)  // 0.4227... (1 - gamma)
 * digamma(0.5)  // -1.9635... (-gamma - 2*ln(2))
 * ```
 *
 * @param x the point at which to evaluate. Must not be zero or a negative integer (poles of the gamma function).
 * @return the value of the digamma function at [x].
 * @throws InvalidParameterException if [x] is zero or a negative integer.
 */
public fun digamma(x: Double): Double {
    // NaN/Inf fast paths
    if (x.isNaN()) return Double.NaN
    if (x == Double.POSITIVE_INFINITY) return Double.POSITIVE_INFINITY
    if (x == Double.NEGATIVE_INFINITY) return Double.NaN

    // Poles: x = 0 or negative integer
    if (x == 0.0 || (x < 0.0 && x == floor(x))) {
        throw InvalidParameterException("digamma is not defined at x=$x (pole of the gamma function)")
    }

    // Negative non-integer: reflection formula psi(x) = psi(1-x) - pi/tan(pi*x)
    if (x < 0.0) {
        return digamma(1.0 - x) - PI / tan(PI * x)
    }

    // Recurrence relation: shift x up until x >= 6
    var result = 0.0
    var xx = x
    while (xx < 6.0) {
        result -= 1.0 / xx
        xx += 1.0
    }

    // Asymptotic expansion for x >= 6:
    // psi(x) ~ ln(x) - 1/(2x) - sum_{k=1}^{6} B_{2k}/(2k * x^{2k})
    val invX = 1.0 / xx
    val invX2 = invX * invX
    result += ln(xx) - 0.5 * invX

    var xPow = invX2 // x^(-2)
    for (coeff in DIGAMMA_ASYMPTOTIC_COEFFICIENTS) {
        result -= coeff * xPow
        xPow *= invX2
    }

    return result
}

/**
 * Computes the trigamma function at [x].
 *
 * The trigamma function is the second derivative of ln(Gamma(x)), or equivalently
 * the derivative of the digamma function: psi'(x) = d/dx [psi(x)].
 * It appears in the variance of sufficient statistics for exponential-family distributions
 * and in Newton-Raphson updates for maximum-likelihood estimation.
 *
 * Uses the asymptotic expansion for x >= 8, with recurrence relation to shift
 * smaller arguments upward, and a reflection formula for negative non-integer arguments.
 *
 * ### Example:
 * ```kotlin
 * trigamma(1.0)  // 1.6449... (pi^2/6)
 * trigamma(0.5)  // 4.9348... (pi^2/2)
 * trigamma(2.0)  // 0.6449... (pi^2/6 - 1)
 * ```
 *
 * @param x the point at which to evaluate. Must not be zero or a negative integer (poles of the gamma function).
 * @return the value of the trigamma function at [x].
 * @throws InvalidParameterException if [x] is zero or a negative integer.
 */
public fun trigamma(x: Double): Double {
    // NaN/Inf fast paths
    if (x.isNaN()) return Double.NaN
    if (x == Double.POSITIVE_INFINITY) return 0.0
    if (x == Double.NEGATIVE_INFINITY) return Double.NaN

    // Poles: x = 0 or negative integer
    if (x == 0.0 || (x < 0.0 && x == floor(x))) {
        throw InvalidParameterException("trigamma is not defined at x=$x (pole of the gamma function)")
    }

    // Negative non-integer: reflection formula psi'(x) = pi^2/sin^2(pi*x) - psi'(1-x)
    if (x < 0.0) {
        val sinPiX = sin(PI * x)
        return (PI * PI) / (sinPiX * sinPiX) - trigamma(1.0 - x)
    }

    // Recurrence relation: shift x up until x >= 8
    var result = 0.0
    var xx = x
    while (xx < 8.0) {
        result += 1.0 / (xx * xx)
        xx += 1.0
    }

    // Asymptotic expansion for x >= 8:
    // psi'(x) ~ 1/x + 1/(2x^2) + sum_{k=1}^{6} B_{2k}/x^{2k+1}
    val invX = 1.0 / xx
    val invX2 = invX * invX
    result += invX + 0.5 * invX2

    var xPow = invX2 * invX // x^(-3)
    for (coeff in TRIGAMMA_ASYMPTOTIC_COEFFICIENTS) {
        result += coeff * xPow
        xPow *= invX2
    }

    return result
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

/**
 * Computes the natural logarithm of the number of k-permutations of n items.
 *
 * The number of k-permutations P(n, k) is the number of ways to choose and arrange [k] items
 * from a set of [n] items, where order matters. The logarithmic form avoids overflow for large
 * [n] and [k]. Computed as lnFactorial(n) - lnFactorial(n - k). Returns 0.0 when [k] is 0
 * (since P(n, 0) = 1).
 *
 * ### Example:
 * ```kotlin
 * lnPermutation(5, 2)  // 2.9957... (ln(20), since P(5,2) = 20)
 * lnPermutation(5, 0)  // 0.0 (ln(1))
 * lnPermutation(5, 5)  // 4.7874... (ln(120), since P(5,5) = 5! = 120)
 * ```
 *
 * @param n the total number of items. Must be non-negative.
 * @param k the number of items to arrange. Must satisfy 0 <= k <= n.
 * @return the natural logarithm of P([n], [k]).
 * @see lnCombination
 */
public fun lnPermutation(n: Int, k: Int): Double {
    if (n < 0 || k < 0 || k > n) throw InvalidParameterException("lnPermutation requires 0 <= k <= n, got n=$n, k=$k")
    if (k == 0) return 0.0
    return lnFactorial(n) - lnFactorial(n - k)
}

/**
 * Computes the greatest common divisor of two integers using the Euclidean algorithm.
 *
 * The GCD is the largest positive integer that divides both [a] and [b] without a remainder.
 * Negative inputs are treated as their absolute values. Returns 0 when both inputs are 0.
 * Uses a tail-recursive implementation for stack safety with large inputs.
 *
 * ### Example:
 * ```kotlin
 * gcd(12, 8)   // 4
 * gcd(7, 13)   // 1 (coprime)
 * gcd(0, 5)    // 5
 * gcd(-12, 8)  // 4 (negatives treated as absolute values)
 * ```
 *
 * @param a the first integer. Must not be [Long.MIN_VALUE].
 * @param b the second integer. Must not be [Long.MIN_VALUE].
 * @return the greatest common divisor of |[a]| and |[b]|.
 * @throws InvalidParameterException if [a] or [b] is [Long.MIN_VALUE].
 * @see lcm
 */
public tailrec fun gcd(a: Long, b: Long): Long {
    if (a == Long.MIN_VALUE || b == Long.MIN_VALUE) throw InvalidParameterException(
        "gcd is not supported for Long.MIN_VALUE (absolute value overflows Long)"
    )
    val absA = if (a < 0) -a else a
    val absB = if (b < 0) -b else b
    return if (absB == 0L) absA else gcd(absB, absA % absB)
}

/**
 * Computes the least common multiple of two integers.
 *
 * The LCM is the smallest positive integer that is divisible by both [a] and [b]. Returns 0
 * when either input is 0. Negative inputs are treated as their absolute values. Computed as
 * |a| / gcd(|a|, |b|) * |b| to avoid intermediate overflow.
 *
 * ### Example:
 * ```kotlin
 * lcm(12, 8)    // 24
 * lcm(7, 13)    // 91 (coprime, so LCM = a * b)
 * lcm(0, 5)     // 0
 * lcm(-12, 8)   // 24 (negatives treated as absolute values)
 * ```
 *
 * @param a the first integer. Must not be [Long.MIN_VALUE].
 * @param b the second integer. Must not be [Long.MIN_VALUE].
 * @return the least common multiple of |[a]| and |[b]|, or 0 if either input is 0.
 * @throws InvalidParameterException if [a] or [b] is [Long.MIN_VALUE].
 * @see gcd
 */
public fun lcm(a: Long, b: Long): Long {
    if (a == 0L || b == 0L) return 0L
    if (a == Long.MIN_VALUE || b == Long.MIN_VALUE) throw InvalidParameterException(
        "lcm is not supported for Long.MIN_VALUE (absolute value overflows Long)"
    )
    val absA = if (a < 0) -a else a
    val absB = if (b < 0) -b else b
    return absA / gcd(absA, absB) * absB
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
