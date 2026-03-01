package org.oremif.kstats.core

import kotlin.math.*

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

internal fun lnGamma(x: Double): Double {
    require(x > 0.0) { "lnGamma requires x > 0, got $x" }
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

internal fun gamma(x: Double): Double = exp(lnGamma(x))

// ── Beta function ───────────────────────────────────────────────────────────

internal fun lnBeta(a: Double, b: Double): Double {
    require(a > 0.0 && b > 0.0) { "lnBeta requires a > 0 and b > 0, got a=$a, b=$b" }
    return lnGamma(a) + lnGamma(b) - lnGamma(a + b)
}

internal fun beta(a: Double, b: Double): Double = exp(lnBeta(a, b))

// ── Regularized incomplete beta function I_x(a, b) ─────────────────────────

private const val BETA_MAX_ITERATIONS = 200
private const val BETA_EPSILON = 1e-14

internal fun regularizedBeta(x: Double, a: Double, b: Double): Double {
    require(a > 0.0 && b > 0.0) { "regularizedBeta requires a > 0 and b > 0" }
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

        if (abs(delta - 1.0) < BETA_EPSILON) break
    }

    return prefactor * result
}

// ── Regularized incomplete gamma functions ──────────────────────────────────

private const val GAMMA_MAX_ITERATIONS = 200
private const val GAMMA_EPSILON = 1e-14

/**
 * Lower regularized incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)
 * Uses series expansion for x < a+1, continued fraction otherwise.
 */
internal fun regularizedGammaP(a: Double, x: Double): Double {
    require(a > 0.0) { "regularizedGammaP requires a > 0, got $a" }
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
 * Upper regularized incomplete gamma function Q(a, x) = 1 - P(a, x)
 */
internal fun regularizedGammaQ(a: Double, x: Double): Double {
    require(a > 0.0) { "regularizedGammaQ requires a > 0, got $a" }
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
    for (n in 1..GAMMA_MAX_ITERATIONS) {
        term *= x / (a + n)
        sum += term
        if (abs(term) < abs(sum) * GAMMA_EPSILON) break
    }
    return sum * exp(lnPrefix)
}

private fun gammaContinuedFractionQ(a: Double, x: Double): Double {
    val lnPrefix = a * ln(x) - x - lnGamma(a)
    // Modified Lentz's method
    var b0 = x + 1.0 - a
    var c = 1.0 / 1e-30
    var d = 1.0 / b0
    var h = d

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

        if (abs(delta - 1.0) < GAMMA_EPSILON) break
    }

    return exp(lnPrefix) * h
}

// ── Error function ──────────────────────────────────────────────────────────

/**
 * Error function using regularized incomplete gamma function for high precision.
 * erf(x) = sign(x) * P(0.5, x^2)
 */
internal fun erf(x: Double): Double {
    if (x.isNaN()) return Double.NaN
    if (x == Double.POSITIVE_INFINITY) return 1.0
    if (x == Double.NEGATIVE_INFINITY) return -1.0
    if (x == 0.0) return 0.0

    val sign = if (x >= 0) 1.0 else -1.0
    val ax = abs(x)

    return sign * regularizedGammaP(0.5, ax * ax)
}

internal fun erfc(x: Double): Double {
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
 * Inverse error function using rational approximation with Newton refinement.
 * Based on Winitzki's approximation with Newton corrections.
 */
internal fun erfInv(x: Double): Double {
    require(x > -1.0 && x < 1.0) { "erfInv requires -1 < x < 1, got $x" }
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
    for (i in 0..3) {
        val err = erf(p) - a
        p -= err / (twoOverSqrtPi * exp(-p * p))
    }

    return sign * p
}

// ── Combinatorics ───────────────────────────────────────────────────────────

internal fun lnFactorial(n: Int): Double {
    require(n >= 0) { "lnFactorial requires n >= 0, got $n" }
    if (n <= 1) return 0.0
    return lnGamma(n.toDouble() + 1.0)
}

internal fun lnCombination(n: Int, k: Int): Double {
    require(n >= 0 && k >= 0 && k <= n) { "lnCombination requires 0 <= k <= n, got n=$n, k=$k" }
    if (k == 0 || k == n) return 0.0
    return lnFactorial(n) - lnFactorial(k) - lnFactorial(n - k)
}
