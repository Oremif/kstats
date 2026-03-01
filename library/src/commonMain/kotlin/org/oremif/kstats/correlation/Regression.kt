package org.oremif.kstats.correlation

import org.oremif.kstats.descriptive.mean
import kotlin.math.sqrt

public data class SimpleLinearRegressionResult(
    val slope: Double,
    val intercept: Double,
    val rSquared: Double,
    val standardErrorSlope: Double,
    val standardErrorIntercept: Double,
    val residuals: DoubleArray,
    val n: Int
) {
    public fun predict(x: Double): Double = intercept + slope * x

    public fun predict(x: DoubleArray): DoubleArray = DoubleArray(x.size) { predict(x[it]) }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SimpleLinearRegressionResult) return false
        return slope == other.slope && intercept == other.intercept &&
            rSquared == other.rSquared && n == other.n
    }

    override fun hashCode(): Int {
        var result = slope.hashCode()
        result = 31 * result + intercept.hashCode()
        result = 31 * result + rSquared.hashCode()
        result = 31 * result + n
        return result
    }
}

/**
 * Simple linear regression: y = intercept + slope * x
 */
public fun simpleLinearRegression(x: DoubleArray, y: DoubleArray): SimpleLinearRegressionResult {
    require(x.size == y.size) { "Arrays must have the same size" }
    val n = x.size
    require(n >= 3) { "Need at least 3 observations" }

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

    require(sxx > 0.0) { "All x values are identical" }

    val slope = sxy / sxx
    val intercept = my - slope * mx

    // Residuals
    val residuals = DoubleArray(n) { y[it] - (intercept + slope * x[it]) }

    // R-squared
    val ssRes = residuals.sumOf { it * it }
    val rSquared = if (syy > 0.0) 1.0 - ssRes / syy else 1.0

    // Standard errors
    val mse = ssRes / (n - 2)
    val seSlope = sqrt(mse / sxx)
    val seIntercept = sqrt(mse * (1.0 / n + mx * mx / sxx))

    return SimpleLinearRegressionResult(
        slope = slope,
        intercept = intercept,
        rSquared = rSquared,
        standardErrorSlope = seSlope,
        standardErrorIntercept = seIntercept,
        residuals = residuals,
        n = n
    )
}
