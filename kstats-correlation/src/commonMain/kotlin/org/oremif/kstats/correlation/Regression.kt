package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.DegenerateDataException
import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.mean
import kotlin.math.sqrt

/**
 * The result of a simple linear regression fit (y = intercept + slope * x).
 *
 * ### Example:
 * ```kotlin
 * val result = simpleLinearRegression(x, y)
 * result.slope     // change in y per unit change in x
 * result.intercept // predicted y when x = 0
 * result.rSquared  // proportion of variance explained (0.0 to 1.0)
 * result.predict(6.0) // predict y for a new x value
 * ```
 *
 * @property slope the change in y for a one-unit increase in x.
 * @property intercept the predicted value of y when x is zero.
 * @property rSquared the coefficient of determination, indicating the proportion of variance
 * in y explained by the linear relationship with x. Ranges from 0.0 (no explanatory power)
 * to 1.0 (perfect fit). Returns 1.0 when y has zero variance (all values identical), since
 * there is no variance left to explain.
 * @property standardErrorSlope the standard error of the slope estimate, measuring the
 * uncertainty in [slope]. Smaller values indicate a more precise estimate.
 * @property standardErrorIntercept the standard error of the intercept estimate, measuring
 * the uncertainty in [intercept].
 * @property residuals the difference between each observed y value and the predicted value
 * (y - predicted). Residuals sum to approximately zero for a correctly fitted model.
 * @property n the number of observations used in the regression.
 */
public data class SimpleLinearRegressionResult(
    val slope: Double,
    val intercept: Double,
    val rSquared: Double,
    val standardErrorSlope: Double,
    val standardErrorIntercept: Double,
    val residuals: DoubleArray,
    val n: Int
) {
    /**
     * Predicts the y value for a single x value using the fitted model.
     *
     * ### Example:
     * ```kotlin
     * val result = simpleLinearRegression(
     *     x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0),
     *     y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
     * )
     * result.predict(6.0) // 13.0
     * ```
     *
     * @param x the input value.
     * @return the predicted y value (intercept + slope * x).
     */
    public fun predict(x: Double): Double = intercept + slope * x

    /**
     * Predicts y values for an array of x values using the fitted model.
     *
     * ### Example:
     * ```kotlin
     * val result = simpleLinearRegression(
     *     x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0),
     *     y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
     * )
     * result.predict(doubleArrayOf(6.0, 7.0)) // [13.0, 15.0]
     * ```
     *
     * @param x the array of input values.
     * @return an array of predicted y values, one per input.
     */
    public fun predict(x: DoubleArray): DoubleArray = DoubleArray(x.size) { predict(x[it]) }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is SimpleLinearRegressionResult) return false
        return slope == other.slope && intercept == other.intercept &&
            rSquared == other.rSquared && standardErrorSlope == other.standardErrorSlope &&
            standardErrorIntercept == other.standardErrorIntercept &&
            residuals.contentEquals(other.residuals) && n == other.n
    }

    override fun hashCode(): Int {
        var result = slope.hashCode()
        result = 31 * result + intercept.hashCode()
        result = 31 * result + rSquared.hashCode()
        result = 31 * result + standardErrorSlope.hashCode()
        result = 31 * result + standardErrorIntercept.hashCode()
        result = 31 * result + residuals.contentHashCode()
        result = 31 * result + n
        return result
    }
}

/**
 * Fits a simple linear regression model (y = intercept + slope * x) using ordinary least squares.
 *
 * Simple linear regression finds the straight line that best fits the data by minimizing the
 * sum of squared residuals (differences between observed and predicted y values). The result
 * includes the fitted coefficients, goodness-of-fit measure (R²), standard errors, and
 * residuals.
 *
 * ### Example:
 * ```kotlin
 * val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
 * val result = simpleLinearRegression(x, y)
 * result.slope     // 2.0
 * result.intercept // 1.0
 * result.rSquared  // 1.0 (perfect fit)
 * result.predict(6.0) // 13.0
 * ```
 *
 * @param x the array of predictor (independent variable) observations.
 * @param y the array of response (dependent variable) observations, must have the same size as [x].
 * @return a [SimpleLinearRegressionResult] containing the fitted model and diagnostics.
 * @throws DegenerateDataException if all x values are identical (no variance to fit a slope).
 */
public fun simpleLinearRegression(x: DoubleArray, y: DoubleArray): SimpleLinearRegressionResult {
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

    if (sxx <= 0.0) throw DegenerateDataException("All x values are identical")

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
