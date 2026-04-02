package org.oremif.kstats.descriptive

/**
 * Selects the algorithm used to compute quantiles and percentiles from sample data.
 *
 * There are nine standard estimation methods defined by Hyndman and Fan (1996), numbered
 * HF1 through HF9. Each method uses a different formula to map a probability to a position
 * in the sorted data, and they differ in how they handle positions that fall between data
 * points. Methods HF1 through HF3 are discontinuous (they return actual data points),
 * while HF4 through HF9 use linear interpolation between neighboring values.
 *
 * In addition, four rounding-based methods ([LOWER], [HIGHER], [NEAREST], [MIDPOINT]) use
 * the same position formula as [LINEAR] (HF7) but apply different rounding instead of
 * linear interpolation. These are useful when you need to select an actual data point
 * rather than an interpolated value.
 *
 * The default method is [LINEAR] (HF7), which is the default in R, S, NumPy, and Pandas.
 *
 * ### Example:
 * ```kotlin
 * val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
 * data.quantile(0.25, QuantileMethod.LINEAR)          // 2.0 (default, HF7)
 * data.quantile(0.25, QuantileMethod.MEDIAN_UNBIASED) // 1.75 (HF8)
 * data.quantile(0.25, QuantileMethod.WEIBULL)         // 1.5 (HF6)
 * ```
 *
 * @see quantile
 * @see percentile
 * @see quantileSelect
 */
public enum class QuantileMethod {
    /**
     * Inverse of the empirical cumulative distribution function (HF1, R type 1).
     *
     * Uses position h = n*p + 0.5 and selects the ceiling order statistic. This is
     * a discontinuous step function that always returns an actual data point.
     */
    INVERTED_CDF,

    /**
     * Averaged inverse of the empirical CDF (HF2, R type 2).
     *
     * Same as [INVERTED_CDF] but averages the two bracketing order statistics at
     * discontinuity points. This smooths the step function of HF1 at its jump points.
     *
     * Note: at q = 0, this returns the average of x[0] and x[1] (not the minimum),
     * because q = 0 is itself a discontinuity point. This differs from all other methods,
     * which return x[0] at q = 0.
     */
    AVERAGED_INVERTED_CDF,

    /**
     * Nearest observation with ties to even, as used by SAS (HF3, R type 3).
     *
     * Uses position h = n*p and rounds to the nearest integer with banker's rounding
     * (ties to the nearest even integer). Always returns an actual data point.
     */
    CLOSEST_OBSERVATION,

    /**
     * Linear interpolation of the empirical CDF (HF4, R type 4).
     *
     * Uses position h = n*p and linear interpolation between the two bracketing order
     * statistics. For small samples, this method can produce values below the minimum
     * for small probabilities.
     */
    INTERPOLATED_INVERTED_CDF,

    /**
     * Piecewise linear function using the Hazen (1914) plotting position (HF5, R type 5).
     *
     * Uses position h = n*p + 0.5 and linear interpolation. Assigns a probability of
     * (i - 0.5) / n to each order statistic, which centers each data point in its
     * probability interval.
     */
    HAZEN,

    /**
     * Linear interpolation using the Weibull plotting position (HF6, R type 6).
     *
     * Uses position h = (n + 1)*p and linear interpolation. This is the method used by
     * Excel's `PERCENTILE.EXC` function and Minitab. Assigns a probability of i / (n + 1)
     * to each order statistic.
     */
    WEIBULL,

    /**
     * Linear interpolation, the default method in most software (HF7, R type 7).
     *
     * Uses position h = (n - 1)*p + 1 and linear interpolation. This is the default in R,
     * S, NumPy, Pandas, and kstats. Assigns a probability of (i - 1) / (n - 1) to each
     * order statistic, placing the minimum at p = 0 and the maximum at p = 1.
     */
    LINEAR,

    /**
     * Approximately median-unbiased quantile estimator (HF8, R type 8).
     *
     * Uses position h = (n + 1/3)*p + 1/3 and linear interpolation. Recommended by
     * Hyndman and Fan (1996) as the preferred method because it is approximately
     * median-unbiased for all continuous distributions.
     */
    MEDIAN_UNBIASED,

    /**
     * Approximately normal-unbiased quantile estimator (HF9, R type 9).
     *
     * Uses position h = (n + 1/4)*p + 3/8 and linear interpolation. This method
     * produces approximately unbiased quantile estimates when the data comes from a
     * normal distribution. It uses the Blom (1958) plotting position.
     */
    NORMAL_UNBIASED,

    /**
     * Returns the lower of the two bracketing data points (floor).
     *
     * Uses the same position formula as [LINEAR] (HF7) but returns the data point at or
     * below the computed position instead of interpolating. Always returns an actual data
     * point from the dataset.
     */
    LOWER,

    /**
     * Returns the higher of the two bracketing data points (ceiling).
     *
     * Uses the same position formula as [LINEAR] (HF7) but returns the data point at or
     * above the computed position instead of interpolating. Always returns an actual data
     * point from the dataset.
     */
    HIGHER,

    /**
     * Returns whichever bracketing data point is closer to the computed position.
     *
     * Uses the same position formula as [LINEAR] (HF7) and rounds to the nearest data
     * point. Ties (when the position falls exactly between two data points) are rounded
     * toward the higher value. Always returns an actual data point from the dataset.
     */
    NEAREST,

    /**
     * Returns the average of the two bracketing data points.
     *
     * Uses the same position formula as [LINEAR] (HF7) but returns the midpoint of the
     * two neighboring values instead of linear interpolation. When the position falls
     * exactly on a data point, returns that data point.
     */
    MIDPOINT,
}
