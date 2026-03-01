package org.oremif.kstats.core.exceptions

/**
 * Base exception for all errors thrown by the kstats library.
 *
 * All kstats exceptions extend this class, so catching [KStatsException] will handle any
 * error originating from the library. The optional [cause] allows chaining with an underlying
 * throwable.
 *
 * ### Example:
 * ```kotlin
 * try {
 *     doubleArrayOf().mean()
 * } catch (e: KStatsException) {
 *     println(e.message) // "array must not be empty"
 * }
 * ```
 *
 * @param message a description of what went wrong.
 * @param cause an optional underlying throwable that caused this exception.
 */
public open class KStatsException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * Thrown when the input data does not contain enough elements for the requested operation.
 *
 * For example, computing variance requires at least two values, and some hypothesis tests
 * require a minimum sample size. The [message] indicates the required minimum and the actual
 * size received.
 *
 * ### Example:
 * ```kotlin
 * try {
 *     doubleArrayOf(1.0).variance() // needs at least 2 values
 * } catch (e: InsufficientDataException) {
 *     println(e.message) // "array must have at least 2 elements, got 1"
 * }
 * ```
 *
 * @param message a description of the size requirement that was not met.
 */
public class InsufficientDataException(message: String) : KStatsException(message)

/**
 * Thrown when a parameter value is outside the valid range for the requested operation.
 *
 * This covers cases such as a negative standard deviation for a distribution, a probability
 * outside [0, 1], or mismatched array sizes. The [message] describes which parameter was
 * invalid and what value was received.
 *
 * ### Example:
 * ```kotlin
 * try {
 *     NormalDistribution(mu = 0.0, sigma = -1.0) // sigma must be positive
 * } catch (e: InvalidParameterException) {
 *     println(e.message) // "sigma must be positive, got -1.0"
 * }
 * ```
 *
 * @param message a description of the parameter constraint that was violated.
 */
public class InvalidParameterException(message: String) : KStatsException(message)

/**
 * Thrown when an iterative numerical algorithm fails to converge within its iteration limit.
 *
 * Some computations in kstats rely on iterative methods (e.g. quantile functions for certain
 * distributions, regularized incomplete gamma/beta functions). If the algorithm does not reach
 * the required precision within the allowed number of iterations, this exception is thrown
 * with diagnostic information to help identify the problem.
 *
 * ### Example:
 * ```kotlin
 * try {
 *     dist.quantile(0.999)
 * } catch (e: ConvergenceException) {
 *     println("Failed after ${e.iterations} iterations, last estimate: ${e.lastEstimate}")
 * }
 * ```
 *
 * @param message a description of the convergence failure.
 * @property iterations the number of iterations performed before the algorithm was stopped.
 * @property lastEstimate the most recent approximation computed before the algorithm was stopped.
 */
public class ConvergenceException(
    message: String,
    public val iterations: Int,
    public val lastEstimate: Double,
) : KStatsException(message)

/**
 * Thrown when the input data is mathematically degenerate for the requested operation.
 *
 * This differs from [InsufficientDataException] in that the data has enough elements but
 * lacks the variation needed for a meaningful result. For example, computing a linear regression
 * when all x-values are identical makes the slope undefined.
 *
 * ### Example:
 * ```kotlin
 * try {
 *     val x = doubleArrayOf(5.0, 5.0, 5.0)
 *     val y = doubleArrayOf(1.0, 2.0, 3.0)
 *     simpleLinearRegression(x, y) // all x-values are identical
 * } catch (e: DegenerateDataException) {
 *     println(e.message) // "All x values are identical"
 * }
 * ```
 *
 * @param message a description of why the data is degenerate.
 */
public class DegenerateDataException(message: String) : KStatsException(message)
