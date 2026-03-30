package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.ConfidenceInterval
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Shared assertion helpers for hypothesis test files.
 */
internal object TestAssertions {

    /** Assert that both statistic and pValue are NaN (non-finite input propagation). */
    fun assertNaNResult(result: TestResult, message: String = "") {
        assertTrue(result.statistic.isNaN(), "statistic should be NaN $message")
        assertTrue(result.pValue.isNaN(), "p-value should be NaN $message")
    }

    /** Assert that isSignificant is consistent with pValue at the given alpha. */
    fun assertIsSignificantConsistency(result: TestResult, alpha: Double = 0.05) {
        assertEquals(
            result.pValue < alpha,
            result.isSignificant(alpha),
            "isSignificant($alpha) should be consistent with pValue=${result.pValue}"
        )
    }

    /** Assert p-value with tolerance. */
    fun assertPValue(expected: Double, actual: Double, tol: Double = 1e-10, message: String = "") {
        assertEquals(expected, actual, tol, "p-value $message")
    }

    /** Assert a confidence interval. */
    fun assertCI(
        expectedLow: Double,
        expectedHigh: Double,
        ci: ConfidenceInterval?,
        tol: Double = 1e-8,
        message: String = ""
    ) {
        requireNotNull(ci) { "CI should not be null $message" }
        assertEquals(expectedLow, ci.lower, tol, "CI lower $message")
        assertEquals(expectedHigh, ci.upper, tol, "CI upper $message")
    }
}
