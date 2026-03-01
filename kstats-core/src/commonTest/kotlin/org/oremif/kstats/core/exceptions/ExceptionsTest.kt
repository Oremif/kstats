package org.oremif.kstats.core.exceptions

import org.oremif.kstats.core.checkConvergence
import org.oremif.kstats.core.lnGamma
import org.oremif.kstats.descriptive.coefficientOfVariation
import org.oremif.kstats.descriptive.mean
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs

class ExceptionsTest {

    // ── Hierarchy tests ─────────────────────────────────────────────────

    @Test
    fun testKStatsExceptionIsRuntimeException() {
        val ex = KStatsException("test")
        assertIs<RuntimeException>(ex)
    }

    @Test
    fun testInsufficientDataExceptionHierarchy() {
        val ex = InsufficientDataException("not enough data")
        assertIs<KStatsException>(ex)
        assertIs<RuntimeException>(ex)
        assertEquals("not enough data", ex.message)
    }

    @Test
    fun testInvalidParameterExceptionHierarchy() {
        val ex = InvalidParameterException("bad param")
        assertIs<KStatsException>(ex)
        assertIs<RuntimeException>(ex)
        assertEquals("bad param", ex.message)
    }

    @Test
    fun testConvergenceExceptionHierarchy() {
        val ex = ConvergenceException("did not converge", iterations = 100, lastEstimate = 3.14)
        assertIs<KStatsException>(ex)
        assertIs<RuntimeException>(ex)
        assertEquals("did not converge", ex.message)
        assertEquals(100, ex.iterations)
        assertEquals(3.14, ex.lastEstimate)
    }

    @Test
    fun testDegenerateDataExceptionHierarchy() {
        val ex = DegenerateDataException("all zeros")
        assertIs<KStatsException>(ex)
        assertIs<RuntimeException>(ex)
        assertEquals("all zeros", ex.message)
    }

    @Test
    fun testKStatsExceptionPreservesCause() {
        val cause = RuntimeException("root cause")
        val ex = KStatsException("wrapper", cause)
        assertEquals(cause, ex.cause)
    }

    // ── Integration tests (one per exception type) ──────────────────────

    @Test
    fun testInsufficientDataIntegration() {
        assertFailsWith<InsufficientDataException> {
            emptyList<Double>().mean()
        }
    }

    @Test
    fun testInvalidParameterIntegration() {
        assertFailsWith<InvalidParameterException> {
            lnGamma(0.0)
        }
    }

    @Test
    fun testConvergenceExceptionIntegration() {
        assertFailsWith<ConvergenceException> {
            checkConvergence(false, 50, 1.23) { "test convergence failure" }
        }
    }

    @Test
    fun testDegenerateDataIntegration() {
        assertFailsWith<DegenerateDataException> {
            listOf(-2.0, -1.0, 1.0, 2.0).coefficientOfVariation()
        }
    }
}
