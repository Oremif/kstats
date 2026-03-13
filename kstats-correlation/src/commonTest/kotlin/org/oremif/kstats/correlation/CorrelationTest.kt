package org.oremif.kstats.correlation

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PearsonCorrelationTest {

    @Test
    fun testPerfectPositive() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testPerfectNegative() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(10.0, 8.0, 6.0, 4.0, 2.0)
        val result = pearsonCorrelation(x, y)
        assertEquals(-1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testSignificance() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val result = pearsonCorrelation(x, y)
        assertTrue(result.pValue < 0.01)
    }
}

class SpearmanCorrelationTest {

    @Test
    fun testMonotonicallyIncreasing() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 4.0, 9.0, 16.0, 25.0) // monotonically increasing
        val result = spearmanCorrelation(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }
}

class KendallTauTest {

    @Test
    fun testConcordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val result = kendallTau(x, y)
        assertEquals(1.0, result.coefficient, 1e-10)
    }

    @Test
    fun testDiscordant() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val result = kendallTau(x, y)
        assertEquals(-1.0, result.coefficient, 1e-10)
    }
}

class CovarianceTest {

    @Test
    fun testCovarianceSample() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        // cov(x, y) = 5.0
        assertEquals(5.0, covariance(x, y), 1e-10)
    }
}

class CorrelationMatrixTest {

    @Test
    fun testDiagonal() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(1.0, matrix[0][0], 1e-10)
        assertEquals(1.0, matrix[1][1], 1e-10)
    }

    @Test
    fun testSymmetry() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(matrix[0][1], matrix[1][0], 1e-10)
    }
}

class RegressionTest {

    @Test
    fun testPerfectFit() {
        // y = 2x + 1
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(2.0, result.slope, 1e-10)
        assertEquals(1.0, result.intercept, 1e-10)
        assertEquals(1.0, result.rSquared, 1e-10)
    }

    @Test
    fun testPredict() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        assertEquals(13.0, result.predict(6.0), 1e-10)
    }

    @Test
    fun testResiduals() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(3.0, 5.0, 7.0, 9.0, 11.0)
        val result = simpleLinearRegression(x, y)
        for (r in result.residuals) {
            assertEquals(0.0, r, 1e-10)
        }
    }

    @Test
    fun testRSquaredImperfect() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 5.0, 6.0, 9.0, 10.0)
        val result = simpleLinearRegression(x, y)
        assertTrue(result.rSquared > 0.9)
        assertTrue(result.rSquared < 1.0)
    }
}
