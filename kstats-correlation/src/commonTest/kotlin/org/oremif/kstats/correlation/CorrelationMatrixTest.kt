package org.oremif.kstats.correlation

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import org.oremif.kstats.descriptive.PopulationKind
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CorrelationMatrixTest {

    private val tol = 1e-10

    // --- correlationMatrix ---

    @Test
    fun testDiagonal() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(1.0, matrix[0][0], tol)
        assertEquals(1.0, matrix[1][1], tol)
    }

    @Test
    fun testSymmetry() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0)
        val y = doubleArrayOf(2.0, 3.0, 5.0, 7.0)
        val matrix = correlationMatrix(x, y)
        assertEquals(matrix[0][1], matrix[1][0], tol)
    }

    @Test
    fun testThreeVariables() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val z = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)
        val matrix = correlationMatrix(x, y, z)
        // 3×3 matrix
        assertEquals(3, matrix.size)
        assertEquals(3, matrix[0].size)
        // Diagonal = 1.0
        for (i in 0..2) assertEquals(1.0, matrix[i][i], tol)
        // Symmetry
        assertEquals(matrix[0][1], matrix[1][0], tol)
        assertEquals(matrix[0][2], matrix[2][0], tol)
        assertEquals(matrix[1][2], matrix[2][1], tol)
        // x and y perfectly correlated
        assertEquals(1.0, matrix[0][1], tol)
        // x and z perfectly negatively correlated
        assertEquals(-1.0, matrix[0][2], tol)
    }

    @Test
    fun testOffDiagonalMatchesPearson() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val matrix = correlationMatrix(x, y)
        val pearson = pearsonCorrelation(x, y)
        assertEquals(pearson.coefficient, matrix[0][1], tol)
    }

    @Test
    fun testNaNForConstantVariable() {
        val x = doubleArrayOf(1.0, 2.0, 3.0)
        val y = doubleArrayOf(5.0, 5.0, 5.0)
        val matrix = correlationMatrix(x, y)
        assertTrue(matrix[0][1].isNaN())
        assertTrue(matrix[1][0].isNaN())
    }

    // --- correlationMatrix validation ---

    @Test
    fun testSingleVariableThrows() {
        assertFailsWith<InsufficientDataException> {
            correlationMatrix(doubleArrayOf(1.0, 2.0, 3.0))
        }
    }

    @Test
    fun testDifferentSizesThrows() {
        assertFailsWith<InvalidParameterException> {
            correlationMatrix(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0)
            )
        }
    }

    @Test
    fun testTooFewObservationsThrows() {
        assertFailsWith<InsufficientDataException> {
            correlationMatrix(
                doubleArrayOf(1.0, 2.0),
                doubleArrayOf(3.0, 4.0)
            )
        }
    }

    // --- covarianceMatrix ---

    @Test
    fun testCovarianceMatrixDiagonalIsVariance() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val matrix = covarianceMatrix(x, y)
        // Diagonal = variance of each variable
        assertEquals(covariance(x, x), matrix[0][0], tol)
        assertEquals(covariance(y, y), matrix[1][1], tol)
    }

    @Test
    fun testCovarianceMatrixSymmetry() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val z = doubleArrayOf(5.0, 3.0, 7.0, 6.0, 8.0)
        val matrix = covarianceMatrix(x, y, z)
        assertEquals(3, matrix.size)
        for (i in 0..2) {
            for (j in 0..2) {
                assertEquals(matrix[i][j], matrix[j][i], tol)
            }
        }
    }

    @Test
    fun testCovarianceMatrixOffDiagonalMatchesCovariance() {
        val x = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)
        val y = doubleArrayOf(2.0, 1.0, 4.0, 3.0, 5.0)
        val matrix = covarianceMatrix(x, y)
        assertEquals(covariance(x, y), matrix[0][1], tol)
    }

    @Test
    fun testCovarianceMatrixPopulationKind() {
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val sample = covarianceMatrix(x, y)
        val population = covarianceMatrix(x, y, kind = PopulationKind.POPULATION)
        // Population covariance is smaller (divides by n, not n-1)
        assertTrue(population[0][1] < sample[0][1])
    }

    // --- covarianceMatrix validation ---

    @Test
    fun testCovarianceMatrixSingleVariableThrows() {
        assertFailsWith<InsufficientDataException> {
            covarianceMatrix(doubleArrayOf(1.0, 2.0, 3.0))
        }
    }

    @Test
    fun testCovarianceMatrixTooFewObservationsThrows() {
        assertFailsWith<InsufficientDataException> {
            covarianceMatrix(
                doubleArrayOf(1.0),
                doubleArrayOf(2.0)
            )
        }
    }

    @Test
    fun testCovarianceMatrixDifferentSizesThrows() {
        assertFailsWith<InvalidParameterException> {
            covarianceMatrix(
                doubleArrayOf(1.0, 2.0, 3.0),
                doubleArrayOf(4.0, 5.0)
            )
        }
    }

    // --- NaN ---

    @Test
    fun testCovarianceMatrixNaNPropagation() {
        val x = doubleArrayOf(1.0, Double.NaN, 3.0)
        val y = doubleArrayOf(4.0, 5.0, 6.0)
        val matrix = covarianceMatrix(x, y)
        assertTrue(matrix[0][1].isNaN())
        assertTrue(matrix[1][0].isNaN())
    }
}
