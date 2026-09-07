package org.oremif.kstats.descriptive

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class RegressionMetricsTest {

    // ── rmse ──────────────────────────────────────────────────────────────

    @Test
    fun testRmseKnownValues() {
        val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
        val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
        // numpy: np.sqrt(np.mean((np.array([3.0, 5.0, 2.5, 7.0]) - np.array([2.8, 5.2, 2.1, 6.8])) ** 2))
        assertEquals(0.264575131106459, rmse(actual, predicted), 1e-10)
    }

    @Test
    fun testRmseIdenticalArraysIsZero() {
        val a = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, rmse(a, a), 1e-10)
    }

    @Test
    fun testRmseSingleElement() {
        assertEquals(2.0, rmse(doubleArrayOf(5.0), doubleArrayOf(7.0)), 1e-10)
    }

    @Test
    fun testRmseMixedSignValues() {
        val actual = doubleArrayOf(-5.0, 0.0, 5.0)
        val predicted = doubleArrayOf(-4.5, 0.5, 4.0)
        // numpy: np.sqrt(np.mean(np.array([-0.5, -0.5, 1.0]) ** 2)) = sqrt(1.5 / 3)
        assertEquals(0.7071067811865476, rmse(actual, predicted), 1e-10)
    }

    @Test
    fun testRmsePenalizesLargeErrorsMoreThanMae() {
        val actual = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0)
        val predictedSmallSpread = doubleArrayOf(1.1, 0.9, 1.1, 0.9, 1.1)
        val predictedOneOutlier = doubleArrayOf(1.0, 1.0, 1.0, 1.0, 3.0)

        assertEquals(0.1, rmse(actual, predictedSmallSpread), 1e-10)
        assertEquals(0.1, mae(actual, predictedSmallSpread), 1e-10)

        val rmseOutlier = rmse(actual, predictedOneOutlier)
        val maeOutlier = mae(actual, predictedOneOutlier)
        // numpy: np.sqrt(np.mean([0.0, 0.0, 0.0, 0.0, 4.0])) = sqrt(0.8)
        assertEquals(0.8944271909999159, rmseOutlier, 1e-10)
        // numpy: np.mean([0.0, 0.0, 0.0, 0.0, 2.0])
        assertEquals(0.4, maeOutlier, 1e-10)
        assertTrue(
            rmseOutlier > maeOutlier,
            "RMSE ($rmseOutlier) should exceed MAE ($maeOutlier) when one large outlier is present"
        )
    }

    @Test
    fun testRmseEqualsMaeWhenAllErrorsHaveEqualMagnitude() {
        val actual = doubleArrayOf(10.0, 20.0, 30.0)
        val predicted = doubleArrayOf(11.0, 21.0, 31.0)
        assertEquals(rmse(actual, predicted), mae(actual, predicted), 1e-10)
        assertEquals(1.0, rmse(actual, predicted), 1e-10)
    }

    @Test
    fun testRmseMismatchedSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            rmse(doubleArrayOf(1.0, 2.0), doubleArrayOf(1.0))
        }
    }

    @Test
    fun testRmseEmptyArraysThrows() {
        assertFailsWith<InsufficientDataException> {
            rmse(doubleArrayOf(), doubleArrayOf())
        }
    }

    @Test
    fun testRmseNaNPropagation() {
        val actual = doubleArrayOf(1.0, Double.NaN, 3.0)
        val predicted = doubleArrayOf(1.0, 2.0, 3.0)
        assertTrue(rmse(actual, predicted).isNaN())
    }

    @Test
    fun testRmseInfinityPropagation() {
        val actual = doubleArrayOf(Double.MAX_VALUE)
        val predicted = doubleArrayOf(-Double.MAX_VALUE)
        assertEquals(Double.POSITIVE_INFINITY, rmse(actual, predicted))
    }

    @Test
    fun testRmseIterableInfinityPropagation() {
        val actual = listOf(Double.MAX_VALUE)
        val predicted = listOf(-Double.MAX_VALUE)
        assertEquals(Double.POSITIVE_INFINITY, rmse(actual, predicted))
    }

    @Test
    fun testRmseInfinityMinusInfinityProducesNaN() {
        val actual = doubleArrayOf(Double.POSITIVE_INFINITY)
        val predicted = doubleArrayOf(Double.POSITIVE_INFINITY)
        assertTrue(rmse(actual, predicted).isNaN())
    }

    @Test
    fun testRmseNaNWinsOverInfinity() {
        // A NaN term must dominate an infinite one, in every overload.
        val actual = doubleArrayOf(Double.MAX_VALUE, Double.NaN)
        val predicted = doubleArrayOf(-Double.MAX_VALUE, 0.0)
        assertTrue(rmse(actual, predicted).isNaN())
        assertTrue(rmse(actual.toList(), predicted.toList()).isNaN())
        assertTrue(rmse(actual.asSequence(), predicted.asSequence()).isNaN())
    }

    // ── mae ───────────────────────────────────────────────────────────────

    @Test
    fun testMaeKnownValues() {
        val actual = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
        val predicted = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
        // numpy: np.mean(np.abs(np.array([3.0, 5.0, 2.5, 7.0]) - np.array([2.8, 5.2, 2.1, 6.8])))
        assertEquals(0.25, mae(actual, predicted), 1e-10)
    }

    @Test
    fun testMaeIdenticalArraysIsZero() {
        val a = doubleArrayOf(1.0, 2.0, 3.0)
        assertEquals(0.0, mae(a, a), 1e-10)
    }

    @Test
    fun testMaeSingleElement() {
        assertEquals(2.0, mae(doubleArrayOf(5.0), doubleArrayOf(7.0)), 1e-10)
    }

    @Test
    fun testMaeMixedSignValues() {
        val actual = doubleArrayOf(-5.0, 0.0, 5.0)
        val predicted = doubleArrayOf(-4.5, 0.5, 4.0)
        // numpy: np.mean(np.abs([-0.5, -0.5, 1.0])) = 2.0 / 3
        assertEquals(0.6666666666666666, mae(actual, predicted), 1e-10)
    }

    @Test
    fun testMaeMismatchedSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            mae(doubleArrayOf(1.0, 2.0), doubleArrayOf(1.0))
        }
    }

    @Test
    fun testMaeEmptyArraysThrows() {
        assertFailsWith<InsufficientDataException> {
            mae(doubleArrayOf(), doubleArrayOf())
        }
    }

    @Test
    fun testMaeNaNPropagation() {
        val actual = doubleArrayOf(1.0, Double.NaN, 3.0)
        val predicted = doubleArrayOf(1.0, 2.0, 3.0)
        assertTrue(mae(actual, predicted).isNaN())
    }

    @Test
    fun testMaeInfinityPropagation() {
        val actual = doubleArrayOf(Double.MAX_VALUE)
        val predicted = doubleArrayOf(-Double.MAX_VALUE)
        assertEquals(Double.POSITIVE_INFINITY, mae(actual, predicted))
    }

    @Test
    fun testMaeIterableInfinityPropagation() {
        val actual = listOf(Double.MAX_VALUE)
        val predicted = listOf(-Double.MAX_VALUE)
        assertEquals(Double.POSITIVE_INFINITY, mae(actual, predicted))
    }

    @Test
    fun testMaeInfinityMinusInfinityProducesNaN() {
        val actual = doubleArrayOf(Double.POSITIVE_INFINITY)
        val predicted = doubleArrayOf(Double.POSITIVE_INFINITY)
        assertTrue(mae(actual, predicted).isNaN())
    }

    @Test
    fun testMaeNaNWinsOverInfinity() {
        // A NaN term must dominate an infinite one, in every overload.
        val actual = doubleArrayOf(Double.MAX_VALUE, Double.NaN)
        val predicted = doubleArrayOf(-Double.MAX_VALUE, 0.0)
        assertTrue(mae(actual, predicted).isNaN())
        assertTrue(mae(actual.toList(), predicted.toList()).isNaN())
        assertTrue(mae(actual.asSequence(), predicted.asSequence()).isNaN())
    }

    // ── overload consistency (DoubleArray / Iterable / Sequence) ───────────

    @Test
    fun testRmseOverloadsAgree() {
        val actualArr = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
        val predictedArr = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
        val actualList = actualArr.toList()
        val predictedList = predictedArr.toList()

        val fromArray = rmse(actualArr, predictedArr)
        val fromIterable = rmse(actualList, predictedList)
        val fromSequence = rmse(actualList.asSequence(), predictedList.asSequence())

        assertEquals(fromArray, fromIterable, 1e-10, "Iterable overload differs from DoubleArray overload")
        assertEquals(fromArray, fromSequence, 1e-10, "Sequence overload differs from DoubleArray overload")
    }

    @Test
    fun testMaeOverloadsAgree() {
        val actualArr = doubleArrayOf(3.0, 5.0, 2.5, 7.0)
        val predictedArr = doubleArrayOf(2.8, 5.2, 2.1, 6.8)
        val actualList = actualArr.toList()
        val predictedList = predictedArr.toList()

        val fromArray = mae(actualArr, predictedArr)
        val fromIterable = mae(actualList, predictedList)
        val fromSequence = mae(actualList.asSequence(), predictedList.asSequence())

        assertEquals(fromArray, fromIterable, 1e-10, "Iterable overload differs from DoubleArray overload")
        assertEquals(fromArray, fromSequence, 1e-10, "Sequence overload differs from DoubleArray overload")
    }

    @Test
    fun testRmseOverloadsAgreeOnInfiniteError() {
        val actualArr = doubleArrayOf(Double.MAX_VALUE)
        val predictedArr = doubleArrayOf(-Double.MAX_VALUE)

        assertEquals(Double.POSITIVE_INFINITY, rmse(actualArr, predictedArr))
        assertEquals(Double.POSITIVE_INFINITY, rmse(actualArr.toList(), predictedArr.toList()))
        assertEquals(Double.POSITIVE_INFINITY, rmse(actualArr.asSequence(), predictedArr.asSequence()))
    }

    @Test
    fun testMaeOverloadsAgreeOnInfiniteError() {
        val actualArr = doubleArrayOf(Double.MAX_VALUE)
        val predictedArr = doubleArrayOf(-Double.MAX_VALUE)

        assertEquals(Double.POSITIVE_INFINITY, mae(actualArr, predictedArr))
        assertEquals(Double.POSITIVE_INFINITY, mae(actualArr.toList(), predictedArr.toList()))
        assertEquals(Double.POSITIVE_INFINITY, mae(actualArr.asSequence(), predictedArr.asSequence()))
    }

    @Test
    fun testRmseIterableMismatchedSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            rmse(listOf(1.0, 2.0), listOf(1.0))
        }
    }

    @Test
    fun testMaeIterableMismatchedSizeThrows() {
        assertFailsWith<InvalidParameterException> {
            mae(listOf(1.0, 2.0), listOf(1.0))
        }
    }

    @Test
    fun testRmseIterableMismatchedSizeThrowsWhenErrorIsInfinite() {
        // An infinite error term must not short-circuit the size check.
        assertFailsWith<InvalidParameterException> {
            rmse(listOf(Double.MAX_VALUE), listOf(-Double.MAX_VALUE, 5.0))
        }
    }

    @Test
    fun testMaeIterableMismatchedSizeThrowsWhenErrorIsInfinite() {
        assertFailsWith<InvalidParameterException> {
            mae(listOf(Double.MAX_VALUE), listOf(-Double.MAX_VALUE, 5.0))
        }
    }

    @Test
    fun testRmseIterableMismatchedSizeThrowsWhenErrorIsNaN() {
        // A NaN error term must not short-circuit the size check either.
        assertFailsWith<InvalidParameterException> {
            rmse(listOf(Double.NaN, 2.0), listOf(1.0))
        }
    }

    @Test
    fun testMaeIterableMismatchedSizeThrowsWhenErrorIsNaN() {
        assertFailsWith<InvalidParameterException> {
            mae(listOf(Double.NaN, 2.0), listOf(1.0))
        }
    }

    @Test
    fun testRmseIterableEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            rmse(emptyList<Double>(), emptyList<Double>())
        }
    }

    @Test
    fun testMaeIterableEmptyThrows() {
        assertFailsWith<InsufficientDataException> {
            mae(emptyList<Double>(), emptyList<Double>())
        }
    }
}
