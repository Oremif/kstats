package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.*

class GTestTest {

    private val tolStat = 1e-10
    private val tolP = 1e-3

    // ── 1. Basic correctness (scipy reference values) ──────────────────────

    @Test
    fun gofUniformFairDie() {
        // scipy: power_divergence([8,12,11,9,10,10], lambda_='log-likelihood')
        //   → (1.00575521588183, 0.962100240831901)
        val observed = intArrayOf(8, 12, 11, 9, 10, 10)
        val result = gTest(observed)
        assertEquals("G-Test", result.testName)
        assertEquals(1.00575521588183, result.statistic, tolStat)
        assertEquals(0.962100240831901, result.pValue, tolP)
        assertEquals(5.0, result.degreesOfFreedom)
        assertFalse(result.isSignificant())
    }

    @Test
    fun gofUniformSignificant() {
        // scipy: power_divergence([50,5,5,5,5,30], lambda_='log-likelihood')
        //   → (96.9695165879007, 2.29865644095191e-19)
        val observed = intArrayOf(50, 5, 5, 5, 5, 30)
        val result = gTest(observed)
        assertEquals(96.9695165879007, result.statistic, tolStat)
        assertTrue(result.pValue < 1e-15)
        assertTrue(result.isSignificant())
    }

    @Test
    fun gofExplicitExpected() {
        // scipy: power_divergence([10,20,30], f_exp=[20,20,20], lambda_='log-likelihood')
        //   → (10.464962875291, 0.00534025733206688)
        val observed = intArrayOf(10, 20, 30)
        val expected = doubleArrayOf(20.0, 20.0, 20.0)
        val result = gTest(observed, expected)
        assertEquals(10.464962875291, result.statistic, tolStat)
        assertEquals(0.00534025733206688, result.pValue, tolP)
        assertEquals(2.0, result.degreesOfFreedom)
    }

    @Test
    fun independence2x2() {
        // scipy: chi2_contingency([[10,30],[20,40]], lambda_='log-likelihood', correction=False)
        //   → (0.804348646096483, 0.369796367929896, df=1)
        val table = arrayOf(intArrayOf(10, 30), intArrayOf(20, 40))
        val result = gIndependenceTest(table)
        assertEquals("G-Test of Independence", result.testName)
        assertEquals(0.804348646096483, result.statistic, tolStat)
        assertEquals(0.369796367929896, result.pValue, tolP)
        assertEquals(1.0, result.degreesOfFreedom)
        assertFalse(result.isSignificant())
    }

    @Test
    fun independence3x3() {
        // scipy: chi2_contingency([[10,20,30],[40,50,60],[70,80,90]], lambda_='log-likelihood')
        //   → (4.91730890573126, 0.295887836892569, df=4)
        val table = arrayOf(
            intArrayOf(10, 20, 30),
            intArrayOf(40, 50, 60),
            intArrayOf(70, 80, 90)
        )
        val result = gIndependenceTest(table)
        assertEquals(4.91730890573126, result.statistic, tolStat)
        assertEquals(0.295887836892569, result.pValue, tolP)
        assertEquals(4.0, result.degreesOfFreedom)
    }

    @Test
    fun independence2x3() {
        // scipy: chi2_contingency([[5,10,15],[20,25,30]], lambda_='log-likelihood')
        //   → (1.45134607849345, 0.483998709182811, df=2)
        val table = arrayOf(
            intArrayOf(5, 10, 15),
            intArrayOf(20, 25, 30)
        )
        val result = gIndependenceTest(table)
        assertEquals(1.45134607849345, result.statistic, tolStat)
        assertEquals(0.483998709182811, result.pValue, tolP)
        assertEquals(2.0, result.degreesOfFreedom)
    }

    // ── 2. Edge cases ──────────────────────────────────────────────────────

    @Test
    fun gofMinimum2Categories() {
        // scipy: power_divergence([30,70], lambda_='log-likelihood')
        //   → (16.4565757010104, 4.9777224294076e-05)
        val observed = intArrayOf(30, 70)
        val result = gTest(observed)
        assertEquals(16.4565757010104, result.statistic, tolStat)
        assertTrue(result.pValue < 0.001)
        assertEquals(1.0, result.degreesOfFreedom)
    }

    @Test
    fun gofZeroObservedCell() {
        // scipy: power_divergence([0,50,50], lambda_='log-likelihood')
        //   → (81.0930216216329, 2.45965442657983e-18)
        val observed = intArrayOf(0, 50, 50)
        val result = gTest(observed)
        assertEquals(81.0930216216329, result.statistic, tolStat)
        assertTrue(result.pValue < 1e-15)
    }

    @Test
    fun gofMultipleZeroObservedCells() {
        // scipy: power_divergence([0,0,100], lambda_='log-likelihood')
        //   → (219.722457733622, 1.94032521748262e-48)
        val observed = intArrayOf(0, 0, 100)
        val result = gTest(observed)
        assertEquals(219.722457733622, result.statistic, tolStat)
        assertTrue(result.pValue < 1e-15)
    }

    @Test
    fun gofLargeCountsUniform() {
        // scipy: power_divergence([10000,10000,10000], lambda_='log-likelihood')
        //   → (0.0, 1.0)
        val observed = intArrayOf(10000, 10000, 10000)
        val result = gTest(observed)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun independenceZeroCell() {
        // scipy: chi2_contingency([[0,30],[20,40]], lambda_='log-likelihood', correction=False)
        //   → (18.9654156350003, 1.33109424003722e-05, df=1)
        val table = arrayOf(intArrayOf(0, 30), intArrayOf(20, 40))
        val result = gIndependenceTest(table)
        assertEquals(18.9654156350003, result.statistic, tolStat)
        assertTrue(result.pValue < 0.001)
    }

    // ── 3. Degenerate cases ──────────────────────────────────────────────

    @Test
    fun gofObservedEqualsExpected() {
        // All observed equal to expected → G=0, p=1
        val observed = intArrayOf(25, 25, 25, 25)
        val expected = doubleArrayOf(25.0, 25.0, 25.0, 25.0)
        val result = gTest(observed, expected)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun gofUniformAllEqual() {
        // Uniform expected, all categories equal → G=0, p=1
        val observed = intArrayOf(25, 25, 25, 25)
        val result = gTest(observed)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    @Test
    fun independenceUniform() {
        // scipy: chi2_contingency([[25,25],[25,25]], lambda_='log-likelihood')
        //   → (0.0, 1.0, df=1)
        val table = arrayOf(intArrayOf(25, 25), intArrayOf(25, 25))
        val result = gIndependenceTest(table)
        assertEquals(0.0, result.statistic, tolStat)
        assertEquals(1.0, result.pValue, tolP)
    }

    // ── 4. Extreme parameters ────────────────────────────────────────────

    @Test
    fun gofLargeCountsSlightDifference() {
        // scipy: power_divergence([9000,10000,11000], lambda_='log-likelihood')
        //   → (200.334673854275, 3.14686640172804e-44)
        val observed = intArrayOf(9000, 10000, 11000)
        val result = gTest(observed)
        assertEquals(200.334673854275, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-15)
    }

    @Test
    fun gofVerySkewedDistribution() {
        // scipy: power_divergence([1,1,1,1,996], lambda_='log-likelihood')
        //   → (3155.62980401245, 0.0)
        val observed = intArrayOf(1, 1, 1, 1, 996)
        val result = gTest(observed)
        assertEquals(3155.62980401245, result.statistic, 1e-3)
        assertEquals(0.0, result.pValue)
    }

    @Test
    fun independenceLargeCounts() {
        // scipy: chi2_contingency([[1000,2000],[3000,4000]], lambda_='log-likelihood', correction=False)
        //   → (80.4348646096483, 3.00447077210396e-19, df=1)
        val table = arrayOf(intArrayOf(1000, 2000), intArrayOf(3000, 4000))
        val result = gIndependenceTest(table)
        assertEquals(80.4348646096483, result.statistic, 1e-6)
        assertTrue(result.pValue < 1e-15)
    }

    // ── 5. Non-finite input ──────────────────────────────────────────────

    @Test
    fun nanInExpectedThrows() {
        val observed = intArrayOf(10, 20, 30)
        val expected = doubleArrayOf(20.0, Double.NaN, 20.0)
        // NaN is not positive, so validation rejects it
        assertFailsWith<InvalidParameterException> {
            gTest(observed, expected)
        }
    }

    // ── 6. Input validation ─────────────────────────────────────────────

    @Test
    fun gofFewerThan2Categories() {
        assertFailsWith<InsufficientDataException> {
            gTest(intArrayOf(10))
        }
    }

    @Test
    fun gofSizeMismatch() {
        assertFailsWith<InvalidParameterException> {
            gTest(intArrayOf(10, 20), doubleArrayOf(15.0, 15.0, 15.0))
        }
    }

    @Test
    fun gofExpectedNotPositive() {
        assertFailsWith<InvalidParameterException> {
            gTest(intArrayOf(10, 20), doubleArrayOf(15.0, 0.0))
        }
    }

    @Test
    fun gofNegativeExpected() {
        assertFailsWith<InvalidParameterException> {
            gTest(intArrayOf(10, 20), doubleArrayOf(15.0, -5.0))
        }
    }

    @Test
    fun independenceFewerThan2Rows() {
        assertFailsWith<InsufficientDataException> {
            gIndependenceTest(arrayOf(intArrayOf(10, 20)))
        }
    }

    @Test
    fun independenceFewerThan2Cols() {
        assertFailsWith<InsufficientDataException> {
            gIndependenceTest(arrayOf(intArrayOf(10), intArrayOf(20)))
        }
    }

    @Test
    fun independenceJaggedRows() {
        assertFailsWith<InvalidParameterException> {
            gIndependenceTest(arrayOf(intArrayOf(10, 20), intArrayOf(30)))
        }
    }

    // ── 7. Property-based ───────────────────────────────────────────────

    @Test
    fun gStatisticAlwaysNonNegative() {
        val result1 = gTest(intArrayOf(5, 10, 15, 20))
        assertTrue(result1.statistic >= 0.0)

        val result2 = gTest(intArrayOf(25, 25, 25, 25))
        assertTrue(result2.statistic >= 0.0)
    }

    @Test
    fun pValueAlwaysInUnitInterval() {
        val result1 = gTest(intArrayOf(8, 12, 11, 9, 10, 10))
        assertTrue(result1.pValue in 0.0..1.0)

        val result2 = gTest(intArrayOf(50, 5, 5, 5, 5, 30))
        assertTrue(result2.pValue in 0.0..1.0)

        val result3 = gIndependenceTest(arrayOf(intArrayOf(10, 30), intArrayOf(20, 40)))
        assertTrue(result3.pValue in 0.0..1.0)
    }

    @Test
    fun moreUnevenDataGivesLargerStatistic() {
        val even = intArrayOf(10, 10, 10, 10, 10)
        val uneven = intArrayOf(5, 5, 5, 5, 30)
        val veryUneven = intArrayOf(1, 1, 1, 1, 46)

        val r1 = gTest(even)
        val r2 = gTest(uneven)
        val r3 = gTest(veryUneven)

        assertTrue(r2.statistic > r1.statistic)
        assertTrue(r3.statistic > r2.statistic)
        assertTrue(r3.pValue < r2.pValue)
        assertTrue(r2.pValue < r1.pValue)
    }

    @Test
    fun isSignificantConsistency() {
        val result = gTest(intArrayOf(10, 20, 30), doubleArrayOf(20.0, 20.0, 20.0))
        // p ≈ 0.00534, should be significant at alpha=0.05 but not at 0.001
        assertTrue(result.isSignificant(0.05))
        assertTrue(result.isSignificant(0.01))
        assertFalse(result.isSignificant(0.001))
    }
}
