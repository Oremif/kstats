package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.exceptions.InsufficientDataException
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.abs
import kotlin.test.*

class ChiSquaredTestTest {

    // ===== Goodness-of-fit: existing tests =====

    @Test
    fun testGoodnessOfFitUniform() {
        // Fair die: 6 outcomes, 60 rolls total, ~10 each
        val observed = intArrayOf(8, 12, 11, 9, 10, 10)
        val result = chiSquaredTest(observed)
        assertFalse(result.isSignificant(), "Near-uniform should not be significant")
    }

    @Test
    fun testGoodnessOfFitSignificant() {
        // Very unequal
        val observed = intArrayOf(50, 5, 5, 5, 5, 30)
        val result = chiSquaredTest(observed)
        assertTrue(result.isSignificant())
    }

    // ===== Goodness-of-fit: scipy reference values =====

    @Test
    fun testGoodnessOfFitScipyUniform() {
        // observed=[8,12,11,9,10,10], sum=60, expected=10 each (uniform)
        // chi2 = (8-10)^2/10 + (12-10)^2/10 + (11-10)^2/10 + (9-10)^2/10 + 0 + 0 = 1.0
        // scipy.stats.chisquare([8,12,11,9,10,10]) → chi2=1.0, p=0.9626, df=5
        val observed = intArrayOf(8, 12, 11, 9, 10, 10)
        val result = chiSquaredTest(observed)
        assertEquals(1.0, result.statistic, 1e-10, "chi2 statistic")
        assertTrue(
            abs(result.pValue - 0.9626) < 0.01,
            "p-value: expected~0.9626, actual=${result.pValue}"
        )
        assertEquals(5.0, result.degreesOfFreedom, 1e-10, "degrees of freedom")
        assertEquals("Chi-Squared Goodness-of-Fit Test", result.testName)
    }

    @Test
    fun testGoodnessOfFitScipyCustomExpected() {
        // observed=[50,30,20], expected=[40,40,20]
        // chi2 = (50-40)^2/40 + (30-40)^2/40 + (20-20)^2/20 = 2.5 + 2.5 + 0 = 5.0
        // scipy.stats.chisquare([50,30,20], f_exp=[40,40,20]) → chi2=5.0, p=0.08209, df=2
        val observed = intArrayOf(50, 30, 20)
        val expected = doubleArrayOf(40.0, 40.0, 20.0)
        val result = chiSquaredTest(observed, expected)
        assertEquals(5.0, result.statistic, 1e-10, "chi2 statistic with custom expected")
        assertTrue(
            abs(result.pValue - 0.08209) < 0.01,
            "p-value: expected~0.0821, actual=${result.pValue}"
        )
        assertEquals(2.0, result.degreesOfFreedom, 1e-10, "degrees of freedom")
        assertFalse(result.isSignificant(), "Should not be significant at alpha=0.05")
    }

    @Test
    fun testGoodnessOfFitTwoCategories() {
        // Minimum: 2 categories
        val observed = intArrayOf(30, 70)
        val result = chiSquaredTest(observed)
        assertEquals(1.0, result.degreesOfFreedom, 1e-10, "df = k-1 = 1")
        assertTrue(result.pValue in 0.0..1.0)
    }

    // ===== Independence: existing test =====

    @Test
    fun testIndependence() {
        val table = arrayOf(intArrayOf(10, 30), intArrayOf(20, 40))
        val result = chiSquaredIndependenceTest(table)
        assertEquals(1.0, result.degreesOfFreedom, 1e-10)
    }

    // ===== Independence: scipy reference values =====

    @Test
    fun testIndependenceScipyTwoByTwo() {
        // Table: [[10,30],[20,40]], total=100
        // Row totals: 40, 60. Col totals: 30, 70.
        // Expected: [[12,28],[18,42]]
        // chi2 = (10-12)^2/12 + (30-28)^2/28 + (20-18)^2/18 + (40-42)^2/42
        //      = 0.3333 + 0.1429 + 0.2222 + 0.0952 = 0.7937
        // scipy.stats.chi2_contingency([[10,30],[20,40]], correction=False) → chi2=0.7937, p=0.3730
        val table = arrayOf(intArrayOf(10, 30), intArrayOf(20, 40))
        val result = chiSquaredIndependenceTest(table)
        assertTrue(
            abs(result.statistic - 0.7937) < 0.01,
            "chi2 statistic: expected~0.7937, actual=${result.statistic}"
        )
        assertTrue(
            abs(result.pValue - 0.3730) < 0.02,
            "p-value: expected~0.3730, actual=${result.pValue}"
        )
        assertEquals(1.0, result.degreesOfFreedom, 1e-10, "df = (2-1)*(2-1) = 1")
        assertFalse(result.isSignificant(), "Should not be significant")
        assertEquals("Chi-Squared Test of Independence", result.testName)
    }

    @Test
    fun testIndependenceScipyThreeByThree() {
        // Table: [[10,20,30],[20,15,25],[30,25,5]], total=180
        // Row totals: 60, 60, 60. Col totals: 60, 60, 60. All expected=20.
        // chi2 = sum((o-20)^2/20) = (100+0+100+0+25+25+100+25+225)/20 = 600/20 = 30.0
        // scipy.stats.chi2_contingency([[10,20,30],[20,15,25],[30,25,5]], correction=False)
        // → chi2=30.0, p=4.73e-6, df=4
        val table = arrayOf(
            intArrayOf(10, 20, 30),
            intArrayOf(20, 15, 25),
            intArrayOf(30, 25, 5)
        )
        val result = chiSquaredIndependenceTest(table)
        assertEquals(30.0, result.statistic, 1e-10, "chi2 statistic")
        assertTrue(
            result.pValue < 0.001,
            "p-value should be very small, actual=${result.pValue}"
        )
        assertEquals(4.0, result.degreesOfFreedom, 1e-10, "df = (3-1)*(3-1) = 4")
        assertTrue(result.isSignificant(), "Should be significant")
    }

    // ===== Goodness-of-fit: error cases =====

    @Test
    fun testGoodnessOfFitLessThanTwoCategories() {
        assertFailsWith<InsufficientDataException> {
            chiSquaredTest(intArrayOf(10))
        }
    }

    @Test
    fun testGoodnessOfFitEmptyObserved() {
        assertFailsWith<InsufficientDataException> {
            chiSquaredTest(intArrayOf())
        }
    }

    @Test
    fun testGoodnessOfFitMismatchedSizes() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredTest(intArrayOf(10, 20, 30), doubleArrayOf(15.0, 25.0))
        }
    }

    @Test
    fun testGoodnessOfFitNonPositiveExpected() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredTest(intArrayOf(10, 20, 30), doubleArrayOf(15.0, 0.0, 15.0))
        }
    }

    @Test
    fun testGoodnessOfFitNegativeExpected() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredTest(intArrayOf(10, 20, 30), doubleArrayOf(15.0, -5.0, 20.0))
        }
    }

    // ===== Independence: error cases =====

    @Test
    fun testIndependenceLessThanTwoRows() {
        assertFailsWith<InsufficientDataException> {
            chiSquaredIndependenceTest(arrayOf(intArrayOf(10, 20)))
        }
    }

    @Test
    fun testIndependenceLessThanTwoColumns() {
        assertFailsWith<InsufficientDataException> {
            chiSquaredIndependenceTest(arrayOf(intArrayOf(10), intArrayOf(20)))
        }
    }

    @Test
    fun testIndependenceUnequalRowLengths() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredIndependenceTest(arrayOf(intArrayOf(10, 20), intArrayOf(30)))
        }
    }

    @Test
    fun testIndependenceZeroRowTotal() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredIndependenceTest(
                arrayOf(intArrayOf(0, 0), intArrayOf(10, 20))
            )
        }
    }

    @Test
    fun testIndependenceZeroColumnTotal() {
        assertFailsWith<InvalidParameterException> {
            chiSquaredIndependenceTest(
                arrayOf(intArrayOf(10, 0), intArrayOf(20, 0))
            )
        }
    }
}
