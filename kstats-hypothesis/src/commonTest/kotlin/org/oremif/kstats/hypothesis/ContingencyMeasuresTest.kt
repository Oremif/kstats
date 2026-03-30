package org.oremif.kstats.hypothesis

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class ContingencyMeasuresTest {

    private val tol = 1e-10
    private val ciTol = 1e-8

    // ===== oddsRatio: Basic correctness =====

    @Test
    fun testOddsRatioKnownValues() {
        // Table [[10,5],[3,12]]: OR = (10*12)/(5*3) = 8
        val r1 = oddsRatio(arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)))
        assertEquals(8.0, r1.estimate, tol, "OR [[10,5],[3,12]]")
        assertEquals(0.95, r1.confidenceLevel, 0.0)

        // Table [[20,10],[15,25]]: OR = (20*25)/(10*15) = 3.333...
        // manual: (20*25)/(10*15) = 500/150 = 10/3
        val r2 = oddsRatio(arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)))
        assertEquals(10.0 / 3.0, r2.estimate, tol, "OR [[20,10],[15,25]]")

        // Table [[30,70],[40,60]]: OR = (30*60)/(70*40) = 1800/2800 = 9/14
        val r3 = oddsRatio(arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)))
        assertEquals(9.0 / 14.0, r3.estimate, tol, "OR [[30,70],[40,60]]")

        // Table [[100,50],[25,75]]: OR = (100*75)/(50*25) = 6
        val r4 = oddsRatio(arrayOf(intArrayOf(100, 50), intArrayOf(25, 75)))
        assertEquals(6.0, r4.estimate, tol, "OR [[100,50],[25,75]]")
    }

    @Test
    fun testOddsRatioCIKnownValues() {
        // Woolf logit CI: exp(log(OR) +/- z * sqrt(1/a + 1/b + 1/c + 1/d))
        // scipy/manual: z_0.975 = 1.95996398454005

        // [[10,5],[3,12]]: CI95 = (1.52228295380014, 42.0421182804644)
        val r1 = oddsRatio(arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)))
        assertEquals(1.52228295380014, r1.ci.lower, ciTol, "OR CI lower [[10,5],[3,12]]")
        assertEquals(42.0421182804644, r1.ci.upper, ciTol, "OR CI upper [[10,5],[3,12]]")

        // [[20,10],[15,25]]: CI95 = (1.23492513961453, 8.99739648557105)
        val r2 = oddsRatio(arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)))
        assertEquals(1.23492513961453, r2.ci.lower, ciTol, "OR CI lower [[20,10],[15,25]]")
        assertEquals(8.99739648557105, r2.ci.upper, ciTol, "OR CI upper [[20,10],[15,25]]")

        // [[30,70],[40,60]]: CI95 = (0.357906317718833, 1.15467452141235)
        val r3 = oddsRatio(arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)))
        assertEquals(0.357906317718833, r3.ci.lower, ciTol, "OR CI lower [[30,70],[40,60]]")
        assertEquals(1.15467452141235, r3.ci.upper, ciTol, "OR CI upper [[30,70],[40,60]]")
    }

    @Test
    fun testOddsRatioCustomConfidenceLevel() {
        val table = arrayOf(intArrayOf(20, 10), intArrayOf(15, 25))

        // CI90 = (1.44867949128189, 7.66982012099808)
        val r90 = oddsRatio(table, confidenceLevel = 0.90)
        assertEquals(0.90, r90.confidenceLevel, 0.0)
        assertEquals(1.44867949128189, r90.ci.lower, ciTol, "OR CI90 lower")
        assertEquals(7.66982012099808, r90.ci.upper, ciTol, "OR CI90 upper")

        // CI99 = (0.903932072054568, 12.2919757519571)
        val r99 = oddsRatio(table, confidenceLevel = 0.99)
        assertEquals(0.99, r99.confidenceLevel, 0.0)
        assertEquals(0.903932072054568, r99.ci.lower, ciTol, "OR CI99 lower")
        assertEquals(12.2919757519571, r99.ci.upper, ciTol, "OR CI99 upper")
    }

    // ===== oddsRatio: Edge cases =====

    @Test
    fun testOddsRatioSymmetricTable() {
        // [[10,10],[10,10]]: OR = 1, CI95 = (0.289502871089946, 3.45419717681939)
        val r = oddsRatio(arrayOf(intArrayOf(10, 10), intArrayOf(10, 10)))
        assertEquals(1.0, r.estimate, tol, "OR symmetric")
        assertEquals(0.289502871089946, r.ci.lower, ciTol, "OR CI lower symmetric")
        assertEquals(3.45419717681939, r.ci.upper, ciTol, "OR CI upper symmetric")
    }

    @Test
    fun testOddsRatioMinimalNonzeroTable() {
        // [[1,1],[1,1]]: OR = 1, CI95 = (0.0198425239681499, 50.3968145184122)
        val r = oddsRatio(arrayOf(intArrayOf(1, 1), intArrayOf(1, 1)))
        assertEquals(1.0, r.estimate, tol, "OR [[1,1],[1,1]]")
        assertEquals(0.0198425239681499, r.ci.lower, ciTol, "OR CI lower [[1,1],[1,1]]")
        assertEquals(50.3968145184122, r.ci.upper, ciTol, "OR CI upper [[1,1],[1,1]]")
    }

    @Test
    fun testOddsRatioZeroCellAD() {
        // [[0,5],[5,0]]: a=0, d=0 -> OR = (0*0)/(5*5) = 0
        val r = oddsRatio(arrayOf(intArrayOf(0, 5), intArrayOf(5, 0)))
        assertEquals(0.0, r.estimate, 0.0, "OR with a=0, d=0")
        // CI should be NaN (not all cells > 0)
        assertTrue(r.ci.lower.isNaN(), "CI lower should be NaN when zero cells")
        assertTrue(r.ci.upper.isNaN(), "CI upper should be NaN when zero cells")
    }

    @Test
    fun testOddsRatioZeroCellBC() {
        // [[5,0],[0,5]]: b=0, c=0 -> OR = (5*5)/(0*0) = Inf (IEEE 754)
        val r = oddsRatio(arrayOf(intArrayOf(5, 0), intArrayOf(0, 5)))
        assertEquals(Double.POSITIVE_INFINITY, r.estimate, "OR with b=0, c=0")
        // CI should be NaN (not all cells > 0)
        assertTrue(r.ci.lower.isNaN(), "CI lower should be NaN when zero cells")
        assertTrue(r.ci.upper.isNaN(), "CI upper should be NaN when zero cells")
    }

    @Test
    fun testOddsRatioSingleZeroCell() {
        // [[0,5],[5,5]]: a=0 -> OR = 0, CI = NaN (a not > 0)
        val r1 = oddsRatio(arrayOf(intArrayOf(0, 5), intArrayOf(5, 5)))
        assertEquals(0.0, r1.estimate, 0.0, "OR with a=0")
        assertTrue(r1.ci.lower.isNaN(), "CI lower when a=0")
        assertTrue(r1.ci.upper.isNaN(), "CI upper when a=0")

        // [[5,0],[5,5]]: b=0 -> OR = Inf, CI = NaN (b not > 0)
        val r2 = oddsRatio(arrayOf(intArrayOf(5, 0), intArrayOf(5, 5)))
        assertEquals(Double.POSITIVE_INFINITY, r2.estimate, "OR with b=0")
        assertTrue(r2.ci.lower.isNaN(), "CI lower when b=0")
        assertTrue(r2.ci.upper.isNaN(), "CI upper when b=0")

        // [[5,5],[0,5]]: c=0 -> OR = Inf, CI = NaN (c not > 0)
        val r3 = oddsRatio(arrayOf(intArrayOf(5, 5), intArrayOf(0, 5)))
        assertEquals(Double.POSITIVE_INFINITY, r3.estimate, "OR with c=0")
        assertTrue(r3.ci.lower.isNaN(), "CI lower when c=0")
        assertTrue(r3.ci.upper.isNaN(), "CI upper when c=0")

        // [[5,5],[5,0]]: d=0 -> OR = 0, CI = NaN (d not > 0)
        val r4 = oddsRatio(arrayOf(intArrayOf(5, 5), intArrayOf(5, 0)))
        assertEquals(0.0, r4.estimate, 0.0, "OR with d=0")
        assertTrue(r4.ci.lower.isNaN(), "CI lower when d=0")
        assertTrue(r4.ci.upper.isNaN(), "CI upper when d=0")
    }

    @Test
    fun testOddsRatioAllZeroCells() {
        // [[0,0],[0,0]]: OR = (0*0)/(0*0) = NaN (IEEE 754)
        val r = oddsRatio(arrayOf(intArrayOf(0, 0), intArrayOf(0, 0)))
        assertTrue(r.estimate.isNaN(), "OR all zeros should be NaN")
        assertTrue(r.ci.lower.isNaN(), "CI lower all zeros")
        assertTrue(r.ci.upper.isNaN(), "CI upper all zeros")
    }

    // ===== oddsRatio: Degenerate input =====

    @Test
    fun testOddsRatioInvalidTableShape() {
        assertFailsWith<InvalidParameterException> {
            oddsRatio(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(arrayOf(intArrayOf(1, 2)))
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(arrayOf(intArrayOf(1, 2), intArrayOf(3, 4), intArrayOf(5, 6)))
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(emptyArray())
        }
    }

    @Test
    fun testOddsRatioNegativeValues() {
        assertFailsWith<InvalidParameterException> {
            oddsRatio(arrayOf(intArrayOf(-1, 2), intArrayOf(3, 4)))
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(arrayOf(intArrayOf(1, 2), intArrayOf(3, -4)))
        }
    }

    @Test
    fun testOddsRatioInvalidConfidenceLevel() {
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        assertFailsWith<InvalidParameterException> {
            oddsRatio(table, confidenceLevel = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(table, confidenceLevel = 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(table, confidenceLevel = -0.5)
        }
        assertFailsWith<InvalidParameterException> {
            oddsRatio(table, confidenceLevel = 1.5)
        }
    }

    // ===== oddsRatio: Extreme parameters =====

    @Test
    fun testOddsRatioLargeCounts() {
        // [[200,300],[400,100]]: OR = (200*100)/(300*400) = 20000/120000 = 1/6
        // CI95 = (0.125599638548958, 0.221161287553786)
        val r = oddsRatio(arrayOf(intArrayOf(200, 300), intArrayOf(400, 100)))
        assertEquals(1.0 / 6.0, r.estimate, tol, "OR large counts")
        assertEquals(0.125599638548958, r.ci.lower, ciTol, "OR CI lower large counts")
        assertEquals(0.221161287553786, r.ci.upper, ciTol, "OR CI upper large counts")
    }

    @Test
    fun testOddsRatioVeryLargeOddsRatio() {
        // [[1000,1],[1,1000]]: OR = 1e6
        // CI95 = (62462.2311676586, 16009674.6674937)
        val r = oddsRatio(arrayOf(intArrayOf(1000, 1), intArrayOf(1, 1000)))
        assertEquals(1e6, r.estimate, tol, "OR very large")
        assertTrue(r.ci.lower.isFinite(), "CI lower should be finite for large OR")
        assertTrue(r.ci.upper.isFinite(), "CI upper should be finite for large OR")
        assertEquals(62462.2311676586, r.ci.lower, 1.0, "OR CI lower very large")
        assertEquals(16009674.6674937, r.ci.upper, 10.0, "OR CI upper very large")
    }

    // ===== oddsRatio: Non-finite input =====

    @Test
    fun testOddsRatioNaNConfidenceLevel() {
        // NaN confidenceLevel: 1.0 - NaN = NaN, should not throw per validation
        // (NaN <= 0.0 is false, NaN >= 1.0 is false) -> passes validation
        // alpha = 1.0 - NaN = NaN -> CI should be (NaN, NaN)
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        val r = oddsRatio(table, confidenceLevel = Double.NaN)
        assertEquals(8.0, r.estimate, tol, "OR estimate unaffected by NaN CL")
        assertTrue(r.ci.lower.isNaN(), "CI lower should be NaN for NaN CL")
        assertTrue(r.ci.upper.isNaN(), "CI upper should be NaN for NaN CL")
    }

    // ===== oddsRatio: Property-based =====

    @Test
    fun testOddsRatioCIContainsEstimate() {
        // For tables with all cells > 0, the CI should contain the point estimate
        val tables = listOf(
            arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)),
            arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)),
            arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)),
            arrayOf(intArrayOf(100, 50), intArrayOf(25, 75)),
            arrayOf(intArrayOf(10, 10), intArrayOf(10, 10)),
        )
        for (table in tables) {
            val r = oddsRatio(table)
            assertTrue(
                r.ci.lower <= r.estimate && r.estimate <= r.ci.upper,
                "CI should contain OR for ${table.map { it.toList() }}"
            )
        }
    }

    @Test
    fun testOddsRatioCIWidensWithLowerConfidence() {
        // Higher confidence level -> wider CI
        val table = arrayOf(intArrayOf(20, 10), intArrayOf(15, 25))
        val r90 = oddsRatio(table, confidenceLevel = 0.90)
        val r95 = oddsRatio(table, confidenceLevel = 0.95)
        val r99 = oddsRatio(table, confidenceLevel = 0.99)

        // Same point estimate
        assertEquals(r90.estimate, r95.estimate, 0.0)
        assertEquals(r95.estimate, r99.estimate, 0.0)

        // CI widths: 90% < 95% < 99%
        val w90 = r90.ci.upper - r90.ci.lower
        val w95 = r95.ci.upper - r95.ci.lower
        val w99 = r99.ci.upper - r99.ci.lower
        assertTrue(w90 < w95, "90% CI should be narrower than 95%")
        assertTrue(w95 < w99, "95% CI should be narrower than 99%")
    }

    @Test
    fun testOddsRatioEstimateNonNegative() {
        // OR is always >= 0 for non-negative input
        val tables = listOf(
            arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)),
            arrayOf(intArrayOf(1, 1), intArrayOf(1, 1)),
            arrayOf(intArrayOf(0, 5), intArrayOf(5, 0)),
            arrayOf(intArrayOf(100, 1), intArrayOf(1, 100)),
        )
        for (table in tables) {
            val r = oddsRatio(table)
            assertTrue(r.estimate >= 0.0 || r.estimate.isNaN(), "OR should be >= 0")
        }
    }

    @Test
    fun testOddsRatioSymmetryProperty() {
        // Transposing the table should give the reciprocal OR
        // If table = [[a,b],[c,d]], transposed = [[a,c],[b,d]]
        // OR = (a*d)/(b*c), transposed OR = (a*d)/(c*b) = same
        // Actually transposing gives: [[a,c],[b,d]] -> OR_t = (a*d)/(c*b) = OR
        // Swapping rows: [[c,d],[a,b]] -> OR_swap = (c*b)/(d*a) = 1/OR
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        val swapped = arrayOf(intArrayOf(3, 12), intArrayOf(10, 5))
        val r1 = oddsRatio(table)
        val r2 = oddsRatio(swapped)
        assertEquals(1.0, r1.estimate * r2.estimate, tol, "OR * OR_swapped = 1")
    }

    // ===== relativeRisk: Basic correctness =====

    @Test
    fun testRelativeRiskKnownValues() {
        // Table [[10,5],[3,12]]: RR = (10/15)/(3/15) = 10/3
        // scipy: relative_risk(10, 15, 3, 15).relative_risk = 3.33333333333333
        val r1 = relativeRisk(arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)))
        assertEquals(10.0 / 3.0, r1.estimate, tol, "RR [[10,5],[3,12]]")
        assertEquals(0.95, r1.confidenceLevel, 0.0)

        // Table [[20,10],[15,25]]: RR = (20/30)/(15/40) = (2/3)/(3/8) = 16/9
        // scipy: relative_risk(20, 30, 15, 40).relative_risk = 1.77777777777778
        val r2 = relativeRisk(arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)))
        assertEquals(16.0 / 9.0, r2.estimate, tol, "RR [[20,10],[15,25]]")

        // Table [[30,70],[40,60]]: RR = (30/100)/(40/100) = 0.75
        // scipy: relative_risk(30, 100, 40, 100).relative_risk = 0.75
        val r3 = relativeRisk(arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)))
        assertEquals(0.75, r3.estimate, tol, "RR [[30,70],[40,60]]")

        // Table [[100,50],[25,75]]: RR = (100/150)/(25/100) = (2/3)/(1/4) = 8/3
        // scipy: relative_risk(100, 150, 25, 100).relative_risk = 2.66666666666667
        val r4 = relativeRisk(arrayOf(intArrayOf(100, 50), intArrayOf(25, 75)))
        assertEquals(8.0 / 3.0, r4.estimate, tol, "RR [[100,50],[25,75]]")
    }

    @Test
    fun testRelativeRiskCIKnownValues() {
        // Log-based CI: exp(log(RR) +/- z * sqrt(b/(a*(a+b)) + d/(c*(c+d))))

        // [[10,5],[3,12]]: CI95 = (1.13934816990876, 9.75216479436741)
        val r1 = relativeRisk(arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)))
        assertEquals(1.13934816990876, r1.ci.lower, ciTol, "RR CI lower [[10,5],[3,12]]")
        assertEquals(9.75216479436741, r1.ci.upper, ciTol, "RR CI upper [[10,5],[3,12]]")

        // [[20,10],[15,25]]: CI95 = (1.10737003897624, 2.8540539439575)
        val r2 = relativeRisk(arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)))
        assertEquals(1.10737003897624, r2.ci.lower, ciTol, "RR CI lower [[20,10],[15,25]]")
        assertEquals(2.8540539439575, r2.ci.upper, ciTol, "RR CI upper [[20,10],[15,25]]")

        // [[30,70],[40,60]]: CI95 = (0.510981718633414, 1.10082216151366)
        val r3 = relativeRisk(arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)))
        assertEquals(0.510981718633414, r3.ci.lower, ciTol, "RR CI lower [[30,70],[40,60]]")
        assertEquals(1.10082216151366, r3.ci.upper, ciTol, "RR CI upper [[30,70],[40,60]]")
    }

    @Test
    fun testRelativeRiskCustomConfidenceLevel() {
        val table = arrayOf(intArrayOf(20, 10), intArrayOf(15, 25))

        // CI90 = (1.19493794069492, 2.64490206522567)
        val r90 = relativeRisk(table, confidenceLevel = 0.90)
        assertEquals(0.90, r90.confidenceLevel, 0.0)
        assertEquals(1.19493794069492, r90.ci.lower, ciTol, "RR CI90 lower")
        assertEquals(2.64490206522567, r90.ci.upper, ciTol, "RR CI90 upper")

        // CI99 = (0.954318566343933, 3.31178071832825)
        val r99 = relativeRisk(table, confidenceLevel = 0.99)
        assertEquals(0.99, r99.confidenceLevel, 0.0)
        assertEquals(0.954318566343933, r99.ci.lower, ciTol, "RR CI99 lower")
        assertEquals(3.31178071832825, r99.ci.upper, ciTol, "RR CI99 upper")
    }

    // ===== relativeRisk: Edge cases =====

    @Test
    fun testRelativeRiskSymmetricTable() {
        // [[10,10],[10,10]]: RR = (10/20)/(10/20) = 1
        // CI95 = (0.53805471012709, 1.85854706069537)
        val r = relativeRisk(arrayOf(intArrayOf(10, 10), intArrayOf(10, 10)))
        assertEquals(1.0, r.estimate, tol, "RR symmetric")
        assertEquals(0.53805471012709, r.ci.lower, ciTol, "RR CI lower symmetric")
        assertEquals(1.85854706069537, r.ci.upper, ciTol, "RR CI upper symmetric")
    }

    @Test
    fun testRelativeRiskMinimalNonzeroTable() {
        // [[1,1],[1,1]]: RR = (1/2)/(1/2) = 1
        // CI95 = (0.140863494093217, 7.09907138423134)
        val r = relativeRisk(arrayOf(intArrayOf(1, 1), intArrayOf(1, 1)))
        assertEquals(1.0, r.estimate, tol, "RR [[1,1],[1,1]]")
        assertEquals(0.140863494093217, r.ci.lower, ciTol, "RR CI lower [[1,1],[1,1]]")
        assertEquals(7.09907138423134, r.ci.upper, ciTol, "RR CI upper [[1,1],[1,1]]")
    }

    @Test
    fun testRelativeRiskZeroExposedInGroup1() {
        // [[0,10],[5,5]]: a=0 -> RR = 0/(5/10) = 0
        val r = relativeRisk(arrayOf(intArrayOf(0, 10), intArrayOf(5, 5)))
        assertEquals(0.0, r.estimate, 0.0, "RR with a=0")
        // CI should be NaN (a not > 0 for log-based CI)
        assertTrue(r.ci.lower.isNaN(), "CI lower when a=0")
        assertTrue(r.ci.upper.isNaN(), "CI upper when a=0")
    }

    @Test
    fun testRelativeRiskZeroExposedInGroup2() {
        // [[5,5],[0,10]]: c=0 -> RR = (5/10)/(0/10) = Inf
        val r = relativeRisk(arrayOf(intArrayOf(5, 5), intArrayOf(0, 10)))
        assertEquals(Double.POSITIVE_INFINITY, r.estimate, "RR with c=0")
        // CI should be NaN (c not > 0)
        assertTrue(r.ci.lower.isNaN(), "CI lower when c=0")
        assertTrue(r.ci.upper.isNaN(), "CI upper when c=0")
    }

    @Test
    fun testRelativeRiskZeroRowSum() {
        // [[0,0],[5,5]]: row1 = 0 -> a/row1 = 0/0 = NaN (IEEE 754)
        val r = relativeRisk(arrayOf(intArrayOf(0, 0), intArrayOf(5, 5)))
        assertTrue(r.estimate.isNaN(), "RR with zero row sum should be NaN")
        assertTrue(r.ci.lower.isNaN(), "CI lower zero row sum")
        assertTrue(r.ci.upper.isNaN(), "CI upper zero row sum")
    }

    @Test
    fun testRelativeRiskAllZeros() {
        // [[0,0],[0,0]]: both row sums = 0 -> NaN
        val r = relativeRisk(arrayOf(intArrayOf(0, 0), intArrayOf(0, 0)))
        assertTrue(r.estimate.isNaN(), "RR all zeros should be NaN")
        assertTrue(r.ci.lower.isNaN(), "CI lower all zeros")
        assertTrue(r.ci.upper.isNaN(), "CI upper all zeros")
    }

    // ===== relativeRisk: Degenerate input =====

    @Test
    fun testRelativeRiskInvalidTableShape() {
        assertFailsWith<InvalidParameterException> {
            relativeRisk(arrayOf(intArrayOf(1, 2, 3), intArrayOf(4, 5, 6)))
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(arrayOf(intArrayOf(1, 2)))
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(arrayOf(intArrayOf(1, 2), intArrayOf(3, 4), intArrayOf(5, 6)))
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(emptyArray())
        }
    }

    @Test
    fun testRelativeRiskNegativeValues() {
        assertFailsWith<InvalidParameterException> {
            relativeRisk(arrayOf(intArrayOf(-1, 2), intArrayOf(3, 4)))
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(arrayOf(intArrayOf(1, 2), intArrayOf(3, -4)))
        }
    }

    @Test
    fun testRelativeRiskInvalidConfidenceLevel() {
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        assertFailsWith<InvalidParameterException> {
            relativeRisk(table, confidenceLevel = 0.0)
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(table, confidenceLevel = 1.0)
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(table, confidenceLevel = -0.5)
        }
        assertFailsWith<InvalidParameterException> {
            relativeRisk(table, confidenceLevel = 1.5)
        }
    }

    // ===== relativeRisk: Extreme parameters =====

    @Test
    fun testRelativeRiskLargeCounts() {
        // [[200,300],[400,100]]: RR = (200/500)/(400/500) = 0.5
        // CI95 = (0.445258523594507, 0.561471564837854)
        val r = relativeRisk(arrayOf(intArrayOf(200, 300), intArrayOf(400, 100)))
        assertEquals(0.5, r.estimate, tol, "RR large counts")
        assertEquals(0.445258523594507, r.ci.lower, ciTol, "RR CI lower large counts")
        assertEquals(0.561471564837854, r.ci.upper, ciTol, "RR CI upper large counts")
    }

    @Test
    fun testRelativeRiskVeryLargeRatio() {
        // [[1000,1],[1,1000]]: RR = (1000/1001)/(1/1001) = 1000
        // CI95 = (141.001363785318, 7092.12998480322)
        val r = relativeRisk(arrayOf(intArrayOf(1000, 1), intArrayOf(1, 1000)))
        assertEquals(1000.0, r.estimate, tol, "RR very large")
        assertTrue(r.ci.lower.isFinite(), "CI lower should be finite for large RR")
        assertTrue(r.ci.upper.isFinite(), "CI upper should be finite for large RR")
        assertEquals(141.001363785318, r.ci.lower, 0.1, "RR CI lower very large")
        assertEquals(7092.12998480322, r.ci.upper, 1.0, "RR CI upper very large")
    }

    // ===== relativeRisk: Non-finite input =====

    @Test
    fun testRelativeRiskNaNConfidenceLevel() {
        // NaN confidenceLevel passes validation (NaN > 0.0 is false, NaN < 1.0 is false)
        // alpha = 1.0 - NaN = NaN -> CI = (NaN, NaN)
        val table = arrayOf(intArrayOf(10, 5), intArrayOf(3, 12))
        val r = relativeRisk(table, confidenceLevel = Double.NaN)
        assertEquals(10.0 / 3.0, r.estimate, tol, "RR estimate unaffected by NaN CL")
        assertTrue(r.ci.lower.isNaN(), "CI lower should be NaN for NaN CL")
        assertTrue(r.ci.upper.isNaN(), "CI upper should be NaN for NaN CL")
    }

    // ===== relativeRisk: Property-based =====

    @Test
    fun testRelativeRiskCIContainsEstimate() {
        val tables = listOf(
            arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)),
            arrayOf(intArrayOf(20, 10), intArrayOf(15, 25)),
            arrayOf(intArrayOf(30, 70), intArrayOf(40, 60)),
            arrayOf(intArrayOf(100, 50), intArrayOf(25, 75)),
            arrayOf(intArrayOf(10, 10), intArrayOf(10, 10)),
        )
        for (table in tables) {
            val r = relativeRisk(table)
            assertTrue(
                r.ci.lower <= r.estimate && r.estimate <= r.ci.upper,
                "CI should contain RR for ${table.map { it.toList() }}"
            )
        }
    }

    @Test
    fun testRelativeRiskCIWidensWithConfidence() {
        val table = arrayOf(intArrayOf(20, 10), intArrayOf(15, 25))
        val r90 = relativeRisk(table, confidenceLevel = 0.90)
        val r95 = relativeRisk(table, confidenceLevel = 0.95)
        val r99 = relativeRisk(table, confidenceLevel = 0.99)

        assertEquals(r90.estimate, r95.estimate, 0.0)
        assertEquals(r95.estimate, r99.estimate, 0.0)

        val w90 = r90.ci.upper - r90.ci.lower
        val w95 = r95.ci.upper - r95.ci.lower
        val w99 = r99.ci.upper - r99.ci.lower
        assertTrue(w90 < w95, "90% CI should be narrower than 95%")
        assertTrue(w95 < w99, "95% CI should be narrower than 99%")
    }

    @Test
    fun testRelativeRiskEstimateNonNegative() {
        val tables = listOf(
            arrayOf(intArrayOf(10, 5), intArrayOf(3, 12)),
            arrayOf(intArrayOf(1, 1), intArrayOf(1, 1)),
            arrayOf(intArrayOf(0, 10), intArrayOf(5, 5)),
            arrayOf(intArrayOf(100, 1), intArrayOf(1, 100)),
        )
        for (table in tables) {
            val r = relativeRisk(table)
            assertTrue(r.estimate >= 0.0 || r.estimate.isNaN(), "RR should be >= 0")
        }
    }

    @Test
    fun testRelativeRiskEqualRowProportionsGivesOne() {
        // If both groups have the same proportion of events, RR = 1
        // [[a,b],[c,d]] with a/(a+b) = c/(c+d) -> RR = 1
        val table = arrayOf(intArrayOf(5, 10), intArrayOf(10, 20))
        val r = relativeRisk(table)
        assertEquals(1.0, r.estimate, tol, "RR should be 1 when proportions are equal")
    }

    // ===== RiskEstimate data class =====

    @Test
    fun testRiskEstimateDataClassEquality() {
        val r1 = RiskEstimate(estimate = 2.0, ci = ConfidenceInterval(1.0, 3.0), confidenceLevel = 0.95)
        val r2 = RiskEstimate(estimate = 2.0, ci = ConfidenceInterval(1.0, 3.0), confidenceLevel = 0.95)
        assertEquals(r1, r2)
        assertEquals(r1.hashCode(), r2.hashCode())
    }

    @Test
    fun testRiskEstimateFields() {
        val r = RiskEstimate(estimate = 5.0, ci = ConfidenceInterval(2.0, 10.0), confidenceLevel = 0.99)
        assertEquals(5.0, r.estimate, 0.0)
        assertEquals(2.0, r.ci.lower, 0.0)
        assertEquals(10.0, r.ci.upper, 0.0)
        assertEquals(0.99, r.confidenceLevel, 0.0)
    }

    @Test
    fun testRiskEstimateCopy() {
        val r1 = RiskEstimate(estimate = 3.0, ci = ConfidenceInterval(1.0, 5.0), confidenceLevel = 0.95)
        val r2 = r1.copy(estimate = 4.0)
        assertEquals(4.0, r2.estimate, 0.0)
        assertEquals(r1.ci, r2.ci)
        assertEquals(r1.confidenceLevel, r2.confidenceLevel, 0.0)
    }
}
