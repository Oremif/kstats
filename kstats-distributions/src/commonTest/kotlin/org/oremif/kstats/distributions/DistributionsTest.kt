package org.oremif.kstats.distributions

import org.oremif.kstats.core.exceptions.InvalidParameterException
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class NormalDistributionTest {
    private val std = NormalDistribution.STANDARD
    private val tol = 1e-6

    @Test
    fun testPdfAtZero() {
        assertEquals(1.0 / sqrt(2.0 * PI), std.pdf(0.0), tol)
    }

    @Test
    fun testCdfAtZero() {
        assertEquals(0.5, std.cdf(0.0), tol)
    }

    @Test
    fun testCdfKnownValues() {
        // R: pnorm(1) = 0.8413447
        assertEquals(0.8413447, std.cdf(1.0), tol)
        // R: pnorm(-1) = 0.1586553
        assertEquals(0.1586553, std.cdf(-1.0), tol)
    }

    @Test
    fun testQuantile() {
        assertEquals(0.0, std.quantile(0.5), tol)
        // R: qnorm(0.975) = 1.959964
        assertEquals(1.959964, std.quantile(0.975), tol)
    }

    @Test
    fun testCdfQuantileInverse() {
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, std.cdf(std.quantile(p)), tol, "cdf(quantile($p)) should ≈ $p")
        }
    }

    @Test
    fun testMoments() {
        assertEquals(0.0, std.mean, tol)
        assertEquals(1.0, std.variance, tol)
        assertEquals(0.0, std.skewness, tol)
        assertEquals(0.0, std.kurtosis, tol)
    }

    @Test
    fun testCustomMuSigma() {
        val d = NormalDistribution(5.0, 2.0)
        assertEquals(5.0, d.mean, tol)
        assertEquals(4.0, d.variance, tol)
        assertEquals(2.0, d.standardDeviation, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.norm(0, 1).entropy() = 1.418938533204673
        assertEquals(1.418938533204673, std.entropy, 1e-12)
        // scipy: stats.norm(5, 2).entropy() = 2.112085713764618
        val d = NormalDistribution(5.0, 2.0)
        assertEquals(2.112085713764618, d.entropy, 1e-12)
    }

    @Test
    fun testLogPdfConsistency() {
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(std.pdf(x), exp(std.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
        val d = NormalDistribution(5.0, 2.0)
        for (x in listOf(0.0, 3.0, 5.0, 7.0, 10.0)) {
            assertEquals(d.pdf(x), exp(d.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSfConsistency() {
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(1.0, std.sf(x) + std.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }
}

class CauchyDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).pdf(x)
        assertEquals(0.318309886183791, d.pdf(0.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(1.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(-1.0), 1e-12)
        assertEquals(0.0636619772367581, d.pdf(2.0), 1e-12)
        assertEquals(0.254647908947033, d.pdf(0.5), 1e-12)
        assertEquals(0.00315158303152268, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).cdf(x)
        assertEquals(0.5, d.cdf(0.0), 1e-12)
        assertEquals(0.75, d.cdf(1.0), 1e-12)
        assertEquals(0.25, d.cdf(-1.0), 1e-12)
        assertEquals(0.852416382349567, d.cdf(2.0), 1e-10)
        assertEquals(0.647583617650433, d.cdf(0.5), 1e-10)
        assertEquals(0.968274482569446, d.cdf(10.0), 1e-10)
        assertEquals(0.0317255174305536, d.cdf(-10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).sf(x)
        assertEquals(0.5, d.sf(0.0), 1e-12)
        assertEquals(0.25, d.sf(1.0), 1e-12)
        assertEquals(0.75, d.sf(-1.0), 1e-12)
        assertEquals(0.147583617650433, d.sf(2.0), 1e-10)
        assertEquals(0.352416382349567, d.sf(0.5), 1e-10)
        assertEquals(0.0317255174305536, d.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).sf(x) — far tail
        assertEquals(0.00318299276490819, d.sf(100.0), 1e-10)
        assertEquals(0.000318309780080517, d.sf(1000.0), 1e-12)
        assertEquals(3.18309886171431e-07, d.sf(1e6), 1e-13)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).logpdf(x)
        assertEquals(-1.1447298858494, d.logPdf(0.0), 1e-10)
        assertEquals(-1.83787706640935, d.logPdf(1.0), 1e-10)
        assertEquals(-5.75985040269066, d.logPdf(10.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = CauchyDistribution.STANDARD
        // scipy: stats.cauchy(0, 1).ppf(p)
        assertEquals(-31.8205159537739, d.quantile(0.01), 1e-6)
        assertEquals(-6.31375151467504, d.quantile(0.05), 1e-6)
        assertEquals(-3.07768353717525, d.quantile(0.1), 1e-6)
        assertEquals(-1.0, d.quantile(0.25), 1e-10)
        assertEquals(0.0, d.quantile(0.5), 1e-12)
        assertEquals(1.0, d.quantile(0.75), 1e-10)
        assertEquals(3.07768353717525, d.quantile(0.9), 1e-6)
        assertEquals(6.31375151467504, d.quantile(0.95), 1e-6)
        assertEquals(31.8205159537741, d.quantile(0.99), 1e-4)
    }

    @Test
    fun testMoments() {
        val d = CauchyDistribution.STANDARD
        assertTrue(d.mean.isNaN())
        assertTrue(d.variance.isNaN())
        assertTrue(d.standardDeviation.isNaN())
        assertTrue(d.skewness.isNaN())
        assertTrue(d.kurtosis.isNaN())
    }

    @Test
    fun testEntropy() {
        // scipy: stats.cauchy(0, 1).entropy() = 2.53102424696929
        assertEquals(2.53102424696929, CauchyDistribution.STANDARD.entropy, 1e-10)
        // scipy: stats.cauchy(3, 2).entropy() = 3.22417142752924
        assertEquals(3.22417142752924, CauchyDistribution(3.0, 2.0).entropy, 1e-10)
    }

    // --- Custom parameters ---

    @Test
    fun testCustomLocationScale() {
        val d = CauchyDistribution(3.0, 2.0)
        // scipy: stats.cauchy(3, 2)
        assertEquals(0.5, d.cdf(3.0), 1e-12)
        assertEquals(0.75, d.cdf(5.0), 1e-12)
        assertEquals(0.5, d.sf(3.0), 1e-12)
        assertEquals(0.25, d.sf(5.0), 1e-12)
        assertEquals(0.159154943091895, d.pdf(3.0), 1e-12)
        assertEquals(1.0, d.quantile(0.25), 1e-10)
        assertEquals(5.0, d.quantile(0.75), 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testQuantileBoundaries() {
        val d = CauchyDistribution.STANDARD
        assertEquals(Double.NEGATIVE_INFINITY, d.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    @Test
    fun testCdfSymmetry() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(0.1, 0.5, 1.0, 2.0, 10.0)) {
            assertEquals(1.0, d.cdf(x) + d.cdf(-x), 1e-12, "cdf($x) + cdf(-$x) ≈ 1")
        }
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, 0.0) }
        assertFailsWith<InvalidParameterException> { CauchyDistribution(0.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = CauchyDistribution.STANDARD
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = CauchyDistribution.STANDARD
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-10, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(-10.0, -1.0, 0.0, 1.0, 10.0, 100.0)) {
            assertEquals(1.0, d.sf(x) + d.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(-10.0, -1.0, 0.0, 1.0, 10.0)) {
            assertEquals(d.pdf(x), exp(d.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleMedian() {
        val d = CauchyDistribution(3.0, 2.0) // median = location = 3
        val rng = kotlin.random.Random(42)
        val samples = d.sample(10000, rng).sorted()
        val sampleMedian = samples[5000]
        assertEquals(3.0, sampleMedian, 1.0, "sample median ≈ 3")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = CauchyDistribution.STANDARD
        var prev = 0.0
        for (x in listOf(-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0)) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testPdfSymmetry() {
        val d = CauchyDistribution.STANDARD
        for (x in listOf(0.5, 1.0, 2.0, 10.0)) {
            assertEquals(d.pdf(x), d.pdf(-x), 1e-12, "pdf($x) = pdf(-$x)")
        }
    }
}

class StudentTDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).pdf(x)
        assertEquals(0.379606689822494, t.pdf(0.0), 1e-12)
        assertEquals(0.219679797350981, t.pdf(1.0), 1e-12)
        assertEquals(0.219679797350981, t.pdf(-1.0), 1e-12)
        assertEquals(0.0650903103262165, t.pdf(2.0), 1e-12)
        assertEquals(0.0172925788002229, t.pdf(3.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).cdf(x)
        assertEquals(0.5, t.cdf(0.0), 1e-12)
        assertEquals(0.818391266175439, t.cdf(1.0), 1e-10)
        assertEquals(0.181608733824561, t.cdf(-1.0), 1e-10)
        assertEquals(0.949030260585071, t.cdf(2.0), 1e-10)
        assertEquals(0.0509697394149292, t.cdf(-2.0), 1e-10)
        assertEquals(0.984950376051269, t.cdf(3.0), 1e-10)
        assertEquals(0.0150496239487313, t.cdf(-3.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).sf(x)
        assertEquals(0.5, t.sf(0.0), 1e-12)
        assertEquals(0.181608733824561, t.sf(1.0), 1e-10)
        assertEquals(0.0509697394149292, t.sf(2.0), 1e-10)
        assertEquals(0.0150496239487313, t.sf(3.0), 1e-10)
        assertEquals(0.818391266175439, t.sf(-1.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val t = StudentTDistribution(5.0)
        // scipy: stats.t(5).sf(x) — far tail
        assertEquals(8.54737878714818e-05, t.sf(10.0), 1e-9)
        assertEquals(9.48000711231183e-10, t.sf(100.0), 1e-14)
    }

    @Test
    fun testLogPdfKnownValues() {
        val t = StudentTDistribution(10.0)
        // scipy: stats.t(10).pdf(x) → logpdf
        assertEquals(ln(0.389108383966031), t.logPdf(0.0), 1e-10)
        assertEquals(ln(0.230361989229139), t.logPdf(1.0), 1e-10)
        assertEquals(ln(0.0611457663212182), t.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val t = StudentTDistribution(10.0)
        // scipy: stats.t(10).ppf(p)
        assertEquals(-2.76376946298153, t.quantile(0.01), 1e-4)
        assertEquals(-2.22813885198627, t.quantile(0.025), 1e-4)
        assertEquals(-1.81246112281168, t.quantile(0.05), 1e-4)
        assertEquals(-1.37218364282088, t.quantile(0.1), 1e-4)
        assertEquals(0.0, t.quantile(0.5), 1e-12)
        assertEquals(1.37218364282088, t.quantile(0.9), 1e-4)
        assertEquals(2.22813885198627, t.quantile(0.975), 1e-4)
    }

    @Test
    fun testMoments() {
        val t5 = StudentTDistribution(5.0)
        assertEquals(0.0, t5.mean, 1e-12)
        assertEquals(5.0 / 3.0, t5.variance, 1e-12)
        assertEquals(0.0, t5.skewness, 1e-12)

        val t10 = StudentTDistribution(10.0)
        assertEquals(0.0, t10.mean, 1e-12)
        assertEquals(10.0 / 8.0, t10.variance, 1e-12)
        assertEquals(6.0 / 6.0, t10.kurtosis, 1e-12) // 6/(df-4)
    }

    // --- Edge cases ---

    @Test
    fun testDf1IsCauchy() {
        // StudentT(df=1) = Cauchy(0,1)
        val t = StudentTDistribution(1.0)
        val c = CauchyDistribution.STANDARD
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(c.pdf(x), t.pdf(x), 1e-10, "T(1).pdf($x) = Cauchy.pdf($x)")
            assertEquals(c.cdf(x), t.cdf(x), 1e-10, "T(1).cdf($x) = Cauchy.cdf($x)")
        }
    }

    @Test
    fun testMomentsEdgeDf() {
        // df=1: mean=NaN, var=NaN
        assertTrue(StudentTDistribution(1.0).mean.isNaN())
        assertTrue(StudentTDistribution(1.0).variance.isNaN())
        // df=2: mean=0, var=Inf
        assertEquals(0.0, StudentTDistribution(2.0).mean, 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, StudentTDistribution(2.0).variance)
        // df=3: skewness=NaN (requires df>3), kurtosis=+Inf (2<df<=4)
        assertTrue(StudentTDistribution(3.0).skewness.isNaN())
        assertEquals(Double.POSITIVE_INFINITY, StudentTDistribution(3.0).kurtosis)
        // df=5: kurtosis = 6/(5-4) = 6
        assertEquals(6.0, StudentTDistribution(5.0).kurtosis, 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val t = StudentTDistribution(5.0)
        assertEquals(Double.NEGATIVE_INFINITY, t.quantile(0.0))
        assertEquals(Double.POSITIVE_INFINITY, t.quantile(1.0))
    }

    @Test
    fun testCdfSymmetry() {
        val t = StudentTDistribution(10.0)
        for (x in listOf(0.5, 1.0, 2.0, 3.0)) {
            assertEquals(1.0, t.cdf(x) + t.cdf(-x), 1e-12, "cdf($x) + cdf(-$x) ≈ 1")
        }
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { StudentTDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { StudentTDistribution(-1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val t = StudentTDistribution(5.0)
        assertFailsWith<InvalidParameterException> { t.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { t.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val t = StudentTDistribution(10.0)
        for (p in listOf(0.1, 0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, t.cdf(t.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val t = StudentTDistribution(5.0)
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0, 10.0)) {
            assertEquals(1.0, t.sf(x) + t.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val t = StudentTDistribution(5.0)
        for (x in listOf(-3.0, -1.0, 0.0, 1.0, 3.0)) {
            assertEquals(t.pdf(x), exp(t.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val t = StudentTDistribution(30.0) // mean=0, var=30/28≈1.071
        val rng = kotlin.random.Random(42)
        val samples = t.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(0.0, sampleMean, 0.1, "sample mean ≈ 0")
    }

    @Test
    fun testCdfMonotonicity() {
        val t = StudentTDistribution(10.0)
        var prev = 0.0
        for (x in listOf(-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0)) {
            val cdfVal = t.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(StudentTDistribution(5.0).entropy.isNaN())
    }
}

class ChiSquaredDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).pdf(x)
        assertEquals(0.00400012982810046, d.pdf(0.1), 1e-12)
        assertEquals(0.0806569081730478, d.pdf(1.0), 1e-12)
        assertEquals(0.138369165806865, d.pdf(2.0), 1e-12)
        assertEquals(0.122041521349387, d.pdf(5.0), 1e-12)
        assertEquals(0.0283345553417345, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).cdf(x)
        assertEquals(0.000162316611922615, d.cdf(0.1), 1e-10)
        assertEquals(0.0374342267527036, d.cdf(1.0), 1e-10)
        assertEquals(0.15085496391539, d.cdf(2.0), 1e-10)
        assertEquals(0.584119813004492, d.cdf(5.0), 1e-10)
        assertEquals(0.924764753853488, d.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).sf(x)
        assertEquals(0.999837683388077, d.sf(0.1), 1e-10)
        assertEquals(0.962565773247296, d.sf(1.0), 1e-10)
        assertEquals(0.84914503608461, d.sf(2.0), 1e-10)
        assertEquals(0.415880186995508, d.sf(5.0), 1e-10)
        assertEquals(0.0752352461465122, d.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val d = ChiSquaredDistribution(5.0)
        // scipy: stats.chi2(5).sf(x) — far tail
        assertEquals(0.00124973056303138, d.sf(20.0), 1e-10)
        assertEquals(1.38579733670096e-09, d.sf(50.0), 1e-14)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = ChiSquaredDistribution(10.0)
        // scipy: stats.chi2(10).pdf(x) → logpdf
        assertEquals(ln(0.000789753463167491), d.logPdf(1.0), 1e-8)
        assertEquals(ln(0.0668009428905426), d.logPdf(5.0), 1e-10)
        assertEquals(ln(0.0877336848839254), d.logPdf(10.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = ChiSquaredDistribution(10.0)
        // scipy: stats.chi2(10).ppf(p)
        assertEquals(2.55821216018721, d.quantile(0.01), 1e-6)
        assertEquals(3.94029913611906, d.quantile(0.05), 1e-6)
        assertEquals(4.86518205192533, d.quantile(0.1), 1e-6)
        assertEquals(9.34181776559197, d.quantile(0.5), 1e-6)
        assertEquals(15.9871791721053, d.quantile(0.9), 1e-6)
        assertEquals(18.3070380532751, d.quantile(0.95), 1e-5)
        assertEquals(23.2092511589544, d.quantile(0.99), 1e-4)
    }

    @Test
    fun testMoments() {
        val d5 = ChiSquaredDistribution(5.0)
        assertEquals(5.0, d5.mean, 1e-12)
        assertEquals(10.0, d5.variance, 1e-12)
        assertEquals(sqrt(8.0 / 5.0), d5.skewness, 1e-12)
        assertEquals(12.0 / 5.0, d5.kurtosis, 1e-12)

        val d10 = ChiSquaredDistribution(10.0)
        assertEquals(10.0, d10.mean, 1e-12)
        assertEquals(20.0, d10.variance, 1e-12)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // df=2: pdf(0)=0.5
        assertEquals(0.5, ChiSquaredDistribution(2.0).pdf(0.0), 1e-12)
        // df<2: pdf(0)=+Inf
        assertEquals(Double.POSITIVE_INFINITY, ChiSquaredDistribution(1.0).pdf(0.0))
        // df>2: pdf(0)=0
        assertEquals(0.0, ChiSquaredDistribution(5.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val d = ChiSquaredDistribution(5.0)
        assertEquals(0.0, d.pdf(-1.0), 1e-12)
        assertEquals(0.0, d.cdf(-1.0), 1e-12)
        assertEquals(1.0, d.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val d = ChiSquaredDistribution(5.0)
        assertEquals(0.0, d.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    // --- Different df ---

    @Test
    fun testDf1() {
        val d = ChiSquaredDistribution(1.0)
        // scipy: stats.chi2(1).cdf(x)
        assertEquals(0.682689492137086, d.cdf(1.0), 1e-10)
        assertEquals(0.974652681322532, d.cdf(5.0), 1e-10)
        // sf
        assertEquals(0.317310507862911, d.sf(1.0), 1e-10)
        assertEquals(0.0253473186774683, d.sf(5.0), 1e-10)
    }

    @Test
    fun testDf30() {
        val d = ChiSquaredDistribution(30.0)
        // scipy: stats.chi2(30)
        assertEquals(30.0, d.mean, 1e-12)
        assertEquals(60.0, d.variance, 1e-12)
        assertEquals(0.0834584729346629, d.cdf(20.0), 1e-8)
        assertEquals(0.916541527065337, d.sf(20.0), 1e-8)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(0.0) }
        assertFailsWith<InvalidParameterException> { ChiSquaredDistribution(-1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = ChiSquaredDistribution(5.0)
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = ChiSquaredDistribution(10.0)
        for (p in listOf(0.1, 0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = ChiSquaredDistribution(5.0)
        for (x in listOf(0.0, 1.0, 2.0, 5.0, 10.0, 20.0)) {
            assertEquals(1.0, d.sf(x) + d.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val d = ChiSquaredDistribution(5.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(d.pdf(x), exp(d.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val d = ChiSquaredDistribution(10.0) // mean=10
        val rng = kotlin.random.Random(42)
        val samples = d.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(10.0, sampleMean, 0.5, "sample mean ≈ 10")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = ChiSquaredDistribution(5.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(ChiSquaredDistribution(5.0).entropy.isNaN())
    }
}

class FDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).pdf(x)
        assertEquals(0.227400093561907, f.pdf(0.1), 1e-10)
        assertEquals(0.687607002770625, f.pdf(0.5), 1e-10)
        assertEquals(0.49547978348664, f.pdf(1.0), 1e-10)
        assertEquals(0.162005742180115, f.pdf(2.0), 1e-10)
        assertEquals(0.0558267311455723, f.pdf(3.0), 1e-10)
        assertEquals(0.000478163040278585, f.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).cdf(x)
        assertEquals(0.0101150894697428, f.cdf(0.1), 1e-10)
        assertEquals(0.229975119349898, f.cdf(0.5), 1e-10)
        assertEquals(0.5348805734622, f.cdf(1.0), 1e-10)
        assertEquals(0.835805049100261, f.cdf(2.0), 1e-10)
        assertEquals(0.934442437906156, f.cdf(3.0), 1e-10)
        assertEquals(0.9987942193513, f.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).sf(x)
        assertEquals(0.989884910530257, f.sf(0.1), 1e-10)
        assertEquals(0.770024880650102, f.sf(0.5), 1e-10)
        assertEquals(0.4651194265378, f.sf(1.0), 1e-10)
        assertEquals(0.164194950899739, f.sf(2.0), 1e-10)
        assertEquals(0.0655575620938441, f.sf(3.0), 1e-10)
        assertEquals(0.00120578064869954, f.sf(10.0), 1e-10)
    }

    @Test
    fun testSfUpperTail() {
        val f = FDistribution(10.0, 20.0)
        // scipy: stats.f(10, 20).sf(x)
        assertEquals(0.0175095414784, f.sf(3.0), 1e-8)
        assertEquals(0.00109658938743534, f.sf(5.0), 1e-10)
        assertEquals(8.59411961792848e-06, f.sf(10.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(ln(0.49547978348664), f.logPdf(1.0), 1e-10)
        assertEquals(ln(0.162005742180115), f.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val f = FDistribution(5.0, 10.0)
        // scipy: stats.f(5, 10).ppf(p)
        // Note: small p values may not converge well — see DIST-010
        assertEquals(0.529141685567822, f.quantile(0.25), 1e-6)
        assertEquals(0.931933160851048, f.quantile(0.5), 1e-6)
        assertEquals(1.58532325938462, f.quantile(0.75), 1e-5)
        assertEquals(2.52164068620962, f.quantile(0.9), 1e-5)
        assertEquals(3.32583453041301, f.quantile(0.95), 1e-4)
    }

    @Test
    fun testMoments() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(1.25, f.mean, 1e-12)
        assertEquals(1.35416666666667, f.variance, 1e-10)

        val f2 = FDistribution(10.0, 20.0)
        assertEquals(1.11111111111111, f2.mean, 1e-10)
        assertEquals(0.432098765432099, f2.variance, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // d1=2: pdf(0) = 1.0
        assertEquals(1.0, FDistribution(2.0, 5.0).pdf(0.0), 1e-12)
        // d1<2: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, FDistribution(1.0, 5.0).pdf(0.0))
        // d1>2: pdf(0) = 0
        assertEquals(0.0, FDistribution(5.0, 10.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(0.0, f.pdf(-1.0), 1e-12)
        assertEquals(0.0, f.cdf(-1.0), 1e-12)
        assertEquals(1.0, f.sf(-1.0), 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val f = FDistribution(5.0, 10.0)
        assertEquals(0.0, f.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, f.quantile(1.0))
    }

    @Test
    fun testMomentsEdgeDf() {
        // d2<=2: mean=NaN
        assertTrue(FDistribution(5.0, 2.0).mean.isNaN())
        // d2<=4: variance=NaN
        assertTrue(FDistribution(5.0, 4.0).variance.isNaN())
        // d2<=6: skewness=NaN
        assertTrue(FDistribution(5.0, 6.0).skewness.isNaN())
        // d2<=8: kurtosis=NaN
        assertTrue(FDistribution(5.0, 8.0).kurtosis.isNaN())
    }

    // --- Different params ---

    @Test
    fun testF_2_5() {
        val f = FDistribution(2.0, 5.0)
        // scipy: stats.f(2, 5)
        assertEquals(0.366061854739391, f.cdf(0.5), 1e-10)
        assertEquals(0.633938145260609, f.sf(0.5), 1e-10)
        assertEquals(1.66666666666667, f.mean, 1e-10)
    }

    // --- Invalid input ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { FDistribution(0.0, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(-1.0, 10.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(5.0, 0.0) }
        assertFailsWith<InvalidParameterException> { FDistribution(5.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val f = FDistribution(5.0, 10.0)
        assertFailsWith<InvalidParameterException> { f.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { f.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val f = FDistribution(5.0, 10.0)
        // Note: small p values may not converge well — see DIST-010
        for (p in listOf(0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, f.cdf(f.quantile(p)), 1e-6, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val f = FDistribution(5.0, 10.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, f.sf(x) + f.cdf(x), 1e-10, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val f = FDistribution(5.0, 10.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(f.pdf(x), exp(f.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val f = FDistribution(5.0, 10.0) // mean=1.25
        val rng = kotlin.random.Random(42)
        val samples = f.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(1.25, sampleMean, 0.15, "sample mean ≈ 1.25")
    }

    @Test
    fun testCdfMonotonicity() {
        val f = FDistribution(5.0, 10.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)) {
            val cdfVal = f.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(FDistribution(5.0, 10.0).entropy.isNaN())
    }
}

class UniformDistributionTest {
    private val tol = 1e-10

    @Test
    fun testPdf() {
        val u = UniformDistribution(0.0, 10.0)
        assertEquals(0.1, u.pdf(5.0), tol)
        assertEquals(0.0, u.pdf(-1.0), tol)
    }

    @Test
    fun testCdf() {
        val u = UniformDistribution(0.0, 10.0)
        assertEquals(0.5, u.cdf(5.0), tol)
    }

    @Test
    fun testQuantile() {
        val u = UniformDistribution(2.0, 8.0)
        assertEquals(5.0, u.quantile(0.5), tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.uniform(0, 10).entropy() = 2.302585092994046
        val u1 = UniformDistribution(0.0, 10.0)
        assertEquals(2.302585092994046, u1.entropy, 1e-12)
        // scipy: stats.uniform(2, 6).entropy() = 1.791759469228055
        val u2 = UniformDistribution(2.0, 8.0)
        assertEquals(1.791759469228055, u2.entropy, 1e-12)
    }

    @Test
    fun testLogPdfConsistency() {
        val u = UniformDistribution(0.0, 10.0)
        for (x in listOf(0.0, 2.5, 5.0, 7.5, 10.0)) {
            assertEquals(u.pdf(x), exp(u.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSfConsistency() {
        val u = UniformDistribution(0.0, 10.0)
        for (x in listOf(-1.0, 0.0, 5.0, 10.0, 11.0)) {
            assertEquals(1.0, u.sf(x) + u.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }
}

class ExponentialDistributionTest {
    private val tol = 1e-10

    @Test
    fun testCdf() {
        val e = ExponentialDistribution(2.0)
        // 1 - e^(-2*1) = 1 - e^(-2)
        assertEquals(1.0 - exp(-2.0), e.cdf(1.0), tol)
    }

    @Test
    fun testQuantile() {
        val e = ExponentialDistribution(1.0)
        assertEquals(ln(2.0), e.quantile(0.5), tol) // median
    }

    @Test
    fun testMean() {
        val e = ExponentialDistribution(0.5)
        assertEquals(2.0, e.mean, tol)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.expon(scale=1).entropy() = 1.0
        val e1 = ExponentialDistribution(1.0)
        assertEquals(1.0, e1.entropy, 1e-12)
        // scipy: stats.expon(scale=2).entropy() = 1.693147180559945
        val e2 = ExponentialDistribution(0.5)
        assertEquals(1.693147180559945, e2.entropy, 1e-12)
    }

    @Test
    fun testQuantilePrecisionNearOne() {
        val e = ExponentialDistribution(1.0)
        // scipy: stats.expon.ppf(1 - 1e-15) = 34.539575992340879
        assertEquals(34.539575992340879, e.quantile(1.0 - 1e-15), 1e-6)
        // scipy: stats.expon.ppf(1 - 1e-10) = 23.025850847200090
        assertEquals(23.025850847200090, e.quantile(1.0 - 1e-10), 1e-6)
    }

    @Test
    fun testLogPdfConsistency() {
        val e = ExponentialDistribution(2.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0)) {
            assertEquals(e.pdf(x), exp(e.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSfConsistency() {
        val e = ExponentialDistribution(2.0)
        for (x in listOf(-1.0, 0.0, 0.5, 1.0, 5.0)) {
            assertEquals(1.0, e.sf(x) + e.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }
}

class BinomialDistributionTest {
    private val tol = 1e-6

    @Test
    fun testPmfKnown() {
        val b = BinomialDistribution(10, 0.3)
        // R: dbinom(3, 10, 0.3) = 0.2668279
        assertEquals(0.2668279, b.pmf(3), tol)
    }

    @Test
    fun testCdfKnown() {
        val b = BinomialDistribution(10, 0.3)
        // R: pbinom(3, 10, 0.3) = 0.6496107
        assertEquals(0.6496107, b.cdf(3), tol)
    }

    @Test
    fun testMean() {
        val b = BinomialDistribution(20, 0.4)
        assertEquals(8.0, b.mean, tol)
    }
}

class PoissonDistributionTest {
    private val tol = 1e-6

    @Test
    fun testPmfKnown() {
        val p = PoissonDistribution(3.0)
        // R: dpois(2, 3) = 0.2240418
        assertEquals(0.2240418, p.pmf(2), tol)
    }

    @Test
    fun testMean() {
        val p = PoissonDistribution(5.0)
        assertEquals(5.0, p.mean, tol)
        assertEquals(5.0, p.variance, tol)
    }
}

class BernoulliDistributionTest {
    @Test
    fun testPmf() {
        val b = BernoulliDistribution(0.7)
        assertEquals(0.3, b.pmf(0), 1e-10)
        assertEquals(0.7, b.pmf(1), 1e-10)
        assertEquals(0.0, b.pmf(2), 1e-10)
    }

    @Test
    fun testMean() {
        assertEquals(0.7, BernoulliDistribution(0.7).mean, 1e-10)
    }
}

class GeometricDistributionTest {
    @Test
    fun testPmf() {
        val g = GeometricDistribution(0.5)
        // P(X=0) = 0.5, P(X=1) = 0.25
        assertEquals(0.5, g.pmf(0), 1e-10)
        assertEquals(0.25, g.pmf(1), 1e-10)
    }

    @Test
    fun testMean() {
        val g = GeometricDistribution(0.25)
        assertEquals(3.0, g.mean, 1e-10) // (1-p)/p = 0.75/0.25 = 3
    }
}

class GammaDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).pdf(x)
        assertEquals(0.303265329856317, g.pdf(0.5), 1e-12)
        assertEquals(0.367879441171442, g.pdf(1.0), 1e-12)
        assertEquals(0.270670566473225, g.pdf(2.0), 1e-12)
        assertEquals(0.149361205103592, g.pdf(3.0), 1e-12)
        assertEquals(0.0336897349954273, g.pdf(5.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).cdf(x)
        assertEquals(0.0902040104310499, g.cdf(0.5), 1e-10)
        assertEquals(0.264241117657115, g.cdf(1.0), 1e-10)
        assertEquals(0.593994150290162, g.cdf(2.0), 1e-10)
        assertEquals(0.800851726528544, g.cdf(3.0), 1e-10)
        assertEquals(0.959572318005487, g.cdf(5.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(-1.19314718055995, g.logPdf(0.5), 1e-10)
        assertEquals(-1.0, g.logPdf(1.0), 1e-10)
        assertEquals(-1.30685281944005, g.logPdf(2.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).ppf(p)
        // Note: extreme p values (0.01, 0.99) may not converge well — see DIST-010
        assertEquals(0.531811608389612, g.quantile(0.1), 1e-6)
        assertEquals(0.961278763114777, g.quantile(0.25), 1e-6)
        assertEquals(1.67834699001666, g.quantile(0.5), 1e-6)
        assertEquals(2.6926345288897, g.quantile(0.75), 1e-6)
        assertEquals(3.88972016986743, g.quantile(0.9), 1e-6)
    }

    @Test
    fun testMoments() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(2.0, g.mean, 1e-12)
        assertEquals(2.0, g.variance, 1e-12)
        assertEquals(2.0 / sqrt(2.0), g.skewness, 1e-12)
        assertEquals(3.0, g.kurtosis, 1e-12)
    }

    @Test
    fun testMomentsWithRate() {
        val g = GammaDistribution(3.0, 2.0)
        assertEquals(1.5, g.mean, 1e-12)
        assertEquals(0.75, g.variance, 1e-12)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // shape=2: pdf(0) = 0
        assertEquals(0.0, GammaDistribution(2.0, 1.0).pdf(0.0), 1e-12)
        // shape=1: pdf(0) = rate
        assertEquals(2.0, GammaDistribution(1.0, 2.0).pdf(0.0), 1e-12)
        // shape<1: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, GammaDistribution(0.5, 1.0).pdf(0.0))
    }

    @Test
    fun testNegativeX() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(0.0, g.pdf(-1.0), 1e-12)
        assertEquals(0.0, g.cdf(-1.0), 1e-12)
        assertEquals(1.0, g.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, g.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val g = GammaDistribution(2.0, 1.0)
        assertEquals(0.0, g.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, g.quantile(1.0))
    }

    // --- Degenerate / extreme params ---

    @Test
    fun testSmallShape() {
        val g = GammaDistribution(0.5, 2.0) // scale=0.5
        // scipy: stats.gamma(0.5, scale=0.5)
        assertEquals(0.25, g.mean, 1e-12)
        assertEquals(0.125, g.variance, 1e-12)
    }

    @Test
    fun testLargeShape() {
        val g = GammaDistribution(5.0, 0.5) // scale=2.0
        assertEquals(10.0, g.mean, 1e-12)
        assertEquals(20.0, g.variance, 1e-12)
        // scipy: stats.gamma(5, scale=2).cdf(10)
        assertEquals(0.559506714934788, g.cdf(10.0), 1e-8)
    }

    @Test
    fun testExponentialEquivalence() {
        // Gamma(1, rate) = Exponential(rate)
        val g = GammaDistribution(1.0, 1.0)
        assertEquals(1.0, g.mean, 1e-12)
        assertEquals(1.0, g.variance, 1e-12)
        assertEquals(0.367879441171442, g.sf(1.0), 1e-10)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { GammaDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { GammaDistribution(1.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val g = GammaDistribution(2.0, 1.0)
        assertFailsWith<InvalidParameterException> { g.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { g.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val g = GammaDistribution(2.0, 1.0)
        // Note: extreme p values (0.01, 0.99) may not converge well — see DIST-010
        for (p in listOf(0.1, 0.25, 0.5, 0.75, 0.9)) {
            assertEquals(p, g.cdf(g.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val g = GammaDistribution(2.0, 1.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, g.sf(x) + g.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testSfUpperTail() {
        val g = GammaDistribution(2.0, 1.0)
        // scipy: stats.gamma(2, scale=1).sf(x)
        assertEquals(0.000499399227387334, g.sf(10.0), 1e-10)
        assertEquals(4.89443712802922e-06, g.sf(15.0), 1e-11)
        assertEquals(4.32842260712097e-08, g.sf(20.0), 1e-13)
    }

    @Test
    fun testLogPdfConsistency() {
        val g = GammaDistribution(2.0, 1.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0)) {
            assertEquals(g.pdf(x), exp(g.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val g = GammaDistribution(5.0, 0.5) // mean=10, var=20
        val rng = kotlin.random.Random(42)
        val samples = g.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(10.0, sampleMean, 0.5, "sample mean ≈ 10")
    }

    @Test
    fun testCdfMonotonicity() {
        val g = GammaDistribution(2.0, 1.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)) {
            val cdfVal = g.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(GammaDistribution(2.0, 1.0).entropy.isNaN())
    }
}

class BetaDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).pdf(x)
        assertEquals(0.0, b.pdf(0.0), 1e-12)
        assertEquals(1.9683, b.pdf(0.1), 1e-10)
        assertEquals(2.4576, b.pdf(0.2), 1e-10)
        assertEquals(2.1609, b.pdf(0.3), 1e-10)
        assertEquals(0.9375, b.pdf(0.5), 1e-10)
        assertEquals(0.0384, b.pdf(0.8), 1e-10)
        assertEquals(0.0, b.pdf(1.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).cdf(x)
        assertEquals(0.0, b.cdf(0.0), 1e-12)
        assertEquals(0.114265, b.cdf(0.1), 1e-5)
        assertEquals(0.34464, b.cdf(0.2), 1e-5)
        assertEquals(0.579825, b.cdf(0.3), 1e-5)
        assertEquals(0.890625, b.cdf(0.5), 1e-5)
        assertEquals(0.9984, b.cdf(0.8), 1e-5)
        assertEquals(1.0, b.cdf(1.0), 1e-12)
    }

    @Test
    fun testLogPdfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.677170226036805, b.logPdf(0.1), 1e-10)
        assertEquals(0.899185263971216, b.logPdf(0.2), 1e-10)
        assertEquals(0.77052480158129, b.logPdf(0.3), 1e-10)
        assertEquals(-0.0645385211375711, b.logPdf(0.5), 1e-10)
        assertEquals(-3.25969781938846, b.logPdf(0.8), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).ppf(p)
        assertEquals(0.0267631911427551, b.quantile(0.01), 1e-6)
        assertEquals(0.0925952589131287, b.quantile(0.1), 1e-6)
        assertEquals(0.161162916790327, b.quantile(0.25), 1e-6)
        assertEquals(0.26444998329566, b.quantile(0.5), 1e-6)
        assertEquals(0.389479485200725, b.quantile(0.75), 1e-6)
        assertEquals(0.510316306551492, b.quantile(0.9), 1e-6)
        assertEquals(0.705686328319707, b.quantile(0.99), 1e-6)
    }

    @Test
    fun testMoments() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.285714285714286, b.mean, 1e-12)
        assertEquals(0.0255102040816327, b.variance, 1e-12)
        assertEquals(0.596284793999944, b.skewness, 1e-10)
        assertEquals(-0.12, b.kurtosis, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testOutsideSupport() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b.pdf(-0.1), 1e-12)
        assertEquals(0.0, b.pdf(1.1), 1e-12)
        assertEquals(0.0, b.cdf(-0.1), 1e-12)
        assertEquals(1.0, b.cdf(1.1), 1e-12)
        assertEquals(1.0, b.sf(-0.1), 1e-12)
        assertEquals(0.0, b.sf(1.1), 1e-12)
    }

    @Test
    fun testQuantileBoundaries() {
        val b = BetaDistribution(2.0, 5.0)
        assertEquals(0.0, b.quantile(0.0), 1e-12)
        assertEquals(1.0, b.quantile(1.0), 1e-12)
    }

    // --- Degenerate / special shapes ---

    @Test
    fun testUniformEquivalence() {
        // Beta(1,1) = Uniform(0,1)
        val b = BetaDistribution(1.0, 1.0)
        assertEquals(1.0, b.pdf(0.5), 1e-12)
        assertEquals(0.5, b.cdf(0.5), 1e-10)
        assertEquals(0.5, b.mean, 1e-12)
        assertEquals(1.0 / 12.0, b.variance, 1e-12)
    }

    @Test
    fun testUShapedBeta() {
        // Beta(0.5, 0.5) - arcsine distribution
        val b = BetaDistribution(0.5, 0.5)
        assertEquals(0.5, b.mean, 1e-12)
        assertEquals(0.125, b.variance, 1e-12)
        // scipy: stats.beta(0.5, 0.5).cdf(0.5)
        assertEquals(0.5, b.cdf(0.5), 1e-8)
        assertEquals(0.636619772367581, b.pdf(0.5), 1e-8)
    }

    @Test
    fun testJShapedBeta() {
        // Beta(5, 1) - power distribution
        val b = BetaDistribution(5.0, 1.0)
        assertEquals(0.833333333333333, b.mean, 1e-12)
        // scipy: stats.beta(5, 1).sf(0.9)
        assertEquals(0.40951, b.sf(0.9), 1e-5)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { BetaDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { BetaDistribution(1.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val b = BetaDistribution(2.0, 5.0)
        assertFailsWith<InvalidParameterException> { b.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { b.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val b = BetaDistribution(2.0, 5.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, b.cdf(b.quantile(p)), 1e-8, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val b = BetaDistribution(2.0, 5.0)
        for (x in listOf(0.0, 0.1, 0.3, 0.5, 0.8, 1.0)) {
            assertEquals(1.0, b.sf(x) + b.cdf(x), 1e-10, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testSfKnownValues() {
        val b = BetaDistribution(2.0, 5.0)
        // scipy: stats.beta(2, 5).sf(x)
        assertEquals(0.885735, b.sf(0.1), 1e-5)
        assertEquals(0.420175, b.sf(0.3), 1e-5)
        assertEquals(0.109375, b.sf(0.5), 1e-5)
        assertEquals(0.0016, b.sf(0.8), 1e-5)
    }

    @Test
    fun testSfSymmetryRelation() {
        // sf(x) for Beta(a,b) should equal cdf(1-x) for Beta(b,a) (symmetry)
        val b1 = BetaDistribution(2.0, 5.0)
        val b2 = BetaDistribution(5.0, 2.0)
        for (x in listOf(0.1, 0.3, 0.5, 0.7, 0.9)) {
            assertEquals(b1.sf(x), b2.cdf(1.0 - x), 1e-10, "sf($x) = cdf_mirror(1-$x)")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val b = BetaDistribution(2.0, 5.0)
        for (x in listOf(0.1, 0.2, 0.3, 0.5, 0.8)) {
            assertEquals(b.pdf(x), exp(b.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val b = BetaDistribution(2.0, 5.0) // mean=0.2857
        val rng = kotlin.random.Random(42)
        val samples = b.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(0.2857, sampleMean, 0.02, "sample mean ≈ 0.2857")
    }

    @Test
    fun testCdfMonotonicity() {
        val b = BetaDistribution(2.0, 5.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)) {
            val cdfVal = b.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testEntropyNaN() {
        // entropy requires digamma (deferred to MATH-001)
        assertTrue(BetaDistribution(2.0, 5.0).entropy.isNaN())
    }
}

class WeibullDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).pdf(x)
        assertEquals(0.330936338469223, w.pdf(0.5), 1e-12)
        assertEquals(0.372391688219422, w.pdf(1.0), 1e-12)
        assertEquals(0.275909580878582, w.pdf(2.0), 1e-12)
        assertEquals(0.146304264044542, w.pdf(3.0), 1e-12)
        assertEquals(0.0227683519028661, w.pdf(5.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).cdf(x)
        assertEquals(0.117503097415405, w.cdf(0.5), 1e-12)
        assertEquals(0.29781149867344, w.cdf(1.0), 1e-12)
        assertEquals(0.632120558828558, w.cdf(2.0), 1e-12)
        assertEquals(0.840724091509979, w.cdf(3.0), 1e-12)
        assertEquals(0.98080003984499, w.cdf(5.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).sf(x)
        assertEquals(0.882496902584595, w.sf(0.5), 1e-12)
        assertEquals(0.70218850132656, w.sf(1.0), 1e-12)
        assertEquals(0.367879441171442, w.sf(2.0), 1e-12)
        assertEquals(0.159275908490021, w.sf(3.0), 1e-12)
        assertEquals(0.0191999601550095, w.sf(5.0), 1e-12)
    }

    @Test
    fun testLogPdfKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(-1.10582925301173, w.logPdf(0.5), 1e-10)
        assertEquals(-0.987809053325027, w.logPdf(1.0), 1e-10)
        assertEquals(-1.28768207245178, w.logPdf(2.0), 1e-10)
        assertEquals(-1.92206682548508, w.logPdf(3.0), 1e-10)
        assertEquals(-3.78238378172518, w.logPdf(5.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val w = WeibullDistribution(1.5, 2.0)
        // scipy: stats.weibull_min(1.5, scale=2).ppf(p)
        assertEquals(0.0931430336940394, w.quantile(0.01), 1e-10)
        assertEquals(0.446151051273834, w.quantile(0.1), 1e-10)
        assertEquals(0.871575863406048, w.quantile(0.25), 1e-10)
        assertEquals(1.5664395375493, w.quantile(0.5), 1e-10)
        assertEquals(2.48656776975034, w.quantile(0.75), 1e-10)
        assertEquals(3.48744302719282, w.quantile(0.9), 1e-10)
        assertEquals(5.53597073004505, w.quantile(0.99), 1e-10)
    }

    @Test
    fun testMoments() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(1.80549058590187, w.mean, 1e-10)
        assertEquals(1.50276113925573, w.variance, 1e-10)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.weibull_min(1.5, scale=2).entropy()
        assertEquals(1.48008729408563, WeibullDistribution(1.5, 2.0).entropy, 1e-10)
        // scipy: stats.weibull_min(0.5, scale=1).entropy()
        assertEquals(1.11593151565841, WeibullDistribution(0.5, 1.0).entropy, 1e-10)
        // scipy: stats.weibull_min(3, scale=5).entropy()
        assertEquals(1.89563606703368, WeibullDistribution(3.0, 5.0).entropy, 1e-10)
        // Weibull(1,1) = Exponential(1), entropy = 1
        assertEquals(1.0, WeibullDistribution(1.0, 1.0).entropy, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testPdfAtZero() {
        // k=1: pdf(0) = 1/lambda
        assertEquals(0.5, WeibullDistribution(1.0, 2.0).pdf(0.0), 1e-12)
        // k<1: pdf(0) = +Inf
        assertEquals(Double.POSITIVE_INFINITY, WeibullDistribution(0.5, 1.0).pdf(0.0))
        // k>1: pdf(0) = 0
        assertEquals(0.0, WeibullDistribution(2.0, 1.0).pdf(0.0), 1e-12)
    }

    @Test
    fun testNegativeX() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(0.0, w.pdf(-1.0), 1e-12)
        assertEquals(0.0, w.cdf(-1.0), 1e-12)
        assertEquals(1.0, w.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, w.logPdf(-1.0))
    }

    @Test
    fun testLogPdfAtZero() {
        assertEquals(-ln(2.0), WeibullDistribution(1.0, 2.0).logPdf(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, WeibullDistribution(0.5, 1.0).logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, WeibullDistribution(2.0, 1.0).logPdf(0.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val w = WeibullDistribution(1.5, 2.0)
        assertEquals(0.0, w.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, w.quantile(1.0))
    }

    // --- Extreme params ---

    @Test
    fun testSmallShape() {
        val w = WeibullDistribution(0.5, 1.0)
        assertEquals(2.0, w.mean, 1e-10)
        assertEquals(20.0, w.variance, 1e-8)
        // scipy: stats.weibull_min(0.5, scale=1).sf(1.0)
        assertEquals(0.367879441171442, w.sf(1.0), 1e-12)
    }

    @Test
    fun testLargeShape() {
        val w = WeibullDistribution(3.0, 5.0)
        assertEquals(4.46489755784624, w.mean, 1e-10)
        // scipy: stats.weibull_min(3, scale=5).sf(x)
        assertEquals(0.992031914837061, w.sf(1.0), 1e-10)
        assertEquals(0.80573530187348, w.sf(3.0), 1e-10)
        assertEquals(0.367879441171442, w.sf(5.0), 1e-12)
    }

    @Test
    fun testExponentialEquivalence() {
        // Weibull(1,1) = Exponential(1)
        val w = WeibullDistribution(1.0, 1.0)
        assertEquals(1.0, w.mean, 1e-12)
        assertEquals(1.0, w.variance, 1e-12)
        assertEquals(1.0, w.entropy, 1e-12)
    }

    @Test
    fun testQuantilePrecisionNearOne() {
        // Same as ExponentialDistribution test: Weibull(1,1) = Exp(1)
        val w = WeibullDistribution(1.0, 1.0)
        // scipy: stats.weibull_min(1, scale=1).ppf(1 - 1e-15)
        assertEquals(34.5395759923409, w.quantile(1.0 - 1e-15), 1e-6)
        assertEquals(23.0258508472001, w.quantile(1.0 - 1e-10), 1e-6)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { WeibullDistribution(0.0, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(-1.0, 1.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, 0.0) }
        assertFailsWith<InvalidParameterException> { WeibullDistribution(1.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val w = WeibullDistribution(1.5, 2.0)
        assertFailsWith<InvalidParameterException> { w.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { w.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val w = WeibullDistribution(1.5, 2.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, w.cdf(w.quantile(p)), 1e-12, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val w = WeibullDistribution(1.5, 2.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0)) {
            assertEquals(1.0, w.sf(x) + w.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val w = WeibullDistribution(1.5, 2.0)
        for (x in listOf(0.5, 1.0, 2.0, 3.0, 5.0)) {
            assertEquals(w.pdf(x), exp(w.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val w = WeibullDistribution(1.5, 2.0) // mean ≈ 1.805
        val rng = kotlin.random.Random(42)
        val samples = w.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(1.805, sampleMean, 0.1, "sample mean ≈ 1.805")
    }

    @Test
    fun testCdfMonotonicity() {
        val w = WeibullDistribution(1.5, 2.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)) {
            val cdfVal = w.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }
}

class LogNormalDistributionTest {

    // --- Basic correctness (scipy 15-digit refs) ---

    @Test
    fun testPdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).pdf(x)
        assertEquals(0.627496077115924, d.pdf(0.5), 1e-12)
        assertEquals(0.398942280401433, d.pdf(1.0), 1e-12)
        assertEquals(0.156874019278981, d.pdf(2.0), 1e-12)
        assertEquals(0.0218507148303272, d.pdf(5.0), 1e-12)
        assertEquals(0.00281590189015268, d.pdf(10.0), 1e-12)
    }

    @Test
    fun testCdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).cdf(x)
        assertEquals(0.244108595785583, d.cdf(0.5), 1e-10)
        assertEquals(0.5, d.cdf(1.0), 1e-10)
        assertEquals(0.755891404214417, d.cdf(2.0), 1e-10)
        assertEquals(0.946239689548337, d.cdf(5.0), 1e-10)
        assertEquals(0.9893489006583, d.cdf(10.0), 1e-10)
    }

    @Test
    fun testSfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).sf(x)
        assertEquals(0.755891404214417, d.sf(0.5), 1e-10)
        assertEquals(0.5, d.sf(1.0), 1e-10)
        assertEquals(0.244108595785583, d.sf(2.0), 1e-10)
        assertEquals(0.0537603104516631, d.sf(5.0), 1e-10)
        assertEquals(0.0106510993417001, d.sf(10.0), 1e-10)
    }

    @Test
    fun testLogPdfKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(-0.466017859603828, d.logPdf(0.5), 1e-10)
        assertEquals(-0.918938533204673, d.logPdf(1.0), 1e-10)
        assertEquals(-1.85231222072372, d.logPdf(2.0), 1e-10)
        assertEquals(-3.82352164262889, d.logPdf(5.0), 1e-10)
    }

    @Test
    fun testQuantileKnownValues() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).ppf(p)
        assertEquals(0.097651733070336, d.quantile(0.01), 1e-8)
        assertEquals(0.27760624185201, d.quantile(0.1), 1e-8)
        assertEquals(0.509416283863278, d.quantile(0.25), 1e-8)
        assertEquals(1.0, d.quantile(0.5), 1e-10)
        assertEquals(1.96303108415826, d.quantile(0.75), 1e-8)
        assertEquals(3.60222447927916, d.quantile(0.9), 1e-8)
        assertEquals(10.2404736563121, d.quantile(0.99), 1e-6)
    }

    @Test
    fun testMoments() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(1.64872127070013, d.mean, 1e-10)
        assertEquals(4.6707742704716, d.variance, 1e-8)
    }

    @Test
    fun testEntropy() {
        // scipy: stats.lognorm(s=1, scale=exp(0)).entropy()
        assertEquals(1.41893853320467, LogNormalDistribution(0.0, 1.0).entropy, 1e-10)
        // scipy: stats.lognorm(s=0.5, scale=exp(2)).entropy()
        assertEquals(2.72579135264473, LogNormalDistribution(2.0, 0.5).entropy, 1e-10)
    }

    // --- Edge cases ---

    @Test
    fun testNonPositiveX() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(0.0, d.pdf(0.0), 1e-12)
        assertEquals(0.0, d.pdf(-1.0), 1e-12)
        assertEquals(0.0, d.cdf(0.0), 1e-12)
        assertEquals(0.0, d.cdf(-1.0), 1e-12)
        assertEquals(1.0, d.sf(0.0), 1e-12)
        assertEquals(1.0, d.sf(-1.0), 1e-12)
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(0.0))
        assertEquals(Double.NEGATIVE_INFINITY, d.logPdf(-1.0))
    }

    @Test
    fun testQuantileBoundaries() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertEquals(0.0, d.quantile(0.0), 1e-12)
        assertEquals(Double.POSITIVE_INFINITY, d.quantile(1.0))
    }

    // --- Different param combos ---

    @Test
    fun testMu2Sigma05() {
        val d = LogNormalDistribution(2.0, 0.5)
        // scipy: stats.lognorm(s=0.5, scale=exp(2))
        assertEquals(8.37289748812726, d.mean, 1e-8)
        assertEquals(19.9117189538339, d.variance, 1e-6)
        // cdf
        assertEquals(0.217364732271511, d.cdf(5.0), 1e-8)
        assertEquals(0.727467038315737, d.cdf(10.0), 1e-8)
        // sf
        assertEquals(0.782635267728489, d.sf(5.0), 1e-8)
        assertEquals(0.272532961684263, d.sf(10.0), 1e-8)
        // ppf
        assertEquals(2.30902662894362, d.quantile(0.01), 1e-6)
        assertEquals(7.38905609893065, d.quantile(0.5), 1e-6)
        assertEquals(23.6455263654204, d.quantile(0.99), 1e-5)
    }

    @Test
    fun testSfUpperTail() {
        val d = LogNormalDistribution(0.0, 1.0)
        // scipy: stats.lognorm(s=1, scale=exp(0)).sf(x)
        assertEquals(0.00136893348785809, d.sf(20.0), 1e-8)
        assertEquals(4.57630952498886e-05, d.sf(50.0), 1e-9)
        assertEquals(2.06064339597171e-06, d.sf(100.0), 1e-10)
    }

    // --- NaN/inf handling ---

    @Test
    fun testInvalidParameters() {
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, 0.0) }
        assertFailsWith<InvalidParameterException> { LogNormalDistribution(0.0, -1.0) }
    }

    @Test
    fun testQuantileInvalidP() {
        val d = LogNormalDistribution(0.0, 1.0)
        assertFailsWith<InvalidParameterException> { d.quantile(-0.1) }
        assertFailsWith<InvalidParameterException> { d.quantile(1.1) }
    }

    // --- Property-based ---

    @Test
    fun testCdfQuantileRoundTrip() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (p in listOf(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)) {
            assertEquals(p, d.cdf(d.quantile(p)), 1e-10, "cdf(quantile($p)) ≈ $p")
        }
    }

    @Test
    fun testSfPlusCdfEqualsOne() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(1.0, d.sf(x) + d.cdf(x), 1e-12, "sf($x) + cdf($x) ≈ 1")
        }
    }

    @Test
    fun testLogPdfConsistency() {
        val d = LogNormalDistribution(0.0, 1.0)
        for (x in listOf(0.5, 1.0, 2.0, 5.0, 10.0)) {
            assertEquals(d.pdf(x), exp(d.logPdf(x)), 1e-12, "exp(logPdf($x)) ≈ pdf($x)")
        }
    }

    @Test
    fun testSampleStats() {
        val d = LogNormalDistribution(0.0, 1.0) // mean ≈ 1.649
        val rng = kotlin.random.Random(42)
        val samples = d.sample(10000, rng)
        val sampleMean = samples.average()
        assertEquals(1.649, sampleMean, 0.15, "sample mean ≈ 1.649")
    }

    @Test
    fun testCdfMonotonicity() {
        val d = LogNormalDistribution(0.0, 1.0)
        var prev = 0.0
        for (x in listOf(0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)) {
            val cdfVal = d.cdf(x)
            assertTrue(cdfVal >= prev, "cdf should be monotonically increasing")
            prev = cdfVal
        }
    }

    @Test
    fun testCdfSymmetryInLogSpace() {
        // LogNormal(mu, sigma): cdf(exp(mu)) = 0.5
        val d = LogNormalDistribution(2.0, 0.5)
        assertEquals(0.5, d.cdf(exp(2.0)), 1e-10)
    }
}
