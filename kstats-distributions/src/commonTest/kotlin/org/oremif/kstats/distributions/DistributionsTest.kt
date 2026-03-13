package org.oremif.kstats.distributions

import kotlin.math.*
import kotlin.test.Test
import kotlin.test.assertEquals
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
}

class StudentTDistributionTest {
    private val tol = 1e-5

    @Test
    fun testCdfSymmetry() {
        val t = StudentTDistribution(10.0)
        assertEquals(0.5, t.cdf(0.0), tol)
    }

    @Test
    fun testCdfKnown() {
        // R: pt(2.228, df=10) ≈ 0.9750
        val t = StudentTDistribution(10.0)
        assertEquals(0.975, t.cdf(2.228139), tol)
    }

    @Test
    fun testQuantile() {
        val t = StudentTDistribution(10.0)
        // R: qt(0.975, 10) = 2.228139
        assertEquals(2.228139, t.quantile(0.975), 1e-4)
    }

    @Test
    fun testMoments() {
        val t = StudentTDistribution(5.0)
        assertEquals(0.0, t.mean, tol)
        assertEquals(5.0 / 3.0, t.variance, tol)
    }
}

class ChiSquaredDistributionTest {
    private val tol = 1e-5

    @Test
    fun testCdfKnown() {
        val chi2 = ChiSquaredDistribution(5.0)
        // R: pchisq(11.07, 5) ≈ 0.95
        assertEquals(0.95, chi2.cdf(11.0705), tol)
    }

    @Test
    fun testMoments() {
        val chi2 = ChiSquaredDistribution(10.0)
        assertEquals(10.0, chi2.mean, tol)
        assertEquals(20.0, chi2.variance, tol)
    }
}

class FDistributionTest {
    private val tol = 1e-4

    @Test
    fun testCdfKnown() {
        val f = FDistribution(5.0, 10.0)
        // R: pf(3.3258, 5, 10) ≈ 0.95
        assertEquals(0.95, f.cdf(3.3258), tol)
    }

    @Test
    fun testMean() {
        val f = FDistribution(5.0, 10.0)
        // mean = d2/(d2-2) = 10/8 = 1.25
        assertEquals(1.25, f.mean, tol)
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
    private val tol = 1e-5

    @Test
    fun testCdfKnown() {
        val g = GammaDistribution(2.0, 1.0)
        // R: pgamma(2, shape=2, rate=1) = 0.5939942
        assertEquals(0.5939942, g.cdf(2.0), tol)
    }

    @Test
    fun testMean() {
        val g = GammaDistribution(3.0, 2.0)
        assertEquals(1.5, g.mean, tol) // shape/rate
    }
}

class BetaDistributionTest {
    private val tol = 1e-5

    @Test
    fun testCdfKnown() {
        val b = BetaDistribution(2.0, 5.0)
        // R: pbeta(0.3, 2, 5) = 0.5798250 (verified)
        assertEquals(0.5798250, b.cdf(0.3), tol)
    }

    @Test
    fun testMean() {
        val b = BetaDistribution(2.0, 3.0)
        assertEquals(0.4, b.mean, tol)
    }
}
