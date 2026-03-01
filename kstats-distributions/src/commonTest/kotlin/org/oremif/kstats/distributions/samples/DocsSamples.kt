package org.oremif.kstats.distributions.samples

import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.distributions.*
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

class DocsSamples {

    // ── distributions/overview.mdx ──────────────────────────────────────

    @Test
    fun distWorkingWithDistribution() {
        // SampleStart
        val normal = NormalDistribution(mu = 0.0, sigma = 1.0)

        // Statistical properties
        normal.mean                  // 0.0
        normal.variance              // 1.0
        normal.standardDeviation     // 1.0
        normal.skewness              // 0.0
        normal.kurtosis              // 0.0
        normal.entropy               // 1.4189

        // Evaluate
        normal.pdf(0.0)              // 0.3989 — density at x = 0
        normal.cdf(1.96)             // 0.9750 — P(X ≤ 1.96)
        normal.sf(1.96)              // 0.0250 — P(X > 1.96) = 1 - cdf

        // Invert
        normal.quantile(0.975)       // 1.9600 — value at the 97.5th percentile

        // Sample
        normal.sample(Random(42))           // single random draw
        normal.sample(5, Random(42))        // 5 random draws
        // SampleEnd
        assertEquals(0.0, normal.mean, 1e-4)
        assertEquals(1.0, normal.variance, 1e-4)
        assertEquals(1.0, normal.standardDeviation, 1e-4)
        assertEquals(0.0, normal.skewness, 1e-4)
        assertEquals(0.0, normal.kurtosis, 1e-4)
        assertEquals(1.4189, normal.entropy, 1e-4)
        assertEquals(0.3989, normal.pdf(0.0), 1e-4)
        assertEquals(0.9750, normal.cdf(1.96), 1e-4)
        assertEquals(0.0250, normal.sf(1.96), 1e-4)
        assertEquals(1.9600, normal.quantile(0.975), 1e-4)
    }

    @Test
    fun distNormal() {
        // SampleStart
        val d = NormalDistribution(mu = 100.0, sigma = 15.0)
        d.mean       // 100.0
        d.cdf(115.0) // 0.8413
        d.quantile(0.975) // 129.3994
        // SampleEnd
        assertEquals(100.0, d.mean, 1e-4)
        assertEquals(0.8413, d.cdf(115.0), 1e-4)
        assertEquals(129.3994, d.quantile(0.975), 1e-4)
    }

    @Test
    fun distStudentT() {
        // SampleStart
        val d = StudentTDistribution(degreesOfFreedom = 10.0)
        d.mean       // 0.0
        d.cdf(2.228) // ≈ 0.975
        d.quantile(0.975) // 2.2281
        // SampleEnd
        assertEquals(0.0, d.mean, 1e-4)
        assertEquals(0.975, d.cdf(2.228), 0.01)
        assertEquals(2.2281, d.quantile(0.975), 1e-4)
    }

    @Test
    fun distLogistic() {
        // SampleStart
        val d = LogisticDistribution(mu = 0.0, scale = 1.0)
        d.mean       // 0.0
        d.cdf(0.0)   // 0.5
        d.pdf(0.0)   // 0.25
        // SampleEnd
        assertEquals(0.0, d.mean, 1e-4)
        assertEquals(0.5, d.cdf(0.0), 1e-4)
        assertEquals(0.25, d.pdf(0.0), 1e-4)
    }

    @Test
    fun distCauchy() {
        // SampleStart
        val d = CauchyDistribution(location = 0.0, scale = 1.0)
        d.pdf(0.0)        // 0.3183
        d.cdf(0.0)        // 0.5
        d.quantile(0.75)  // 1.0
        // SampleEnd
        assertEquals(0.3183, d.pdf(0.0), 1e-4)
        assertEquals(0.5, d.cdf(0.0), 1e-4)
        assertEquals(1.0, d.quantile(0.75), 1e-4)
    }

    @Test
    fun distLaplace() {
        // SampleStart
        val d = LaplaceDistribution(mu = 0.0, scale = 1.0)
        d.mean       // 0.0
        d.variance   // 2.0
        d.pdf(0.0)   // 0.5
        // SampleEnd
        assertEquals(0.0, d.mean, 1e-4)
        assertEquals(2.0, d.variance, 1e-4)
        assertEquals(0.5, d.pdf(0.0), 1e-4)
    }

    @Test
    fun distExponential() {
        // SampleStart
        val d = ExponentialDistribution(rate = 2.0)
        d.mean       // 0.5
        d.cdf(1.0)   // 0.8647
        d.quantile(0.5) // 0.3466
        // SampleEnd
        assertEquals(0.5, d.mean, 1e-4)
        assertEquals(0.8647, d.cdf(1.0), 1e-4)
        assertEquals(0.3466, d.quantile(0.5), 1e-4)
    }

    @Test
    fun distGamma() {
        // SampleStart
        val d = GammaDistribution(shape = 2.0, rate = 0.5)
        d.mean       // 4.0
        d.variance   // 8.0
        d.cdf(4.0)   // 0.5940
        // SampleEnd
        assertEquals(4.0, d.mean, 1e-4)
        assertEquals(8.0, d.variance, 1e-4)
        assertEquals(0.5940, d.cdf(4.0), 1e-4)
    }

    @Test
    fun distWeibull() {
        // SampleStart
        val d = WeibullDistribution(shape = 1.5, scale = 1.0)
        d.mean       // 0.9027
        d.cdf(1.0)   // 0.6321
        // SampleEnd
        assertEquals(0.9027, d.mean, 1e-4)
        assertEquals(0.6321, d.cdf(1.0), 1e-4)
    }

    @Test
    fun distLogNormal() {
        // SampleStart
        val d = LogNormalDistribution(mu = 0.0, sigma = 1.0)
        d.mean          // 1.6487
        d.quantile(0.5) // 1.0
        d.cdf(1.0)      // 0.5
        // SampleEnd
        assertEquals(1.6487, d.mean, 1e-4)
        assertEquals(1.0, d.quantile(0.5), 1e-4)
        assertEquals(0.5, d.cdf(1.0), 1e-4)
    }

    @Test
    fun distNakagami() {
        // SampleStart
        val d = NakagamiDistribution(mu = 1.0, omega = 1.0)
        d.mean       // 0.8862
        d.variance   // 0.2146
        // SampleEnd
        assertEquals(0.8862, d.mean, 1e-4)
        assertEquals(0.2146, d.variance, 1e-4)
    }

    @Test
    fun distLevy() {
        // SampleStart
        val d = LevyDistribution(mu = 0.0, c = 1.0)
        d.cdf(1.0)   // 0.3173
        d.sf(1.0)    // 0.6827
        // SampleEnd
        assertEquals(0.3173, d.cdf(1.0), 1e-4)
        assertEquals(0.6827, d.sf(1.0), 1e-4)
    }

    @Test
    fun distBeta() {
        // SampleStart
        val d = BetaDistribution(alpha = 2.0, beta = 5.0)
        d.mean       // 0.2857
        d.cdf(0.3)   // 0.5798
        d.pdf(0.2)   // 2.4576
        // SampleEnd
        assertEquals(0.2857, d.mean, 1e-4)
        assertEquals(0.5798, d.cdf(0.3), 1e-4)
        assertEquals(2.4576, d.pdf(0.2), 1e-4)
    }

    @Test
    fun distUniform() {
        // SampleStart
        val d = UniformDistribution(min = 0.0, max = 10.0)
        d.mean       // 5.0
        d.variance   // 8.3333
        d.cdf(3.0)   // 0.3
        // SampleEnd
        assertEquals(5.0, d.mean, 1e-4)
        assertEquals(8.3333, d.variance, 1e-4)
        assertEquals(0.3, d.cdf(3.0), 1e-4)
    }

    @Test
    fun distTriangular() {
        // SampleStart
        val d = TriangularDistribution(a = 0.0, b = 10.0, c = 3.0)
        d.mean       // 4.3333
        d.cdf(3.0)   // 0.3
        // SampleEnd
        assertEquals(4.3333, d.mean, 1e-4)
        assertEquals(0.3, d.cdf(3.0), 1e-4)
    }

    @Test
    fun distPareto() {
        // SampleStart
        val d = ParetoDistribution(shape = 2.0, scale = 1.0)
        d.mean       // 2.0
        d.cdf(2.0)   // 0.75
        // SampleEnd
        assertEquals(2.0, d.mean, 1e-4)
        assertEquals(0.75, d.cdf(2.0), 1e-4)
    }

    @Test
    fun distGumbel() {
        // SampleStart
        val d = GumbelDistribution(mu = 0.0, beta = 1.0)
        d.mean       // 0.5772
        d.cdf(0.0)   // 0.3679
        // SampleEnd
        assertEquals(0.5772, d.mean, 1e-4)
        assertEquals(0.3679, d.cdf(0.0), 1e-4)
    }

    @Test
    fun distChiSquared() {
        // SampleStart
        val d = ChiSquaredDistribution(degreesOfFreedom = 5.0)
        d.mean       // 5.0
        d.variance   // 10.0
        d.cdf(11.07) // ≈ 0.95
        // SampleEnd
        assertEquals(5.0, d.mean, 1e-4)
        assertEquals(10.0, d.variance, 1e-4)
        assertEquals(0.95, d.cdf(11.07), 0.01)
    }

    @Test
    fun distF() {
        // SampleStart
        val d = FDistribution(dfNumerator = 5.0, dfDenominator = 10.0)
        d.mean       // 1.25
        d.cdf(3.33)  // ≈ 0.95
        // SampleEnd
        assertEquals(1.25, d.mean, 1e-4)
        assertEquals(0.95, d.cdf(3.33), 0.01)
    }

    @Test
    fun distPoisson() {
        // SampleStart
        val d = PoissonDistribution(rate = 3.0)
        d.mean          // 3.0
        d.pmf(5)        // 0.1008
        d.cdf(5)        // 0.9161
        d.quantileInt(0.95) // 6
        // SampleEnd
        assertEquals(3.0, d.mean, 1e-4)
        assertEquals(0.1008, d.pmf(5), 1e-4)
        assertEquals(0.9161, d.cdf(5), 1e-4)
        assertEquals(6, d.quantileInt(0.95))
    }

    @Test
    fun distBinomial() {
        // SampleStart
        val d = BinomialDistribution(trials = 10, probability = 0.3)
        d.mean          // 3.0
        d.pmf(3)        // 0.2668
        d.cdf(3)        // 0.6496
        d.quantileInt(0.5) // 3
        // SampleEnd
        assertEquals(3.0, d.mean, 1e-4)
        assertEquals(0.2668, d.pmf(3), 1e-4)
        assertEquals(0.6496, d.cdf(3), 1e-4)
        assertEquals(3, d.quantileInt(0.5))
    }

    @Test
    fun distNegativeBinomial() {
        // SampleStart
        val d = NegativeBinomialDistribution(successes = 5, probability = 0.5)
        d.mean          // 5.0
        d.variance      // 10.0
        d.pmf(3)        // probability of exactly 3 failures before 5 successes
        // SampleEnd
        assertEquals(5.0, d.mean, 1e-4)
        assertEquals(10.0, d.variance, 1e-4)
    }

    @Test
    fun distGeometric() {
        // SampleStart
        val d = GeometricDistribution(probability = 0.3)
        d.mean          // 3.3333
        d.pmf(1)        // 0.3
        d.cdf(3)        // 0.657
        // SampleEnd
        assertEquals(2.3333, d.mean, 1e-4)
        assertEquals(0.21, d.pmf(1), 1e-4)
        assertEquals(0.7599, d.cdf(3), 1e-4)
    }

    @Test
    fun distHypergeometric() {
        // SampleStart
        val d = HypergeometricDistribution(population = 50, successes = 10, draws = 5)
        d.mean          // 1.0
        d.pmf(2)        // probability of exactly 2 successes in 5 draws
        // SampleEnd
        assertEquals(1.0, d.mean, 1e-4)
    }

    @Test
    fun distBetaBinomial() {
        // SampleStart
        val d = BetaBinomialDistribution(trials = 10, alpha = 2.0, beta = 3.0)
        d.mean          // 4.0
        d.pmf(4)        // probability of exactly 4 successes
        // SampleEnd
        assertEquals(4.0, d.mean, 1e-4)
    }

    @Test
    fun distBernoulli() {
        // SampleStart
        val d = BernoulliDistribution(probability = 0.7)
        d.mean          // 0.7
        d.pmf(1)        // 0.7
        d.pmf(0)        // 0.3
        // SampleEnd
        assertEquals(0.7, d.mean, 1e-4)
        assertEquals(0.7, d.pmf(1), 1e-4)
        assertEquals(0.3, d.pmf(0), 1e-4)
    }

    @Test
    fun distUniformDiscrete() {
        // SampleStart
        val d = UniformDiscreteDistribution(min = 1, max = 6)
        d.mean          // 3.5
        d.pmf(3)        // 0.1667
        d.cdf(3)        // 0.5
        // SampleEnd
        assertEquals(3.5, d.mean, 1e-4)
        assertEquals(0.1667, d.pmf(3), 1e-4)
        assertEquals(0.5, d.cdf(3), 1e-4)
    }

    @Test
    fun distZipf() {
        // SampleStart
        val d = ZipfDistribution(numberOfElements = 100, exponent = 1.0)
        d.pmf(1)        // probability of rank 1 (the most common)
        d.pmf(100)      // probability of rank 100 (the least common)
        // SampleEnd
    }

    @Test
    fun distLogarithmic() {
        // SampleStart
        val d = LogarithmicDistribution(probability = 0.5)
        d.mean          // 1.4427
        d.pmf(1)        // 0.7213
        d.pmf(2)        // 0.1803
        // SampleEnd
        assertEquals(1.4427, d.mean, 1e-4)
        assertEquals(0.7213, d.pmf(1), 1e-4)
        assertEquals(0.1803, d.pmf(2), 1e-4)
    }

    // ── quickstart.mdx ─────────────────────────────────────────────────

    @Test
    fun quickstartFit() {
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val stats = sample.describe()
        // SampleStart
        val normal = NormalDistribution(mu = stats.mean, sigma = stats.standardDeviation)
        normal.cdf(6.0)          // probability that X ≤ 6.0
        normal.quantile(0.95)    // value below which 95% of the distribution falls
        // SampleEnd
    }

    @Test
    fun quickstartTabDistributions() {
        // SampleStart
        val normal = NormalDistribution(mu = 0.0, sigma = 1.0)
        normal.pdf(0.0)           // 0.3989
        normal.cdf(1.96)          // 0.9750
        normal.quantile(0.975)    // 1.9600

        val poisson = PoissonDistribution(rate = 3.0)
        poisson.pmf(5)            // 0.1008
        poisson.cdf(5)            // 0.9161
        // SampleEnd
        assertEquals(0.3989, normal.pdf(0.0), 1e-4)
        assertEquals(0.9750, normal.cdf(1.96), 1e-4)
        assertEquals(1.9600, normal.quantile(0.975), 1e-4)
        assertEquals(0.1008, poisson.pmf(5), 1e-4)
        assertEquals(0.9161, poisson.cdf(5), 1e-4)
    }

    // ── choosing-a-distribution.mdx ─────────────────────────────────────

    @Test
    fun choosingResponseTimes() {
        // SampleStart
        // API response times — often right-skewed with a long tail
        val responseTime = LogNormalDistribution(mu = 4.5, sigma = 0.8)

        responseTime.mean              // expected average in ms
        responseTime.quantile(0.95)    // P95 latency
        responseTime.quantile(0.99)    // P99 latency
        responseTime.cdf(200.0)        // probability of responding under 200ms
        // SampleEnd
    }

    @Test
    fun choosingTimeBetweenEvents() {
        // SampleStart
        // Time between server errors — memoryless waiting process
        val timeBetweenErrors = ExponentialDistribution(rate = 0.5)

        timeBetweenErrors.mean          // average time between errors
        timeBetweenErrors.sf(10.0)      // probability of waiting longer than 10 units
        timeBetweenErrors.quantile(0.5) // median waiting time
        // SampleEnd
    }

    @Test
    fun choosingComponentLifetime() {
        // SampleStart
        // Hardware component lifetime — models wear-out (shape > 1) or burn-in (shape < 1)
        val componentLifetime = WeibullDistribution(shape = 2.0, scale = 5000.0)

        componentLifetime.quantile(0.1)  // 10% of components fail before this time
        componentLifetime.cdf(3000.0)    // probability of failure before 3000 hours
        componentLifetime.sf(4000.0)     // probability of surviving past 4000 hours
        // SampleEnd
    }

    @Test
    fun choosingEventCounts() {
        // SampleStart
        // Errors per hour on a production server
        val errorsPerHour = PoissonDistribution(rate = 3.2)

        errorsPerHour.pmf(0)            // probability of zero errors
        errorsPerHour.cdf(5)            // probability of at most 5 errors
        errorsPerHour.quantileInt(0.99) // error count exceeded only 1% of the time
        // SampleEnd
    }

    @Test
    fun choosingFixedTrials() {
        // SampleStart
        // Conversions out of 1000 page views with 3.5% conversion rate
        val conversions = BinomialDistribution(trials = 1000, probability = 0.035)

        conversions.mean    // expected number of conversions
        conversions.pmf(40) // probability of exactly 40 conversions
        conversions.sf(50)  // probability of more than 50 conversions
        // SampleEnd
    }

    @Test
    fun choosingOverdispersed() {
        // SampleStart
        // Support tickets until 5th resolution (variance > mean)
        val tickets = NegativeBinomialDistribution(successes = 5, probability = 0.1)

        tickets.mean     // expected ticket count
        tickets.variance // larger than mean — overdispersion
        // SampleEnd
    }

    @Test
    fun choosingProportions() {
        // SampleStart
        // Click-through rate estimated from 120 clicks in 4000 impressions
        val ctr = BetaDistribution(alpha = 120.0, beta = 3880.0)

        ctr.mean            // point estimate of CTR
        ctr.quantile(0.025) // lower bound of 95% credible interval
        ctr.quantile(0.975) // upper bound
        ctr.cdf(0.035)      // probability that true CTR is below 3.5%
        // SampleEnd
    }

    @Test
    fun choosingSymmetric() {
        // SampleStart
        // User session duration in minutes (roughly symmetric)
        val sessionDuration = NormalDistribution(mu = 12.5, sigma = 3.2)

        sessionDuration.cdf(15.0)      // probability of session under 15 min
        sessionDuration.quantile(0.95) // 95th percentile

        // Small-sample estimate — heavier tails give more conservative intervals
        val smallSampleEstimate = StudentTDistribution(degreesOfFreedom = 8.0)
        smallSampleEstimate.quantile(0.975) // critical value for 95% CI
        // SampleEnd
    }
}
