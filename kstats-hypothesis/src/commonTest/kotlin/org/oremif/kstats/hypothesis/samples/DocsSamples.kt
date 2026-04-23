package org.oremif.kstats.hypothesis.samples

import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.descriptive.mean
import org.oremif.kstats.descriptive.standardDeviation
import org.oremif.kstats.distributions.NormalDistribution
import org.oremif.kstats.hypothesis.*
import kotlin.test.Test
import kotlin.test.assertEquals

class DocsSamples {

    companion object {
        val abControlDuration = doubleArrayOf(
            34.2, 41.5, 38.7, 45.1, 36.9, 42.3, 39.8, 44.6, 37.4, 40.1,
            43.2, 35.8, 41.9, 38.3, 46.0, 39.5, 42.7, 37.1, 40.8, 44.3
        )
        val abTreatmentDuration = doubleArrayOf(
            29.1, 33.8, 31.5, 35.2, 28.7, 32.4, 30.9, 34.6, 29.8, 33.1,
            31.2, 27.5, 34.0, 30.3, 36.1, 31.8, 33.5, 28.9, 32.7, 35.8
        )
        val qcSensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        val edaResponseTimeMs = doubleArrayOf(
            89.2, 95.1, 87.6, 102.3, 91.8, 88.4, 96.7, 103.5, 90.1, 94.3,
            88.9, 97.2, 105.8, 91.4, 93.6, 87.1, 99.0, 92.5, 96.1, 104.2,
            90.7, 88.3, 101.6, 93.9, 95.4, 89.8, 98.3, 106.1, 91.0, 94.7
        )
        val edaErrorsPerHour = doubleArrayOf(
            2.0, 3.0, 1.0, 5.0, 2.0, 1.0, 4.0, 6.0, 2.0, 3.0,
            1.0, 4.0, 7.0, 2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 6.0,
            2.0, 1.0, 5.0, 3.0, 3.0, 1.0, 4.0, 8.0, 2.0, 3.0
        )
        val edaMemoryUsageMb = doubleArrayOf(
            512.3, 528.1, 505.7, 545.2, 519.6, 508.4, 534.8, 551.3, 515.0, 526.7,
            509.2, 537.1, 558.4, 517.8, 524.3, 503.1, 541.6, 520.9, 531.5, 549.7,
            514.2, 506.8, 543.9, 522.5, 529.0, 511.4, 539.3, 561.2, 516.3, 527.4
        )
        val edaThroughputRps = doubleArrayOf(
            245.0, 238.0, 251.0, 225.0, 242.0, 249.0, 232.0, 218.0, 244.0, 236.0,
            250.0, 230.0, 212.0, 243.0, 237.0, 253.0, 227.0, 241.0, 233.0, 220.0,
            246.0, 252.0, 224.0, 239.0, 235.0, 248.0, 228.0, 210.0, 243.0, 234.0
        )
    }

    // =====================================================================
    // introduction.mdx
    // =====================================================================

    @Test
    fun introOverview() {
        // SampleStart
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)

        val summary = sample.describe()
        summary.mean              // 5.1667
        summary.standardDeviation // 2.4833

        val normality = shapiroWilkTest(sample)
        normality.pValue          // 0.8933

        val fitted = NormalDistribution(mu = summary.mean, sigma = summary.standardDeviation)
        fitted.cdf(6.0)           // 0.6335
        // SampleEnd
        assertEquals(5.1667, summary.mean, 1e-4)
        assertEquals(2.4833, summary.standardDeviation, 1e-4)
        assertEquals(0.7939, normality.pValue, 1e-4)
        assertEquals(0.6314, fitted.cdf(6.0), 1e-4)
    }

    // =====================================================================
    // quickstart.mdx
    // =====================================================================

    @Test
    fun quickstartNormality() {
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        // SampleStart
        val normality = shapiroWilkTest(sample)
        normality.statistic          // W statistic
        normality.pValue             // > 0.05 → no evidence against normality
        normality.isSignificant()    // false
        // SampleEnd
        assertEquals(false, normality.isSignificant())
    }

    @Test
    fun quickstartHypothesis() {
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        // SampleStart
        val result = tTest(sample, mu = 5.0)
        result.statistic         // 0.0
        result.pValue            // 1.0
        result.isSignificant()   // false — cannot reject H₀: μ = 5.0
        result.confidenceInterval // (3.21, 6.79)
        // SampleEnd
        assertEquals(0.1644, result.statistic, 1e-4)
        assertEquals(0.8759, result.pValue, 1e-4)
        assertEquals(false, result.isSignificant())
    }

    @Test
    fun quickstartTabHypothesis() {
        // SampleStart
        val sample = doubleArrayOf(5.1, 4.9, 5.3, 5.0, 4.8)
        val result = tTest(sample, mu = 5.0, alternative = Alternative.GREATER)
        result.statistic          // t value
        result.pValue             // one-sided p-value
        result.isSignificant()    // true or false at α = 0.05
        result.confidenceInterval // one-sided CI
        // SampleEnd
    }

    // =====================================================================
    // hypothesis/overview.mdx
    // =====================================================================

    @Test
    fun hypTestResult() {
        // SampleStart
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 5.0)

        result.testName          // "One-Sample T-Test"
        result.statistic         // 2.8284
        result.pValue            // 0.0474
        result.degreesOfFreedom  // 4.0
        result.isSignificant()   // true (p < 0.05)
        result.confidenceInterval // (5.02, 6.98)
        // SampleEnd
        assertEquals("One-Sample t-Test", result.testName)
        assertEquals(2.8284, result.statistic, 1e-4)
        assertEquals(0.0474, result.pValue, 1e-4)
        assertEquals(4.0, result.degreesOfFreedom, 1e-4)
        assertEquals(true, result.isSignificant())
        assertEquals(5.02, result.confidenceInterval!!.lower, 0.01)
        assertEquals(6.98, result.confidenceInterval.upper, 0.01)
    }

    @Test
    fun hypOneSample() {
        // SampleStart
        val sample = doubleArrayOf(5.1, 4.9, 5.3, 5.0, 4.8)

        // Two-sided: is the mean different from 5.0?
        val two = tTest(sample, mu = 5.0)
        two.statistic            // t value
        two.pValue               // two-sided p-value

        // One-sided: is the mean greater than 5.0?
        val one = tTest(sample, mu = 5.0, alternative = Alternative.GREATER)
        one.pValue               // one-sided p-value
        // SampleEnd
    }

    @Test
    fun hypTwoSample() {
        // SampleStart
        val group1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val group2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)

        val result = tTest(group1, group2)
        result.statistic         // -5.0
        result.pValue            // 0.0011
        result.isSignificant()   // true
        // SampleEnd
        assertEquals(-5.0, result.statistic, 1e-4)
        assertEquals(0.0011, result.pValue, 1e-4)
        assertEquals(true, result.isSignificant())
    }

    @Test
    fun hypPaired() {
        // SampleStart
        val before = doubleArrayOf(200.0, 190.0, 210.0, 180.0, 195.0)
        val after = doubleArrayOf(190.0, 180.0, 195.0, 170.0, 185.0)

        val result = pairedTTest(before, after)
        result.statistic         // positive t (before > after)
        result.pValue            // p-value for the difference
        result.isSignificant()   // true if the change is significant
        // SampleEnd
    }

    @Test
    fun hypMannWhitney() {
        // SampleStart
        val group1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val group2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)

        val result = mannWhitneyUTest(group1, group2)
        result.statistic         // U = 0.0
        result.pValue            // < 0.02
        result.isSignificant()   // true
        // SampleEnd
        assertEquals(0.0, result.statistic, 1e-4)
        assertEquals(true, result.isSignificant())
    }

    @Test
    fun hypWilcoxon() {
        // SampleStart
        val before = doubleArrayOf(10.0, 12.0, 14.0, 16.0, 18.0)
        val after = doubleArrayOf(8.0, 9.0, 11.0, 12.0, 13.0)

        val result = wilcoxonSignedRankTest(before, after)
        result.statistic         // W+ = 15.0
        result.pValue            // p-value with continuity correction
        // SampleEnd
        assertEquals(15.0, result.statistic, 1e-4)
    }

    @Test
    fun hypAnova() {
        // SampleStart
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val g3 = doubleArrayOf(11.0, 12.0, 13.0, 14.0, 15.0)

        val anova = oneWayAnova(g1, g2, g3)
        anova.fStatistic         // 50.0
        anova.pValue             // < 0.00001
        anova.dfBetween          // 2
        anova.dfWithin           // 12
        anova.ssBetween          // 250.0
        anova.ssWithin           // 30.0
        anova.msBetween          // 125.0
        anova.msWithin           // 2.5
        // SampleEnd
        assertEquals(50.0, anova.fStatistic, 1e-4)
        assertEquals(2, anova.dfBetween)
        assertEquals(12, anova.dfWithin)
        assertEquals(250.0, anova.ssBetween, 1e-4)
        assertEquals(30.0, anova.ssWithin, 1e-4)
        assertEquals(125.0, anova.msBetween, 1e-4)
        assertEquals(2.5, anova.msWithin, 1e-4)
    }

    @Test
    fun hypFriedman() {
        // SampleStart
        val treatment1 = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val treatment2 = doubleArrayOf(4.0, 5.0, 6.0, 4.5, 5.5)
        val treatment3 = doubleArrayOf(7.0, 8.0, 9.0, 7.5, 8.5)

        val result = friedmanTest(treatment1, treatment2, treatment3)
        result.statistic         // Friedman chi-squared
        result.pValue            // p-value
        // SampleEnd
    }

    @Test
    fun hypNormality() {
        // SampleStart
        val sample = doubleArrayOf(
            -1.2, -0.5, 0.0, 0.5, 1.2, 0.3, -0.1, 0.8, -0.4, 0.6,
            -0.8, 0.2, 0.9, -0.3, 0.4, -0.6, 1.0, -0.9, 0.1, 0.7
        )

        shapiroWilkTest(sample).pValue       // high p → consistent with normality
        andersonDarlingTest(sample).pValue   // high p → consistent with normality
        dagostinoPearsonTest(sample).pValue  // combines skewness and kurtosis tests
        jarqueBeraTest(sample).pValue        // asymptotic test based on skewness and kurtosis
        // SampleEnd
    }

    @Test
    fun hypChiSquared() {
        // SampleStart
        val observed = intArrayOf(50, 30, 20)
        val expected = doubleArrayOf(40.0, 40.0, 20.0)

        val result = chiSquaredTest(observed, expected)
        result.statistic         // 5.0
        result.pValue            // 0.0821
        result.isSignificant()   // false at α = 0.05
        // SampleEnd
        assertEquals(5.0, result.statistic, 1e-4)
        assertEquals(0.0821, result.pValue, 1e-4)
        assertEquals(false, result.isSignificant())
    }

    @Test
    fun hypGTest() {
        // SampleStart
        val observed = intArrayOf(50, 30, 20)
        val expected = doubleArrayOf(40.0, 40.0, 20.0)

        val result = gTest(observed, expected)
        result.statistic         // G statistic
        result.pValue            // p-value
        // SampleEnd
    }

    @Test
    fun hypBinomial() {
        // SampleStart
        val result = binomialTest(successes = 60, trials = 100, probability = 0.5)
        result.pValue            // p-value for H₀: p = 0.5
        result.isSignificant()   // true if 60/100 is significantly different from 0.5
        // SampleEnd
        assertEquals(false, result.isSignificant())
    }

    @Test
    fun hypFisher() {
        // SampleStart
        // 2×2 table: [[10, 30], [20, 40]]
        val result = fisherExactTest(arrayOf(intArrayOf(10, 30), intArrayOf(20, 40)))
        result.pValue            // exact p-value
        result.isSignificant()   // true or false
        // SampleEnd
    }

    @Test
    fun hypVarianceHomogeneity() {
        // SampleStart
        val g1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val g2 = doubleArrayOf(2.0, 4.0, 6.0, 8.0, 10.0)
        val g3 = doubleArrayOf(1.0, 3.0, 5.0, 7.0, 9.0)

        leveneTest(g1, g2, g3).pValue         // robust to non-normality
        bartlettTest(g1, g2, g3).pValue       // most powerful when normality holds
        flignerKilleenTest(g1, g2, g3).pValue // rank-based, robust to outliers
        // SampleEnd
    }

    @Test
    fun hypKolmogorovSmirnov() {
        // SampleStart
        val sample1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val sample2 = doubleArrayOf(1.5, 2.5, 3.5, 4.5, 5.5)

        val result = kolmogorovSmirnovTest(sample1, sample2)
        result.statistic         // D = max difference between ECDFs
        result.pValue            // p-value
        // SampleEnd
    }

    @Test
    fun hypGrubbsSingle() {
        // SampleStart
        // Response times (ms) with a suspected outlier
        val latencies = doubleArrayOf(12.0, 14.0, 11.0, 13.0, 15.0, 98.0, 12.0)

        val result = grubbsTest(latencies)
        result.statistic                       // G statistic
        result.pValue                          // Bonferroni-corrected p-value
        result.additionalInfo["outlierIndex"]  // index of the suspected outlier
        result.additionalInfo["outlierValue"]  // the suspected outlier's value
        result.isSignificant()                 // true if outlier is significant at α = 0.05
        // SampleEnd
    }

    @Test
    fun hypGrubbsDirection() {
        // SampleStart
        // Only test for a suspiciously large value (upper tail)
        val data = doubleArrayOf(2.1, 2.5, 2.3, 2.8, 10.0, 2.4, 2.2)
        val upper = grubbsTest(data, alternative = Alternative.GREATER)
        upper.additionalInfo["outlierValue"] // 10.0 — the maximum

        // Only test for a suspiciously small value (lower tail)
        val dataLow = doubleArrayOf(2.1, 2.5, 2.3, 2.8, -5.0, 2.4, 2.2)
        val lower = grubbsTest(dataLow, alternative = Alternative.LESS)
        lower.additionalInfo["outlierValue"] // -5.0 — the minimum
        // SampleEnd
    }

    @Test
    fun hypGrubbsIterative() {
        // SampleStart
        // Remove multiple outliers by repeatedly applying the test
        val data = doubleArrayOf(10.0, 11.0, 12.0, 13.0, 14.0, 80.0, 90.0)
        val cleaned = grubbsTestIterative(data, alpha = 0.05)

        cleaned.outlierIndices // indices (in the original array) that were removed
        cleaned.cleanedData    // observations after removing all detected outliers
        cleaned.iterations     // TestResult from each round (last one is non-significant)
        // SampleEnd
    }

    // =====================================================================
    // choosing-a-distribution.mdx
    // =====================================================================

    @Test
    fun choosingVerifyFit() {
        // SampleStart
        val processingTimesMs = doubleArrayOf(
            45.2, 51.8, 48.1, 52.3, 47.6, 49.9, 53.1, 46.5, 50.7, 48.8,
            51.2, 47.3, 49.1, 52.8, 46.9, 50.3, 48.5, 51.6, 47.8, 49.4
        )

        // Fit a Normal from sample statistics
        val fitted = NormalDistribution(
            mu = processingTimesMs.mean(),
            sigma = processingTimesMs.standardDeviation()
        )

        val ks = kolmogorovSmirnovTest(processingTimesMs, fitted)
        ks.statistic // KS statistic — smaller means better fit
        ks.pValue    // high p-value means data does not contradict the distribution
        // SampleEnd
    }

    // =====================================================================
    // testing-assumptions.mdx
    // =====================================================================

    @Test
    fun testingNormality() {
        // SampleStart
        val sensorReadings = doubleArrayOf(
            150.2, 151.8, 149.6, 152.1, 150.9, 151.3, 149.8, 152.5, 150.4, 151.1,
            150.7, 149.5, 151.6, 150.0, 152.3, 151.0, 149.9, 150.8, 151.5, 150.3,
            151.2, 149.7, 150.6, 152.0, 150.1, 151.4, 149.4, 151.9, 150.5, 151.7
        )

        val shapiro = shapiroWilkTest(sensorReadings)
        val anderson = andersonDarlingTest(sensorReadings)
        val dagostino = dagostinoPearsonTest(sensorReadings)
        val jarqueBera = jarqueBeraTest(sensorReadings)

        shapiro.pValue    // Shapiro-Wilk
        anderson.pValue   // Anderson-Darling
        dagostino.pValue  // D'Agostino-Pearson
        jarqueBera.pValue // Jarque-Bera
        // SampleEnd
    }

    @Test
    fun testingNormalityDescriptive() {
        val sensorReadings = doubleArrayOf(
            150.2, 151.8, 149.6, 152.1, 150.9, 151.3, 149.8, 152.5, 150.4, 151.1,
            150.7, 149.5, 151.6, 150.0, 152.3, 151.0, 149.9, 150.8, 151.5, 150.3,
            151.2, 149.7, 150.6, 152.0, 150.1, 151.4, 149.4, 151.9, 150.5, 151.7
        )
        // SampleStart
        val summary = sensorReadings.describe()
        summary.skewness // close to 0 for symmetric data
        summary.kurtosis // close to 0 (excess) for Normal-like tails
        // SampleEnd
    }

    @Test
    fun testingVarianceHomogeneity() {
        // SampleStart
        val batchA = doubleArrayOf(48.2, 47.8, 49.1, 48.5, 47.9, 48.7, 48.3, 49.0, 48.1, 48.6)
        val batchB = doubleArrayOf(51.3, 50.8, 52.1, 51.0, 51.7, 50.5, 51.9, 51.2, 50.9, 51.5)
        val batchC = doubleArrayOf(49.5, 50.2, 49.8, 50.0, 49.3, 50.4, 49.7, 50.1, 49.6, 50.3)

        val levene = leveneTest(batchA, batchB, batchC)
        val bartlett = bartlettTest(batchA, batchB, batchC)
        val fligner = flignerKilleenTest(batchA, batchB, batchC)

        levene.pValue   // Levene
        bartlett.pValue // Bartlett
        fligner.pValue  // Fligner-Killeen
        // SampleEnd
    }

    @Test
    fun testingVarianceThenAnova() {
        val batchA = doubleArrayOf(48.2, 47.8, 49.1, 48.5, 47.9, 48.7, 48.3, 49.0, 48.1, 48.6)
        val batchB = doubleArrayOf(51.3, 50.8, 52.1, 51.0, 51.7, 50.5, 51.9, 51.2, 50.9, 51.5)
        val batchC = doubleArrayOf(49.5, 50.2, 49.8, 50.0, 49.3, 50.4, 49.7, 50.1, 49.6, 50.3)
        // SampleStart
        val anova = oneWayAnova(batchA, batchB, batchC)
        anova.fStatistic
        anova.pValue
        // SampleEnd
    }

    @Test
    fun testingKsOneSample() {
        // SampleStart
        val temperatureReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6
        )

        // Fit Normal from sample
        val fitted = NormalDistribution(
            mu = temperatureReadings.mean(),
            sigma = temperatureReadings.standardDeviation()
        )

        val ks = kolmogorovSmirnovTest(temperatureReadings, fitted)
        ks.statistic // smaller means better fit
        ks.pValue
        // SampleEnd
    }

    @Test
    fun testingChiSquared() {
        // SampleStart
        // Defect counts across 5 product categories
        val observedDefects = intArrayOf(12, 18, 25, 15, 30)

        // Test against uniform expectation (null = equal probability per category)
        val uniform = chiSquaredTest(observedDefects)
        uniform.pValue

        // Test against specific expected counts
        val expectedCounts = doubleArrayOf(20.0, 20.0, 20.0, 20.0, 20.0)
        val specific = chiSquaredTest(observedDefects, expectedCounts)
        specific.pValue
        // SampleEnd
    }

    @Test
    fun testingKsTwoSample() {
        // SampleStart
        val morningReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4
        )
        val nightReadings = doubleArrayOf(
            156.1, 155.3, 157.0, 156.5, 155.8, 156.8, 155.5, 157.2, 155.9, 156.3
        )

        val twoSampleKs = kolmogorovSmirnovTest(morningReadings, nightReadings)
        twoSampleKs.pValue // low p-value suggests different underlying distributions
        // SampleEnd
    }

    // =====================================================================
    // ab-testing.mdx
    // =====================================================================

    @Test
    fun abTestingData() {
        // SampleStart
        // Variant A (control): original checkout flow
        val controlDurationSec = doubleArrayOf(
            34.2, 41.5, 38.7, 45.1, 36.9, 42.3, 39.8, 44.6, 37.4, 40.1,
            43.2, 35.8, 41.9, 38.3, 46.0, 39.5, 42.7, 37.1, 40.8, 44.3
        )

        // Variant B (treatment): simplified checkout flow
        val treatmentDurationSec = doubleArrayOf(
            29.1, 33.8, 31.5, 35.2, 28.7, 32.4, 30.9, 34.6, 29.8, 33.1,
            31.2, 27.5, 34.0, 30.3, 36.1, 31.8, 33.5, 28.9, 32.7, 35.8
        )
        // SampleEnd
    }

    @Test
    fun abTestingSummarize() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val controlSummary = controlDurationSec.describe()
        val treatmentSummary = treatmentDurationSec.describe()

        controlSummary.mean                // control average
        treatmentSummary.mean              // treatment average
        controlSummary.standardDeviation   // control spread
        treatmentSummary.standardDeviation // treatment spread
        // SampleEnd
    }

    @Test
    fun abTestingNormality() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val controlNormality = shapiroWilkTest(controlDurationSec)
        val treatmentNormality = shapiroWilkTest(treatmentDurationSec)

        controlNormality.pValue
        treatmentNormality.pValue
        // SampleEnd
    }

    @Test
    fun abTestingVariance() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val variances = leveneTest(controlDurationSec, treatmentDurationSec)
        variances.pValue
        // SampleEnd
    }

    @Test
    fun abTestingTTest() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        // Welch's t-test (default: equalVariances = false)
        val result = tTest(controlDurationSec, treatmentDurationSec)

        result.statistic
        result.pValue
        result.confidenceInterval // 95% CI for the difference in means
        result.isSignificant()    // true if p < 0.05
        // SampleEnd
    }

    @Test
    fun abTestingTTestEqual() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val equalVar = tTest(
            controlDurationSec,
            treatmentDurationSec,
            equalVariances = true
        )
        equalVar.pValue
        // SampleEnd
    }

    @Test
    fun abTestingMannWhitney() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val result = mannWhitneyUTest(controlDurationSec, treatmentDurationSec)

        result.statistic
        result.pValue
        result.isSignificant()
        // SampleEnd
    }

    @Test
    fun abTestingOneSided() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        val oneSided = tTest(
            controlDurationSec,
            treatmentDurationSec,
            alternative = Alternative.GREATER // control > treatment
        )
        oneSided.pValue
        // SampleEnd
    }

    @Test
    fun abTestingEffectSize() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        // SampleStart
        // Cohen's d: how large is the difference in standard-deviation units?
        val d = cohensD(controlDurationSec, treatmentDurationSec)
        d // ~2.9 → large effect (|d| ≥ 0.8)
        // SampleEnd
    }

    @Test
    fun abTestingSecondMetric() {
        // SampleStart
        // Number of completed checkout steps per session
        val controlSteps = doubleArrayOf(
            3.0, 4.0, 3.0, 5.0, 3.0, 4.0, 4.0, 5.0, 3.0, 4.0,
            4.0, 3.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 4.0, 5.0
        )
        val treatmentSteps = doubleArrayOf(
            5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0,
            4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0
        )

        // Discrete step counts are typically non-normal
        shapiroWilkTest(controlSteps).pValue

        val stepsResult = mannWhitneyUTest(controlSteps, treatmentSteps)
        stepsResult.pValue
        stepsResult.isSignificant()
        // SampleEnd
    }

    @Test
    fun abTestingMultipleComparison() {
        val controlDurationSec = abControlDuration
        val treatmentDurationSec = abTreatmentDuration
        val result = tTest(controlDurationSec, treatmentDurationSec)
        val controlSteps = doubleArrayOf(
            3.0, 4.0, 3.0, 5.0, 3.0, 4.0, 4.0, 5.0, 3.0, 4.0,
            4.0, 3.0, 4.0, 3.0, 5.0, 4.0, 4.0, 3.0, 4.0, 5.0
        )
        val treatmentSteps = doubleArrayOf(
            5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0,
            4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0
        )
        val stepsResult = mannWhitneyUTest(controlSteps, treatmentSteps)
        // SampleStart
        // Correct for testing two metrics (duration + steps)
        val rawPValues = doubleArrayOf(result.pValue, stepsResult.pValue)
        val corrected = holmBonferroniCorrection(rawPValues)

        corrected[0] // adjusted p-value for duration
        corrected[1] // adjusted p-value for steps
        // SampleEnd
    }

    @Test
    fun abTestingPaired() {
        // SampleStart
        val beforeMs = doubleArrayOf(
            340.2, 415.0, 387.1, 451.3, 369.5, 423.8, 398.0, 446.2, 374.1, 401.5
        )
        val afterMs = doubleArrayOf(
            310.5, 380.2, 355.8, 410.7, 335.1, 392.4, 365.3, 405.9, 340.8, 371.6
        )

        val paired = pairedTTest(beforeMs, afterMs)
        paired.pValue
        paired.confidenceInterval

        // Non-parametric alternative
        val wilcoxon = wilcoxonSignedRankTest(beforeMs, afterMs)
        wilcoxon.pValue

        // Paired effect size: Cohen's dz = mean(diff) / sd(diff)
        val differences = DoubleArray(beforeMs.size) { beforeMs[it] - afterMs[it] }
        val dz = differences.mean() / differences.standardDeviation()
        dz // ~6.1 → large effect
        // SampleEnd
    }

    // =====================================================================
    // quality-control.mdx
    // =====================================================================

    @Test
    fun qcBatchStability() {
        // SampleStart
        val morningBatch = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4
        )
        val eveningBatch = doubleArrayOf(
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6
        )

        // Check normality
        shapiroWilkTest(morningBatch).pValue
        shapiroWilkTest(eveningBatch).pValue

        // Check variance homogeneity
        leveneTest(morningBatch, eveningBatch).pValue

        // Compare means
        val batchComparison = tTest(morningBatch, eveningBatch)
        batchComparison.pValue
        batchComparison.isSignificant() // false means no significant shift
        // SampleEnd
    }

    @Test
    fun qcMultipleBatches() {
        // SampleStart
        val batch1 = doubleArrayOf(155.2, 154.8, 156.1, 155.5, 154.3)
        val batch2 = doubleArrayOf(155.9, 155.0, 156.3, 154.7, 155.4)
        val batch3 = doubleArrayOf(155.8, 154.5, 156.0, 155.3, 154.9)

        val anova = oneWayAnova(batch1, batch2, batch3)
        anova.fStatistic
        anova.pValue
        // SampleEnd
    }

    @Test
    fun qcDistributionFit() {
        val sensorReadings = qcSensorReadings
        // SampleStart
        // Exclude known outliers for fitting
        val cleanReadings = sensorReadings.filter { it in 150.0..160.0 }.toDoubleArray()

        val fitted = NormalDistribution(
            mu = cleanReadings.mean(),
            sigma = cleanReadings.standardDeviation()
        )

        val ks = kolmogorovSmirnovTest(cleanReadings, fitted)
        ks.pValue // high p-value supports the normal process model
        // SampleEnd
    }

    // =====================================================================
    // exploratory-analysis.mdx
    // =====================================================================

    @Test
    fun edaNormalityTests() {
        val responseTimeMs = edaResponseTimeMs
        val errorsPerHour = edaErrorsPerHour
        val memoryUsageMb = edaMemoryUsageMb
        val throughputRps = edaThroughputRps
        // SampleStart
        shapiroWilkTest(responseTimeMs).pValue
        shapiroWilkTest(errorsPerHour).pValue
        shapiroWilkTest(memoryUsageMb).pValue
        shapiroWilkTest(throughputRps).pValue
        // SampleEnd
    }

    @Test
    fun edaFitDistribution() {
        val responseTimeMs = edaResponseTimeMs
        val memoryUsageMb = edaMemoryUsageMb
        // SampleStart
        val rtFit = NormalDistribution(
            mu = responseTimeMs.mean(),
            sigma = responseTimeMs.standardDeviation()
        )
        kolmogorovSmirnovTest(responseTimeMs, rtFit).pValue

        val memFit = NormalDistribution(
            mu = memoryUsageMb.mean(),
            sigma = memoryUsageMb.standardDeviation()
        )
        kolmogorovSmirnovTest(memoryUsageMb, memFit).pValue
        // SampleEnd
    }

    @Test
    fun edaComparePeriods() {
        val responseTimeMs = edaResponseTimeMs
        // SampleStart
        val firstHalfRt = responseTimeMs.sliceArray(0 until 15)
        val secondHalfRt = responseTimeMs.sliceArray(15 until 30)

        val periodComparison = tTest(firstHalfRt, secondHalfRt)
        periodComparison.pValue
        periodComparison.isSignificant()

        // Non-parametric alternative
        val periodRank = mannWhitneyUTest(firstHalfRt, secondHalfRt)
        periodRank.pValue
        // SampleEnd
    }

    @Test
    fun edaCompareThroughput() {
        val throughputRps = edaThroughputRps
        // SampleStart
        val firstHalfTp = throughputRps.sliceArray(0 until 15)
        val secondHalfTp = throughputRps.sliceArray(15 until 30)

        val tpComparison = tTest(firstHalfTp, secondHalfTp)
        tpComparison.pValue
        // SampleEnd
    }
}
