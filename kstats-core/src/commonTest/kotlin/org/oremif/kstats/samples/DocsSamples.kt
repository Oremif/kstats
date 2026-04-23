package org.oremif.kstats.samples

import org.oremif.kstats.descriptive.*
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DocsSamples {

    @Test
    fun coreSummarySnapshot() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        val stats = data.describe()

        stats.count              // 8
        stats.mean               // 5.0
        stats.median             // 4.5
        stats.standardDeviation  // 2.1380
        stats.variance           // 4.5714
        stats.q1                 // 4.0
        stats.q3                 // 5.5
        stats.interquartileRange // 1.5
        stats.skewness           // 0.6563
        stats.kurtosis           // -0.1640
        stats.range              // 7.0
        stats.standardError      // 0.7559
        // SampleEnd
        assertEquals(8L, stats.count)
        assertEquals(5.0, stats.mean, 1e-4)
        assertEquals(4.5, stats.median, 1e-4)
        assertEquals(2.1380, stats.standardDeviation, 1e-4)
        assertEquals(4.5714, stats.variance, 1e-4)
        assertEquals(4.0, stats.q1, 1e-4)
        assertEquals(5.5, stats.q3, 1e-4)
        assertEquals(1.5, stats.interquartileRange, 1e-4)
        assertEquals(0.8185, stats.skewness, 1e-4)
        assertEquals(0.9406, stats.kurtosis, 1e-4)
        assertEquals(7.0, stats.range, 1e-4)
        assertEquals(0.7559, stats.standardError, 1e-4)
    }

    @Test
    fun coreCentralTendency() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

        data.mean()                    // 5.0
        data.median()                  // 4.5
        data.toList().mode()           // {4.0}

        data.trimmedMean(0.1)          // 4.8333 — trims 10% from each tail

        val positive = doubleArrayOf(1.0, 2.0, 4.0, 8.0)
        positive.geometricMean()       // 2.8284
        positive.harmonicMean()        // 2.1333

        val values = doubleArrayOf(1.0, 2.0, 3.0)
        val weights = doubleArrayOf(3.0, 1.0, 1.0)
        values.weightedMean(weights)   // 1.6
        // SampleEnd
        assertEquals(5.0, data.mean(), 1e-4)
        assertEquals(4.5, data.median(), 1e-4)
        assertEquals(setOf(4.0), data.toList().mode())
        assertEquals(5.0, data.trimmedMean(0.1), 1e-4)
        assertEquals(2.8284, positive.geometricMean(), 1e-4)
        assertEquals(2.1333, positive.harmonicMean(), 1e-4)
        assertEquals(1.6, values.weightedMean(weights), 1e-4)
    }

    @Test
    fun coreDispersion() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

        data.variance()                                    // 4.5714 (sample)
        data.variance(PopulationKind.POPULATION)            // 4.0
        data.standardDeviation()                            // 2.1380
        data.range()                                        // 7.0
        data.interquartileRange()                           // 1.5
        data.meanAbsoluteDeviation()                        // 1.5

        data.trimmedVariance(0.1)                           // trimmed sample variance

        data.semiVariance(5.0, SemiVarianceDirection.DOWNSIDE) // 1.7143
        data.semiVariance(5.0, SemiVarianceDirection.UPSIDE)   // 2.8571
        // SampleEnd
        assertEquals(4.5714, data.variance(), 1e-4)
        assertEquals(4.0, data.variance(PopulationKind.POPULATION), 1e-4)
        assertEquals(2.1380, data.standardDeviation(), 1e-4)
        assertEquals(7.0, data.range(), 1e-4)
        assertEquals(1.5, data.interquartileRange(), 1e-4)
        assertEquals(1.5, data.meanAbsoluteDeviation(), 1e-4)
        assertEquals(1.7143, data.semiVariance(5.0, SemiVarianceDirection.DOWNSIDE), 1e-4)
        assertEquals(2.8571, data.semiVariance(5.0, SemiVarianceDirection.UPSIDE), 1e-4)
    }

    @Test
    fun coreQuantiles() {
        // SampleStart
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

        data.quantile(0.5)         // 5.5 (median)
        data.quantile(0.25)        // 3.25 (Q1)
        data.percentile(90.0)      // 9.1 (90th percentile)

        val (q1, median, q3) = data.quartiles()
        // q1 = 3.25, median = 5.5, q3 = 7.75
        // SampleEnd
        assertEquals(5.5, data.quantile(0.5), 1e-4)
        assertEquals(3.25, data.quantile(0.25), 1e-4)
        assertEquals(9.1, data.percentile(90.0), 1e-4)
        assertEquals(3.25, q1, 1e-4)
        assertEquals(5.5, median, 1e-4)
        assertEquals(7.75, q3, 1e-4)
    }

    @Test
    fun coreShape() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)

        data.skewness()            // 0.6563 — positive: right tail is longer
        data.kurtosis()            // -0.1640 — negative excess: lighter tails than normal
        data.kurtosis(excess = false) // raw kurtosis (not centered at 0)

        data.centralMoment(2)      // 4.0 (equals population variance)
        data.centralMoment(3)      // 5.25
        data.centralMoment(4)      // 44.5

        data.kStatistic(1)          // 5.0 (equals mean)
        data.kStatistic(2)          // 4.5714 (equals sample variance)
        // SampleEnd
        assertEquals(0.8185, data.skewness(), 1e-4)
        assertEquals(0.9406, data.kurtosis(), 1e-4)
        assertEquals(4.0, data.centralMoment(2), 1e-4)
        assertEquals(5.25, data.centralMoment(3), 1e-4)
        assertEquals(44.5, data.centralMoment(4), 1e-4)
        assertEquals(5.0, data.kStatistic(1), 1e-4)
        assertEquals(4.5714, data.kStatistic(2), 1e-4)
    }

    @Test
    fun coreFrequency() {
        // SampleStart
        val freq = listOf("a", "a", "b", "b", "b", "c").toFrequency()
        freq.totalCount            // 6
        freq.count("b")            // 3
        freq.proportion("b")       // 0.5
        freq.cumulativeCount("b")  // 5 (a=2 + b=3)
        freq.mode                  // {b}
        // SampleEnd
        assertEquals(6L, freq.totalCount)
        assertEquals(3L, freq.count("b"))
        assertEquals(0.5, freq.proportion("b"), 1e-4)
        assertEquals(5L, freq.cumulativeCount("b"))
        assertEquals(setOf("b"), freq.mode)
    }

    @Test
    fun coreStreaming() {
        // SampleStart
        val online = OnlineStatistics()
        online.addAll(doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0))

        online.count               // 8
        online.mean                // 5.0
        online.sum                 // 40.0
        online.min                 // 2.0
        online.max                 // 9.0
        online.variance()          // 4.5714 (sample)
        online.standardDeviation() // 2.1380
        online.skewness()          // 0.6563

        // Add more data later
        online.add(3.0)
        online.count               // 9
        online.mean                // 4.7778
        // SampleEnd
        assertEquals(9L, online.count)
        assertEquals(4.7778, online.mean, 1e-4)
    }

    @Test
    fun quickstartDefine() {
        // SampleStart
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        // SampleEnd
    }

    @Test
    fun quickstartSummarize() {
        val sample = doubleArrayOf(2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0)
        // SampleStart
        val stats = sample.describe()
        stats.mean              // 5.0
        stats.median            // 4.5
        stats.standardDeviation // 2.1380
        stats.skewness          // 0.6563
        stats.interquartileRange // 1.5
        // SampleEnd
        assertEquals(5.0, stats.mean, 1e-4)
        assertEquals(4.5, stats.median, 1e-4)
        assertEquals(2.1380, stats.standardDeviation, 1e-4)
        assertEquals(0.8185, stats.skewness, 1e-4)
        assertEquals(1.5, stats.interquartileRange, 1e-4)
    }

    @Test
    fun quickstartTabCore() {
        // SampleStart
        val data = doubleArrayOf(2.0, 4.0, 4.0, 5.0, 7.0, 9.0)
        data.mean()               // 5.1667
        data.median()             // 4.5
        data.standardDeviation()  // 2.4833
        val summary = data.describe()
        summary.skewness          // 0.3942
        // SampleEnd
        assertEquals(5.1667, data.mean(), 1e-4)
        assertEquals(4.5, data.median(), 1e-4)
        assertEquals(2.4833, data.standardDeviation(), 1e-4)
        assertEquals(0.5398, summary.skewness, 1e-4)
    }

    @Test
    fun qcProcessData() {
        // SampleStart
        // Temperature readings (°C) from sensor on production line
        val sensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        // Two potential outliers: 162.5 and 141.3
        // SampleEnd
    }

    @Test
    fun qcBaselineStats() {
        val sensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        // SampleStart
        val summary = sensorReadings.describe()

        summary.mean
        summary.standardDeviation
        summary.min // check for unusually low values
        summary.max // check for unusually high values
        summary.interquartileRange
        // SampleEnd
    }

    @Test
    fun qcPercentileDetection() {
        val sensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        // SampleStart
        val lowerBound = sensorReadings.quantile(0.01)
        val upperBound = sensorReadings.quantile(0.99)

        val outliers = sensorReadings.filter { it < lowerBound || it > upperBound }
        // SampleEnd
    }

    @Test
    fun qcControlLimits() {
        val sensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        // SampleStart
        val centerLine = sensorReadings.mean()
        val sigma = sensorReadings.standardDeviation()

        val upperControlLimit = centerLine + 3 * sigma
        val lowerControlLimit = centerLine - 3 * sigma

        val outOfControl = sensorReadings.filter { it > upperControlLimit || it < lowerControlLimit }
        // SampleEnd
    }

    @Test
    fun edaDataset() {
        // SampleStart
        val responseTimeMs = doubleArrayOf(
            89.2, 95.1, 87.6, 102.3, 91.8, 88.4, 96.7, 103.5, 90.1, 94.3,
            88.9, 97.2, 105.8, 91.4, 93.6, 87.1, 99.0, 92.5, 96.1, 104.2,
            90.7, 88.3, 101.6, 93.9, 95.4, 89.8, 98.3, 106.1, 91.0, 94.7
        )

        val errorsPerHour = doubleArrayOf(
            2.0, 3.0, 1.0, 5.0, 2.0, 1.0, 4.0, 6.0, 2.0, 3.0,
            1.0, 4.0, 7.0, 2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 6.0,
            2.0, 1.0, 5.0, 3.0, 3.0, 1.0, 4.0, 8.0, 2.0, 3.0
        )

        val memoryUsageMb = doubleArrayOf(
            512.3, 528.1, 505.7, 545.2, 519.6, 508.4, 534.8, 551.3, 515.0, 526.7,
            509.2, 537.1, 558.4, 517.8, 524.3, 503.1, 541.6, 520.9, 531.5, 549.7,
            514.2, 506.8, 543.9, 522.5, 529.0, 511.4, 539.3, 561.2, 516.3, 527.4
        )

        val throughputRps = doubleArrayOf(
            245.0, 238.0, 251.0, 225.0, 242.0, 249.0, 232.0, 218.0, 244.0, 236.0,
            250.0, 230.0, 212.0, 243.0, 237.0, 253.0, 227.0, 241.0, 233.0, 220.0,
            246.0, 252.0, 224.0, 239.0, 235.0, 248.0, 228.0, 210.0, 243.0, 234.0
        )
        // SampleEnd
    }

    @Test
    fun edaSummaryStats() {
        val responseTimeMs = doubleArrayOf(
            89.2, 95.1, 87.6, 102.3, 91.8, 88.4, 96.7, 103.5, 90.1, 94.3,
            88.9, 97.2, 105.8, 91.4, 93.6, 87.1, 99.0, 92.5, 96.1, 104.2,
            90.7, 88.3, 101.6, 93.9, 95.4, 89.8, 98.3, 106.1, 91.0, 94.7
        )
        val errorsPerHour = doubleArrayOf(
            2.0, 3.0, 1.0, 5.0, 2.0, 1.0, 4.0, 6.0, 2.0, 3.0,
            1.0, 4.0, 7.0, 2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 6.0,
            2.0, 1.0, 5.0, 3.0, 3.0, 1.0, 4.0, 8.0, 2.0, 3.0
        )
        val memoryUsageMb = doubleArrayOf(
            512.3, 528.1, 505.7, 545.2, 519.6, 508.4, 534.8, 551.3, 515.0, 526.7,
            509.2, 537.1, 558.4, 517.8, 524.3, 503.1, 541.6, 520.9, 531.5, 549.7,
            514.2, 506.8, 543.9, 522.5, 529.0, 511.4, 539.3, 561.2, 516.3, 527.4
        )
        val throughputRps = doubleArrayOf(
            245.0, 238.0, 251.0, 225.0, 242.0, 249.0, 232.0, 218.0, 244.0, 236.0,
            250.0, 230.0, 212.0, 243.0, 237.0, 253.0, 227.0, 241.0, 233.0, 220.0,
            246.0, 252.0, 224.0, 239.0, 235.0, 248.0, 228.0, 210.0, 243.0, 234.0
        )
        // SampleStart
        val rtSummary = responseTimeMs.describe()
        val errSummary = errorsPerHour.describe()
        val memSummary = memoryUsageMb.describe()
        val tpSummary = throughputRps.describe()

        rtSummary.mean; rtSummary.standardDeviation; rtSummary.min; rtSummary.max
        errSummary.mean; errSummary.standardDeviation; errSummary.min; errSummary.max
        memSummary.mean; memSummary.standardDeviation; memSummary.min; memSummary.max
        tpSummary.mean; tpSummary.standardDeviation; tpSummary.min; tpSummary.max
        // SampleEnd
    }

    @Test
    fun coreProcessCapability() {
        // SampleStart
        // Ten parts measured against a spec window of [48, 52]
        val measurements = doubleArrayOf(
            50.0, 50.5, 49.5, 50.2, 49.8, 50.1, 49.9, 50.3, 49.7, 50.0
        )
        val capability = measurements.processCapability(lsl = 48.0, usl = 52.0)

        capability.cp   // 2.2646 — potential capability (spread vs tolerance)
        capability.cpk  // 2.2646 — actual capability (penalizes off-centering)
        capability.pp   // 2.3870 — overall (population σ) counterpart of Cp
        capability.ppk  // 2.3870 — overall counterpart of Cpk
        // SampleEnd
        assertEquals(2.26455406828919, capability.cp, 1e-4)
        assertEquals(2.26455406828918, capability.cpk, 1e-4)
        assertEquals(2.38704958013144, capability.pp, 1e-4)
        assertEquals(2.38704958013144, capability.ppk, 1e-4)
    }

    @Test
    fun coreXBarRChart() {
        // SampleStart
        // Five subgroups of four parts; bracket width monitored per batch
        val subgroups = listOf(
            doubleArrayOf(72.0, 84.0, 79.0, 49.0),
            doubleArrayOf(56.0, 87.0, 33.0, 42.0),
            doubleArrayOf(55.0, 73.0, 22.0, 60.0),
            doubleArrayOf(44.0, 80.0, 54.0, 74.0),
            doubleArrayOf(97.0, 26.0, 48.0, 58.0),
        )
        val chart = xBarRChart(subgroups)

        chart.centerLine     // 59.65 — grand mean (x-double-bar)
        chart.ucl            // 95.6626 — upper control limit for the mean
        chart.lcl            // 23.6374 — lower control limit for the mean
        chart.rChart.centerLine // 49.4 — average range (R-bar)
        chart.rChart.ucl     // 112.7308 — upper limit for within-subgroup range
        chart.rChart.lcl     // 0.0 — lower limit (D₃ = 0 for n ≤ 6)
        // SampleEnd
        assertEquals(59.65, chart.centerLine, 1e-4)
        assertEquals(95.6626, chart.ucl, 1e-4)
        assertEquals(23.6374, chart.lcl, 1e-4)
        assertEquals(49.4, chart.rChart.centerLine, 1e-4)
        assertEquals(112.7308, chart.rChart.ucl, 1e-4)
        assertEquals(0.0, chart.rChart.lcl, 1e-4)
    }

    @Test
    fun coreXBarSChart() {
        // SampleStart
        // Same subgroups, S chart uses sample standard deviation instead of range
        val subgroups = listOf(
            doubleArrayOf(10.0, 12.0, 11.0, 13.0, 9.0),
            doubleArrayOf(11.0, 10.0, 12.0, 11.0, 14.0),
            doubleArrayOf(9.0, 13.0, 10.0, 12.0, 11.0),
        )
        val chart = xBarSChart(subgroups)

        chart.centerLine      // 11.2 — grand mean
        chart.ucl             // upper control limit for the mean (uses A₃)
        chart.lcl             // lower control limit for the mean
        chart.sChart.centerLine // S-bar — average subgroup standard deviation
        chart.sChart.ucl      // upper limit for within-subgroup spread (B₄)
        chart.sChart.lcl      // lower limit (B₃ = 0 for n ≤ 5)
        // SampleEnd
        assertEquals(11.2, chart.centerLine, 1e-4)
        assertTrue(chart.ucl > chart.centerLine)
        assertTrue(chart.lcl < chart.centerLine)
        assertTrue(chart.sChart.centerLine > 0.0)
        assertTrue(chart.sChart.ucl >= chart.sChart.lcl)
    }

    @Test
    fun coreSpcConstants() {
        // SampleStart
        val c = spcConstants(subgroupSize = 5)
        c.a2  // 0.577 — x-bar factor from R-bar
        c.a3  // 1.427 — x-bar factor from S-bar
        c.d3  // 0.000 — R-chart lower factor (zero for n ≤ 6)
        c.d4  // 2.114 — R-chart upper factor
        c.b3  // 0.000 — S-chart lower factor
        c.b4  // 2.089 — S-chart upper factor
        c.c4  // 0.9400 — bias correction for sample σ
        // SampleEnd
        assertEquals(0.577, c.a2, 0.0)
        assertEquals(1.427, c.a3, 0.0)
        assertEquals(0.0, c.d3, 0.0)
        assertEquals(2.114, c.d4, 0.0)
        assertEquals(0.0, c.b3, 0.0)
        assertEquals(2.089, c.b4, 0.0)
        assertEquals(0.9400, c.c4, 0.0)
    }

    @Test
    fun coreCusum() {
        // SampleStart
        // Individual measurements from a process with target 10, drifting upward
        val observations = doubleArrayOf(10.2, 10.4, 10.6, 10.9, 11.2, 11.5, 11.8, 12.0)
        val result = cusum(observations, target = 10.0, k = 0.5, h = 3.0)

        result.sPlus      // [0.0, 0.0, 0.1, 0.5, 1.2, 2.2, 3.5, 5.0]
        result.sMinus     // all zero — no downward drift
        result.alarmIndex // 6 — first index where C⁺ > H
        // SampleEnd
        assertEquals(0.0, result.sPlus[0], 1e-10)
        assertEquals(3.5, result.sPlus[6], 1e-10)
        assertEquals(5.0, result.sPlus[7], 1e-10)
        for (v in result.sMinus) assertEquals(0.0, v, 1e-10)
        assertEquals(6, result.alarmIndex)
    }

    @Test
    fun coreEwma() {
        // SampleStart
        // EWMA chart: target = 25, σ = 1, λ = 0.2, L = 3
        val observations = doubleArrayOf(25.0, 24.5, 25.2, 26.1, 25.8, 27.0, 26.5, 28.0)
        val result = ewma(
            observations,
            target = 25.0,
            sigma = 1.0,
            lambda = 0.2,
            controlLimitWidth = 3.0
        )

        result.smoothedValues[0] // 25.0 — Z₀ = λ·x + (1-λ)·target
        result.smoothedValues[7] // 26.2549 — smoothed statistic at t = 7
        result.ucl[0]            // 25.6 — narrow at first, widens with t
        result.ucl[7]            // 25.9858 — approaching steady state
        result.outOfControl      // [7] — Z₇ exceeds UCL₇
        // SampleEnd
        assertEquals(25.0, result.smoothedValues[0], 1e-10)
        assertEquals(26.2549248, result.smoothedValues[7], 1e-6)
        assertEquals(25.6, result.ucl[0], 1e-10)
        assertEquals(25.9858257971513, result.ucl[7], 1e-6)
        assertTrue(result.outOfControl.contentEquals(intArrayOf(7)))
    }

    @Test
    fun coreWesternElectricRules() {
        // SampleStart
        // Process drifting upward in the last four observations
        val observations = doubleArrayOf(
            0.1, 0.2, -0.3, 0.0, 1.4, 1.2, 2.4, 2.6, 3.5, 2.2
        )
        val violations = westernElectricRules(observations, center = 0.0, sigma = 1.0)

        violations.rule1 // indices of points beyond ±3σ
        violations.rule2 // indices where 2 of last 3 points are beyond ±2σ (same side)
        violations.rule3 // indices where 4 of last 5 points are beyond ±1σ (same side)
        violations.rule4 // indices where 8 consecutive points fall on the same side
        // SampleEnd
        assertTrue(violations.rule1.isNotEmpty() || violations.rule2.isNotEmpty() ||
            violations.rule3.isNotEmpty() || violations.rule4.isNotEmpty())
    }

    @Test
    fun edaDistributionShape() {
        val responseTimeMs = doubleArrayOf(
            89.2, 95.1, 87.6, 102.3, 91.8, 88.4, 96.7, 103.5, 90.1, 94.3,
            88.9, 97.2, 105.8, 91.4, 93.6, 87.1, 99.0, 92.5, 96.1, 104.2,
            90.7, 88.3, 101.6, 93.9, 95.4, 89.8, 98.3, 106.1, 91.0, 94.7
        )
        val errorsPerHour = doubleArrayOf(
            2.0, 3.0, 1.0, 5.0, 2.0, 1.0, 4.0, 6.0, 2.0, 3.0,
            1.0, 4.0, 7.0, 2.0, 3.0, 1.0, 5.0, 2.0, 4.0, 6.0,
            2.0, 1.0, 5.0, 3.0, 3.0, 1.0, 4.0, 8.0, 2.0, 3.0
        )
        val memoryUsageMb = doubleArrayOf(
            512.3, 528.1, 505.7, 545.2, 519.6, 508.4, 534.8, 551.3, 515.0, 526.7,
            509.2, 537.1, 558.4, 517.8, 524.3, 503.1, 541.6, 520.9, 531.5, 549.7,
            514.2, 506.8, 543.9, 522.5, 529.0, 511.4, 539.3, 561.2, 516.3, 527.4
        )
        val throughputRps = doubleArrayOf(
            245.0, 238.0, 251.0, 225.0, 242.0, 249.0, 232.0, 218.0, 244.0, 236.0,
            250.0, 230.0, 212.0, 243.0, 237.0, 253.0, 227.0, 241.0, 233.0, 220.0,
            246.0, 252.0, 224.0, 239.0, 235.0, 248.0, 228.0, 210.0, 243.0, 234.0
        )
        // SampleStart
        responseTimeMs.skewness() // positive = right-skewed
        responseTimeMs.kurtosis() // positive excess = heavier tails than Normal

        errorsPerHour.skewness()
        errorsPerHour.kurtosis()

        memoryUsageMb.skewness()
        throughputRps.skewness()
        // SampleEnd
    }
}
