package org.oremif.kstats.sampling.samples

import org.oremif.kstats.sampling.*
import org.oremif.kstats.descriptive.Frequency
import org.oremif.kstats.descriptive.toFrequency
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class DocsSamples {

    // ── sampling/overview.mdx ────────────────────────────────────────────

    @Test
    fun sampRanking() {
        // SampleStart
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)

        data.rank()                        // [3.0, 1.5, 4.0, 1.5, 5.0]
        data.rank(TieMethod.MIN)           // [3.0, 1.0, 4.0, 1.0, 5.0]
        data.rank(TieMethod.MAX)           // [3.0, 2.0, 4.0, 2.0, 5.0]
        data.rank(TieMethod.DENSE)         // [2.0, 1.0, 3.0, 1.0, 4.0]
        data.rank(TieMethod.ORDINAL)       // [3.0, 1.0, 4.0, 2.0, 5.0]

        data.percentileRank()              // ranks scaled to 0-100
        // SampleEnd
        assertContentEquals(doubleArrayOf(3.0, 1.5, 4.0, 1.5, 5.0), data.rank())
        assertContentEquals(doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0), data.rank(TieMethod.MIN))
        assertContentEquals(doubleArrayOf(3.0, 2.0, 4.0, 2.0, 5.0), data.rank(TieMethod.MAX))
        assertContentEquals(doubleArrayOf(2.0, 1.0, 3.0, 1.0, 4.0), data.rank(TieMethod.DENSE))
        assertContentEquals(doubleArrayOf(3.0, 1.0, 4.0, 2.0, 5.0), data.rank(TieMethod.ORDINAL))
    }

    @Test
    fun sampNormalization() {
        // SampleStart
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)

        data.zScore()                      // [-1.2649, -0.6325, 0.0, 0.6325, 1.2649]

        data.minMaxNormalize()             // [0.0, 0.25, 0.5, 0.75, 1.0]
        data.minMaxNormalize(0.0, 100.0)   // [0.0, 25.0, 50.0, 75.0, 100.0]
        // SampleEnd
        val zScores = data.zScore()
        assertEquals(-1.2649, zScores[0], 1e-4)
        assertEquals(-0.6325, zScores[1], 1e-4)
        assertEquals(0.0, zScores[2], 1e-4)
        assertEquals(0.6325, zScores[3], 1e-4)
        assertEquals(1.2649, zScores[4], 1e-4)
        assertContentEquals(doubleArrayOf(0.0, 0.25, 0.5, 0.75, 1.0), data.minMaxNormalize())
        assertContentEquals(doubleArrayOf(0.0, 25.0, 50.0, 75.0, 100.0), data.minMaxNormalize(0.0, 100.0))
    }

    @Test
    fun sampBinning() {
        // SampleStart
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

        // By bin count
        val bins = data.asIterable().bin(3)
        bins.size                          // 3
        bins[0].range                      // interval of the first bin
        bins[0].items                      // values that fall in the first bin
        bins[0].count                      // number of values

        // By bin width
        val wideBins = data.asIterable().bin(5.0)

        // Frequency table -- counts and proportions instead of items
        val freq = data.asIterable().frequencyTable(3)
        freq[0].count                      // number of values in the first bin
        freq[0].relativeFrequency          // proportion of total
        freq[0].cumulativeFrequency        // running total of relative frequencies
        // SampleEnd
        assertEquals(3, bins.size)
    }

    @Test
    fun sampRandomBootstrap() {
        // SampleStart
        val items = listOf("A", "B", "C", "D", "E")

        items.randomSample(3, Random(42))     // 3 distinct items
        items.bootstrapSample(6, Random(42))  // 6 items, may have repeats
        // SampleEnd
    }

    @Test
    fun sampWeightedRandom() {
        // SampleStart
        val coin = WeightedCoin(probability = 0.7)
        coin.flip()                        // true with 70% probability

        val dice = WeightedDice(mapOf("A" to 3.0, "B" to 1.0))
        dice.roll()                        // "A" with 75% probability, "B" with 25%
        // SampleEnd
    }

    // ── core/overview.mdx ────────────────────────────────────────────────

    @Test
    fun coreFrequencyTable() {
        // SampleStart
        val data = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        val bins = data.asIterable().frequencyTable(3)
        // Each FrequencyBin has: range, count, relativeFrequency, cumulativeFrequency
        bins[0].count              // number of values in the first bin
        bins[0].relativeFrequency  // proportion of total
        // SampleEnd
    }

    // ── quickstart.mdx ──────────────────────────────────────────────────

    @Test
    fun quickstartTabSampling() {
        // SampleStart
        val data = doubleArrayOf(3.0, 1.0, 4.0, 1.0, 5.0)
        data.rank()               // [3.0, 1.5, 4.0, 1.5, 5.0]
        data.zScore()             // [-0.16, -1.47, 0.49, -1.47, 1.14]
        data.minMaxNormalize()    // [0.5, 0.0, 0.75, 0.0, 1.0]
        // SampleEnd
        assertContentEquals(doubleArrayOf(3.0, 1.5, 4.0, 1.5, 5.0), data.rank())
        val zScores = data.zScore()
        assertEquals(0.1118, zScores[0], 1e-3)
        assertEquals(-1.0062, zScores[1], 1e-3)
        assertEquals(0.6708, zScores[2], 1e-3)
        assertEquals(-1.0062, zScores[3], 1e-3)
        assertEquals(1.2298, zScores[4], 1e-3)
        assertContentEquals(doubleArrayOf(0.5, 0.0, 0.75, 0.0, 1.0), data.minMaxNormalize())
    }

    // ── quality-control.mdx ─────────────────────────────────────────────

    @Test
    fun qcZScore() {
        val sensorReadings = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4,
            155.8, 154.5, 156.0, 155.3, 154.9, 155.7, 155.1, 156.2, 154.6, 155.6,
            155.3, 154.4, 155.8, 162.5, 155.1, 155.7, 154.2, 155.5, 155.9, 141.3
        )
        // SampleStart
        val zScores = sensorReadings.zScore()

        val anomalyIndices = zScores.indices.filter { kotlin.math.abs(zScores[it]) > 3.0 }
        val anomalies = anomalyIndices.map { sensorReadings[it] }
        // SampleEnd
    }

    // ── exploratory-analysis.mdx ────────────────────────────────────────

    companion object {
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
    }

    @Test
    fun edaFrequencyDistribution() {
        val responseTimeMs = DocsSamples.responseTimeMs
        val errorsPerHour = DocsSamples.errorsPerHour
        // SampleStart
        val rtBins = responseTimeMs.frequencyTable(binCount = 5)
        rtBins.forEach { bin ->
            // bin.range, bin.count, bin.relativeFrequency
        }

        val errBins = errorsPerHour.frequencyTable(binSize = 2.0)
        errBins.forEach { bin ->
            // bin.range, bin.count, bin.cumulativeFrequency
        }
        // SampleEnd
    }

    @Test
    fun edaNormalizeRank() {
        val responseTimeMs = DocsSamples.responseTimeMs
        val memoryUsageMb = DocsSamples.memoryUsageMb
        val throughputRps = DocsSamples.throughputRps
        // SampleStart
        // Z-score: values become standard deviations from mean
        val rtNormalized = responseTimeMs.zScore()
        val memNormalized = memoryUsageMb.zScore()
        // Both are now on the same scale and can be compared directly

        // Min-max scaling to [0, 1]
        val rtScaled = responseTimeMs.minMaxNormalize()
        val tpScaled = throughputRps.minMaxNormalize()

        // Rank the days by response time (worst days get highest rank)
        val rtRanked = responseTimeMs.rank()
        // SampleEnd
    }
}
