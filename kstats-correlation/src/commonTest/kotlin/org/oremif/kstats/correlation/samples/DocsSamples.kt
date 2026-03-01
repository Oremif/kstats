package org.oremif.kstats.correlation.samples

import org.oremif.kstats.correlation.*
import org.oremif.kstats.descriptive.describe
import kotlin.test.Test
import kotlin.test.assertEquals

class DocsSamples {

    companion object {
        // AB-testing treatment data
        val treatmentDurationSec = doubleArrayOf(
            29.1, 33.8, 31.5, 35.2, 28.7, 32.4, 30.9, 34.6, 29.8, 33.1,
            31.2, 27.5, 34.0, 30.3, 36.1, 31.8, 33.5, 28.9, 32.7, 35.8
        )
        val treatmentSteps = doubleArrayOf(
            5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0,
            4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 5.0, 4.0, 5.0, 5.0
        )

        // EDA metric arrays
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

    // =====================================================================
    // correlation/overview.mdx
    // =====================================================================

    @Test
    fun corrPearson() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        val r = pearsonCorrelation(x, y)
        r.coefficient            // 0.9987
        r.pValue                 // 0.0001
        r.n                      // 5
        // SampleEnd
        assertEquals(0.9987, r.coefficient, 1e-4)
        assertEquals(0.0001, r.pValue, 1e-4)
        assertEquals(5, r.n)
    }

    @Test
    fun corrSpearman() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        val y = doubleArrayOf(2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0)

        val r = spearmanCorrelation(x, y)
        r.coefficient            // 1.0 — perfect monotonic relationship
        r.pValue                 // 0.0
        // SampleEnd
        assertEquals(1.0, r.coefficient, 1e-4)
        assertEquals(0.0, r.pValue, 1e-4)
    }

    @Test
    fun corrKendall() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(1.0, 3.0, 2.0, 5.0, 4.0)

        val tau = kendallTau(x, y)
        tau.coefficient          // 0.6
        tau.pValue               // p-value for tau
        // SampleEnd
        assertEquals(0.6, tau.coefficient, 1e-4)
    }

    @Test
    fun corrPointBiserial() {
        // SampleStart
        val binary     = intArrayOf(0, 0, 0, 1, 1, 1, 1)
        val continuous = doubleArrayOf(1.0, 2.0, 1.5, 4.0, 5.0, 4.5, 3.5)

        val r = pointBiserialCorrelation(binary, continuous)
        r.coefficient            // positive — group 1 has higher values
        r.pValue                 // p-value
        // SampleEnd
    }

    @Test
    fun corrPartial() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.0, 4.0, 5.0, 4.0, 5.0)
        val z = doubleArrayOf(1.0, 1.0, 2.0, 3.0, 3.0)

        val r = partialCorrelation(x, y, z)
        r.coefficient            // correlation between x and y, controlling for z
        r.pValue                 // p-value
        // SampleEnd
    }

    @Test
    fun corrMatrices() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)
        val z = doubleArrayOf(5.0, 4.0, 3.0, 2.0, 1.0)

        val corr = correlationMatrix(x, y, z)
        corr[0][1]               // Pearson r between x and y ≈ 0.9987
        corr[0][2]               // Pearson r between x and z = -1.0

        val cov = covarianceMatrix(x, y, z)
        cov[0][0]                // variance of x = 2.5
        cov[0][1]                // covariance of x and y
        // SampleEnd
        assertEquals(0.9987, corr[0][1], 1e-4)
        assertEquals(-1.0, corr[0][2], 1e-4)
        assertEquals(2.5, cov[0][0], 1e-4)
    }

    @Test
    fun corrRegression() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        val model = simpleLinearRegression(x, y)
        model.slope                  // 1.99
        model.intercept              // 0.06
        model.rSquared               // 0.9973
        model.standardErrorSlope     // standard error of the slope estimate
        model.standardErrorIntercept // standard error of the intercept estimate
        model.n                      // 5
        model.residuals              // [0.05, -0.07, 0.15, -0.17, 0.05]

        // Prediction
        model.predict(6.0)           // 11.99
        model.predict(doubleArrayOf(6.0, 7.0, 8.0)) // batch prediction
        // SampleEnd
        assertEquals(1.99, model.slope, 0.01)
        assertEquals(0.06, model.intercept, 0.01)
        assertEquals(0.9973, model.rSquared, 1e-4)
        assertEquals(5, model.n)
        assertEquals(11.99, model.predict(6.0), 0.01)
    }

    // =====================================================================
    // quickstart.mdx
    // =====================================================================

    @Test
    fun quickstartAssociation() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        val r = pearsonCorrelation(x, y)
        r.coefficient            // 0.9987
        r.pValue                 // 0.0001

        val model = simpleLinearRegression(x, y)
        model.slope              // 1.99
        model.rSquared           // 0.9973
        model.predict(6.0)       // 11.99
        // SampleEnd
        assertEquals(0.9987, r.coefficient, 1e-4)
        assertEquals(0.0001, r.pValue, 1e-4)
        assertEquals(1.99, model.slope, 0.01)
        assertEquals(0.9973, model.rSquared, 1e-4)
        assertEquals(11.99, model.predict(6.0), 0.01)
    }

    @Test
    fun quickstartTabCorrelation() {
        // SampleStart
        val x = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val y = doubleArrayOf(2.1, 3.9, 6.2, 7.8, 10.1)

        pearsonCorrelation(x, y).coefficient  // 0.9987
        simpleLinearRegression(x, y).slope    // 1.99
        // SampleEnd
        assertEquals(0.9987, pearsonCorrelation(x, y).coefficient, 1e-4)
        assertEquals(1.99, simpleLinearRegression(x, y).slope, 0.01)
    }

    // =====================================================================
    // ab-testing.mdx
    // =====================================================================

    @Test
    fun abTestingCorrelation() {
        val treatmentDurationSec = treatmentDurationSec
        val treatmentSteps = treatmentSteps
        // SampleStart
        // Within the treatment group: do faster sessions correlate with more completed steps?
        val correlation = spearmanCorrelation(treatmentDurationSec, treatmentSteps)

        correlation.coefficient // negative means shorter sessions correlate with more steps
        correlation.pValue
        // SampleEnd
    }

    // =====================================================================
    // quality-control.mdx
    // =====================================================================

    @Test
    fun qcMultiParameter() {
        // SampleStart
        val temperatureSensor = doubleArrayOf(
            155.2, 154.8, 156.1, 155.5, 154.3, 155.9, 155.0, 156.3, 154.7, 155.4
        )
        val pressureSensor = doubleArrayOf(
            2.12, 2.08, 2.15, 2.11, 2.05, 2.14, 2.09, 2.16, 2.07, 2.10
        )
        val flowRateSensor = doubleArrayOf(
            45.1, 44.8, 45.5, 45.2, 44.5, 45.4, 44.9, 45.6, 44.7, 45.3
        )

        temperatureSensor.describe()
        pressureSensor.describe()
        flowRateSensor.describe()

        // Correlation matrix — identifies linked parameters
        val matrix = correlationMatrix(temperatureSensor, pressureSensor, flowRateSensor)
        // matrix[0][1] = temperature-pressure correlation
        // matrix[0][2] = temperature-flow correlation
        // matrix[1][2] = pressure-flow correlation
        // SampleEnd
    }

    // =====================================================================
    // exploratory-analysis.mdx
    // =====================================================================

    @Test
    fun edaCorrelations() {
        val responseTimeMs = responseTimeMs
        val errorsPerHour = errorsPerHour
        val memoryUsageMb = memoryUsageMb
        val throughputRps = throughputRps
        // SampleStart
        // Correlation matrix across all four metrics
        val matrix = correlationMatrix(responseTimeMs, errorsPerHour, memoryUsageMb, throughputRps)
        // matrix[i][j] gives Pearson r between metrics i and j

        // Deeper look at specific relationships
        val rtVsErrors = pearsonCorrelation(responseTimeMs, errorsPerHour)
        rtVsErrors.coefficient // positive = errors rise with latency
        rtVsErrors.pValue

        val rtVsThroughput = spearmanCorrelation(responseTimeMs, throughputRps)
        rtVsThroughput.coefficient // negative = latency rises when throughput drops
        rtVsThroughput.pValue
        // SampleEnd
    }

    @Test
    fun edaRegression() {
        val errorsPerHour = errorsPerHour
        val responseTimeMs = responseTimeMs
        // SampleStart
        // Model: how does error count relate to response time?
        val regression = simpleLinearRegression(errorsPerHour, responseTimeMs)

        regression.slope     // ms increase per additional error/hour
        regression.intercept // baseline response time at zero errors
        regression.rSquared  // proportion of variance explained

        regression.predict(4.0) // expected latency at 4 errors/hour
        // SampleEnd
    }
}
