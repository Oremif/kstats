package org.oremif.kstats.correlation.samples

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.correlation.pearsonCorrelation
import org.oremif.kstats.correlation.simpleLinearRegression
import org.oremif.kstats.descriptive.DescriptiveStatistics
import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.hypothesis.leveneTest
import org.oremif.kstats.hypothesis.mannWhitneyUTest
import org.oremif.kstats.hypothesis.shapiroWilkTest
import org.oremif.kstats.hypothesis.tTest
import kotlin.test.Test

class PipelineSamples {

    // Duplicated data classes from the hypothesis PipelineSamples (needed for block 6)
    private data class AssumptionCheck(
        val normalityPValue: Double,
        val isNormal: Boolean,
        val varianceEqualityPValue: Double,
        val isVarianceEqual: Boolean
    )

    private data class GroupComparison(
        val testName: String,
        val pValue: Double,
        val isSignificant: Boolean,
        val confidenceInterval: ConfidenceInterval?
    )

    private data class AnalysisReport(
        val controlSummary: DescriptiveStatistics,
        val treatmentSummary: DescriptiveStatistics,
        val assumptions: AssumptionCheck,
        val comparison: GroupComparison
    )

    // Duplicated helper functions from the hypothesis PipelineSamples (needed for block 6)
    private fun checkAssumptions(
        control: DoubleArray,
        treatment: DoubleArray,
        alpha: Double = 0.05
    ): AssumptionCheck {
        val controlNormality = shapiroWilkTest(control)
        val treatmentNormality = shapiroWilkTest(treatment)
        val normality = minOf(controlNormality.pValue, treatmentNormality.pValue)

        val variance = leveneTest(control, treatment)

        return AssumptionCheck(
            normalityPValue = normality,
            isNormal = normality >= alpha,
            varianceEqualityPValue = variance.pValue,
            isVarianceEqual = variance.pValue >= alpha
        )
    }

    private fun compareGroups(
        control: DoubleArray,
        treatment: DoubleArray,
        assumptions: AssumptionCheck,
        alpha: Double = 0.05
    ): GroupComparison {
        val result = if (assumptions.isNormal) {
            tTest(control, treatment, equalVariances = assumptions.isVarianceEqual)
        } else {
            mannWhitneyUTest(control, treatment)
        }

        return GroupComparison(
            testName = result.testName,
            pValue = result.pValue,
            isSignificant = result.isSignificant(alpha),
            confidenceInterval = result.confidenceInterval
        )
    }

    private fun analyze(
        control: DoubleArray,
        treatment: DoubleArray,
        alpha: Double = 0.05
    ): AnalysisReport {
        val assumptions = checkAssumptions(control, treatment, alpha)
        val comparison = compareGroups(control, treatment, assumptions, alpha)

        return AnalysisReport(
            controlSummary = control.describe(),
            treatmentSummary = treatment.describe(),
            assumptions = assumptions,
            comparison = comparison
        )
    }

    // =====================================================================
    // building-a-pipeline.mdx — Block 6: Extending the Pipeline
    // =====================================================================

    @Test
    fun pipelineExtended() {
        // SampleStart
        data class ExtendedReport(
            val base: AnalysisReport,
            val correlationCoefficient: Double,
            val correlationPValue: Double,
            val regressionSlope: Double,
            val regressionRSquared: Double
        )

        fun analyzeWithCorrelation(
            control: DoubleArray,
            treatment: DoubleArray,
            metricX: DoubleArray,
            metricY: DoubleArray
        ): ExtendedReport {
            val base = analyze(control, treatment)
            val correlation = pearsonCorrelation(metricX, metricY)
            val regression = simpleLinearRegression(metricX, metricY)

            return ExtendedReport(
                base = base,
                correlationCoefficient = correlation.coefficient,
                correlationPValue = correlation.pValue,
                regressionSlope = regression.slope,
                regressionRSquared = regression.rSquared
            )
        }
        // SampleEnd
    }
}
