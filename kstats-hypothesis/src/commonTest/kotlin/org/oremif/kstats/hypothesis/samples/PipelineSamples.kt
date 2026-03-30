package org.oremif.kstats.hypothesis.samples

import org.oremif.kstats.core.ConfidenceInterval
import org.oremif.kstats.descriptive.DescriptiveStatistics
import org.oremif.kstats.descriptive.describe
import org.oremif.kstats.hypothesis.leveneTest
import org.oremif.kstats.hypothesis.mannWhitneyUTest
import org.oremif.kstats.hypothesis.shapiroWilkTest
import org.oremif.kstats.hypothesis.tTest
import kotlin.test.Test

class PipelineSamples {

    // Private data classes that mirror the docs code
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

    // Private helper functions
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
    // building-a-pipeline.mdx — Block 1: Result Types
    // =====================================================================

    @Test
    fun pipelineResultTypes() {
        // SampleStart
        data class AssumptionCheck(
            val normalityPValue: Double,
            val isNormal: Boolean,
            val varianceEqualityPValue: Double,
            val isVarianceEqual: Boolean
        )

        data class GroupComparison(
            val testName: String,
            val pValue: Double,
            val isSignificant: Boolean,
            val confidenceInterval: ConfidenceInterval?
        )

        data class AnalysisReport(
            val controlSummary: DescriptiveStatistics,
            val treatmentSummary: DescriptiveStatistics,
            val assumptions: AssumptionCheck,
            val comparison: GroupComparison
        )
        // SampleEnd
    }

    // =====================================================================
    // building-a-pipeline.mdx — Block 2: Check Assumptions
    // =====================================================================

    @Test
    fun pipelineCheckAssumptions() {
        // SampleStart
        fun checkAssumptions(
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
        // SampleEnd
    }

    // =====================================================================
    // building-a-pipeline.mdx — Block 3: Compare Groups
    // =====================================================================

    @Test
    fun pipelineCompareGroups() {
        // SampleStart
        fun compareGroups(
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
        // SampleEnd
    }

    // =====================================================================
    // building-a-pipeline.mdx — Block 4: Full Report
    // =====================================================================

    @Test
    fun pipelineFullReport() {
        // SampleStart
        fun analyze(
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
        // SampleEnd
    }

    // =====================================================================
    // building-a-pipeline.mdx — Block 5: Usage
    // =====================================================================

    @Test
    fun pipelineUsage() {
        // SampleStart
        val pageLoadControl = doubleArrayOf(
            1.23, 1.45, 1.31, 1.52, 1.38, 1.41, 1.29, 1.47, 1.35, 1.44,
            1.33, 1.50, 1.27, 1.42, 1.36, 1.48, 1.30, 1.46, 1.39, 1.43
        )
        val pageLoadTreatment = doubleArrayOf(
            1.10, 1.25, 1.18, 1.32, 1.15, 1.22, 1.12, 1.28, 1.19, 1.26,
            1.14, 1.30, 1.11, 1.24, 1.17, 1.29, 1.13, 1.27, 1.20, 1.23
        )

        val report = analyze(pageLoadControl, pageLoadTreatment)

        report.controlSummary.mean
        report.treatmentSummary.mean
        report.assumptions.isNormal
        report.assumptions.isVarianceEqual
        report.comparison.testName
        report.comparison.pValue
        report.comparison.isSignificant
        report.comparison.confidenceInterval
        // SampleEnd
    }
}
