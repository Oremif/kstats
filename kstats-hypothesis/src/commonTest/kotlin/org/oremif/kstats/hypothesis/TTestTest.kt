package org.oremif.kstats.hypothesis

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class TTestTest {

    @Test
    fun testOneSampleTTest() {
        // Data: sample mean close to 0 with large variance -> not significant
        val sample = doubleArrayOf(1.0, -1.0, 2.0, -2.0, 0.5, -0.5)
        val result = tTest(sample, mu = 0.0)
        assertFalse(result.isSignificant())
        assertEquals("One-Sample t-Test", result.testName)
    }

    @Test
    fun testOneSampleTTestSignificant() {
        // Data: clearly above 0
        val sample = doubleArrayOf(5.0, 6.0, 7.0, 5.5, 6.5)
        val result = tTest(sample, mu = 0.0)
        assertTrue(result.isSignificant())
    }

    @Test
    fun testTwoSampleWelch() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = tTest(s1, s2)
        assertTrue(result.isSignificant(), "Clearly different means should be significant")
        assertTrue(result.testName.contains("Welch"))
    }

    @Test
    fun testTwoSampleEqualVariances() {
        val s1 = doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0)
        val s2 = doubleArrayOf(6.0, 7.0, 8.0, 9.0, 10.0)
        val result = tTest(s1, s2, equalVariances = true)
        assertTrue(result.isSignificant())
        assertTrue(result.testName.contains("Equal Variances"))
    }

    @Test
    fun testPairedTTest() {
        val before = doubleArrayOf(200.0, 190.0, 210.0, 180.0, 195.0)
        val after = doubleArrayOf(190.0, 180.0, 195.0, 170.0, 185.0)
        val result = pairedTTest(before, after)
        assertEquals("Paired t-Test", result.testName)
        assertTrue(result.statistic > 0) // before > after
    }

    @Test
    fun testConfidenceInterval() {
        val sample = doubleArrayOf(10.0, 12.0, 11.0, 13.0, 14.0)
        val result = tTest(sample, mu = 0.0, confidenceLevel = 0.95)
        val ci = result.confidenceInterval!!
        val mean = sample.average()
        assertTrue(ci.first < mean && mean < ci.second)
    }
}
