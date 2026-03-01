package org.oremif.kstats.benchmark.util

import kotlin.random.Random

object DataGenerators {

    fun generateDoubleArray(size: Int, seed: Long = 42L): DoubleArray {
        val random = Random(seed)
        return DoubleArray(size) {
            // Box-Muller transform for normal distribution
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)
        }
    }

    fun generatePositiveArray(size: Int, seed: Long = 42L): DoubleArray {
        val random = Random(seed)
        return DoubleArray(size) {
            // Exponential distribution to guarantee positive values
            -kotlin.math.ln(random.nextDouble()) + 0.01
        }
    }

    fun generateCorrelatedPair(size: Int, seed: Long = 42L): Pair<DoubleArray, DoubleArray> {
        val random = Random(seed)
        val x = DoubleArray(size) {
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)
        }
        val noise = DoubleArray(size) {
            val u1 = random.nextDouble()
            val u2 = random.nextDouble()
            kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * Math.PI * u2)
        }
        // y = 0.8*x + 0.2*noise => correlation ~0.97
        val y = DoubleArray(size) { i -> 0.8 * x[i] + 0.2 * noise[i] }
        return x to y
    }
}
