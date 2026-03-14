package org.oremif.kstats.distributions

internal fun trapezoidalIntegral(pdf: (Double) -> Double, a: Double, b: Double, n: Int = 100_000): Double {
    val h = (b - a) / n
    var sum = 0.5 * (pdf(a) + pdf(b))
    for (i in 1 until n) {
        sum += pdf(a + i * h)
    }
    return sum * h
}
