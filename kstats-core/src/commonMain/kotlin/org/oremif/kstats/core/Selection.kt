package org.oremif.kstats.core

/**
 * Rearranges elements in-place so that the k-th smallest value is at index [k].
 *
 * After this call, `this[k]` holds the k-th order statistic, all elements before index k are
 * less than or equal to `this[k]`, and all elements after are greater than or equal. Uses
 * introselect (quickselect with median-of-three pivot and heapsort fallback when recursion
 * depth exceeds 2 * floor(log2(n))). Expected O(n) time, worst-case O(n log n), O(1) extra
 * memory. NaN values sort last, consistent with [Double.compareTo] total order.
 */
internal fun DoubleArray.introSelect(k: Int) {
    val n = size
    if (n <= 1) return
    val depthLimit = 2 * floorLog2(n)
    introSelectRange(k, 0, n - 1, depthLimit)
}

private fun DoubleArray.introSelectRange(k: Int, lo: Int, hi: Int, depthLimit: Int) {
    if (lo >= hi) return

    if (depthLimit == 0) {
        heapSort(lo, hi)
        return
    }

    medianOfThree(lo, hi)
    val p = lomutoPartition(lo, hi)

    when {
        k == p -> return
        k < p -> introSelectRange(k, lo, p - 1, depthLimit - 1)
        else -> introSelectRange(k, p + 1, hi, depthLimit - 1)
    }
}

/**
 * Sorts `arr[lo]`, `arr[mid]`, `arr[hi]` and places the median at `arr[hi]` as pivot.
 */
private fun DoubleArray.medianOfThree(lo: Int, hi: Int) {
    val mid = lo + (hi - lo) / 2
    // Sort the three elements: lo <= mid <= hi (using compareTo for NaN safety)
    if (this[lo].compareTo(this[mid]) > 0) swap(lo, mid)
    if (this[lo].compareTo(this[hi]) > 0) swap(lo, hi)
    if (this[mid].compareTo(this[hi]) > 0) swap(mid, hi)
    // Place median at hi (pivot position for Lomuto)
    swap(mid, hi)
}

/**
 * Lomuto partition around pivot at arr[hi].
 * Returns the final index of the pivot.
 */
private fun DoubleArray.lomutoPartition(lo: Int, hi: Int): Int {
    val pivot = this[hi]
    var i = lo
    for (j in lo until hi) {
        if (this[j].compareTo(pivot) <= 0) {
            swap(i, j)
            i++
        }
    }
    swap(i, hi)
    return i
}

/**
 * In-place heapsort of subarray `[lo..hi]`.
 */
private fun DoubleArray.heapSort(lo: Int, hi: Int) {
    val n = hi - lo + 1
    // Build max-heap
    for (i in n / 2 - 1 downTo 0) {
        siftDown(lo, n, i)
    }
    // Extract elements
    for (i in n - 1 downTo 1) {
        swap(lo, lo + i)
        siftDown(lo, i, 0)
    }
}

/**
 * Max-heap sift down for subarray starting at [base].
 */
private fun DoubleArray.siftDown(base: Int, heapSize: Int, i: Int) {
    var parent = i
    while (true) {
        var largest = parent
        val left = 2 * parent + 1
        val right = 2 * parent + 2
        if (left < heapSize && this[base + left].compareTo(this[base + largest]) > 0) {
            largest = left
        }
        if (right < heapSize && this[base + right].compareTo(this[base + largest]) > 0) {
            largest = right
        }
        if (largest == parent) break
        swap(base + parent, base + largest)
        parent = largest
    }
}

private fun DoubleArray.swap(i: Int, j: Int) {
    val tmp = this[i]
    this[i] = this[j]
    this[j] = tmp
}

private fun floorLog2(n: Int): Int {
    var result = 0
    var v = n
    while (v > 1) {
        result++
        v = v shr 1
    }
    return result
}
