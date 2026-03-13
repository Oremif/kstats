package org.oremif.kstats.core.exceptions

public open class KStatsException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

public class InsufficientDataException(message: String) : KStatsException(message)

public class InvalidParameterException(message: String) : KStatsException(message)

public class ConvergenceException(
    message: String,
    public val iterations: Int,
    public val lastEstimate: Double,
) : KStatsException(message)

public class DegenerateDataException(message: String) : KStatsException(message)
