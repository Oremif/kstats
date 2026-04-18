# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

kstats is a Kotlin Multiplatform statistics library published to Maven Central (`org.oremif:kstats`). All math is implemented from scratch in pure Kotlin common code — no platform-specific source sets exist for the library itself.

## Build commands

```bash
# Fastest feedback loop — single module JVM tests
./gradlew :kstats-core:jvmTest
./gradlew :kstats-distributions:jvmTest
./gradlew :kstats-hypothesis:jvmTest
./gradlew :kstats-correlation:jvmTest
./gradlew :kstats-sampling:jvmTest

# Single test class or method
./gradlew :kstats-core:jvmTest --tests "org.oremif.kstats.DescriptiveStatisticsTest"

# Specific non-JVM platforms
./gradlew :kstats-core:iosSimulatorArm64Test
./gradlew :kstats-core:jsNodeTest

# All platforms, all modules
./gradlew allTests

# API documentation (Dokka)
./gradlew :dokkaGenerate                # aggregated → build/dokka/html/
./gradlew :kstats-core:dokkaGenerate    # single module

# Benchmarks (JMH, JVM-only, kstats vs Apache Commons Math)
./gradlew :benchmark:benchmark          # full run
./gradlew :benchmark:smokeBenchmark     # quick smoke run

# Binary compatibility (kotlinx.binary-compatibility-validator)
./gradlew apiCheck                      # verify API hasn't changed unexpectedly
./gradlew apiDump                       # regenerate .api files after intentional changes

# Insert code snippets in documentation
./gradlew korro
```

## Architecture

Seven Gradle modules: five library modules, a BOM, and a benchmark module.

```
kstats-core
  ├── kstats-distributions
  │     ├── kstats-hypothesis
  │     └── kstats-correlation
  └── kstats-sampling
kstats-bom    (version alignment only)
benchmark/    (JMH comparisons, JVM-only)
```

Shared Gradle config lives in `build-logic/src/main/kotlin/` — the `kstats.kmp-library`, `kstats.maven-publish`, and `kstats.dokka` convention plugins.

## Key conventions

- **`explicitApi()`** is enabled — all public declarations must have explicit visibility modifiers.
- **No external dependencies in library modules** — math is implemented from scratch. Check `core/MathUtils.kt` before reimplementing special functions (gamma, beta, erf, digamma, etc.).
- **Typed exceptions** (`KStatsException` hierarchy in `core/exceptions/Exceptions.kt`) — use instead of `require()` / `check()` for all user-facing errors. Validation helpers in `core/Validation.kt`.

## Public API workflow

after any change to public API:

1. Add KDoc to every new public declaration (required).
2. Run `./gradlew apiDump`
3. Bug fixes must include a test that fails without the fix.

## Numerical precision

IMPORTANT: these rules apply to all distribution, test, and correlation implementations:

- **`sf()` must be computed directly, never as `1 - cdf(x)`** — avoids catastrophic cancellation in the upper tail.
- **Use `(1 - r) * (1 + r)` instead of `1 - r * r`** — same cancellation issue when r approaches +/-1.
- **Neumaier compensated summation** for mean/sum — not `array.sum()`.
- **`logPdf`/`logPmf` must use a direct formula**, not `ln(pdf(x))`.
- **NaN passes validation intentionally** — `requirePositive(NaN)` does not throw because `NaN <= 0.0` is `false` per IEEE 754. NaN propagates through computation.
- **Non-finite input in hypothesis tests** returns `TestResult(statistic=NaN, pValue=NaN)`, not an exception.

## Korro sample embedding

Code samples in documentation are extracted from test files via the Korro Gradle plugin — samples are always compiled and tested.

Sample files in `<module>/src/commonTest/.../samples/`:

| File | Embeds into |
|---|---|
| `DokkaSamples.kt` | `Module.md` (Dokka API docs) |
| `ReadmeSamples.kt` | `README.md` |
| `DocsSamples.kt` | `docs/**/*.mdx` (Mintlify) |
| `PipelineSamples.kt` | `docs/guides/how-to/*.mdx` |

All are `@Test` classes with `// SampleStart` / `// SampleEnd` markers. Embedding syntax in `.md`/`.mdx`:

```markdown
<!---IMPORT org.oremif.kstats.distributions.samples.DokkaSamples-->
<!---FUN dokkaDistributions-->
```

## Testing conventions

Golden values from scipy — comment format: `// scipy: stats.norm(0, 1).cdf(1.96)`. Tolerance tiers:

- `1e-10` to `1e-15` — scipy reference values (pdf, cdf, quantile, entropy)
- `1e-5` to `1e-10` — derived properties (moments, property-based checks)
- Adaptive for Monte Carlo: `maxOf(variance * 0.1, 0.05)`

Property-based tests (no external library): `exp(logPmf(k)) ~ pmf(k)`, `sf(x) + cdf(x) ~ 1`, CDF monotonicity, PMF sums to 1, CDF/quantile roundtrip.

Test structure for distributions: known values -> consistency properties -> moments/entropy -> degenerate cases -> invalid input -> NaN/Infinity -> property-based -> sample statistics.

New statistical methods must cite their mathematical reference and be validated against a reference implementation (R, SciPy, or Apache Commons Math).

## CI/CD

- **Build CI** runs `jvmTest linuxX64Test wasmJsNodeTest` only (not all 20+ targets).
- **Publish** uses macOS-latest (needed for Apple native), `--max-workers=2`.
- **Renovate** — patch updates auto-merge, Kotlin deps grouped into one PR.
- User documentation: Mintlify in `docs/`, config in `docs/docs.json` (English + German).
