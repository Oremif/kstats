# Contributing to kstats

Thank you for your interest in contributing to kstats!

## Submitting Issues

Before creating an issue, please search [existing issues](https://github.com/oremif/kstats/issues)
to avoid duplicates. Use thumbs-up reactions on existing issues to show interest
instead of posting "+1" comments.

### Bug Reports

When reporting a bug, include:

- kstats version (please test against the [latest release](https://github.com/oremif/kstats/releases))
- Kotlin version and target platform (JVM, JS, Native/iOS, Native/Linux, Wasm, etc.)
- Minimal reproducing code snippet
- Expected vs actual behavior

### Feature Requests

- Explain your use case and domain — focus on the problem, not the solution.
- For new statistical methods, include a reference to the formula or algorithm
  (paper, textbook, or Wikipedia link).

## Submitting Pull Requests

### Before You Start

Discuss large changes or new public API in an issue before starting work.

### Code Changes

- Target the `master` branch.
- Follow [Kotlin Coding Conventions](https://kotlinlang.org/docs/coding-conventions.html).
- Bug fixes must include a test that fails without the fix.
- New public API declarations must have explicit visibility modifiers (`public`, `internal`, etc.) —
  the project has `explicitApi()` enabled.
- New public API must include KDoc documentation.
- After adding or changing public API, run `./gradlew apiDump` and commit the updated `.api` files.
- Do not add external dependencies to library modules — kstats is a **zero-dependency** library.

### New Statistical Methods

- Include a reference to the mathematical definition (paper, textbook, or Wikipedia).
- Include numerical tests validated against a reference implementation (R, SciPy, or Apache Commons Math).

## Building & Testing

JDK 21 is required (set via `jvmToolchain(21)`).

```bash
# Run all JVM tests (fastest feedback loop)
./gradlew jvmTest

# Run tests for a specific module
./gradlew :kstats-core:jvmTest
./gradlew :kstats-distributions:jvmTest
./gradlew :kstats-hypothesis:jvmTest
./gradlew :kstats-correlation:jvmTest
./gradlew :kstats-sampling:jvmTest

# Run all platform tests
./gradlew allTests

# Full build
./gradlew build

# Regenerate API dump files after public API changes
./gradlew apiDump

# Verify API compatibility
./gradlew apiCheck

# Generate API docs locally
./gradlew :dokkaGenerate

# Run benchmarks (JVM-only)
./gradlew :benchmark:benchmark
```

CI runs `jvmTest`, `linuxX64Test`, and `wasmJsNodeTest` on pull requests.

## Code Style

- `DoubleArray` is the primary data type for public API (not `List<Double>`).
- Use typed exceptions from `core/exceptions/Exceptions.kt` instead of `require()`.
- Use validation helpers from `core/Validation.kt`.
- Check `core/MathUtils.kt` before reimplementing special functions (gamma, beta, erf, etc.).

## Project Structure

| Module                 | Description                                          |
|------------------------|------------------------------------------------------|
| `kstats-core`          | Math foundations, descriptive statistics              |
| `kstats-distributions` | Continuous & discrete probability distributions      |
| `kstats-hypothesis`    | Statistical tests (t-test, ANOVA, chi-squared, etc.) |
| `kstats-correlation`   | Correlation coefficients & simple linear regression  |
| `kstats-sampling`      | Ranking, normalization, bootstrap, weighted sampling |
| `kstats-bom`           | Bill of Materials for version alignment              |

## Contact

Use [GitHub Issues](https://github.com/oremif/kstats/issues) for questions and discussions.
