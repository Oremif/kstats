# Contributing to kstats

Thank you for your interest in contributing to kstats!

## Submitting Issues

Before creating an issue, please search [existing issues](https://github.com/oremif/kstats/issues) to avoid duplicates.

When reporting a bug, include:

- kstats version
- Kotlin version and target platform (JVM, iOS, Linux, etc.)
- Minimal reproducing code snippet
- Expected vs actual behavior

## Submitting Pull Requests

- Target the `master` branch.
- Follow [Kotlin Coding Conventions](https://kotlinlang.org/docs/coding-conventions.html).
- Bug fixes must include a test that fails without the fix.
- New public API declarations must have explicit visibility modifiers (`public`, `internal`, etc.) — the project has
  `explicitApi()` enabled.
- Discuss large changes in an issue before starting work.

## Building & Testing

```bash
# Run all JVM tests (fastest feedback loop)
./gradlew jvmTest

# Run tests for a specific module
./gradlew :kstats-core:jvmTest
./gradlew :kstats-distributions:jvmTest

# Run all platform tests
./gradlew allTests

# Full build
./gradlew build

# Run benchmarks (JVM-only)
./gradlew :benchmark:benchmark
```

## Project Structure

| Module                 | Description                                          |
|------------------------|------------------------------------------------------|
| `kstats-core`          | Math foundations, descriptive statistics             |
| `kstats-distributions` | Continuous & discrete probability distributions      |
| `kstats-hypothesis`    | Statistical tests (t-test, ANOVA, chi-squared, etc.) |
| `kstats-correlation`   | Correlation coefficients & simple linear regression  |
| `kstats-sampling`      | Ranking, normalization, bootstrap, weighted sampling |
| `kstats-bom`           | Bill of Materials for version alignment              |

## Contact

Use [GitHub Issues](https://github.com/oremif/kstats/issues) for questions and discussions.
