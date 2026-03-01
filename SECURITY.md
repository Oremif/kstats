# Security Policy

## Scope

kstats is a pure computational library with no network access, no file I/O,
and no server components. The security surface is limited to:

- Numerical correctness (overflow, underflow, edge-case inputs)
- The `SecureRandom` implementation in `kstats-core` (delegates to platform primitives)
- Algorithmic complexity (inputs that cause excessive computation)

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.3.x   | :white_check_mark: |
| < 0.3   | :x:                |

Security fixes are applied to the latest release only.

## Reporting a Vulnerability

If you discover a security issue, please report it **privately** via
[GitHub Security Advisories](https://github.com/oremif/kstats/security/advisories/new)
rather than opening a public issue.

You can expect an initial response within 7 days. Since kstats is maintained by a solo
developer, please allow reasonable time for investigation and patching.

## Disclosure Policy

Once a fix is released, the advisory will be published with full details.
