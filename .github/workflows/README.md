# Continous Integration Workflows

This package implements different workflows for CI, based on our organisation's common workflows.
They are organised as follows.

## Documentation

The `documentation` workflow triggers on any push to master, builds the documentation and pushes it to the `gh-pages` branch (if the build is successful).

## Testing Suite

Tests are ensured in the `tests` workflow, which triggers on all pushes.
It runs on a matrix of all supported operating systems for all supported Python versions.

## Test Coverage

Test coverage is calculated in the `coverage` wokflow, which triggers on pushes to `master` and any push to a `pull request`.
It reports the coverage results of the test suite to `CodeClimate`.

## Regular Testing

A `cron` workflow triggers every Monday at 3am (UTC time) and runs the full testing suite, on all available operating systems and supported Python versions.
It also runs on `Python 3.x` so that newly released Python versions that would break tests are automatically included.

## Publishing

Publishing to `PyPI` is done through the `publish` workflow, which triggers anytime a `release` is made of the GitHub repository.
It builds a `wheel`, checks it, and pushes to `PyPI` if checks are successful.
