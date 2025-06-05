# Algorithmic Trading Dashboard

This project provides a Streamlit dashboard for collecting market data, generating features, backtesting strategies and running live trading.

## Symbol Format

Currency pairs should be specified with a slash separator, e.g. `EUR/USD`.

## Running Tests

Use the `run_tests.sh` script to install the minimal dependencies required for
the test suite. The script upgrades `pip`, installs packages listed in
`requirements-test.txt`, and then launches `pytest`.

```bash
./run_tests.sh
```
