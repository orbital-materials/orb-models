### Setting up a development environment

The `orb_models` repository uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the package and its dependencies, run the following command:

```bash
pipx install uv  # If you don't have uv, we recommend installing it into an isolated environment with pipx: https://docs.astral.sh/uv/getting-started/installation/#pypi
uv sync --group dev  # Install orb-models and development packages
```

### Running linters

The `orb_models` repository uses `ruff` for formatting and linting, and `mypy` for type checking. To run the linters, use the following commands:

```bash
ruff format .   # Format code
ruff check .    # Check for linting errors
mypy .          # Run type checking
```

### Running tests

The `orb_models` repository uses `pytest` for testing. To run the tests, navigate to the root directory of the package and run the following command:

```bash
pytest -n auto ./tests/
```

### Publishing

The `orb_models` package is published using [trusted publishers](https://docs.pypi.org/trusted-publishers/). Whenever a new release is created on GitHub, the package is automatically published to PyPI using GitHub Actions.
