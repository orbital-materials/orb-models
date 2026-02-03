### Setting up a development environment

The `orb_models` repository uses [uv](https://docs.astral.sh/uv/) for dependency management. To install the package and its dependencies, run the following command:

```bash
pipx install uv  # If you don't have uv, we recommend installing it into an isolated environment with pipx: https://docs.astral.sh/uv/getting-started/installation/#pypi
uv sync --group dev  # Install orb-models and development packages
```

Optionally, also install [cuML](https://docs.rapids.ai/install/) (requires CUDA):
```bash
uv pip install "cuml-cu11==25.2.*"  # For cuda versions >=11.4, <11.8
uv pip install "cuml-cu12==25.2.*"  # For cuda versions >=12.0, <13.0
```

### Running tests

The `orb_models` repository uses `pytest` for testing. To run the tests, navigate to the root directory of the package and run the following command:

```bash
pytest ./tests/
```

### Publishing

The `orb_models` package is published using [trusted publishers](https://docs.pypi.org/trusted-publishers/). Whenever a new release is created on GitHub, the package is automatically published to PyPI using GitHub Actions.
