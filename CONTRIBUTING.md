### Setting up a development environment

The `orb_models` package uses [Poetry](https://python-poetry.org/) for dependency management. To install the package and its dependencies, run the following command:

```bash
pip install poetry  # Install Poetry if you don't have it
poetry install
```

Optionally, also install [cuML](https://docs.rapids.ai/install/) (requires CUDA):
```bash
pip install "cuml-cu11==25.2.*"  # For cuda versions >=11.4, <11.8
pip install "cuml-cu12==25.2.*"  # For cuda versions >=12.0, <13.0
```

### Running tests

The `orb_models` package uses `pytest` for testing. To run the tests, navigate to the root directory of the package and run the following command:

```bash
pytest
```

### Publishing

The `orb_models` package is published using [trusted publishers](https://docs.pypi.org/trusted-publishers/). Whenever a new release is created on GitHub, the package is automatically published to PyPI using GitHub Actions.