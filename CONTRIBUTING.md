# Contributor's Guide

## Setting up dev environment

Create a conda environment and install dev requirements

```sh
conda create --name databricks-ai-dev-env python=3.10
conda activate databricks-ai-dev-env
pip install -e ".[dev]"
pip install -r requirements/lint-requirements.txt
```

If you are working with integration packages install them as well

```sh
pip install -e "integrations/langchain[dev]"
```

## Running tests

To run tests for the bridge library, use the following command:

```sh
pytest tests 
```

To run tests for the langchain integration library, use the following command:

```sh
cd integrations/langchain
pytest tests 
```

## Linting and formatting

For formatting code run:

```sh
ruff format
```

For checking linting rules run (required for CI checks):

```sh
ruff check .
```

For fixing the linting issues such as import order, etc run:

```sh
ruff check . --fix
```
