# Contributor's Guide

## Setting up dev environment

Create a conda environement and install dev requirements

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

## Publishing to PyPI

Note: this section is for maitainers only.

We recommend first uploading to test-PyPI

### Publishing core package


```sh
python3 -m build --wheel
twine upload dist/*
```

### Publishing integration packages

```sh
cd integrations/langchain
python3 -m build --wheel
twine upload dist/*
```
