# Contributor's Guide

## Setting up dev environment

Create a conda environment and install dev requirements

```sh
conda create --name databricks-ai-dev-env python=3.10
conda activate databricks-ai-dev-env
pip install -e ".[dev]"
pip install -r requirements/lint-requirements.txt
pip install -r requirements/dev-requirements.txt
```

If you are working with integration packages install them as well

```sh
pip install -e "integrations/langchain[dev]"
```

### Build API docs

See the documentation in docs/README.md for how to build docs. When releasing a new wheel, please send a pull request to change the API reference published in [docs-api-ref](https://github.com/databricks-eng/docs-api-ref/tree/main/content-publish/python/databricks-agents).
