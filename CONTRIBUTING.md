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
