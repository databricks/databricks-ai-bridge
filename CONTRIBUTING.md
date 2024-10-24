Setting up dev environment

Create a conda environement and install dev requirements

```
conda create --name databricks-ai-dev-env python=3.10
conda activate databricks-ai-dev-env
pip install -e ".[dev]"
pip install -r requirements/lint-requirements.txt
```
