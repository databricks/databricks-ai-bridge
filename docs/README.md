# Databricks AI Bridge Documentation

We generate our API docs with Sphinx, and they get published to the [Databricks AI Bridge docs](https://docs.databricks.com/en/ai/databricks-ai-bridge/index.html).

## Setup

Install `uv`: https://docs.astral.sh/uv/getting-started/installation/

## Develop the docs locally

Navigate to the `docs` directory and run the make command to start a local server:

```sh
cd ~/databricks-ai-bridge/docs
uv run --group doc make livehtml
```

## Build for production

To build for production, run:

```sh
uv run --group doc make html
```

This will output a set of static files in build/.

To check the build, you can use a python http server:

```sh
uv run --group doc python3 -m http.server --directory build/html
```
