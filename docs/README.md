# Databricks AI Bridge Documentation

We generate our API docs with Sphinx, and they get published to [this directory](https://github.com/databricks-eng/docs-api-ref/tree/main/content-publish/python).

## Setup
Requirements:
- Follow the steps in ../CONTRIBUTING.md to set up the development environment.

## Develop the docs locally
Once you have activated the conda environment, navigate to this directory and run:

```sh
make livehtml
```

## Build for production
To build for production, run:

```sh
make html
```

This will output a set of static files in build/.

To check the build, you can use a python http server:

```sh
python3 -m http.server --directory build/html
```
