# Databricks AI Bridge library

The Databricks AI Bridge library provides a shared layer of APIs to interact with Databricks AI features, such as [Databricks AI/BI Genie ](https://www.databricks.com/product/ai-bi/genie) and [Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html).

## Integrations

This library also contains the source code for the following integration packages. These integration packages provide seamless integration of Databricks AI features to use in AI authoring frameworks.

* [`databricks-langchain`](./integrations/langchain/README.md)
* [`databricks-openai`](./integrations/openai/README.md)

## Installation

### Install from PyPI
```sh
pip install databricks-ai-bridge
```

### Install from source

With https:

```sh
pip install git+https://github.com/databricks/databricks-ai-bridge.git
```

With ssh creds:

```sh
pip install git+ssh://git@github.com/databricks/databricks-ai-bridge.git
```

