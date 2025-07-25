# Databricks DSPy Integration

The `databricks-dspy` package provides Databricks extensions for DSPy, with a purpose of facilitating the usage
of DSPy on Databricks platform.

## Installation

### From PyPI
```sh
pip install databricks-dspy
```

### From Source
```sh
pip install git+https://git@github.com/databricks/databricks-ai-bridge.git#subdirectory=integrations/dspy
```

## Key Features

- **LLMs Integration:** Use Databricks-hosted LLM endpoints.

## Getting Started

### Use LLMs on Databricks

```python
import databricks_dspy
import dspy

dspy.configure(lm=databricks_dspy.DatabricksLM(model="databricks/databricks-llama-4-maverick"))

predict = dspy.Predict("question->answer")

print(predict(question="why did a chicken cross the kitchen?"))
```


