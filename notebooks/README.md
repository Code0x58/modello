## Running a notebook

Use uv 0.13.1 or newer and Python 3.11 or newer. From the repository root:

```sh
uv sync --locked --group notebooks --python 3.14
cd notebooks
uv run --locked --group notebooks jupyter notebook
```

The root `uv.lock` records the notebook dependencies alongside the development
environment. The local `modello` checkout is installed automatically.

## Importing examples

```python
import os, sys

sys.path.append(os.path.realpath("../examples"))
import jobs, streaming, geometry
```
