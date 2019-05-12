## Running a notebook
```sh
pipenv install
pipenv run jupyter notebook --matplotlib=notebook
```
## Importing examples
```python
import os, sys
sys.path.append(os.path.realpath("../examples"))
import jobs, streaming, geometry
```