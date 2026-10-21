"""Keep the core-only README walkthrough executable as one progressive example."""

import re
from pathlib import Path


def test_readme_python_examples() -> None:
    readme = Path(__file__).parents[1] / "README.md"
    namespace: dict[str, object] = {}
    blocks = re.findall(r"^```python\n(.*?)^```", readme.read_text(), re.MULTILINE | re.DOTALL)
    assert blocks
    for source in blocks:
        exec(compile(source, str(readme), "exec"), namespace)
