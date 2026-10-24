"""One progression of practical questions works across all four fundamentals."""

import re
from xml.etree import ElementTree

import pytest
from sympy import sympify
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.test import Client
from werkzeug.wrappers import Response

from examples.gallery.app import create_gallery
from examples.gallery.catalog import EXAMPLES, TUTORIAL

QUESTION_TITLES = (
    "Calculate results",
    "Infer missing inputs",
    "Try different inputs",
    "Leave inputs unknown",
    "Check the observations",
)
ANSWERS = {
    "simple": ({"c": 5}, {"a": 6, "c": 10}, {"c": 10}),
    "components": ({"base.c": 5, "diagonal": 13}, {"base.a": 3}, {"base.c": 10, "diagonal": 26}),
    "trend": (
        {"N": 4, "squared_error": 1},
        {"slope": "9/10", "intercept": "9/10", "squared_error": "7/10"},
        {"N": 6, "slope": 2, "intercept": 1, "squared_error": 0},
    ),
    "sequence": (
        {"N": 2, "terminal.quantity": 4, "terminal.cash": 908, "total_fees": 2},
        {"fills[1].price": 110, "terminal.cash": 908},
        {"N": 3, "terminal.quantity": 5, "terminal.cash": 802, "total_fees": 3},
    ),
}
UNKNOWN = {"simple": "a", "components": "base.a", "trend": "slope", "sequence": "fills[1].price"}


@pytest.fixture(scope="module")
def gallery_client():
    return create_gallery().test_client()


@pytest.mark.parametrize("prefix", ["", "/work/gallery"], ids=["root", "mounted"])
def test_one_gallery_serves_fundamentals_and_only_genuine_extensions(gallery_client, prefix):
    client = (
        Client(DispatcherMiddleware(Response(status=404), {prefix: gallery_client.application}), Response)
        if prefix
        else gallery_client
    )
    home = client.get(prefix + "/")
    assert home.status_code == 200
    assert re.findall(rb'class="gallery-card" href="([^"]+)"', home.data) == [
        f"{prefix}/{slug}/".encode() for slug in ("simple", "components", "trend", "sequence")
    ]
    assert f'href="{prefix}/extensions/"'.encode() in home.data
    extensions = client.get(prefix + "/extensions/")
    assert extensions.status_code == 200
    assert re.findall(rb'class="gallery-card" href="([^"]+)"', extensions.data) == [f"{prefix}/diffusion/".encode()]
    assert client.get(prefix + "/demos/").location == prefix + "/extensions/"
    for slug in EXAMPLES:
        page = client.get(f"{prefix}/{slug}/")
        assert page.status_code == 200
        assert f'data-explore-url="{prefix}/{slug}/api/explore"'.encode() in page.data
        assert f'src="{prefix}/{slug}/assets/explorer.js"'.encode() in page.data
        assert b'aria-label="Example gallery"' in page.data
        if slug in TUTORIAL:
            assert b"Step " in page.data
            assert b'class="gallery-next"' in page.data
    assert client.get(prefix + "/assets/gallery.css").status_code == 200
    assert client.get(prefix + "/simple/assets/fonts/STIXTwoMath-Regular.woff2").data.startswith(b"wOF2")
    for retired in ("vectors", "relative", "exposure", "missing"):
        assert client.get(f"{prefix}/{retired}/").status_code == 404


@pytest.mark.parametrize(
    "slug, index",
    [
        pytest.param(slug, index, id=f"{slug}:{scenario.title if scenario is not None else 'initial'}")
        for slug, example in EXAMPLES.items()
        for index, scenario in [(-1, None), *enumerate(example.info.scenarios)]
    ],
)
def test_gallery_questions_match_their_promised_results_and_behaviours(gallery_client, slug, index):
    initial = gallery_client.post(f"/{slug}/api/explore", json={"bindings": {}})
    assert initial.status_code == 200
    if slug in TUTORIAL:
        assert tuple(item["title"] for item in initial.json["scenarios"]) == QUESTION_TITLES
    bindings = {} if index == -1 else initial.json["scenarios"][index]["bindings"]
    response = gallery_client.post(f"/{slug}/api/explore", json={"bindings": bindings})
    if slug in TUTORIAL and index == 4:
        assert response.status_code == 422
        assert initial.json["scenarios"][index]["select"] in response.json["paths"]
        assert response.json["error"]
        return
    assert response.status_code == 200, response.json
    fields = {field["path"]: field for field in response.json["fields"]}
    if slug in TUTORIAL:
        if index == 3:
            target = initial.json["scenarios"][index]["select"]
            assert fields[UNKNOWN[slug]]["value_state"] == "free"
            assert fields[target]["value_state"] == "symbolic"
        else:
            for path, value in ANSWERS[slug][max(index, 0)].items():
                assert sympify(fields[path]["value"]) == sympify(value)
    for equation in response.json["equations"]:
        ElementTree.fromstring(equation["mathml"])


@pytest.mark.parametrize("slug", TUTORIAL)
def test_questions_restore_the_baseline_after_changes_unknowns_and_conflicts(gallery_client, slug):
    url = f"/{slug}/api/explore"
    data = gallery_client.post(url, json={"bindings": {}}).json
    for index in (2, 3, 4, 0):
        response = gallery_client.post(url, json={"bindings": data["scenarios"][index]["bindings"]})
        assert response.status_code == (422 if index == 4 else 200)
        if response.status_code == 200:
            data = response.json
    fields = {field["path"]: field for field in data["fields"]}
    for path, value in ANSWERS[slug][0].items():
        assert sympify(fields[path]["value"]) == sympify(value)
    assert data["bindings"] == {}


def test_browser_proposals_are_local_to_each_fundamental(gallery_client):
    changed = gallery_client.post(
        "/trend/api/explore", json={"bindings": {"times": "[0, 1, 2]", "observations": "[1, 2, 3]"}}
    )
    assert changed.status_code == 200
    assert next(field["value"] for field in changed.json["fields"] if field["path"] == "N") == "3"
    assert gallery_client.post("/simple/api/explore", json={"bindings": {"a": "6"}}).status_code == 200
    restored = gallery_client.post("/trend/api/explore", json={"bindings": {}})
    assert restored.status_code == 200
    assert next(field["value"] for field in restored.json["fields"] if field["path"] == "N") == "4"
    assert restored.json["bindings"] == {}
