"""Mounted explorers share navigation while keeping proposals independent."""

from xml.etree import ElementTree

import pytest
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.test import Client
from werkzeug.wrappers import Response

from examples.gallery.app import create_gallery
from examples.gallery.catalog import EXAMPLES


@pytest.fixture(scope="module")
def gallery_client():
    return create_gallery().test_client()


@pytest.mark.parametrize("prefix", ["", "/work/gallery"], ids=["root", "mounted"])
def test_one_gallery_serves_every_explorer_and_its_local_assets(gallery_client, prefix):
    client = (
        Client(DispatcherMiddleware(Response(status=404), {prefix: gallery_client.application}), Response)
        if prefix
        else gallery_client
    )
    home = client.get(prefix + "/")
    assert home.status_code == 200
    for slug in EXAMPLES:
        assert f'href="{prefix}/{slug}/"'.encode() in home.data
        page = client.get(f"{prefix}/{slug}/")
        assert page.status_code == 200
        assert f'data-explore-url="{prefix}/{slug}/api/explore"'.encode() in page.data
        assert f'src="{prefix}/{slug}/assets/explorer.js"'.encode() in page.data
        assert b'aria-label="Example gallery"' in page.data
    assert client.get(prefix + "/assets/gallery.css").status_code == 200
    assert client.get(prefix + "/vectors/assets/fonts/STIXTwoMath-Regular.woff2").data.startswith(b"wOF2")
    assert client.get(prefix + "/missing/").status_code == 404


@pytest.mark.parametrize(
    "slug, index",
    [
        pytest.param(slug, index, id=f"{slug}:{scenario.title if scenario is not None else 'initial'}")
        for slug, example in EXAMPLES.items()
        for index, scenario in [(-1, None), *enumerate(example.info.scenarios)]
    ],
)
def test_gallery_questions_solve_and_preserve_mathematical_markup(gallery_client, slug, index):
    initial = gallery_client.post(f"/{slug}/api/explore", json={"bindings": {}})
    assert initial.status_code == 200
    bindings = {} if index == -1 else initial.json["scenarios"][index]["bindings"]
    response = gallery_client.post(f"/{slug}/api/explore", json={"bindings": bindings})
    assert response.status_code == 200, response.json
    assert response.json["fields"]
    for equation in response.json["equations"]:
        ElementTree.fromstring(equation["mathml"])


def test_edits_are_local_to_the_example_and_each_browser_proposal(gallery_client):
    relative = "/relative/api/explore"
    changed = gallery_client.post(relative, json={"bindings": {"levels": "[1, 2, 3]"}})
    assert changed.status_code == 200
    assert next(field["value"] for field in changed.json["fields"] if field["path"] == "N") == "2"
    other = gallery_client.post("/vectors/api/explore", json={"bindings": {"total": "6"}})
    assert other.status_code == 200
    restored = gallery_client.post(relative, json={"bindings": {}})
    assert restored.status_code == 200
    assert next(field["value"] for field in restored.json["fields"] if field["path"] == "N") == "4"
    assert restored.json["bindings"] == {}
