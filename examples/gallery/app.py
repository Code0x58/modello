"""Mount the ordinary explorer once per example, with shared gallery navigation."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

from examples.gallery.catalog import DEMOS, EXAMPLES, TUTORIAL
from modello.gui import create_app

if TYPE_CHECKING:
    from _typeshed.wsgi import WSGIApplication
    from flask import Flask


def create_gallery() -> Flask:
    """Create one local app; each example retains its own equations and inputs."""
    from flask import Flask, render_template, request
    from werkzeug.middleware.dispatcher import DispatcherMiddleware

    app = Flask(__name__, template_folder="templates", static_folder="static", static_url_path="/assets")
    app.config["TRUSTED_HOSTS"] = ["localhost", "127.0.0.1", "[::1]"]

    @app.get("/")
    def index() -> str:
        return render_template("gallery.html", examples=TUTORIAL, tutorial=True)

    @app.get("/demos/")
    def demos() -> str:
        return render_template("gallery.html", examples=DEMOS, tutorial=False)

    def navigation() -> dict[str, Any]:
        prefix, _, selected = request.script_root.rpartition("/")
        steps = tuple(TUTORIAL)
        position = steps.index(selected) if selected in TUTORIAL else None
        next_page = None
        if position is not None:
            if position + 1 < len(steps):
                following = steps[position + 1]
                next_page = {"title": TUTORIAL[following].info.title, "url": f"{prefix}/{following}/"}
            else:
                next_page = {"title": "Demos and extensions", "url": prefix + "/demos/"}
        return {
            "gallery_home": prefix + "/",
            "gallery_css": prefix + "/assets/gallery.css",
            "gallery_links": [
                {"title": example.info.title, "url": f"{prefix}/{slug}/", "selected": slug == selected, "group": group}
                for group, examples in (("Features", TUTORIAL), ("Demos and extensions", DEMOS))
                for slug, example in examples.items()
            ],
            "gallery_step": f"Step {position + 1} of {len(steps)}" if position is not None else None,
            "gallery_next": next_page,
        }

    mounts: dict[str, WSGIApplication] = {}
    for slug, example in EXAMPLES.items():
        explorer = create_app(example.model, inputs=example.inputs)
        explorer.context_processor(navigation)
        mounts["/" + slug] = explorer.wsgi_app
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, mounts)  # type: ignore[method-assign]  # Flask's middleware hook.
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the Modello example gallery in one local app.")
    parser.add_argument("--port", type=int, default=5070)
    args = parser.parse_args()
    create_gallery().run(host="127.0.0.1", port=args.port, debug=False)
