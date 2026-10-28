"""Checks for the single-container UI serving (core/ui_static.py plus the API-token exemption in core/api_auth.py).

Run from the backend folder:  python test_ui_static.py        (needs the backend's requirements, nothing else; no database, no network)
Builds a throwaway "UI" in a temp folder, then drives a small app that uses the real token middleware.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("BACKEND_DIR", HERE))
sys.path.insert(0, str(ROOT))

tmp = Path(tempfile.mkdtemp())
(tmp / "ui" / "assets").mkdir(parents=True)
(tmp / "ui" / "index.html").write_text("<!doctype html><title>ui</title>")
(tmp / "ui" / "assets" / "app-123.css").write_text("a{}")
(tmp / "ui" / "favicon.svg").write_text("<svg/>")
(tmp / "ui" / ".hidden").write_text("no")
(tmp / "outside.txt").write_text("outside")
os.environ.update(STATIC_DIR=str(tmp / "ui"), VITE_API_TOKEN='tok</script>"', VITE_DEFAULT_COUNTRY="IN", API_STATIC_TOKENS="good-token",
                  MONGODB_URI="mongodb://private-marker", GEMINI_API_KEY="private-marker", UI_API_TOKEN="legacy-name")
os.environ.pop("UI_DIR", None)

from fastapi import FastAPI, Request                    # noqa: E402
from fastapi.responses import JSONResponse              # noqa: E402
from fastapi.testclient import TestClient               # noqa: E402

import core.api_auth as auth                            # noqa: E402
import core.ui_static as ui                             # noqa: E402

app = FastAPI()


@app.get("/")
def root(request: Request):
    page = ui.serve_index_if_browser(request)
    return page if page is not None else JSONResponse({"status": "ok"})


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})


@app.get("/taken")
def taken():
    return JSONResponse({"api": True})


@app.get("/api-example")
def samples():
    return JSONResponse({"ok": 1})


os.environ["UI_PAGE_PATHS"] = "/source, /taken, /app"
assert ui.mount_ui(app)
app.middleware("http")(auth.api_token_middleware)
from core.path_prefix import StripPrefixMiddleware, normalise    # noqa: E402

app.add_middleware(StripPrefixMiddleware, prefix="/source")       # outermost, as server.py adds it
client = TestClient(app)
BEARER = {"Authorization": "Bearer good-token"}
HTML = {"accept": "text/html,application/xhtml+xml"}

failures = 0


def check(name: str, got, want) -> None:
    global failures
    ok = got == want
    failures += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got!r}, wanted {want!r}"))


r = client.get("/", headers=HTML)
check("browser gets the page", (r.status_code, "<title>ui</title>" in r.text, r.headers["cache-control"]), (200, True, "no-cache"))
check("page cannot be framed", r.headers.get("x-frame-options"), "DENY")
check("API client gets JSON at /", client.get("/", headers={"accept": "application/json"}).json(), {"status": "ok"})
check("a plain */* request (probe, curl) gets JSON", client.get("/").json(), {"status": "ok"})
check("/health open", client.get("/health").status_code, 200)

cfg = client.get("/runtime-config.js")
check("runtime config open, never cached", (cfg.status_code, cfg.headers["cache-control"]), (200, "no-store"))
check("runtime config carries the token", '"VITE_API_TOKEN": "tok' in cfg.text and cfg.text.startswith("window.__APP_CONFIG__ = {"), True)
check("a '<' in the token cannot close the script tag", "</script>" in cfg.text, False)
check("STATIC_DIR is accepted as the UI folder", ui.UI_DIR, tmp / "ui")
check("only VITE_* settings are sent, never other variables", "private-marker" in cfg.text or "MONGODB" in cfg.text or "GEMINI" in cfg.text, False)
check("other VITE_* settings are sent", '"VITE_DEFAULT_COUNTRY": "IN"' in cfg.text, True)
check("nothing configured -> empty settings", ui.runtime_config_body({}), "window.__APP_CONFIG__ = {};\n")
check("blank values are left out", ui.runtime_settings({"VITE_A": "  ", "VITE_B": "x"}), {"VITE_B": "x"})
check("UI_API_TOKEN stands in for VITE_API_TOKEN", ui.runtime_settings({"UI_API_TOKEN": "t"}), {"VITE_API_TOKEN": "t"})
check("VITE_API_TOKEN wins over UI_API_TOKEN", ui.runtime_settings({"UI_API_TOKEN": "t", "VITE_API_TOKEN": "v"}), {"VITE_API_TOKEN": "v"})

check("asset served without a token", client.get("/assets/app-123.css").status_code, 200)
check("top-level file served without a token", client.get("/favicon.svg").status_code, 200)
check("dot-files are never served", client.get("/.hidden", headers=BEARER).status_code, 404)
check("path traversal under /assets", client.get("/assets/..%2f..%2foutside.txt").status_code, 404)

src = client.get("/app")
check("a path in UI_PAGE_PATHS shows the UI without a token, whatever the Accept header", (src.status_code, "<title>ui</title>" in src.text), (200, True))
check("...also with a trailing slash", client.get("/app/", follow_redirects=True).status_code, 200)
check("/source in a browser shows the UI", "<title>ui</title>" in client.get("/source", headers=HTML).text, True)
check("an API route listed in UI_PAGE_PATHS is NOT taken over (still needs a token)", client.get("/taken").status_code, 401)
check("...and still works with a token", client.get("/taken", headers=BEARER).json(), {"api": True})
check("is_ui_path knows /app but not /taken", (ui.is_ui_path("/app"), ui.is_ui_path("/taken")), (True, False))
check("path list parsing", (ui._page_paths(), os.environ.pop("UI_PAGE_PATHS") is not None, ui._page_paths()), (["/source", "/taken", "/app"], True, ["/source"]))
# ---- the app served under a path prefix by a shared load balancer (URL_PREFIX=/source) ----
check("prefix: page at /source/", client.get("/source/", headers=HTML).text.count("<title>ui</title>"), 1)
check("prefix: assets under /source/assets", client.get("/source/assets/app-123.css").status_code, 200)
check("prefix: /source/runtime-config.js", client.get("/source/runtime-config.js").status_code, 200)
check("prefix: top-level file", client.get("/source/favicon.svg").status_code, 200)
check("prefix: API still needs the token", client.get("/source/api-example").status_code, 401)
check("prefix: API with token", client.get("/source/api-example", headers=BEARER).status_code, 200)
check("prefix: API under the prefix is not a way round the token check", client.get("/source/api-example", headers=HTML).status_code, 401)
check("prefix: unprefixed paths (Cloud Run's own probes) still work", client.get("/health").status_code, 200)
check("prefix: a different word that merely starts with it is left alone", client.get("/sourcefoo", headers=BEARER).status_code, 404)
check("prefix: normalise", [normalise(x) for x in ("source", "/source/", " /source ", "", None, "/")], ["/source", "/source", "/source", "", "", ""])

check("API without token -> 401", client.get("/api-example").status_code, 401)
check("API with wrong token -> 401", client.get("/api-example", headers={"Authorization": "Bearer nope"}).status_code, 401)
check("API with token -> 200", client.get("/api-example", headers=BEARER).status_code, 200)
check("unknown path without token -> 401", client.get("/nothing-here").status_code, 401)

check("is_ui_path: UI files", [ui.is_ui_path(p) for p in ("/runtime-config.js", "/favicon.svg", "/assets/app-123.css")], [True] * 3)
check("is_ui_path: not API / not index.html / not the bare folder",
      [ui.is_ui_path(p) for p in ("/api-example", "/index.html", "/assets", "/.hidden")], [False] * 4)

# With no UI installed nothing is mounted and nothing becomes public.
ui._top_level.clear()
ui._pages.clear()
ui.UI_DIR = tmp / "missing"
check("no UI: mount does nothing", ui.mount_ui(FastAPI()), False)
check("no UI: nothing is public", [ui.is_ui_path(p) for p in ("/runtime-config.js", "/assets/x.js")], [False, False])

print("FAILED" if failures else "all checks passed")
sys.exit(1 if failures else 0)
