# One container: backend and web UI together

`Dockerfile.combined` builds one image in which the FastAPI server also serves the built web UI. The standard `Dockerfile` is unchanged
and still builds the API-only image.

## How it fits together
| Piece | What it does |
|---|---|
| `Dockerfile.combined` | Stage 1 builds the UI with Node (`VITE_API_HOST=same-origin`); stage 2 installs Python packages; the runtime stage has Python only and the UI in `/app/ui`. The UI repository is a named build context: `docker buildx build --build-context ui=../telcoFrontend -f Dockerfile.combined .` |
| `core/ui_static.py` | `mount_ui(app)` serves `/assets/*`, each top-level file of the UI folder, and `/runtime-config.js`. It does nothing when `UI_DIR` (default `/app/ui`) has no `index.html`. |
| `GET /` | Returns the UI to a browser (`Accept: text/html`) and the usual JSON liveness answer to everything else, so probes and API clients are unaffected. |
| `core/api_auth.py` | The UI's own files are exempt from the API token (the page must load before it can send one). Every API route still needs it. |
| `/runtime-config.js` | `window.__APP_CONFIG__ = {"VITE_API_TOKEN": "...", "VITE_DEFAULT_COUNTRY": "IN", ...}`: every `VITE_*` environment variable of the running service, read on each request (never cached). The UI prefers these over the values built into it (`src/lib/appConfig.ts`), so dev, test and prod can differ with one image and a new revision is enough to change a value. Other variables (`MONGODB_URI`, `GEMINI_API_KEY`, ...) are never sent. |

The UI is a single page with no URL routes of its own, so the server needs no catch-all route for page refreshes.

## Settings
| Variable | Purpose |
|---|---|
| `UI_DIR` (or `STATIC_DIR`) | Folder with the built UI (`/app/ui` in `Dockerfile.combined`). |
| `VITE_API_HOST`, `VITE_API_TOKEN`, `VITE_DEFAULT_COUNTRY`, `VITE_PLATFORM_URL` | Frontend settings, sent to the page. `VITE_API_HOST=same-origin` makes it call this server. `VITE_API_TOKEN` must be one of `API_STATIC_TOKENS`; unset, the page sends none and works only while authentication is off. |
| `UI_API_TOKEN` | Older name for `VITE_API_TOKEN`; still accepted. |

## Security note
The VITE_* values, including the token, are given to everyone who can load the page. With a publicly reachable service it only keeps out callers who never opened the
page. For real data, put an identity check (IAP, a load balancer, a VPN) in front of the whole service. The deploy commands that accompany this change explain the options.

## Checks
`test_ui_static.py` (27 checks; delivered with the optional extras, run it from the backend folder; no database or network) drives the real token middleware with a throw-away UI folder: the page and its files
load without a token, every API route still returns 401 without one, dot-files and path traversal are refused, and nothing becomes public
when no UI is installed.
