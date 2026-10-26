"""FastAPI router for the ER proof of concept: mounted in ``server.py`` on this branch.

It carries no logic of its own: each route hands the request to ``er.handlers``. The application's static API-token
middleware protects these routes like every other route.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse, Response

from core.error_handlers import get_request_id
from er import handlers

router = APIRouter(prefix="/er", tags=["ER model (proof of concept)"])


def _reply(request: Request, status: int, payload: dict[str, Any]) -> JSONResponse:
    """A JSON reply; an error carries this request's id, as every other error of the application does."""
    if isinstance(payload.get("error"), dict):
        payload["error"]["request_id"] = get_request_id(request)
    return JSONResponse(payload, status_code=status)


@router.get("/samples", summary="Bundled sample ER models")
def list_samples() -> JSONResponse:
    status, payload = handlers.samples()
    return JSONResponse(payload, status_code=status)


@router.get("/samples/{sample_id}", summary="One bundled sample ER model")
def get_sample(sample_id: str, request: Request) -> JSONResponse:
    return _reply(request, *handlers.sample(sample_id))


@router.post("/validate", summary="Check an ER model and return the plan (order, columns, expected rows)")
def validate(request: Request, body: dict[str, Any] = Body(...)) -> JSONResponse:
    return _reply(request, *handlers.validate(body))


@router.post("/generate", summary="Generate every table of an ER model, with an integrity report")
def generate(request: Request, body: dict[str, Any] = Body(...)) -> JSONResponse:
    return _reply(request, *handlers.generate_tables(body))


@router.post("/export", summary="Generate and download one CSV per table, the integrity report and the model, as a zip")
def export(request: Request, body: dict[str, Any] = Body(...)) -> Response:
    status, data, filename = handlers.export_zip(body)
    if status != 200:
        return _reply(request, status, data)
    return Response(content=data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
