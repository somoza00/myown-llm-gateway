"""Filtro de `status` em GET /api/logs: valor inválido é 400, nunca lista vazia.

Antes o parâmetro era `str` livre: `?status=ERROR` (ou 'erro', 'failed') casava
com nenhuma linha e a rota respondia 200 com `{"total": 0, "logs": []}` — no
dashboard isso é indistinguível de "não há logs", então o operador concluía que
o tráfego tinha parado em vez de perceber o filtro digitado errado.
"""

from __future__ import annotations

AUTH = {"Authorization": "Bearer test-virt-key"}


async def test_logs_rejects_invalid_status_filter(client, api_key) -> None:
    resp = await client.get("/api/logs", headers=AUTH, params={"status": "ERROR"})

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["type"] == "invalid_request_error"
    # O param aponta onde está o erro (loc = ('query', 'status')).
    assert error["param"].endswith("status")


async def test_logs_accepts_the_two_real_statuses(client, api_key) -> None:
    for value in ("ok", "error"):
        resp = await client.get("/api/logs", headers=AUTH, params={"status": value})

        assert resp.status_code == 200, resp.text
        assert isinstance(resp.json()["total"], int)
