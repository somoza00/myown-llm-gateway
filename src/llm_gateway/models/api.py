"""Public request/response schemas for the OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant"]
FinishReason = Literal["stop", "length", "content_filter"]
_RESPONSE_FORMAT_TYPES = frozenset({"text", "json_object", "json_schema"})


class ChatMessage(BaseModel):
    """A single message in the conversation."""

    role: Role
    content: str = Field(max_length=50_000)  # ~12k tokens; guards against a single huge message


class ChatRequest(BaseModel):
    """OpenAI-compatible chat completion request body."""

    model: str
    # max_length bounds worst-case input cost from an unreasonably long conversation
    # history; it's a coarse guard (message count, not token count), not a precise one.
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    # OpenAI-style stop sequences: a single string or up to 4 strings. Providers
    # that name the field differently (Anthropic: `stop_sequences`) translate it
    # in their adapter; the OpenAI-compatible ones pass it straight through.
    stop: str | list[str] | None = Field(default=None)
    # OpenAI `seed`: reproducible sampling. Providers that support it (OpenAI,
    # Groq, DeepSeek, …) pass it through via model_dump; Anthropic doesn't define
    # it, so its manual adapter intentionally omits it. Also part of the cache key
    # (same prompt with a different seed must not reuse a cached response).
    seed: int | None = Field(default=None)
    # OpenAI `user`: end-user identifier for abuse tracking/rate limits.
    user: str | None = Field(default=None, max_length=255)
    # OpenAI `response_format` (JSON mode): `{"type": "json_object"}` faz o
    # provedor devolver JSON válido; `{"type": "json_schema", "json_schema":
    # {...}}` fixa o schema. Vai como está no corpo dos adapters
    # OpenAI-compatíveis (o payload é o `model_dump`) e entra na cache key —
    # o mesmo prompt em modo texto e em modo JSON não podem compartilhar
    # resposta cacheada. O adapter da Anthropic não define o campo e o omite
    # (mesmo tratamento dado a `seed`).
    response_format: dict[str, Any] | None = None
    stream: bool = False

    @field_validator("response_format")
    @classmethod
    def _valid_response_format(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Rejeita um `response_format` que o provedor recusaria (400/422 na borda).

        Um `type` desconhecido ou um `json_schema` sem o objeto de schema viram
        erro no upstream depois de a requisição já ter consumido cota de
        rate-limit — falhar aqui mantém o erro onde ele é acionável.
        """
        if value is None:
            return value
        kind = value.get("type")
        if kind not in _RESPONSE_FORMAT_TYPES:
            raise ValueError(
                "response_format.type deve ser 'text', 'json_object' ou 'json_schema'"
            )
        if kind == "json_schema" and not isinstance(value.get("json_schema"), dict):
            raise ValueError(
                "response_format do tipo 'json_schema' exige o objeto 'json_schema'"
            )
        return value

    @field_validator("stop")
    @classmethod
    def _stop_within_bounds(cls, value: str | list[str] | None) -> str | list[str] | None:
        """Accept one string or up to 4 non-blank stop sequences.

        Providers reject a 5th sequence or a blank entry as an upstream 400;
        failing fast as a 422 here keeps the error at the gateway boundary.
        """
        if value is None:
            return value
        sequences = [value] if isinstance(value, str) else value
        if not sequences or any(not sequence for sequence in sequences):
            raise ValueError("stop sequences não podem ser vazias")
        if len(sequences) > 4:
            raise ValueError("no máximo 4 stop sequences")
        return value

    @model_validator(mode="after")
    def _require_user_message(self: ChatRequest) -> ChatRequest:
        """Reject a conversation with no `user` turn.

        A request of only `system`/`assistant` messages has nothing to answer;
        upstream providers reject it too. Fail fast with a clear, OpenAI-style
        error instead of forwarding a nonsensical payload.
        """
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("messages must include at least one message with role 'user'")
        return self


class ChatResponseChoice(BaseModel):
    """One candidate completion."""

    index: int = 0
    message: ChatMessage
    finish_reason: FinishReason = "stop"


class Usage(BaseModel):
    """Token accounting for a completed request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @field_validator("prompt_tokens", "completion_tokens", "total_tokens", mode="before")
    @classmethod
    def _never_negative(cls, value: object) -> object:
        """Contagem negativa do upstream vira 0.

        O `usage` destas respostas é copiado do JSON do provedor (ou do
        acumulador de streaming) e alimenta `estimated_cost` em
        `services/metrics.py`, que é o que o spend cap soma. Um provedor
        bugado/malicioso que devolvesse `prompt_tokens: -100000` produzia custo
        NEGATIVO no `usage_logs` — ou seja, *abatia* o gasto acumulado e
        enfraquecia tanto o teto por chave quanto o global (`GLOBAL_SPEND_LIMIT_USD`).
        Token não pode ser negativo; clampar na borda é mais seguro que confiar
        no formato do upstream (e não rejeita a resposta, que já foi paga).
        """
        if isinstance(value, int) and not isinstance(value, bool) and value < 0:
            return 0
        return value


class ChatResponse(BaseModel):
    """OpenAI-compatible chat completion response body."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatResponseChoice]
    usage: Usage
