"""Submission entry point for TechJam 2026 Track 4.

Exports ``Agent`` with the interface fixed by ``docs/agent_api_contract.json``.
The implementation lives in :mod:`src`; this module is the thin, stable shim the
official harness imports.

Runs offline on the Python standard library alone. No network access, no model
weights, no API keys, and therefore ``usage`` is reported as zero tokens.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

from src.agent import Options, ShoppingAgent

DEFAULT_CATALOG = os.environ.get("TECHJAM_CATALOG", "data/catalog.jsonl")


class Agent:
    """The scored artifact.

    ``catalog_path`` matches the official evaluator, which constructs the agent
    as ``Agent(args.catalog)``.
    """

    def __init__(
        self,
        catalog_path: Union[str, Path] = DEFAULT_CATALOG,
        limit: Optional[int] = None,
        options: Optional[Options] = None,
        catalog: object = None,
        retriever: object = None,
    ) -> None:
        self._impl = ShoppingAgent(
            str(catalog_path),
            limit=limit,
            options=options,
            catalog=catalog,
            retriever=retriever,
        )

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._impl.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._impl.respond(session_id, user_message, turn, top_k)

    def finalize(self) -> None:
        """Retire the final session into long-term memory.

        Optional, and the official evaluator never calls it -- a session is
        retired when the *next* one starts, so only the last one in a run is
        left open. Exposed because ``src`` documents it as an affordance for a
        harness that wants complete statistics, and a documented method that
        raises AttributeError on the class the harness actually imports is a
        broken promise.
        """
        self._impl.finalize()


__all__ = ["Agent", "Options"]
