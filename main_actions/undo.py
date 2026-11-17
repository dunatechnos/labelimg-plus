from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

from .context import DatasetContext


class ActionUndo:
    """Small in-memory undo stack for label-file snapshots.

    Each snapshot maps absolute label-file paths to the *raw* lines that were
    on disk before an action.  Restoring the raw text (instead of parsing and
    re-writing the boxes) keeps the undo byte-for-byte faithful.
    """

    def __init__(self, max_depth: int = 20):
        self.stack: Deque[Dict[str, List[str]]] = deque(maxlen=max_depth)

    def push(self, snapshot: Dict[str, List[str]]) -> None:
        if snapshot:
            self.stack.append(dict(snapshot))

    def can_undo(self) -> bool:
        return bool(self.stack)

    def undo(self, context: DatasetContext) -> bool:
        if not self.stack:
            return False
        snapshot = self.stack.pop()
        for path, lines in snapshot.items():
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(
                line if line.endswith("\n") else line + "\n" for line in lines
            )
            target.write_text(text, encoding="utf-8")
        context.reload()
        return True
