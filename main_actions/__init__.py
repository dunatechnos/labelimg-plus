"""Main-window Annotation Helper actions.

Everything is driven from the normal LabelImg window: keyboard shortcuts,
an Action Palette, and compact dialogs.  The old Annotation Helper page is
kept only as hidden fallback code.
"""

from .controller import MainActionController

__all__ = ["MainActionController"]
