"""Hook system — lifecycle callbacks for extensions."""

from .hooks import HookRegistry, HookEvent, HookResult

__all__ = ["HookRegistry", "HookEvent", "HookResult"]
