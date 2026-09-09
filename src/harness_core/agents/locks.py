"""
Workspace lock manager.

Provides abstractions for workspace file ownership and concurrency safety.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class ResourceMode(Enum):
    """Access mode for a workspace resource."""
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True)
class WorkspaceResource:
    """A resource requested by an agent task."""
    path: str
    mode: ResourceMode

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "mode": self.mode.value}

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> WorkspaceResource:
        return cls(
            path=data.get("path", ""),
            mode=ResourceMode(data.get("mode", "read").lower())
        )


class WorkspaceLockManager:
    """Manages file and directory locks for concurrent agent access."""
    
    def __init__(self, workspace_path: str = "") -> None:
        self.workspace_path = Path(workspace_path).resolve() if workspace_path else Path.cwd()
        # Maps task_id -> list of acquired resources
        self._locks: dict[str, list[WorkspaceResource]] = {}
        # Mutex to protect internal dictionary mutations
        self._mutex = asyncio.Lock()
        
    async def acquire(self, task_id: str, resources: list[WorkspaceResource]) -> bool:
        """
        Atomically acquire a set of resources.
        Returns True if acquired successfully, False if there is a conflict.
        """
        if not resources:
            return True

        async with self._mutex:
            # Check for conflicts with ALL other tasks
            for other_task, other_resources in self._locks.items():
                if other_task == task_id:
                    continue  # Tasks don't conflict with themselves
                
                for req_res in resources:
                    for owned_res in other_resources:
                        if self._is_conflict(req_res, owned_res):
                            return False

            # All clear, atomic acquisition
            if task_id not in self._locks:
                self._locks[task_id] = []
            
            # Avoid duplicate resources (e.g. same task re-requesting)
            for res in resources:
                if res not in self._locks[task_id]:
                    self._locks[task_id].append(res)
            return True
            
    async def release(self, task_id: str, resources: list[WorkspaceResource]) -> None:
        """Release specific resources for a task."""
        async with self._mutex:
            if task_id in self._locks:
                current = self._locks[task_id]
                for res in resources:
                    if res in current:
                        current.remove(res)
                if not current:
                    del self._locks[task_id]

    async def release_all(self, task_id: str) -> None:
        """Release all resources owned by a task."""
        async with self._mutex:
            if task_id in self._locks:
                del self._locks[task_id]

    async def is_locked(self, resource: WorkspaceResource) -> bool:
        """Check if a resource is currently locked by ANY task in a conflicting way."""
        async with self._mutex:
            for task_id, owned_resources in self._locks.items():
                for owned in owned_resources:
                    if self._is_conflict(resource, owned):
                        return True
            return False

    async def get_owner(self, resource: WorkspaceResource) -> str | None:
        """Get the current owner of a lock, if any."""
        async with self._mutex:
            for task_id, owned_resources in self._locks.items():
                for owned in owned_resources:
                    if self._is_conflict(resource, owned):
                        return task_id
            return None

    def _is_conflict(self, r1: WorkspaceResource, r2: WorkspaceResource) -> bool:
        """Determine if two resources conflict."""
        # Read + Read = No conflict
        if r1.mode == ResourceMode.READ and r2.mode == ResourceMode.READ:
            return False

        # Write vs Read or Write vs Write
        p1 = r1.path.replace("\\", "/")
        p2 = r2.path.replace("\\", "/")

        is_glob1 = "*" in p1 or "?" in p1
        is_glob2 = "*" in p2 or "?" in p2

        if not is_glob1 and not is_glob2:
            # Parent/child or exact match
            path1 = Path(p1)
            path2 = Path(p2)
            try:
                if path1 == path2:
                    return True
                if path2.is_relative_to(path1):
                    return True
                if path1.is_relative_to(path2):
                    return True
            except (ValueError, AttributeError):
                pass
            
            p1_parts = [p for p in p1.strip("/").split("/") if p]
            p2_parts = [p for p in p2.strip("/").split("/") if p]
            
            min_len = min(len(p1_parts), len(p2_parts))
            if min_len > 0 and p1_parts[:min_len] == p2_parts[:min_len]:
                return True
                
            return False

        # If one or both are globs, extract static prefixes
        pref1 = p1.split("*")[0].split("?")[0].rstrip("/")
        pref2 = p2.split("*")[0].split("?")[0].rstrip("/")
        
        # If static prefixes don't overlap, the globs can't overlap (assuming no absolute globbing)
        if pref1 and pref2:
            p1_parts = [p for p in pref1.strip("/").split("/") if p]
            p2_parts = [p for p in pref2.strip("/").split("/") if p]
            min_len = min(len(p1_parts), len(p2_parts))
            if min_len > 0 and p1_parts[:min_len] != p2_parts[:min_len]:
                return False  # Disjoint directories

        # If prefixes overlap or one has no static prefix, assume conflict
        return True
