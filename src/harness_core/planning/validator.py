"""Plan validator to ensure safety and correctness of the generated graph."""

from __future__ import annotations

from harness_core.agents.domain import AgentRole, WorkspaceScope
from harness_core.agents.registry import AgentRegistry
from harness_core.permissions.manager import PermissionManager
from harness_core.planning.domain import Plan


class PlanValidator:
    """Validates an autonomously generated plan."""

    def __init__(self, registry: AgentRegistry | None = None):
        self.registry = registry or AgentRegistry()
        self.permission_manager = PermissionManager()

    def validate(self, plan: Plan) -> list[str]:
        """Validate the entire plan and return a list of error messages."""
        errors = []
        if not plan.tasks:
            return ["Plan contains no tasks."]

        task_ids = set()
        
        # Validate individual tasks and collect IDs
        for task in plan.tasks:
            # 1. Duplicate task IDs
            if task.task_id in task_ids:
                errors.append(f"Duplicate task ID: {task.task_id}")
            task_ids.add(task.task_id)

            # 2. Unknown role
            try:
                role_enum = AgentRole(task.role.lower())
                profiles = self.registry.find_by_role(role_enum)
                if not profiles:
                    errors.append(f"Task {task.task_id} has role '{task.role}' which has no registered profiles.")
            except ValueError:
                errors.append(f"Task {task.task_id} requests unknown role: {task.role}")

            # 3. Workspace scope
            try:
                WorkspaceScope(task.workspace_scope.lower())
            except ValueError:
                errors.append(f"Task {task.task_id} requests unknown workspace scope: {task.workspace_scope}")
            
            # 4. Self dependency
            if task.task_id in task.dependencies:
                errors.append(f"Task {task.task_id} cannot depend on itself.")
                
            # 5. Resources validation
            for res in task.resources:
                path = res.get("path", "")
                mode = res.get("mode", "")
                
                if mode not in ("read", "write"):
                    errors.append(f"Task {task.task_id} has resource with invalid mode '{mode}'. Must be 'read' or 'write'.")
                
                if not path:
                    errors.append(f"Task {task.task_id} has a resource with empty path.")
                    continue
                
                # Check path safety (prevents absolute, UNC, drive letter, and ../ escapes)
                # Removing glob chars for is_within_workspace check since path might be a glob
                clean_path = path.replace("*", "").replace("?", "")
                # Some globs like `**` might reduce to empty string, but if path is valid inside workspace
                # then clean_path should still be inside. If it reduces to empty, it's the root which is safe.
                if not self.permission_manager.is_within_workspace(clean_path):
                    errors.append(f"Task {task.task_id} requests path '{path}' which escapes the workspace root.")
                
                # Workspace Scope validation (if frontent tries to access backend, etc.)
                # This is a light heuristic based on scope name since we shouldn't do unreliable heuristics.
                if task.workspace_scope.lower() != "project" and mode == "write":
                    scope_folder = task.workspace_scope.lower()
                    if scope_folder in ["frontend", "backend", "docs", "database"]:
                        # Very simple heuristic: if it's not starting with the scope or src/scope, warn (or just pass for now to avoid false positives, 
                        # but we can enforce if it's obviously violating, e.g. frontend accessing backend)
                        if scope_folder == "frontend" and ("backend/" in path.lower() or "api/" in path.lower()):
                            errors.append(f"Task {task.task_id} (Scope: FRONTEND) requests write access to backend path '{path}'.")
                        if scope_folder == "backend" and ("frontend/" in path.lower() or "ui/" in path.lower()):
                            errors.append(f"Task {task.task_id} (Scope: BACKEND) requests write access to frontend path '{path}'.")
                        if task.workspace_scope.lower() == "read_only" and mode == "write":
                            errors.append(f"Task {task.task_id} has READ_ONLY scope but requests WRITE resource '{path}'.")

        # 5. Missing dependencies
        for task in plan.tasks:
            for dep in task.dependencies:
                if dep not in task_ids:
                    errors.append(f"Task {task.task_id} depends on unknown task: {dep}")

        # 6. Dependency Cycles
        # Simple DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {tid: WHITE for tid in task_ids}
        task_map = {t.task_id: t for t in plan.tasks}

        def visit(tid: str) -> bool:
            color[tid] = GRAY
            for dep in task_map[tid].dependencies:
                if dep in task_map:
                    if color[dep] == GRAY:
                        return True
                    if color[dep] == WHITE and visit(dep):
                        return True
            color[tid] = BLACK
            return False

        has_cycle = False
        for tid in task_ids:
            if color[tid] == WHITE:
                if visit(tid):
                    has_cycle = True
                    break

        if has_cycle:
            errors.append("Plan contains a dependency cycle.")

        return errors
