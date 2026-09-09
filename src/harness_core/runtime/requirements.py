"""Structured requirements and requirement traceability (Phase 9).

Requirements are runtime-owned state.  They are derived from the user's
request (never from model output) and are the single anchor that planning,
execution and verification all trace back to:

    REQ-00X (requirement)
        └── mapped tasks (Planner proposes, runtime records the mapping)
              └── artifacts produced (task.files_changed)
                    └── evidence (test/review/verifier results)
                          └── VerificationTrace(status=PASS|FAIL|NOT_VERIFIED)

The classes here are deliberately dependency-free so any component (Planner,
Scheduler, Verifier, CLI) can import them without pulling the whole runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class RequirementStatus(enum.Enum):
    """Verification status of a single requirement."""

    UNVERIFIED = "unverified"  # no task evidence collected yet
    IN_PROGRESS = "in_progress"  # some mapped tasks complete
    VERIFIED = "verified"  # mapped tasks done + evidence attached
    FAILED = "failed"  # mapped task failed or verifier rejected


class EvidenceStatus(enum.Enum):
    """Status of a piece of requirement evidence."""

    PASS = "pass"
    FAIL = "fail"


@dataclass
class Requirement:
    """One requirement with structured traceability metadata.

    Fields:
        req_id: Stable identifier (REQ-001).  Assigned by the runtime.
        category: 'functional' | 'non_functional' | 'constraint' | 'acceptance' | 'exclusion'
        statement: Plain-language statement of the requirement.
        task_ids: Task ids that implement this requirement (runtime-owned mapping).
        acceptance_notes: Optional clarification used by the verifier.
    """

    req_id: str
    statement: str
    category: str = "functional"
    task_ids: list[str] = field(default_factory=list)
    acceptance_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.req_id,
            "category": self.category,
            "statement": self.statement,
            "task_ids": list(self.task_ids),
            "acceptance_notes": self.acceptance_notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirement:
        return cls(
            req_id=data.get("req_id", ""),
            category=data.get("category", "functional"),
            statement=data.get("statement", ""),
            task_ids=list(data.get("task_ids", [])),
            acceptance_notes=data.get("acceptance_notes", ""),
        )


@dataclass
class Requirements:
    """A structured capture of what the user asked for.

    This is the single place Planner *and* Verifier read the project's goal
    from — preventing ``Planner: "build X"`` from silently becoming
    ``Verifier: "looks okay"``.
    """

    objective: str = ""
    functional: list[str] = field(default_factory=list)
    non_functional: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    notes: str = ""

    def _items(self) -> list[tuple[str, str]]:
        """Flatten into (category, statement) pairs, exclusions excluded."""
        out: list[tuple[str, str]] = []
        for statement in self.functional:
            out.append(("functional", statement))
        for statement in self.non_functional:
            out.append(("non_functional", statement))
        for statement in self.constraints:
            out.append(("constraint", statement))
        for statement in self.acceptance_criteria:
            out.append(("acceptance", statement))
        return out

    def to_requirements(self) -> list[Requirement]:
        """Expand into individually-addressable Requirement objects."""
        reqs: list[Requirement] = []
        counter = 1
        for category, statement in self._items():
            reqs.append(
                Requirement(req_id=f"REQ-{counter:03d}", category=category, statement=statement)
            )
            counter += 1
        return reqs

    @classmethod
    def from_entries(
        cls,
        objective: str,
        functional: list[str] | None = None,
        non_functional: list[str] | None = None,
        constraints: list[str] | None = None,
        acceptance_criteria: list[str] | None = None,
        exclusions: list[str] | None = None,
        notes: str = "",
    ) -> Requirements:
        """Builder with safe defaults (no mutable default arguments)."""
        return cls(
            objective=objective or "",
            functional=list(functional or []),
            non_functional=list(non_functional or []),
            constraints=list(constraints or []),
            acceptance_criteria=list(acceptance_criteria or []),
            exclusions=list(exclusions or []),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "functional": list(self.functional),
            "non_functional": list(self.non_functional),
            "constraints": list(self.constraints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "exclusions": list(self.exclusions),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Requirements:
        return cls.from_entries(
            objective=data.get("objective", ""),
            functional=data.get("functional"),
            non_functional=data.get("non_functional"),
            constraints=data.get("constraints"),
            acceptance_criteria=data.get("acceptance_criteria"),
            exclusions=data.get("exclusions"),
            notes=data.get("notes", ""),
        )

    @property
    def is_empty(self) -> bool:
        return not (
            self.objective
            or self.functional
            or self.non_functional
            or self.constraints
            or self.acceptance_criteria
        )


@dataclass
class Evidence:
    """One piece of evidence attached to a requirement."""

    source: str  # e.g. task id, "test", "review", "verifier"
    status: EvidenceStatus
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "status": self.status.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Evidence:
        return cls(
            source=data.get("source", ""),
            status=EvidenceStatus(data.get("status", "pass")),
            detail=data.get("detail", ""),
        )


@dataclass
class VerificationTrace:
    """Traceability record: requirement → tasks → evidence → status.

    This is the audit trail answering *"how do we know REQ-001 is done?"*.
    """

    requirement: Requirement
    evidence: list[Evidence] = field(default_factory=list)
    status: RequirementStatus = RequirementStatus.UNVERIFIED

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)
        self._recompute_status()

    def _recompute_status(self) -> None:
        """Derive status from evidence + the requirement's mapped tasks.

        A requirement with mapped tasks is only VERIFIED once **every** mapped
        task contributed PASS evidence (the runtime attaches one PASS per
        completed mapped task). A FAIL from any source overrides. Requirements
        without mapped tasks rely on verifier/acceptance evidence alone.
        """
        mapped = [t for t in self.requirement.task_ids if t]

        if any(e.status == EvidenceStatus.FAIL for e in self.evidence):
            self.status = RequirementStatus.FAILED
            return

        task_pass = {
            e.source
            for e in self.evidence
            if e.status == EvidenceStatus.PASS and e.source in mapped
        }
        if mapped:
            if set(mapped) <= task_pass:
                self.status = RequirementStatus.VERIFIED
            elif not task_pass:
                self.status = RequirementStatus.UNVERIFIED
            else:
                # Some mapped tasks done, some not yet.
                self.status = RequirementStatus.IN_PROGRESS
            return

        # No mapped tasks: any passing evidence (e.g. a verifier verdict or an
        # acceptance-criterion check) verifies the requirement.
        if any(e.status == EvidenceStatus.PASS for e in self.evidence):
            self.status = RequirementStatus.VERIFIED
            return
        self.status = RequirementStatus.UNVERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "req_id": self.requirement.req_id,
            "category": self.requirement.category,
            "statement": self.requirement.statement,
            "mapped_tasks": list(self.requirement.task_ids),
            "status": self.status.value,
            "evidence": [e.to_dict() for e in self.evidence],
        }


@dataclass
class TraceabilityIndex:
    """Runtime-owned map from requirement ids to their traces."""

    traces: dict[str, VerificationTrace] = field(default_factory=dict)
    # Map task_id → the requirement(s) it implements.  Derived from the same
    # source the traces were built from (kept for fast reverse lookups).
    _task_to_req: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def build(cls, requirements: list[Requirement]) -> TraceabilityIndex:
        index = cls()
        for req in requirements:
            index.traces[req.req_id] = VerificationTrace(requirement=req)
            for task_id in req.task_ids:
                index._task_to_req.setdefault(task_id, []).append(req.req_id)
        return index

    def requirements_for_task(self, task_id: str) -> list[Requirement]:
        req_ids = self._task_to_req.get(task_id, [])
        return [self.traces[rid].requirement for rid in req_ids if rid in self.traces]

    def attach_task_evidence(
        self, task_id: str, source: str, status: EvidenceStatus, detail: str = ""
    ) -> None:
        """Record that *task_id* contributed PASS/FAIL evidence to its requirements."""
        for req_id in self._task_to_req.get(task_id, []):
            trace = self.traces.get(req_id)
            if trace:
                trace.add_evidence(Evidence(source=source or task_id, status=status, detail=detail))

    def overall(self) -> tuple[int, int, int, int]:
        """Return (verified, failed, unverified, total) across all traces."""
        total = len(self.traces)
        verified = sum(1 for t in self.traces.values() if t.status == RequirementStatus.VERIFIED)
        failed = sum(1 for t in self.traces.values() if t.status == RequirementStatus.FAILED)
        return verified, failed, total - verified - failed, total

    def to_dict(self) -> dict[str, Any]:
        return {"traces": [t.to_dict() for t in self.traces.values()]}
