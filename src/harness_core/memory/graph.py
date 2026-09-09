"""In-memory knowledge graph with JSON persistence.

Stores relationships between tasks, agents, files, decisions, and
failures.  Supports:
  - O(1) add / lookup
  - Edge deduplication (re-adding the same (src, dst, type) increments
    weight instead of creating a parallel edge)
  - BFS for `related()` and `find_path()`
  - Atomic JSON persistence

Future backends (e.g. Neo4j) can implement the same public surface.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from harness_core.memory.domain import EdgeType, GraphEdge, GraphNode, NodeType


class KnowledgeGraph:
    """A directed, weighted knowledge graph."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else None
        self._nodes: dict[str, GraphNode] = {}
        # src -> (dst -> edge)
        self._out: dict[str, dict[str, GraphEdge]] = defaultdict(dict)
        # dst -> (src -> edge) — for incoming queries
        self._in: dict[str, dict[str, GraphEdge]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        if self.storage_path and self.storage_path.exists():
            self._load()

    # ── Mutation ─────────────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        """Add a node; if it already exists, update its label/properties."""
        if node.id in self._nodes:
            existing = self._nodes[node.id]
            # Merge properties; new values win
            existing.properties.update(node.properties)
            # Refresh label/type if non-empty
            if node.label:
                existing.label = node.label
            if node.type:
                existing.type = node.type
        else:
            self._nodes[node.id] = node
        self._persist_async()

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge; duplicate (src, dst, type) increments weight."""
        existing = self._out[edge.src].get(edge.dst)
        if existing and existing.type == edge.type:
            existing.weight += edge.weight
            existing.properties.update(edge.properties)
        else:
            self._out[edge.src][edge.dst] = edge
            self._in[edge.dst][edge.src] = edge
        self._persist_async()

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and all incident edges."""
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        # Clean up edges
        for dst in list(self._out.get(node_id, {}).keys()):
            self._in.get(dst, {}).pop(node_id, None)
        self._out.pop(node_id, None)
        for src in list(self._in.get(node_id, {}).keys()):
            self._out.get(src, {}).pop(node_id, None)
        self._in.pop(node_id, None)
        self._persist_async()
        return True

    # ── Query ────────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def all_nodes(self) -> list[GraphNode]:
        return list(self._nodes.values())

    def all_edges(self) -> list[GraphEdge]:
        out: list[GraphEdge] = []
        for src_dict in self._out.values():
            out.extend(src_dict.values())
        return out

    def neighbors(
        self,
        node_id: str,
        edge_types: list[EdgeType] | None = None,
        direction: str = "out",
    ) -> list[GraphNode]:
        """Return nodes directly connected to `node_id`.

        Args:
            node_id: The starting node.
            edge_types: If provided, only follow edges of these types.
            direction: "out" (default), "in", or "both".
        """
        results: list[GraphNode] = []
        seen: set[str] = set()

        def _add(neighbor_id: str) -> None:
            if neighbor_id in seen:
                return
            seen.add(neighbor_id)
            node = self._nodes.get(neighbor_id)
            if node:
                results.append(node)

        if direction in ("out", "both"):
            for dst, edge in self._out.get(node_id, {}).items():
                if edge_types is None or edge.type in edge_types:
                    _add(dst)
        if direction in ("in", "both"):
            for src, edge in self._in.get(node_id, {}).items():
                if edge_types is None or edge.type in edge_types:
                    _add(src)
        return results

    def related(
        self,
        node_id: str,
        max_depth: int = 2,
        edge_types: list[EdgeType] | None = None,
    ) -> list[GraphNode]:
        """Return nodes within `max_depth` hops of `node_id` (BFS)."""
        visited: set[str] = {node_id}
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])
        out: list[GraphNode] = []
        while frontier:
            current, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for neighbour in self.neighbors(current, edge_types=edge_types, direction="both"):
                if neighbour.id in visited:
                    continue
                visited.add(neighbour.id)
                out.append(neighbour)
                frontier.append((neighbour.id, depth + 1))
        return out

    def find_path(self, src: str, dst: str) -> list[str] | None:
        """BFS shortest path from `src` to `dst`.  Returns node IDs."""
        if src not in self._nodes or dst not in self._nodes:
            return None
        if src == dst:
            return [src]
        visited = {src}
        queue: deque[tuple[str, list[str]]] = deque([(src, [src])])
        while queue:
            current, path = queue.popleft()
            for neighbour in self.neighbors(current, direction="out"):
                if neighbour.id in visited:
                    continue
                visited.add(neighbour.id)
                new_path = path + [neighbour.id]
                if neighbour.id == dst:
                    return new_path
                queue.append((neighbour.id, new_path))
        return None

    # ── Persistence ──────────────────────────────────────────────────────

    def _persist_async(self) -> None:
        """Schedule a persist (sync write to avoid async-in-callback issues)."""
        if self.storage_path is None:
            return
        try:
            self.persist()
        except OSError:
            # Persistence failures should never break the in-memory graph
            pass

    def persist(self) -> None:
        if self.storage_path is None:
            return
        data = self.to_dict()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.storage_path.with_suffix(self.storage_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.storage_path)

    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            for nd in data.get("nodes", []):
                node = GraphNode.from_dict(nd)
                self._nodes[node.id] = node
            for ed in data.get("edges", []):
                edge = GraphEdge.from_dict(ed)
                self._out[edge.src][edge.dst] = edge
                self._in[edge.dst][edge.src] = edge
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            # Corrupt file: start fresh
            self._nodes.clear()
            self._out.clear()
            self._in.clear()

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": [e.to_dict() for e in self.all_edges()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeGraph":
        g = cls()
        for nd in data.get("nodes", []):
            g.add_node(GraphNode.from_dict(nd))
        for ed in data.get("edges", []):
            g.add_edge(GraphEdge.from_dict(ed))
        return g

    def stats(self) -> dict[str, Any]:
        return {
            "nodes": len(self._nodes),
            "edges": sum(len(d) for d in self._out.values()),
            "node_types": {
                nt.value: sum(1 for n in self._nodes.values() if n.type == nt)
                for nt in NodeType
            },
        }
