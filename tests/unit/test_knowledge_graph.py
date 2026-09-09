"""Tests for the knowledge graph."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_core.memory.domain import EdgeType, GraphEdge, GraphNode, NodeType
from harness_core.memory.graph import KnowledgeGraph


class TestKnowledgeGraphBasics:
    def test_add_node_and_get(self):
        g = KnowledgeGraph()
        node = GraphNode(id="t1", type=NodeType.TASK, label="Implement auth")
        g.add_node(node)
        assert g.get_node("t1") is not None
        assert g.get_node("t1").label == "Implement auth"

    def test_add_node_updates_existing(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="Old label"))
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="New label"))
        assert g.get_node("t1").label == "New label"

    def test_add_edge(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="Task"))
        g.add_node(GraphNode(id="t2", type=NodeType.TASK, label="Other"))
        g.add_edge(GraphEdge(src="t1", dst="t2", type=EdgeType.DEPENDS_ON))
        assert len(g.all_edges()) == 1

    def test_duplicate_edge_increments_weight(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="t2", type=NodeType.TASK, label="B"))
        g.add_edge(GraphEdge(src="t1", dst="t2", type=EdgeType.DEPENDS_ON, weight=1.0))
        g.add_edge(GraphEdge(src="t1", dst="t2", type=EdgeType.DEPENDS_ON, weight=1.0))
        edges = g.all_edges()
        assert len(edges) == 1
        assert edges[0].weight == 2.0

    def test_remove_node_clears_edges(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="t2", type=NodeType.TASK, label="B"))
        g.add_edge(GraphEdge(src="t1", dst="t2", type=EdgeType.DEPENDS_ON))
        g.remove_node("t1")
        assert g.get_node("t1") is None
        assert len(g.all_edges()) == 0


class TestKnowledgeGraphQueries:
    def test_neighbors_outgoing(self):
        g = KnowledgeGraph()
        for i in range(3):
            g.add_node(GraphNode(id=f"n{i}", type=NodeType.TASK, label=f"N{i}"))
        g.add_edge(GraphEdge(src="n0", dst="n1", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="n0", dst="n2", type=EdgeType.DEPENDS_ON))
        neighbours = g.neighbors("n0", direction="out")
        assert len(neighbours) == 2
        assert {n.id for n in neighbours} == {"n1", "n2"}

    def test_neighbors_incoming(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="n0", type=NodeType.TASK, label="Center"))
        g.add_node(GraphNode(id="n1", type=NodeType.TASK, label="From"))
        g.add_edge(GraphEdge(src="n1", dst="n0", type=EdgeType.DEPENDS_ON))
        neighbours = g.neighbors("n0", direction="in")
        assert [n.id for n in neighbours] == ["n1"]

    def test_neighbors_filter_by_edge_type(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="b", type=NodeType.TASK, label="B"))
        g.add_node(GraphNode(id="c", type=NodeType.TASK, label="C"))
        g.add_edge(GraphEdge(src="a", dst="b", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="a", dst="c", type=EdgeType.RELATED_TO))
        only_dep = g.neighbors("a", edge_types=[EdgeType.DEPENDS_ON])
        assert [n.id for n in only_dep] == ["b"]

    def test_related_bfs_to_depth_2(self):
        g = KnowledgeGraph()
        for i in range(5):
            g.add_node(GraphNode(id=f"n{i}", type=NodeType.TASK, label=f"N{i}"))
        # Chain: n0 - n1 - n2 - n3 - n4
        g.add_edge(GraphEdge(src="n0", dst="n1", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="n1", dst="n2", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="n2", dst="n3", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="n3", dst="n4", type=EdgeType.DEPENDS_ON))
        # Depth 1: n1
        depth1 = g.related("n0", max_depth=1)
        assert [n.id for n in depth1] == ["n1"]
        # Depth 2: n1, n2
        depth2 = g.related("n0", max_depth=2)
        assert {n.id for n in depth2} == {"n1", "n2"}

    def test_find_path_direct(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="b", type=NodeType.TASK, label="B"))
        g.add_edge(GraphEdge(src="a", dst="b", type=EdgeType.DEPENDS_ON))
        assert g.find_path("a", "b") == ["a", "b"]

    def test_find_path_indirect(self):
        g = KnowledgeGraph()
        for n in ["a", "b", "c", "d"]:
            g.add_node(GraphNode(id=n, type=NodeType.TASK, label=n))
        g.add_edge(GraphEdge(src="a", dst="b", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="b", dst="c", type=EdgeType.DEPENDS_ON))
        g.add_edge(GraphEdge(src="c", dst="d", type=EdgeType.DEPENDS_ON))
        assert g.find_path("a", "d") == ["a", "b", "c", "d"]

    def test_find_path_no_path(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="b", type=NodeType.TASK, label="B"))
        assert g.find_path("a", "b") is None

    def test_find_path_same_node(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        assert g.find_path("a", "a") == ["a"]

    def test_find_path_missing_node(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        assert g.find_path("a", "nonexistent") is None


class TestKnowledgeGraphPersistence:
    def test_persist_and_reload(self, tmp_path: Path):
        path = tmp_path / "graph.json"
        g1 = KnowledgeGraph(storage_path=path)
        g1.add_node(GraphNode(id="n1", type=NodeType.TASK, label="First"))
        g1.add_node(GraphNode(id="n2", type=NodeType.TASK, label="Second"))
        g1.add_edge(GraphEdge(src="n1", dst="n2", type=EdgeType.DEPENDS_ON))
        g1.persist()
        assert path.exists()

        g2 = KnowledgeGraph(storage_path=path)
        assert g2.get_node("n1") is not None
        assert g2.get_node("n2") is not None
        assert len(g2.all_edges()) == 1

    def test_to_dict_and_from_dict_roundtrip(self):
        g1 = KnowledgeGraph()
        g1.add_node(GraphNode(id="n1", type=NodeType.TASK, label="X", properties={"k": 1}))
        g1.add_edge(GraphEdge(src="n1", dst="n1", type=EdgeType.RELATED_TO))
        data = g1.to_dict()
        g2 = KnowledgeGraph.from_dict(data)
        assert g2.get_node("n1").properties == {"k": 1}
        assert len(g2.all_edges()) == 1

    def test_corrupt_file_loads_empty(self, tmp_path: Path):
        path = tmp_path / "graph.json"
        path.write_text("not json at all", encoding="utf-8")
        g = KnowledgeGraph(storage_path=path)
        assert g.get_node("anything") is None
        assert g.all_edges() == []

    def test_stats(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="T"))
        g.add_node(GraphNode(id="f1", type=NodeType.FAILURE, label="F"))
        g.add_node(GraphNode(id="d1", type=NodeType.DECISION, label="D"))
        stats = g.stats()
        assert stats["nodes"] == 3
        assert stats["node_types"][NodeType.TASK.value] == 1
        assert stats["node_types"][NodeType.FAILURE.value] == 1


# ── Adversarial ────────────────────────────────────────────────────────────


class TestAdversarialGraph:
    def test_poisoning_via_self_loop_does_not_loop_forever(self):
        """A self-loop edge should not cause infinite loops in BFS."""
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="t1", type=NodeType.TASK, label="Poisoned"))
        # Self-loop
        g.add_edge(GraphEdge(src="t1", dst="t1", type=EdgeType.CAUSED_BY))
        # Should terminate
        related = g.related("t1", max_depth=5)
        # Self is excluded from `related` (depth 0)
        assert all(n.id != "t1" for n in related)

    def test_duplicate_node_merge_preserves_edges(self):
        g = KnowledgeGraph()
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A"))
        g.add_node(GraphNode(id="b", type=NodeType.TASK, label="B"))
        g.add_edge(GraphEdge(src="a", dst="b", type=EdgeType.DEPENDS_ON))
        # Re-add node 'a' with new label — edges should remain
        g.add_node(GraphNode(id="a", type=NodeType.TASK, label="A updated"))
        assert g.get_node("a").label == "A updated"
        assert len(g.all_edges()) == 1
