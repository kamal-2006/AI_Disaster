from __future__ import annotations

"""Question-aware graph retrieval over the curated SafeGraph knowledge base.

The current project has no Neo4j dependency or persisted graph. This adapter
materializes the existing structured knowledge into graph nodes and edges so
retrieval has an explicit Knowledge Graph boundary while remaining local and
dependency-free. A Neo4j backend can replace this adapter without changing the
assistant contract.
"""

import re
from typing import Any, Dict, Iterable, List

from services.disaster_knowledge import DISASTER_KNOWLEDGE_BASE


SECTION_TOPIC_NAMES = {
    "explanations": "cause_and_explanation",
    "immediate_actions": "immediate_action",
    "what_to_avoid": "prevention_and_hazards",
    "emergency_actions": "emergency_response",
    "school_student_tips": "school_safety",
}

INTENT_SECTION_WEIGHTS = {
    "symptoms": {"emergency_actions": 8, "immediate_actions": 2},
    "cause": {"explanations": 9, "description": 2},
    "immediate_action": {"immediate_actions": 8, "emergency_actions": 6, "what_to_avoid": 5},
    "preparedness": {"immediate_actions": 6, "what_to_avoid": 3, "school_student_tips": 2},
    "prevention": {"what_to_avoid": 7, "immediate_actions": 3},
    "effects": {"explanations": 7, "emergency_actions": 3},
    "general": {"immediate_actions": 2, "emergency_actions": 2},
}


def _terms(value: str) -> set[str]:
    stop_words = {
        "what", "are", "the", "is", "a", "an", "do", "does", "i", "my", "you", "how",
        "can", "should", "to", "of", "for", "during", "about", "if", "in", "on", "and",
        "or", "why", "happen", "happens", "would", "with", "from", "before", "after",
    }
    return {word for word in re.findall(r"[a-z0-9]+", value.lower()) if word not in stop_words and len(word) > 2}


def _node(node_id: str, label: str, **properties: Any) -> Dict[str, Any]:
    return {"id": node_id, "label": label, "properties": properties}


def _relationship(source: str, relationship: str, target: str) -> Dict[str, str]:
    return {"source": source, "relationship": relationship, "target": target}


def build_knowledge_graph() -> Dict[str, List[Dict[str, Any]]]:
    """Build graph nodes and relationships from the existing knowledge source."""
    nodes: List[Dict[str, Any]] = []
    relationships: List[Dict[str, Any]] = []

    for category, data in DISASTER_KNOWLEDGE_BASE.items():
        disaster_id = f"disaster:{category}"
        nodes.append(_node(disaster_id, "Disaster", category=category, title=data["title"]))
        topic_ids: Dict[str, str] = {}

        for section, content in data.items():
            if section in {"title", "description"}:
                continue
            topic_name = SECTION_TOPIC_NAMES.get(section, section)
            topic_id = topic_ids.setdefault(topic_name, f"topic:{category}:{topic_name}")
            if not any(node["id"] == topic_id for node in nodes):
                nodes.append(_node(topic_id, "Topic", name=topic_name, section=section))
                relationships.append(_relationship(disaster_id, "HAS_TOPIC", topic_id))

            if not isinstance(content, list):
                continue
            relationships.append(_relationship(disaster_id, "HAS_KNOWLEDGE", topic_id))
            for index, text in enumerate(content):
                knowledge_id = f"knowledge:{category}:{section}:{index}"
                nodes.append(_node(
                    knowledge_id,
                    "Knowledge",
                    category=category,
                    section=section,
                    topic=topic_name,
                    text=text,
                ))
                relationships.append(_relationship(topic_id, "CONTAINS", knowledge_id))

    return {"nodes": nodes, "relationships": relationships}


class KnowledgeGraphRetriever:
    def __init__(self, graph: Dict[str, List[Dict[str, Any]]] | None = None) -> None:
        self.graph = graph or build_knowledge_graph()
        self.nodes_by_id = {node["id"]: node for node in self.graph["nodes"]}

    def retrieve(
        self,
        category: str,
        intent: str,
        topic: str,
        query: str,
        entities: Iterable[str] = (),
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return ranked knowledge nodes connected to the detected disaster."""
        disaster_id = f"disaster:{category}"
        if disaster_id not in self.nodes_by_id:
            return []

        query_terms = _terms(" ".join([query, topic, *entities]))
        section_weights = INTENT_SECTION_WEIGHTS.get(intent, INTENT_SECTION_WEIGHTS["general"])
        connected_ids = {
            edge["target"]
            for edge in self.graph["relationships"]
            if edge["source"] == disaster_id and edge["relationship"] == "HAS_TOPIC"
        }
        knowledge_nodes = [
            node for node in self.graph["nodes"]
            if node["label"] == "Knowledge"
            and node["properties"].get("category") == category
            and any(
                edge["source"] in connected_ids
                and edge["target"] == node["id"]
                and edge["relationship"] == "CONTAINS"
                for edge in self.graph["relationships"]
            )
        ]

        ranked = []
        for node in knowledge_nodes:
            properties = node["properties"]
            text = str(properties["text"])
            text_terms = _terms(text)
            if properties["section"] == "school_student_tips" and not query_terms.intersection({"school", "student", "class", "teacher"}):
                continue
            overlap = len(query_terms & text_terms)
            phrase_bonus = sum(5 for term in query_terms if len(term) > 5 and term in text.lower())
            section = properties["section"]
            score = overlap * 2 + phrase_bonus + section_weights.get(section, 0)
            if topic and properties["topic"] == topic:
                score += 4
            graph_relationships = [
                edge for edge in self.graph["relationships"]
                if edge["target"] == node["id"]
                or edge["source"] == node["id"]
            ]
            if score <= 0:
                continue
            ranked.append({
                "node_id": node["id"],
                "node_label": node["label"],
                "category": category,
                "section": section,
                "topic": properties["topic"],
                "text": text,
                "score": score,
                "relationships": graph_relationships,
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]


KNOWLEDGE_GRAPH = KnowledgeGraphRetriever()