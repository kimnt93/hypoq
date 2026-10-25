"""
hypoq Multi-Agent Harness Coordinator.
Orchestrates autonomous research sessions for codex, claude, pi, omp, and hermes.
"""

import os
import sys
import glob
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any

from hypoq.db import (
    create_session,
    search_hypotheses,
    list_hypotheses,
    record_hypothesis,
    update_hypothesis_status,
    record_experiment,
    DEFAULT_DB_PATH,
)

SUPPORTED_AGENTS = ["claude", "codex", "pi", "omp", "hermes"]


class ResearchSession:
    """Manages an autonomous multi-agent research session."""

    def __init__(
        self,
        agents: List[str],
        topic: Optional[str] = None,
        domain: Optional[str] = None,
        db_path: str = DEFAULT_DB_PATH,
    ):
        self.session_id = f"sess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        self.agents = [a.lower().strip() for a in agents if a.lower().strip() in SUPPORTED_AGENTS]
        if not self.agents:
            raise ValueError(f"No valid agents provided. Supported: {SUPPORTED_AGENTS}")
        self.topic = topic
        self.domain = domain or "general_quantitative_finance"
        self.db_path = db_path
        self.created_at = datetime.utcnow().isoformat() + "Z"

        # Register session in SQLite database
        create_session(
            session_id=self.session_id,
            agents=self.agents,
            topic=self.topic,
            notes=f"Launched via hypoq harness for {', '.join(self.agents)}",
            db_path=self.db_path,
        )

    def check_dedup(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Searches SQLite FTS5 to find top matching prior hypotheses."""
        search_target = query or self.topic
        if not search_target:
            return []
        matches = search_hypotheses(search_target, top_k=5, db_path=self.db_path)
        return matches

    def get_available_skills(self) -> List[Dict[str, str]]:
        """Discovers all skills defined in .agents/skills/."""
        skills = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        skills_dir = os.path.join(base_dir, ".agents", "skills")
        if not os.path.exists(skills_dir):
            skills_dir = os.path.join(os.getcwd(), ".agents", "skills")

        if os.path.exists(skills_dir):
            for skill_path in sorted(glob.glob(os.path.join(skills_dir, "*", "SKILL.md"))):
                skill_name = os.path.basename(os.path.dirname(skill_path))
                with open(skill_path, "r", encoding="utf-8") as f:
                    content = f.read(1000)
                desc = "No description"
                for line in content.splitlines():
                    if line.startswith("description:"):
                        desc = line.replace("description:", "").strip()
                        break
                skills.append({"name": skill_name, "path": skill_path, "description": desc})
        return skills

    def get_available_datasets(self) -> List[str]:
        """Discovers all 36 Parquet datasets in data/."""
        datasets = []
        for p in sorted(glob.glob("data/**/*.parquet", recursive=True)):
            datasets.append(p)
        return datasets

    def generate_agent_brief(self, agent_name: str) -> Dict[str, Any]:
        """Builds a comprehensive research briefing package for the designated agent."""
        dedup_matches = self.check_dedup()
        skills = self.get_available_skills()
        datasets = self.get_available_datasets()

        brief = {
            "session_id": self.session_id,
            "target_agent": agent_name,
            "co_agents": [a for a in self.agents if a != agent_name],
            "research_topic": self.topic or "Autonomous Frontier Research Exploration",
            "domain": self.domain,
            "lifecycle": [
                "IDEA", "DEDUP", "PLAN", "CODE", "EXPERIMENT", "ANALYSE", "REPORT", "DB_SYNC"
            ],
            "prior_matching_hypotheses": [
                {
                    "id": m["id"],
                    "title": m["title"],
                    "status": m["status"],
                    "metric": f"{m.get('metric_name')}: {m.get('metric_value')}",
                    "p_val": m.get("p_value"),
                    "relevance": m.get("relevance_score"),
                }
                for m in dedup_matches
            ],
            "skills_available": [s["name"] for s in skills],
            "dataset_count": len(datasets),
            "sample_datasets": datasets[:10],
            "latex_template": "templates/main.tex",
            "tectonic_compiler": "/home/kim/.local/bin/tectonic",
        }
        return brief

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes the session startup, prompts agents, and prepares execution contexts.
        """
        if verbose:
            print("\n" + "=" * 70)
            print(f"🚀 [hypoq Harness] Launching Multi-Agent Research Session")
            print(f"   Session ID : {self.session_id}")
            print(f"   Agents     : {', '.join(self.agents).upper()}")
            print(f"   Topic      : {self.topic or 'Autonomous Discovery'}")
            print(f"   Timestamp  : {self.created_at}")
            print("=" * 70)

        # Check deduplication
        dedup = self.check_dedup()
        if dedup:
            print(f"\n🔍 [Deduplication Search] Found {len(dedup)} existing research match(es):")
            for idx, m in enumerate(dedup, 1):
                status_color = "🟢" if m["status"] == "ACCEPTED" else ("🔴" if m["status"] == "REJECTED" else "🟡")
                print(f"   [{idx}] {status_color} #{m['id']} \"{m['title']}\"")
                print(f"       Status: {m['status']} | Metric: {m.get('metric_name')} = {m.get('metric_value')} (p={m.get('p_value')})")
                print(f"       Score : {m.get('relevance_score')}")
        else:
            print("\n✨ [Deduplication Search] Novel research space! Zero duplicate hypotheses detected.")

        agent_briefs = {}
        for agent in self.agents:
            brief = self.generate_agent_brief(agent)
            agent_briefs[agent] = brief
            if verbose:
                print(f"\n🤖 [Agent Context Loaded: {agent.upper()}]")
                print(f"   Loaded {len(brief['skills_available'])} skills from .agents/skills/")
                print(f"   Connected to {brief['dataset_count']} Parquet datasets under data/")
                print(f"   Ready to execute hypoq cycle.")

        return {
            "session_id": self.session_id,
            "status": "INITIALIZED",
            "agents": self.agents,
            "dedup_count": len(dedup),
            "briefs": agent_briefs,
        }
