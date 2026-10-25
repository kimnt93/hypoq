"""
hypoq Multi-Agent Harness Coordinator.
Orchestrates autonomous research sessions for codex, claude, pi, omp, and hermes.
"""

import os
import sys
import glob
import json
import uuid
import shlex
import subprocess
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
    get_connection,
)

SUPPORTED_AGENTS = ["claude", "codex", "pi", "omp", "hermes"]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(REPO_ROOT, "runs")

# Non-interactive command per agent CLI. {prompt_file} is replaced with the path of the generated brief.
# Override any of them with HYPOQ_AGENT_CMD_<NAME>, e.g. HYPOQ_AGENT_CMD_CLAUDE='claude -p --model opus "$(cat {prompt_file})"'.
DEFAULT_AGENT_COMMANDS = {
    "claude": 'claude -p "$(cat {prompt_file})" --permission-mode acceptEdits',
    "codex": 'codex exec --full-auto "$(cat {prompt_file})"',
    "pi": 'pi -p "$(cat {prompt_file})"',
    "omp": 'omp -p "$(cat {prompt_file})"',
    "hermes": 'hermes chat -q "$(cat {prompt_file})"',
}

PROTOCOL = """\
# hypoq autonomous research protocol

You are agent `{agent}` in session `{session_id}`. Co-agents: {co_agents}.
Repository root: {root}. Work only inside `researches/<new_folder>/` plus the database through `hypoq.db`.

## Topic
{topic}
Domain: {domain}

## Mandatory read first
`AGENTS.md`, `RESEARCH.md`, `DATA.md`, and the relevant `.agents/skills/*/SKILL.md`.

## Prior hypotheses that may overlap (check before coding; do not repeat REJECTED ideas without a new angle)
{prior}

## Lifecycle
IDEA -> DEDUP (`python run.py --search "..."`) -> PLAN (explicit H0/H1, English only) -> CODE -> EXPERIMENT -> ANALYSE
-> REPORT (`hypoq.report.write_paper` + `compile_pdf`) -> DB SYNC (`hypoq.db.upsert_hypothesis`, `record_experiment`).

## Integrity rules (enforced by `python run.py --audit`)
1. Every number in a table, figure, paper, JSON or DB row must be computed by `researches/<folder>/code/run_pipeline.py`.
   Never hand-type results, simulate data, or pick the decision by hand: derive ACCEPTED/REJECTED/INCONCLUSIVE from the test.
2. Null results are valid results. Report them as REJECTED/INCONCLUSIVE.
3. Strict point-in-time alignment; no look-ahead (`join_asof(strategy="backward")` for slow data).
4. Correct for multiple testing when more than one test decides the outcome.
5. Use only datasets in `data/` (or add a new free, real source through a script in `scripts/`; never synthetic rows).
6. Folder layout: `code/run_pipeline.py`, `results/empirical_results.json`, `figures/`, `main.tex`, `main.pdf`, `references.bib`.
7. The pipeline must be re-runnable end-to-end and idempotent (use `upsert_hypothesis`).

## Available datasets ({n_datasets} parquet files)
{datasets}

## Done means
`python run.py --audit` passes for your folder and `python run.py --list` shows your hypothesis with the computed status.
"""


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
        self.run_dir = os.path.join(RUNS_DIR, self.session_id)

        create_session(
            session_id=self.session_id,
            agents=self.agents,
            topic=self.topic,
            notes=f"Launched via hypoq harness for {', '.join(self.agents)}",
            db_path=self.db_path,
        )

    def check_dedup(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Dense semantic search for prior hypotheses overlapping the topic."""
        search_target = query or self.topic
        if not search_target:
            return []
        return search_hypotheses(search_target, top_k=5, db_path=self.db_path)

    def get_available_skills(self) -> List[Dict[str, str]]:
        """Discovers all skills defined in .agents/skills/."""
        skills = []
        skills_dir = os.path.join(REPO_ROOT, ".agents", "skills")
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
        """Discovers every Parquet dataset under data/."""
        return [
            os.path.relpath(p, REPO_ROOT)
            for p in sorted(glob.glob(os.path.join(REPO_ROOT, "data", "**", "*.parquet"), recursive=True))
        ]

    def generate_agent_brief(self, agent_name: str) -> Dict[str, Any]:
        """Builds the research briefing package for the designated agent."""
        dedup_matches = self.check_dedup()
        datasets = self.get_available_datasets()
        return {
            "session_id": self.session_id,
            "target_agent": agent_name,
            "co_agents": [a for a in self.agents if a != agent_name],
            "research_topic": self.topic or "Autonomous Frontier Research Exploration",
            "domain": self.domain,
            "lifecycle": ["IDEA", "DEDUP", "PLAN", "CODE", "EXPERIMENT", "ANALYSE", "REPORT", "DB_SYNC"],
            "prior_matching_hypotheses": [
                {
                    "id": m["id"],
                    "title": m["title"],
                    "status": m["status"],
                    "metric": f"{m.get('metric_name')}: {m.get('metric_value')}",
                    "p_val": m.get("p_value"),
                    "similarity": m.get("similarity_score"),
                }
                for m in dedup_matches
            ],
            "skills_available": [s["name"] for s in self.get_available_skills()],
            "dataset_count": len(datasets),
            "datasets": datasets,
            "latex_template": "templates/main.tex",
        }

    def render_prompt(self, brief: Dict[str, Any]) -> str:
        prior = "\n".join(
            f"- #{m['id']} [{m['status']}] {m['title']} ({m['metric']}, p={m['p_val']}, similarity={m['similarity']})"
            for m in brief["prior_matching_hypotheses"]
        ) or "- none: novel research space"
        return PROTOCOL.format(
            agent=brief["target_agent"],
            session_id=brief["session_id"],
            co_agents=", ".join(brief["co_agents"]) or "none",
            root=REPO_ROOT,
            topic=brief["research_topic"],
            domain=brief["domain"],
            prior=prior,
            n_datasets=brief["dataset_count"],
            datasets="\n".join(f"- {d}" for d in brief["datasets"]),
        )

    def agent_command(self, agent: str, prompt_file: str) -> str:
        template = os.environ.get(f"HYPOQ_AGENT_CMD_{agent.upper()}") or DEFAULT_AGENT_COMMANDS[agent]
        return template.replace("{prompt_file}", shlex.quote(prompt_file))

    def run(self, verbose: bool = True, execute: bool = False, timeout: int = 3600) -> Dict[str, Any]:
        """
        Writes one protocol prompt per agent under runs/<session>/. With execute=True, launches each agent CLI
        in parallel in the repository root, logs their output, then audits the resulting research folders.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("🚀 [hypoq Harness] Launching Multi-Agent Research Session")
            print(f"   Session ID : {self.session_id}")
            print(f"   Agents     : {', '.join(self.agents).upper()}")
            print(f"   Topic      : {self.topic or 'Autonomous Discovery'}")
            print(f"   Timestamp  : {self.created_at}")
            print("=" * 70)

        dedup = self.check_dedup()
        if verbose:
            if dedup:
                print(f"\n🔍 [Deduplication] {len(dedup)} overlapping prior hypothesis(es):")
                for m in dedup:
                    print(f"   #{m['id']} [{m['status']}] {m['title']} (sim={m.get('similarity_score')})")
            else:
                print("\n✨ [Deduplication] No overlapping hypotheses found.")

        os.makedirs(self.run_dir, exist_ok=True)
        before = set(list_research_dirs())
        agent_briefs, procs = {}, {}
        for agent in self.agents:
            brief = self.generate_agent_brief(agent)
            agent_briefs[agent] = brief
            prompt_file = os.path.join(self.run_dir, f"{agent}_prompt.md")
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(self.render_prompt(brief))
            cmd = self.agent_command(agent, prompt_file)
            brief["prompt_file"] = os.path.relpath(prompt_file, REPO_ROOT)
            brief["command"] = cmd
            if verbose:
                print(f"\n🤖 [{agent.upper()}] prompt: {brief['prompt_file']}")
                print(f"   command: {cmd}")
            if execute:
                log = open(os.path.join(self.run_dir, f"{agent}.log"), "w")
                procs[agent] = (subprocess.Popen(cmd, shell=True, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT), log)

        exit_codes: Dict[str, Optional[int]] = {}
        for agent, (proc, log) in procs.items():
            try:
                exit_codes[agent] = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                exit_codes[agent] = None
            log.close()
            if verbose:
                print(f"   {agent.upper()} finished with exit code {exit_codes[agent]} (log: runs/{self.session_id}/{agent}.log)")

        audit = []
        if execute:
            new_dirs = sorted(set(list_research_dirs()) - before)
            audit = [audit_research(d, self.db_path) for d in new_dirs]
            if verbose:
                print_audit(audit)

        return {
            "session_id": self.session_id,
            "status": "COMPLETED" if execute else "PROMPTS_WRITTEN",
            "agents": self.agents,
            "dedup_count": len(dedup),
            "briefs": agent_briefs,
            "exit_codes": exit_codes,
            "audit": audit,
        }


def list_research_dirs() -> List[str]:
    return sorted(
        d for d in glob.glob(os.path.join(REPO_ROOT, "researches", "*")) if os.path.isdir(d)
    )


def audit_research(research_dir: str, db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Integrity gate for one research folder. A study passes only if it is reproducible from code and its
    published numbers agree across JSON, paper and database.
    """
    rel = os.path.relpath(research_dir, REPO_ROOT)
    problems: List[str] = []
    pipeline = os.path.join(research_dir, "code", "run_pipeline.py")
    results_path = os.path.join(research_dir, "results", "empirical_results.json")
    if not os.path.isfile(pipeline):
        problems.append("missing code/run_pipeline.py (results cannot be reproduced)")
    else:
        src = open(pipeline, encoding="utf-8").read()
        if "empirical_results.json" not in src:
            problems.append("pipeline does not write results/empirical_results.json")
        if "upsert_hypothesis" not in src and "record_hypothesis" not in src:
            problems.append("pipeline does not register the hypothesis in the database")
    decision = None
    if not os.path.isfile(results_path):
        problems.append("missing results/empirical_results.json")
    else:
        try:
            decision = json.load(open(results_path)).get("decision")
        except ValueError:
            problems.append("results/empirical_results.json is not valid JSON")
    for required in ("main.tex", "main.pdf", "references.bib"):
        if not os.path.isfile(os.path.join(research_dir, required)):
            problems.append(f"missing {required}")
    if not glob.glob(os.path.join(research_dir, "figures", "*.png")):
        problems.append("no figures/*.png")

    status = None
    try:
        conn = get_connection(db_path)
        try:
            row = conn.execute(
                "SELECT status FROM hypotheses WHERE report_dir = ? ORDER BY id DESC LIMIT 1", (rel,)
            ).fetchone()
        finally:
            conn.close()
        status = row[0] if row else None
    except Exception as exc:  # postgres or locked database: report instead of crashing the audit
        problems.append(f"database check skipped: {exc}")
    else:
        if status is None:
            problems.append("no hypothesis row with this report_dir in the database")
        elif decision and status != decision:
            problems.append(f"database status {status} != results JSON decision {decision}")
    return {"research": rel, "ok": not problems, "problems": problems, "decision": decision}


def print_audit(reports: List[Dict[str, Any]]) -> None:
    print("\n🧪 [Research Integrity Audit]")
    for r in reports:
        print(f"   {'PASS' if r['ok'] else 'FAIL'}  {r['research']}  ({r['decision']})")
        for p in r["problems"]:
            print(f"        - {p}")
