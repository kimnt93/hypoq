from hypoq.harness import DEFAULT_AGENT_COMMANDS, SUPPORTED_AGENTS, ResearchSession


def test_prompt_contains_integrity_rules_and_datasets(db_path):
    s = ResearchSession(agents=["claude"], topic="credit spreads", db_path=db_path)
    prompt = s.render_prompt(s.generate_agent_brief("claude"))
    assert "Never hand-type results" in prompt and "python run.py --audit" in prompt
    assert "data/market/us_equities_daily.parquet" in prompt


def test_unsupported_agents_are_rejected(db_path):
    import pytest

    with pytest.raises(ValueError):
        ResearchSession(agents=["gpt-nonexistent"], db_path=db_path)


def test_every_agent_has_a_command_and_env_override(db_path, monkeypatch):
    assert set(DEFAULT_AGENT_COMMANDS) == set(SUPPORTED_AGENTS)
    s = ResearchSession(agents=["claude"], db_path=db_path)
    assert "claude -p" in s.agent_command("claude", "/tmp/p.md")
    monkeypatch.setenv("HYPOQ_AGENT_CMD_CLAUDE", "echo {prompt_file}")
    assert s.agent_command("claude", "/tmp/p.md") == "echo /tmp/p.md"


def test_execute_runs_command_and_writes_log(db_path, monkeypatch):
    monkeypatch.setenv("HYPOQ_AGENT_CMD_CLAUDE", "echo ran-$((6*7))")
    s = ResearchSession(agents=["claude"], topic="t", db_path=db_path)
    result = s.run(verbose=False, execute=True, timeout=60)
    assert result["exit_codes"]["claude"] == 0
    import os
    assert "ran-42" in open(os.path.join(s.run_dir, "claude.log")).read()
    import shutil
    shutil.rmtree(s.run_dir, ignore_errors=True)
