import pytest

from harness_core.agents.domain import AgentRole, WorkspaceScope, AgentContract
from harness_core.agents.registry import AgentRegistry, AgentProfile
from harness_core.agents.worker import WorkerAgent
from harness_core.observability.events import EventBus

@pytest.fixture
def registry():
    return AgentRegistry()

def test_all_specialist_profiles_exist(registry):
    """Test that every required specialist profile exists and is enabled."""
    expected_roles = [
        AgentRole.ARCHITECT,
        AgentRole.RESEARCHER,
        AgentRole.UI_DESIGNER,
        AgentRole.FRONTEND,
        AgentRole.BACKEND,
        AgentRole.DATABASE,
        AgentRole.INTEGRATION,
        AgentRole.TESTER,
        AgentRole.DEBUGGER,
        AgentRole.REVIEWER,
        AgentRole.SECURITY_REVIEWER,
        AgentRole.VERIFIER,
        AgentRole.GIT_RELEASE
    ]
    
    for role in expected_roles:
        profile = registry.get_default_for_role(role)
        assert profile is not None, f"Missing profile for role {role.value}"
        assert profile.enabled is True

def test_profile_capabilities(registry):
    """Test that profiles contain valid capabilities."""
    frontend = registry.get_default_for_role(AgentRole.FRONTEND)
    assert "frontend" in frontend.capabilities
    assert "styling" in frontend.capabilities
    
    database = registry.get_default_for_role(AgentRole.DATABASE)
    assert "schema_design" in database.capabilities

def test_reviewer_read_only(registry):
    """Test that reviewer has READ_ONLY scope and denies write tools."""
    reviewer = registry.get_default_for_role(AgentRole.REVIEWER)
    assert reviewer.workspace_scope == WorkspaceScope.READ_ONLY
    
    # Denied tools should explicitly block writes
    assert "write_file" in reviewer.denied_tools
    assert "edit_file" in reviewer.denied_tools
    assert "delete_file" in reviewer.denied_tools
    assert "git_commit" in reviewer.denied_tools

def test_model_policy_mapping(registry):
    """Test that model policy maps correctly based on role."""
    architect = registry.get_default_for_role(AgentRole.ARCHITECT)
    assert architect.model_policy == "reasoning_high"
    
    researcher = registry.get_default_for_role(AgentRole.RESEARCHER)
    assert researcher.model_policy == "fast"
    
    ui_designer = registry.get_default_for_role(AgentRole.UI_DESIGNER)
    assert ui_designer.model_policy == "coding"
    
    verifier = registry.get_default_for_role(AgentRole.VERIFIER)
    assert verifier.model_policy == "deterministic"

@pytest.mark.asyncio
async def test_worker_tool_filtering():
    """Test that WorkerAgent filters tools correctly based on allowed/denied."""
    
    class MockTool:
        def __init__(self, name):
            self.schema = type("Schema", (), {"name": name})()
            self.name = name

    tools = [
        MockTool("read_file"),
        MockTool("write_file"),
        MockTool("delete_file"),
        MockTool("run_command")
    ]
    
    # Restrict to read_file and run_command
    contract = AgentContract(
        role=AgentRole.REVIEWER,
        allowed_tools=["read_file", "write_file", "run_command"],
        denied_tools=["write_file", "delete_file"]
    )
    
    worker = WorkerAgent(
        contract=contract,
        provider=None,
        tools=tools,
        event_bus=EventBus(),
        workspace_path="."
    )
    
    filtered = worker._filter_tools()
    filtered_names = [t.schema.name for t in filtered]
    
    assert "read_file" in filtered_names
    assert "run_command" in filtered_names
    assert "write_file" not in filtered_names  # Excluded by denied_tools
    assert "delete_file" not in filtered_names # Excluded by allowed_tools and denied_tools

@pytest.mark.asyncio
async def test_worker_prompt_building():
    """Test that WorkerAgent includes system instructions and scope in prompt."""
    contract = AgentContract(
        role=AgentRole.FRONTEND,
        objective="Build dashboard",
        workspace_scope=WorkspaceScope.FRONTEND,
        system_instructions="You are a frontend engineer.",
        output_requirements=["Generate React component"]
    )
    
    worker = WorkerAgent(
        contract=contract,
        provider=None,
        tools=[],
        event_bus=EventBus(),
        workspace_path="."
    )
    
    prompt = await worker._build_prompt_async()
    
    assert "You are a specialized agent: frontend" in prompt
    assert "Objective: Build dashboard" in prompt
    assert "Workspace Scope: FRONTEND" in prompt
    assert "You are a frontend engineer." in prompt
    assert "Output Requirements:" in prompt
    assert "Generate React component" in prompt
