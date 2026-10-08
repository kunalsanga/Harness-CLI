import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from harness_core.providers.groq import GroqProvider
from harness_core.providers.base import CompletionRequest, ModelInfo
from harness_core.routing.router import ModelRouter
from harness_core.cli.interactive import InteractiveShell
from collections import namedtuple

@pytest.mark.asyncio
async def test_groq_list_models_parsing():
    p = GroqProvider(api_key="test")
    client_mock = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"id": "groq/compound", "context_window": 131072},
            {"id": "openai/gpt-oss-20b"} # no context window
        ]
    }
    client_mock.get.return_value = mock_resp
    p._get_client = AsyncMock(return_value=client_mock)
    
    models = await p.list_models()
    assert len(models) == 2
    assert models[0].id == "groq/compound"
    assert models[0].provider == "groq"
    assert models[0].context_window == 131072
    assert models[1].id == "openai/gpt-oss-20b"
    assert models[1].context_window == 8192

@pytest.mark.asyncio
async def test_groq_list_models_empty():
    p = GroqProvider(api_key="test")
    client_mock = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    client_mock.get.return_value = mock_resp
    p._get_client = AsyncMock(return_value=client_mock)
    
    models = await p.list_models()
    assert len(models) == 0

@pytest.mark.asyncio
async def test_interactive_model_validation():
    shell = InteractiveShell(plain=True)
    shell._router = AsyncMock()
    shell._router.providers = {"groq": MagicMock()}
    
    # Mock refresh_models to return a known list
    shell._router.refresh_models.return_value = [
        ModelInfo(id="compound", name="compound", provider="groq", context_window=8192, supports_tools=True, cost_per_1k_input=0, cost_per_1k_output=0, is_free=False)
    ]
    
    shell.console = MagicMock()
    
    # Test rejection (invalid provider)
    await shell._cmd_model("invalid/model")
    assert shell.current_model == ""
    
    # Test rejection (invalid model for provider)
    await shell._cmd_model("groq/nonexistent")
    assert shell.current_model == ""
    
    # Test rejection (incomplete)
    await shell._cmd_model("groq/")
    assert shell.current_model == ""
    
    # Test acceptance
    await shell._cmd_model("groq/compound")
    assert shell.current_model == "groq/compound"

@pytest.mark.asyncio
async def test_model_router_explicit_pinning():
    provider_mock = MagicMock()
    provider_mock.name = "groq"
    
    config = MagicMock()
    config.routing_mode = "auto"
    config.scoring_weights = {"cost": 1.0}
    config.max_fallback_chain = 6
    config.budget = None
        
    router = ModelRouter(providers=[provider_mock], config=config)
    router.refresh_models = AsyncMock(return_value=[
        ModelInfo(id="compound", name="compound", provider="groq", context_window=8192, supports_tools=True, cost_per_1k_input=0, cost_per_1k_output=0, is_free=False)
    ])
    
    req = CompletionRequest(
        model="groq/compound",
        messages=[]
    )
    
    chain = await router.select_models(req)
    assert len(chain) == 1
    assert chain[0][0] == "compound"
    assert chain[0][1] == provider_mock
