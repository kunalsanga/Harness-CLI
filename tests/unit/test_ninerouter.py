import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from harness_core.providers.ninerouter import NineRouterProvider
from harness_core.config.credentials import CredentialResolver
from harness_core.cli.interactive import InteractiveShell

def test_credential_resolution():
    with patch.dict(os.environ, {"NINE_ROUTER_API_KEY": "test-key-123"}):
        resolver = CredentialResolver()
        cred = resolver.resolve("9router")
        assert cred.api_key == "test-key-123"

@pytest.mark.asyncio
async def test_ninerouter_list_models_parsing():
    provider = NineRouterProvider(api_key="test")
    client_mock = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "object": "list",
        "data": [
            {
                "id": "ag/gemini-3.8-flash",
                "capabilities": {"tools": True},
                "context_length": 1048576,
                "max_completion_tokens": 65536
            },
            {
                "id": "claude-fable",
                "capabilities": {"tools": False},
                "context_window": 8192
            }
        ]
    }
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    provider._get_client = MagicMock(return_value=client_mock)
    
    models = await provider.list_models()
    assert len(models) == 2
    assert models[0].id == "ag/gemini-3.8-flash"
    assert models[0].name == "gemini-3.8-flash"
    assert models[0].supports_tools is True
    assert models[0].context_window == 1048576
    assert models[1].id == "claude-fable"
    assert models[1].name == "claude-fable"
    assert models[1].supports_tools is False
    assert models[1].context_window == 8192

@pytest.mark.asyncio
async def test_ninerouter_list_models_error():
    provider = NineRouterProvider(api_key="test")
    client_mock = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    client_mock.get.return_value = mock_resp
    client_mock.__aenter__.return_value = client_mock
    provider._get_client = MagicMock(return_value=client_mock)
    
    models = await provider.list_models()
    assert len(models) == 0

@pytest.mark.asyncio
async def test_interactive_registration_and_pinning():
    shell = InteractiveShell(plain=True)
    shell._router = AsyncMock()
    shell._router.providers = {"9router": NineRouterProvider(api_key="test")}
    
    # Mock refresh_models to return 9router models
    from harness_core.providers.base import ModelInfo
    shell._router.refresh_models.return_value = [
        ModelInfo(
            id="ag/gemini-3.8-flash", 
            name="gemini", 
            provider="9router", 
            context_window=128000, 
            supports_tools=True, 
            cost_per_1k_input=0, 
            cost_per_1k_output=0, 
            is_free=False
        )
    ]
    
    shell.console = MagicMock()
    
    # Test acceptance
    await shell._cmd_model("9router/ag/gemini-3.8-flash")
    assert shell.current_model == "9router/ag/gemini-3.8-flash"
