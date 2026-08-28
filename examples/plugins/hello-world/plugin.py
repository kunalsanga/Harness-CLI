"""Hello World plugin — minimal Harness extension example."""


def register_tools():
    """Return tool definitions for this plugin."""
    return [
        {
            "name": "hello_world",
            "description": "Returns a greeting message",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name to greet",
                        "default": "World",
                    }
                },
            },
        }
    ]


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool from this plugin."""
    if tool_name == "hello_world":
        name = arguments.get("name", "World")
        return {"content": f"Hello, {name}! From the Harness plugin system."}
    return {"error": f"Unknown tool: {tool_name}"}
