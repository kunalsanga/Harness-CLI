"""Code Quality plugin — static analysis example."""


def register_tools():
    return [
        {
            "name": "check_complexity",
            "description": "Check function complexity in a Python file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                },
                "required": ["file_path"],
            },
        }
    ]


def execute_tool(tool_name: str, arguments: dict) -> dict:
    if tool_name == "check_complexity":
        file_path = arguments.get("file_path", "")
        # Simplified complexity check
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            lines = content.split("\n")
            long_functions = 0
            for i, line in enumerate(lines):
                if line.strip().startswith("def "):
                    # Count lines until next def or end
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith("def ") and not lines[j].strip().startswith("class "):
                        j += 1
                    if j - i > 50:
                        long_functions += 1
            return {
                "file": file_path,
                "total_lines": len(lines),
                "long_functions": long_functions,
                "verdict": "PASS" if long_functions == 0 else "WARN",
            }
        except Exception as e:
            return {"error": str(e)}
    return {"error": f"Unknown tool: {tool_name}"}
