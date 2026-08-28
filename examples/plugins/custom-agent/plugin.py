"""Security Reviewer agent plugin."""


def register_agent():
    """Return agent configuration for this plugin."""
    return {
        "name": "security-reviewer",
        "role": "REVIEWER",
        "description": "Reviews code for security vulnerabilities",
        "capabilities": ["security_analysis", "code_review"],
        "allowed_tools": ["read_file", "list_files", "grep"],
        "system_instructions": (
            "You are a security-focused code reviewer. "
            "Analyze code changes for security vulnerabilities including: "
            "SQL injection, XSS, command injection, path traversal, "
            "hardcoded secrets, insecure deserialization, and broken authentication. "
            "Return structured findings with severity levels."
        ),
        "preferred_task_types": ["security", "review"],
        "budget": {
            "max_iterations": 20,
            "max_tool_calls": 30,
        },
    }
