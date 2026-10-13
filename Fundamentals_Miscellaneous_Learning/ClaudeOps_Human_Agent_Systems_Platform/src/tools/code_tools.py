"""
ClaudeOps: Sandboxed Code Tools.

Implements native tools for Claude:
- Context-efficient line-range reading (Claude Platform 101 token management)
- Unified diff generation
- Sandboxed patch application
"""

import difflib
from typing import Dict, Any, List, Optional


class SandboxedCodeTools:
    """Virtual sandboxed file system tools for the AI agent."""

    def __init__(self, initial_files: Optional[Dict[str, str]] = None):
        self.virtual_fs: Dict[str, str] = initial_files or {
            "src/api/auth.py": (
                "def authenticate(username, password):\n"
                "    # INSECURE: vulnerable to timing attack and missing rate limit\n"
                "    if password == 'admin_secret':\n"
                "        return {'token': 'jwt_mock_token_123'}\n"
                "    return None\n"
            ),
            "src/models/user.py": (
                "class User:\n"
                "    def __init__(self, id, name):\n"
                "        self.id = id\n"
                "        self.name = name\n"
            )
        }

    def read_file(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Reads file contents with optional line range slicing."""
        file_path = args.get("file_path", "")
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line", None)

        if file_path not in self.virtual_fs:
            return {"error": f"File '{file_path}' not found.", "is_error": True}

        lines = self.virtual_fs[file_path].splitlines()
        selected_lines = lines[start_line - 1 : end_line] if end_line else lines[start_line - 1 :]

        return {
            "file_path": file_path,
            "total_lines": len(lines),
            "displayed_lines": len(selected_lines),
            "content": "\n".join(selected_lines)
        }

    def apply_patch(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Applies a patch safely to the virtual sandbox."""
        file_path = args.get("file_path", "")
        patch_description = args.get("patch", "")

        if file_path not in self.virtual_fs:
            return {"error": f"File '{file_path}' not found.", "is_error": True}

        old_content = self.virtual_fs[file_path]

        # Simulate applying a secure hardening patch
        new_content = (
            "import hmac\n"
            "from src.core.rate_limiter import check_rate_limit\n\n"
            "def authenticate(username, password):\n"
            "    check_rate_limit(username)\n"
            "    # Constant-time comparison prevents timing attacks\n"
            "    if hmac.compare_digest(password, 'admin_secret'):\n"
            "        return {'token': 'jwt_mock_token_123'}\n"
            "    return None\n"
        )
        self.virtual_fs[file_path] = new_content

        # Generate diff
        diff_lines = list(difflib.unified_diff(
            old_content.splitlines(),
            new_content.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        ))

        return {
            "status": "PATCH_APPLIED_SUCCESSFULLY",
            "file_path": file_path,
            "description": patch_description,
            "unified_diff": "\n".join(diff_lines)
        }
