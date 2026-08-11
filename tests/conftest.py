import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_INFRA_MCP_DIR = Path(__file__).resolve().parent.parent / "infra_mcp"
if str(_INFRA_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_INFRA_MCP_DIR))
