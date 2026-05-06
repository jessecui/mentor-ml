"""
MCP-backed tools for the MentorML agent.

The retrieval tool runs in a subprocess (mcp_server/diagram_server.py) over
stdio. This module owns the connection lifecycle and exposes
load_persistent_mcp_tools(), which returns LangChain-compatible tool
instances bound to that session.

Tool returns are flattened from MCP content blocks into a JSON string for
ToolMessage.content.
"""

import json
import os
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.sessions import StdioConnection, create_session
from langchain_mcp_adapters.tools import load_mcp_tools as _load_mcp_tools

PROJECT_ROOT = Path(__file__).parent.parent


def _connection() -> StdioConnection:
    """Describe how to launch the diagram MCP server as a stdio subprocess.

    Specifies the executable, module entrypoint, working directory, and
    environment for the subprocess. Consumed by `create_session()` at startup
    to open the persistent ClientSession that backs all tool calls.
    """
    return StdioConnection(
        transport="stdio",
        command=sys.executable,
        args=["-m", "mcp_server.diagram_server"],
        cwd=str(PROJECT_ROOT),
        env=dict(os.environ),
    )


def _flatten_content_blocks(blocks) -> str:
    """Flatten LangChain content blocks (list of dicts) into a JSON string."""
    if isinstance(blocks, str):
        return blocks
    if isinstance(blocks, list):
        parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        return "".join(parts)
    return json.dumps(blocks)


def _wrap_tool(mcp_tool: BaseTool) -> BaseTool:
    """Wrap an MCP-loaded LangChain tool so it returns a flat JSON string."""
    async def _arun(**kwargs):
        result = await mcp_tool.ainvoke(kwargs)
        return _flatten_content_blocks(result)

    def _run(**kwargs):
        result = mcp_tool.invoke(kwargs)
        return _flatten_content_blocks(result)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=mcp_tool.name,
        description=mcp_tool.description,
        args_schema=mcp_tool.args_schema,
    )


async def load_persistent_mcp_tools(exit_stack: AsyncExitStack) -> list[BaseTool]:
    """Open a persistent stdio session and load wrapped LangChain tools.

    The session (and the underlying subprocess) stay alive until exit_stack
    is closed. Caller is responsible for closing the stack on shutdown.
    """
    session = await exit_stack.enter_async_context(create_session(_connection()))
    await session.initialize()
    raw_tools = await _load_mcp_tools(session)
    return [_wrap_tool(t) for t in raw_tools]
