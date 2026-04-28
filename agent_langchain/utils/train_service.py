"""12306 MCP client for train ticket queries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrainService:
    """High-level wrapper around the 12306 MCP server."""

    def __init__(
        self,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
    ) -> None:
        self.command = command or os.getenv("TRAIN_MCP_COMMAND", "")
        raw_args = os.getenv("TRAIN_MCP_ARGS", "")
        if args is not None:
            self.args = args
        elif raw_args:
            self.args = raw_args.split()
        elif self.command:
            self.args = []
        else:
            local_bin = Path(__file__).resolve().parents[1] / ".mcp-node" / "node_modules" / ".bin" / "12306-mcp"
            if local_bin.exists():
                self.command = str(local_bin)
                self.args = []
            else:
                self.command = "npx"
                self.args = ["-y", "12306-mcp"]

    async def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError as exc:
            raise RuntimeError("缺少 MCP Python SDK，请先安装 requirements.txt 中的 mcp 依赖") from exc

        clean_args = {key: value for key, value in arguments.items() if value not in (None, "")}
        env = os.environ.copy()
        bundled_node_bin = Path("/Users/Jelly/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin")
        if bundled_node_bin.exists():
            env["PATH"] = f"{bundled_node_bin}{os.pathsep}{env.get('PATH', '')}"

        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, clean_args)
                return self._decode_tool_result(result)

    def _decode_tool_result(self, result: Any) -> Any:
        content = getattr(result, "content", None)
        if content is None:
            return result

        decoded_items: List[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is None:
                decoded_items.append(item)
                continue
            decoded_items.append(self._parse_text_payload(text))

        if len(decoded_items) == 1:
            return decoded_items[0]
        return decoded_items

    @staticmethod
    def _parse_text_payload(text: str) -> Any:
        stripped = (text or "").strip()
        if not stripped:
            return ""
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start_candidates = [idx for idx in (stripped.find("["), stripped.find("{")) if idx != -1]
            if not start_candidates:
                return stripped
            start = min(start_candidates)
            end = max(stripped.rfind("]"), stripped.rfind("}"))
            if end > start:
                try:
                    return json.loads(stripped[start : end + 1])
                except json.JSONDecodeError:
                    pass
            return stripped

    async def get_tickets(
        self,
        date: str,
        from_station: str,
        to_station: str,
        train_filter_flags: str = "",
        sort_flag: str = "",
        sort_reverse: bool = False,
        limited_num: int = 10,
    ) -> Any:
        """Query tickets by date and station names.

        The argument names match the 12306 MCP ``get-tickets`` tool.
        """

        return await self._call_tool(
            "get-tickets",
            {
                "date": date,
                "fromStation": from_station,
                "toStation": to_station,
                "trainFilterFlags": train_filter_flags,
                "sortFlag": sort_flag,
                "sortReverse": sort_reverse,
                "limitedNum": limited_num,
            },
        )
