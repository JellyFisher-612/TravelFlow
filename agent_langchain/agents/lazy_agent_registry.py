#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
懒加载智能体注册器
基于正式 Python 模块装配生产智能体，保留 .claude/skills 作为能力说明目录
"""
import sys
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, ItemsView, Optional, ValuesView
from rich.console import Console

logger = logging.getLogger(__name__)

class LazyAgentRegistry:
    """
    懒加载智能体注册器。
    
    生产智能体优先从正式模块装配；.claude/skills 只用于发现
    SKILL.md 能力说明，并为旧入口保留兼容回退。
    """

    def __init__(
        self,
        model: Any,
        cache: Dict[str, Any],
        memory_manager: Any = None,
        event_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        初始化懒加载注册器

        Args:
            model: 共享的 LLM 模型实例
            cache: 用于缓存已加载智能体的字典
            memory_manager: 记忆管理器 (可选，用于注入给需要它的 Agent)
        """
        self.model = model
        self.cache = cache
        self.memory_manager = memory_manager
        self.event_callback = event_callback
        self.console = Console()
        
        # 能力说明目录路径
        self.skills_root = Path(__file__).resolve().parents[1] / ".claude" / "skills"
        
        # 能力说明映射表: skill_name -> SKILL.md path
        self._skill_map: Dict[str, Path] = {}

        # 正式模块映射: skill_name -> import path
        self._formal_agent_classes: Dict[str, str] = {
            "memory": "agents.travelflow_agents.MemoryAgent",
            "search": "agents.travelflow_agents.SearchAgent",
            "plan": "agents.travelflow_agents.PlanAgent",
            "clarification": "agents.travelflow_agents.ClarificationAgent",
            "preference": "agents.preference_agent.PreferenceAgent",
            "memory_query": "context.memory_query.MemoryQueryAgent",
            "memory-query": "context.memory_query.MemoryQueryAgent",
            "information_query": "agents.travelflow_agents.SearchAgent",
            "query-info": "agents.search_agent.InformationQueryAgent",
            "itinerary_planning": "agents.travelflow_agents.PlanAgent",
            "plan-trip": "agents.plan_agent.ItineraryPlanningAgent",
        }
        self._hidden_legacy_aliases: Dict[str, str] = {
            "event_collection": "clarification",
            "event-collection": "clarification",
        }
        
        # 发现能力说明
        self._discover_skills()
        
        # 旧版兼容映射 (name -> skill_folder_name)
        self._legacy_mapping = {
            "memory": "memory",
            "search": "search",
            "plan": "plan",
            "clarification": "clarification",
            "memory_query": "memory-query",
            "preference": "preference",
            "information_query": "query-info",
            "itinerary_planning": "plan-trip",
        }

    def _discover_skills(self):
        """扫描 .claude/skills 目录寻找可用的能力说明。"""
        if not self.skills_root.exists():
            self.console.print(f"[yellow]Warning: Skills directory {self.skills_root} not found[/yellow]")
            return

        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            
            skill_doc = skill_dir / "SKILL.md"
            if skill_doc.exists():
                self._skill_map[skill_dir.name] = skill_doc
                

    def _resolve_agent_name(self, agent_name: str) -> Optional[str]:
        """解析智能体名称到技能目录名"""
        if agent_name in self._formal_agent_classes:
            return agent_name
        if agent_name in self._hidden_legacy_aliases:
            return self._hidden_legacy_aliases[agent_name]

        # 1. 直接匹配技能名
        if agent_name in self._skill_map:
            return agent_name
            
        # 2. 尝试遗留映射
        if agent_name in self._legacy_mapping:
            skill_name = self._legacy_mapping[agent_name]
            if skill_name in self._skill_map:
                return skill_name
                
        return None

    def _emit_event(self, message: str):
        """向外部透传加载事件（例如 Web UI）。"""
        if self.event_callback:
            try:
                self.event_callback(message)
            except Exception:
                logger.debug("LazyAgentRegistry event callback failed", exc_info=True)

    def __getitem__(self, agent_name: str):
        """获取智能体 (懒加载)"""
        if agent_name in self.cache:
            return self.cache[agent_name]

        skill_name = self._resolve_agent_name(agent_name)
        if not skill_name:
             raise KeyError(f"Agent '{agent_name}' not found in registry")

        loading_msg = f"🔄 正在加载 {agent_name} (from {skill_name})..."
        self.console.print(f"[dim]{loading_msg}[/dim]")
        self._emit_event(loading_msg)
        
        try:
            agent_class = self._load_agent_class(skill_name)

            init_params = {
                "name": agent_name,
                "model": self.model,
            }

            sig = inspect.signature(agent_class.__init__)
            if "memory_manager" in sig.parameters:
                init_params["memory_manager"] = self.memory_manager

            agent_instance = agent_class(**init_params)
            self.cache[agent_name] = agent_instance
            loaded_msg = f"✓ {agent_name} 加载完成"
            self.console.print(f"[dim]{loaded_msg}[/dim]")
            self._emit_event(loaded_msg)

            return agent_instance
        except Exception as e:
            err_msg = f"✗ 加载 {agent_name} 失败: {e}"
            self.console.print(f"[red]{err_msg}[/red]")
            self._emit_event(err_msg)
            logger.warning("Failed to load agent %s", agent_name, exc_info=True)
            raise

    def _load_agent_class(self, skill_name: str):
        class_path = self._formal_agent_classes.get(skill_name)
        if class_path:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            agent_class = getattr(module, class_name)
            if inspect.isclass(agent_class) and hasattr(agent_class, "run"):
                return agent_class
            raise TypeError(f"Configured class {class_path} does not implement run(state)")

        skill_doc = self._skill_map[skill_name]
        script_path = skill_doc.parent / "script" / "agent.py"
        if not script_path.exists():
            raise KeyError(f"Agent '{skill_name}' has no formal class or legacy script")
        module_name = f"skills.{skill_name}.agent"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec from {script_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        project_root = str(Path(__file__).parent.parent.absolute())
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        spec.loader.exec_module(module)
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "run"):
                return obj
        raise ValueError(f"No runnable agent class found in {script_path}")

    def __contains__(self, agent_name: str) -> bool:
        return self._resolve_agent_name(agent_name) is not None or agent_name in self.cache

    def get(self, agent_name: str, default: Any = None) -> Any:
        try:
            return self[agent_name]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        # 返回所有可能的 key（包括 legacy mapping 的 key，为了兼容 orchestrator）
        keys = set(self._skill_map.keys()) | set(self._formal_agent_classes.keys())
        for legacy_key, skill_val in self._legacy_mapping.items():
            if skill_val in self._skill_map:
                keys.add(legacy_key)
        return list(keys)

    def values(self) -> ValuesView[Any]:
        return self.cache.values()

    def items(self) -> ItemsView[str, Any]:
        return self.cache.items()
        
    def get_loaded_agents(self) -> list:
        return list(self.cache.keys())
