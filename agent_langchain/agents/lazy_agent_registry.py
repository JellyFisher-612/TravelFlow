#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
懒加载智能体注册器
基于 .claude/skills 目录结构的插件化加载机制
"""
import os
import sys
import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from rich.console import Console

class LazyAgentRegistry:
    """
    懒加载智能体注册器 - 插件化版本
    
    自动扫描 .claude/skills 下的技能目录，保留技能发现能力；
    对核心生产智能体优先从正式模块装配，其余能力仍可回退到脚本动态加载。
    """

    def __init__(self, model, cache: Dict, memory_manager=None, event_callback: Optional[Callable[[str], None]] = None):
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
        
        # 技能目录路径
        self.skills_root = Path(".claude/skills")
        
        # 技能映射表: skill_name -> agent_script_path
        self._skill_map: Dict[str, Path] = {}

        # 正式模块映射: skill_name -> import path
        self._formal_agent_classes: Dict[str, str] = {
            "memory": "agents.travelflow_agents.MemoryAgent",
            "search": "agents.travelflow_agents.SearchAgent",
            "plan": "agents.travelflow_agents.PlanAgent",
            "clarification": "agents.travelflow_agents.ClarificationAgent",
            "preference": "agents.preference_agent.PreferenceAgent",
        }
        
        # 发现技能
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
            "event_collection": "event-collection"
        }

    def _discover_skills(self):
        """扫描 .claude/skills 目录寻找可用的 Agent"""
        if not self.skills_root.exists():
            self.console.print(f"[yellow]Warning: Skills directory {self.skills_root} not found[/yellow]")
            return

        count = 0
        for skill_dir in self.skills_root.iterdir():
            if not skill_dir.is_dir():
                continue
            
            # 查找 script/agent.py
            agent_script = skill_dir / "script" / "agent.py"
            if agent_script.exists():
                skill_name = skill_dir.name
                self._skill_map[skill_name] = agent_script
                count += 1
                
        # self.console.print(f"[dim]已发现 {count} 个技能插件[/dim]")

    def _resolve_agent_name(self, agent_name: str) -> Optional[str]:
        """解析智能体名称到技能目录名"""
        if agent_name in self._formal_agent_classes:
            return agent_name

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
                # 不影响主流程
                pass

    def __getitem__(self, agent_name: str):
        """获取智能体 (懒加载)"""
        if agent_name in self.cache:
            return self.cache[agent_name]

        skill_name = self._resolve_agent_name(agent_name)
        if not skill_name:
             raise KeyError(f"Agent '{agent_name}' not found in skills directory")

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
            import traceback
            traceback.print_exc()
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

        script_path = self._skill_map[skill_name]
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

    def get(self, agent_name: str, default=None):
        try:
            return self[agent_name]
        except KeyError:
            return default

    def keys(self):
        # 返回所有可能的 key（包括 legacy mapping 的 key，为了兼容 orchestrator）
        keys = set(self._skill_map.keys()) | set(self._formal_agent_classes.keys())
        for legacy_key, skill_val in self._legacy_mapping.items():
            if skill_val in self._skill_map:
                keys.add(legacy_key)
        return list(keys)

    def values(self):
        return self.cache.values()

    def items(self):
        return self.cache.items()
        
    def get_loaded_agents(self) -> list:
        return list(self.cache.keys())
