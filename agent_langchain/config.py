"""Configuration for the TravelFlow Multi-Agent System."""

import os

# LLM Configuration
LLM_CONFIG = {
    "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
    "model_name": os.getenv("LLM_MODEL_NAME", "deepseek-v4-flash"),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "8192")),
}

# System Configuration
SYSTEM_CONFIG = {
    "enable_llm": True,  # Set to True to use LLM (recommended), False for rule-based
    "log_level": "INFO",
    "max_retries": 3,
    "timeout": 60,  # Increased timeout for better stability
}

# 高德地图 API 配置（用于 POI/天气等查询）
AMAP_CONFIG = {
    # 优先使用环境变量 AMAP_MAPS_API_KEY / AMAP_API_KEY
    "api_key": os.getenv("AMAP_MAPS_API_KEY") or os.getenv("AMAP_API_KEY", ""),
    "base_url": "https://restapi.amap.com",
}

# 连接与可用性：重试、熔断、健康检查
RESILIENCE_CONFIG = {
    "max_retries": 3,              # 单次请求最大重试次数（与 SYSTEM_CONFIG 对齐）
    "retry_base_delay_sec": 1.0,   # 重试退避基数（秒）
    "retry_max_delay_sec": 30.0,   # 重试退避上限（秒）
    "circuit_failure_threshold": 5, # 连续失败多少次后熔断
    "circuit_recovery_timeout_sec": 60.0,  # 熔断后多少秒进入半开
    "circuit_half_open_successes": 2,      # 半开状态下连续成功多少次后关闭
    "health_check_timeout_sec": 10.0,      # 健康检查请求超时（秒）
}

# LangSmith Tracing 配置（默认关闭，按需通过环境变量开启）
LANGSMITH_CONFIG = {
    "enabled": os.getenv("LANGSMITH_TRACING", "false").lower() in {"1", "true", "yes", "on"},
    "api_key": os.getenv("LANGSMITH_API_KEY", ""),
    # 可选：欧盟端可改为 https://api.smith.langchain.com
    "endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
    "project": os.getenv("LANGSMITH_PROJECT", "travelflow-travel-agent"),
}

# 记忆系统配置：Redis 缓存 + PostgreSQL 持久化
MEMORY_CONFIG = {
    # PostgreSQL: 例如 postgresql://user:password@127.0.0.1:5432/travelflow
    # 默认留空：未配置时自动走 JSON fallback
    "postgres_dsn": os.getenv("POSTGRES_DSN", ""),
    # Redis: 例如 redis://127.0.0.1:6379/0
    # 默认留空：未配置时不启用缓存
    "redis_url": os.getenv("REDIS_URL", ""),
    "cache_ttl_sec": int(os.getenv("MEMORY_CACHE_TTL_SEC", "3600")),
    # 当数据库不可用时，是否回退到本地 JSON（开发环境建议 True）
    "allow_json_fallback": os.getenv("ALLOW_JSON_FALLBACK", "true").lower() == "true",
}
