"""Configuration for the TravelFlow Multi-Agent System."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

MIMO_MODEL_NAME = "mimo-v2.5-pro"
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_TOKEN_PLAN_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEEPSEEK_MODEL_NAME = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


def _default_mimo_base_url(api_key: str) -> str:
    if api_key.startswith("tp-"):
        return MIMO_TOKEN_PLAN_BASE_URL
    return MIMO_BASE_URL


def _resolve_llm_config(env=os.environ):
    """Resolve the OpenAI-compatible LLM configuration.

    MiMo is the preferred provider for this project. DeepSeek remains supported
    as a compatibility fallback for existing local environments.
    """

    mimo_api_key = env.get("MIMO_API_KEY", "")
    deepseek_api_key = env.get("DEEPSEEK_API_KEY", "")
    api_key = mimo_api_key or deepseek_api_key

    if mimo_api_key:
        default_model = MIMO_MODEL_NAME
        default_base_url = _default_mimo_base_url(mimo_api_key)
    else:
        default_model = DEEPSEEK_MODEL_NAME
        default_base_url = DEEPSEEK_BASE_URL

    model_name = env.get("LLM_MODEL_NAME", default_model)
    base_url = env.get("LLM_BASE_URL", default_base_url)

    if mimo_api_key:
        if model_name == DEEPSEEK_MODEL_NAME:
            model_name = default_model
        if base_url == DEEPSEEK_BASE_URL:
            base_url = default_base_url

    return {
        "api_key": api_key,
        "model_name": model_name,
        "base_url": base_url,
        "temperature": float(env.get("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(env.get("LLM_MAX_TOKENS", "8192")),
    }


# LLM Configuration
LLM_CONFIG = _resolve_llm_config()

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
