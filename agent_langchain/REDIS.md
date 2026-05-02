# Redis 管理指南

## 快速命令

```bash
# 设置别名（可选，加到 ~/.zshrc 或 ~/.bashrc）
alias redis-cli='/home/users/lhy/miniconda3/bin/redis-cli'
alias redis-server='/home/users/lhy/miniconda3/bin/redis-server'

# 启动
redis-server --port 6379 --daemonize yes --dir /home/users/lhy/redis-data

# 停止
redis-cli shutdown

# 检查是否运行
redis-cli ping
# 返回 PONG 表示正常
```

## 常用运维命令

### 查看状态

```bash
# 服务器信息
redis-cli info server

# 内存使用
redis-cli info memory | grep used_memory_human

# 当前连接数
redis-cli info clients | grep connected_clients

# 当前 key 数量
redis-cli dbsize
```

### 查看数据

```bash
# 列出所有 key（慎用，数据量大时会阻塞）
redis-cli keys '*'

# 按前缀查找（推荐）
redis-cli keys 'travelflow:*'

# 查看 key 类型
redis-cli type <key>

# 查看 string 类型的值
redis-cli get <key>

# 查看 hash 类型的所有字段
redis-cli hgetall <key>

# 查看 key 的剩余过期时间（秒）
redis-cli ttl <key>
```

### 清理数据

```bash
# 删除单个 key
redis-cli del <key>

# 按前缀批量删除（先确认再删）
redis-cli keys 'travelflow:stmem:*' | xargs redis-cli del

# 清空当前数据库（危险！）
redis-cli flushdb

# 清空所有数据库（更危险！）
redis-cli flushall
```

### 持久化

```bash
# 立即保存快照（RDB）
redis-cli bgsave

# 查看上次保存时间
redis-cli lastsave

# 手动触发 AOF 重写
redis-cli bgrewriteaof
```

## 日志与调试

```bash
# 查看日志
cat /home/users/lhy/redis-data/redis.log

# 实时监控所有命令（调试用，生产环境慎用）
redis-cli monitor

# 查看慢查询日志
redis-cli slowlog get 10
```

## TravelFlow 中的 Redis 用途

| Key 模式 | 用途 | TTL |
|----------|------|-----|
| `travelflow:stmem:{user_id}:{session_id}` | 短期对话记忆 | 由 `MEMORY_CACHE_TTL_SEC` 控制 |
| `travelflow:ltmem:pref:{user_id}` | 用户偏好热缓存 | 1 小时 |
| `travelflow:ltmem:session:{user_id}` | 会话元数据缓存 | 1 小时 |

## 故障排查

### Redis 连不上

```bash
# 检查进程是否存在
ps aux | grep redis-server

# 检查端口是否监听
ss -tlnp | grep 6379

# 检查日志有无报错
tail -20 /home/users/lhy/redis-data/redis.log
```

### 内存占用过高

```bash
# 查看内存
redis-cli info memory | grep -E 'used_memory_human|maxmemory_human'

# 设置最大内存限制（可选）
redis-cli config set maxmemory 256mb
redis-cli config set maxmemory-policy allkeys-lru

# 持久化配置到文件
redis-cli config rewrite
```

### 数据恢复

Redis 数据文件位于 `--dir` 指定的目录：

```bash
ls /home/users/lhy/redis-data/
# dump.rdb   — RDB 快照
# appendonlydir/ — AOF 日志
```

重启 redis-server 会自动加载这些文件恢复数据。
