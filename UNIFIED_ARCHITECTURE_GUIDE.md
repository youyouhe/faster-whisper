# 统一多GPU架构部署指南

## 概述

本文档描述了新的统一多GPU架构，它将原有的多实例部署方式整合为每个GPU运行一个unified_api.py实例的新架构。

## 架构对比

### 原架构 (start_multi_instance_per_gpu.sh)
```
GPU 0 ──┐
GPU 1 ──┼── Load Balancer (5001)
GPU 2 ──┘
```
- 每个GPU运行2个faster_whisper_api.py实例
- 通过load_balancer.py分发请求
- 共享内存管理较为复杂

### 新架构 (unified_api.py + deploy_multi_gpu.sh)
```
GPU 0 ── unified_api.py (5002) ──┐
GPU 1 ── unified_api.py (5003) ──┼── Load Balancer (5001)
GPU 2 ── unified_api.py (5004) ──┘
```
- 每个GPU运行1个unified_api.py实例
- unified_api.py内部管理多个worker进程
- 集成的共享内存管理
- 兼容现有的load_balancer.py接口

## 主要优势

1. **简化的架构**: 每个GPU只需要一个unified_api.py进程
2. **更好的资源管理**: 内置的master-worker架构
3. **统一的接口**: 与现有load_balancer.py完全兼容
4. **增强的监控**: 内置健康检查和统计信息
5. **优化的内存管理**: GPU特定的共享内存池

## 文件结构

### 核心文件
- `unified_api.py` - 统一API服务，包含/inference接口
- `master_process.py` - 主进程管理器
- `worker_process.py` - 工作进程
- `shared_memory_manager.py` - 共享内存管理

### 部署脚本
- `deploy_multi_gpu.sh` - 多GPU部署脚本（宿主机）
- `start_unified_multi_gpu.sh` - Docker内启动脚本
- `stop_unified.sh` - 停止脚本

### Docker配置
- `docker-compose-unified.yml` - 新的Docker Compose配置
- `docker/Dockerfile.unified` - 新的Docker镜像配置

## 部署方式

### 1. 宿主机部署
```bash
# 启动所有GPU实例
./deploy_multi_gpu.sh [workers_per_gpu] [max_file_size_mb] [model]

# 检查状态
./show_status.sh

# 停止所有实例
./stop_multi_gpu.sh
```

### 2. Docker部署
```bash
# 构建镜像
docker-compose -f docker-compose-unified.yml build

# 启动服务
docker-compose -f docker-compose-unified.yml up -d

# 查看日志
docker-compose -f docker-compose-unified.yml logs -f faster-whisper-unified

# 停止服务
docker-compose -f docker-compose-unified.yml down
```

## 配置参数

### 环境变量
- `WORKERS_PER_GPU` - 每GPU工作进程数 (默认: 2)
- `MAX_FILE_SIZE` - 最大文件大小MB (默认: 500)
- `WHISPER_MODEL` - Whisper模型 (默认: large-v3-turbo)
- `LOG_LEVEL` - 日志级别 (默认: INFO)
- `STARTUP_MODE` - 启动模式 (unified_multi_gpu)

### 命令行参数
```bash
python unified_api.py --help
--host HOST                 # 绑定主机地址
--port PORT                 # 绑定端口
--gpus GPU_IDS             # GPU ID列表 (逗号分隔)
--workers-per-gpu COUNT    # 每GPU工作进程数
--model MODEL              # Whisper模型
--log-level LEVEL          # 日志级别
--max-file-size SIZE       # 最大文件大小(MB)
```

## API接口

### 兼容接口 (/inference)
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "response_format=srt" \
  -F "language=auto" \
  http://localhost:5001/inference
```

响应格式:
```json
{
  "code": 0,
  "msg": "ok",
  "data": "SRT字幕内容"
}
```

### 原生接口 (/transcribe)
```bash
curl -X POST \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "response_format=json" \
  -F "language=auto" \
  http://localhost:5002/transcribe
```

### 状态接口
- `/health` - 健康检查
- `/stats` - 系统统计信息
- `/task/{task_id}` - 任务状态查询

## 监控和调试

### 健康检查
```bash
# 检查负载均衡器
curl http://localhost:5001/health

# 检查特定GPU实例
curl http://localhost:5002/health  # GPU 0
curl http://localhost:5003/health  # GPU 1
```

### 查看日志
```bash
# Docker环境
docker-compose -f docker-compose-unified.yml logs -f faster-whisper-unified

# 宿主机环境
tail -f logs/gpu_*/unified_api.log
```

### GPU监控
```bash
# 查看GPU使用情况
nvidia-smi

# 查看共享内存使用
ls -la /dev/shm/whisper_*
```

## 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 检查端口使用情况
   netstat -tlnp | grep 5002

   # 停止现有进程
   ./stop_multi_gpu.sh
   ```

2. **共享内存不足**
   ```bash
   # 清理共享内存
   rm -f /dev/shm/whisper_*

   # 调整共享内存池大小
   export SHARED_MEMORY_POOL_SIZE_MB=800
   ```

3. **GPU内存不足**
   ```bash
   # 减少工作进程数
   export WORKERS_PER_GPU=1

   # 使用较小的模型
   export WHISPER_MODEL=base
   ```

### 调试模式
```bash
# 启用调试日志
export LOG_LEVEL=DEBUG

# 查看详细错误信息
tail -f logs/gpu_*/unified_api.log | grep ERROR
```

## 性能优化

### 推荐配置

1. **高性能配置**
   ```bash
   WORKERS_PER_GPU=4
   MAX_FILE_SIZE=1000
   WHISPER_MODEL=large-v3-turbo
   ```

2. **低延迟配置**
   ```bash
   WORKERS_PER_GPU=2
   MAX_FILE_SIZE=100
   WHISPER_MODEL=medium
   ```

3. **资源节省配置**
   ```bash
   WORKERS_PER_GPU=1
   MAX_FILE_SIZE=200
   WHISPER_MODEL=base
   ```

### 性能监控
- 监控GPU利用率: `nvidia-smi`
- 监控内存使用: `free -h`
- 监控共享内存: `df -h /dev/shm`
- 监控API响应时间: 通过/stats接口

## 迁移指南

### 从旧架构迁移

1. **停止旧服务**
   ```bash
   docker-compose down  # 或相应的停止命令
   ```

2. **备份配置**
   ```bash
   cp docker-compose.yml docker-compose.yml.backup
   ```

3. **部署新架构**
   ```bash
   docker-compose -f docker-compose-unified.yml up -d
   ```

4. **验证服务**
   ```bash
   curl http://localhost:5001/health
   ```

### API兼容性
- 新架构完全兼容现有的/inference接口
- 客户端代码无需修改
- 负载均衡器配置保持不变

## 总结

新的统一多GPU架构提供了更简洁、更高效、更易维护的部署方式，同时保持了与现有系统的完全兼容性。通过集成master-worker架构和优化的共享内存管理，新架构在性能和可扩展性方面都有显著提升。