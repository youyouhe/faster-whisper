# Multi-GPU Multi-Instance Docker Deployment

## 概述

这个Docker部署方案基于`docker-compose.multi-gpu-instance.yml`，实现了**单容器多GPU多实例**的Whisper ASR服务架构。一个Docker容器支持所有GPU，每个GPU运行多个Whisper实例，内置负载均衡器统一对外提供服务。

## 架构设计

```
┌─────────────────────────────────────────┐
│             Load Balancer                │ ← 统一API入口 (5001)
│            (Container)                   │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │  Multi-GPU Multi-Instance │ ← 单容器多GPU (5002-5004)
    │        Container          │
    │                           │
    │  ┌─────────────────────┐  │
    │  │     GPU 0,1,2       │  │
    │  │   (4 instances each) │  │
    │  │    Total: 12         │  │
    │  └─────────────────────┘  │
    └───────────────────────────┘
```

### 服务组件

1. **负载均衡器** (端口 5001) - **统一入口**
   - 唯一对外的API接口
   - 请求分发和负载均衡
   - 健康检查和故障转移
   - 自动路由到多GPU服务端口

2. **多GPU多实例容器** (端口 5002-5004)
   - 单容器支持所有GPU (GPU 0,1,2)
   - 每GPU运行4个Whisper实例 = 总共12个实例
   - GPU资源统一管理
   - 内部端口映射: 5002(GPU0), 5003(GPU1), 5004(GPU2)

3. **支持服务**
   - Redis: 消息队列
   - TUS服务器: 文件上传
   - TUS API服务器: 任务管理
   - ASR Worker: 后台处理
   - 回调服务: 结果通知

## 配置参数

### GPU配置
- **GPU数量**: 3个 (可配置: 0,1,2)
- **每GPU实例数**: 4个 (可配置)
- **总实例数**: 12个
- **GPU内存限制**: 48GB 总计 (单容器)

### 模型配置
- **模型**: large-v3-turbo
- **计算类型**: int8
- **设备**: cuda

### 性能配置
- **最大队列大小**: 100 (全局)
- **并发任务数**: 12 (总计)
- **请求超时**: 1800秒 (30分钟)

### 架构优势
- **统一入口**: 5001端口是唯一对外API接口
- **简化部署**: 2个容器代替多个容器
- **资源统一**: GPU资源在单个容器内集中管理
- **负载均衡**: 独立负载均衡器提供智能分发
- **高效通信**: 容器间通信优化

## 快速开始

### 1. 环境要求

- NVIDIA GPU (至少1个，推荐3个)
- CUDA 11.8+
- Docker & Docker Compose
- NVIDIA Docker Runtime (nvidia-docker2)
- 16GB+ GPU内存 per GPU

### 2. 安装NVIDIA Docker

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. 启动服务

```bash
# 使用默认配置启动
chmod +x start_multi_gpu_docker.sh
./start_multi_gpu_docker.sh

# 或使用自定义配置
GPU_COUNT=2 INSTANCES_PER_GPU=2 ./start_multi_gpu_docker.sh
```

### 4. 手动启动

```bash
# 构建并启动所有服务
docker-compose -f docker-compose.multi-gpu-instance.yml up --build

# 后台运行
docker-compose -f docker-compose.multi-gpu-instance.yml up -d --build

# 停止服务
docker-compose -f docker-compose.multi-gpu-instance.yml down
```

## API使用

### 主要端点

- **负载均衡器**: `http://localhost:5001`
- **健康检查**: `http://localhost:5001/health`
- **统计信息**: `http://localhost:5001/stats`
- **推理接口**: `http://localhost:5001/inference`

### 请求示例

```bash
# 音频文件转录
curl -X POST http://localhost:5001/inference \
  -H "Content-Type: multipart/form-data" \
  -F "audio=@audio.wav" \
  -H "X-API-Key: your-secret-api-key-here"
```

## 配置调优

### 1. GPU实例数调整

根据GPU内存调整每GPU实例数：

```bash
# 高内存GPU (>24GB)
INSTANCES_PER_GPU=6

# 中等内存GPU (16-24GB)
INSTANCES_PER_GPU=4

# 低内存GPU (<16GB)
INSTANCES_PER_GPU=2
```

### 2. 环境变量配置

在`docker-compose.multi-gpu-instance.yml`中调整：

```yaml
environment:
  - NUM_WHISPER_INSTANCES=4      # 每容器实例数
  - MAX_QUEUE_SIZE=20           # 队列大小
  - MAX_CONCURRENT_TASKS=4      # 并发任务数
  - REQUEST_TIMEOUT=1800        # 请求超时
  - WHISPER_MODEL=large-v3-turbo # 模型大小
```

### 3. 资源限制

```yaml
deploy:
  resources:
    limits:
      memory: 16G  # 每GPU容器内存限制
```

## 监控和日志

### 查看日志

```bash
# 所有服务日志
docker-compose -f docker-compose.multi-gpu-instance.yml logs -f

# 特定服务日志
docker-compose -f docker-compose.multi-gpu-instance.yml logs -f load-balancer
docker-compose -f docker-compose.multi-gpu-instance.yml logs -f faster-whisper-gpu0
```

### 性能监控

```bash
# GPU使用情况
nvidia-smi

# 负载均衡器状态
curl http://localhost:5001/health

# 容器资源使用
docker stats
```

## 故障排除

### 常见问题

1. **GPU内存不足**
   ```
   解决方案: 减少NUM_WHISPER_INSTANCES值
   ```

2. **容器启动失败**
   ```bash
   # 检查Docker日志
   docker-compose -f docker-compose.multi-gpu-instance.yml logs [service-name]

   # 检查GPU可用性
   nvidia-smi
   ```

3. **负载均衡器无法连接后端**
   ```bash
   # 检查网络连接
   docker network ls
   docker network inspect [network-name]
   ```

### 健康检查

```bash
# 检查所有服务状态
curl http://localhost:5001/health

# 检查特定GPU服务
curl http://localhost:5002/health  # GPU 0
curl http://localhost:5003/health  # GPU 1
curl http://localhost:5004/health  # GPU 2
```

## 扩展和定制

### 添加更多GPU

在`docker-compose.multi-gpu-instance.yml`中添加新的GPU服务：

```yaml
# GPU 3 服务
faster-whisper-gpu3:
  # ... 类似配置，使用GPU 3和端口5005
```

### 自定义负载均衡算法

修改`load_balancer.py`中的负载均衡逻辑：

```python
# 可选算法: round_robin, least_connections, random
LOAD_BALANCER_ALGORITHM=least_connections
```

### 集群部署

对于多机器部署，可以：
1. 在每台机器上运行GPU服务
2. 配置跨网络负载均衡器
3. 使用服务发现机制

## 性能基准

### 预期性能

- **单GPU (4实例)**: ~4-8 并发音频处理
- **3GPU (12实例)**: ~12-24 并发音频处理
- **吞吐量**: 取决于音频长度和GPU性能

### 建议配置

| GPU内存 | 每GPU实例数 | 总实例数 | 适用场景 |
|---------|------------|----------|----------|
| 8GB     | 1-2        | 3-6      | 轻量级部署 |
| 16GB    | 3-4        | 9-12     | 标准部署 |
| 24GB+   | 5-6        | 15-18    | 高性能部署 |

## 安全注意事项

1. **API密钥**: 确保设置强密码作为API_KEY
2. **网络安全**: 考虑使用HTTPS和防火墙
3. **资源限制**: 设置适当的内存和CPU限制
4. **访问控制**: 限制API访问权限

## 许可证

请参考项目根目录的LICENSE文件。