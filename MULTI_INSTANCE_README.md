# GPU多实例部署方案 - 阶段1快速验证

## 🎯 目标

将每个GPU的实例数从1个增加到2个，提高GPU利用率，实现更高的吞吐量。

## 📁 文件说明

- `start_multi_instance_per_gpu.sh` - 多实例启动脚本
- `test_multi_instance.sh` - 部署测试脚本
- `monitor_gpu_usage.sh` - GPU利用率监控脚本
- `test_stats.py` - 统计功能测试脚本
- `monitor_stats.sh` - 实时统计监控脚本
- `docker-compose.yml` - 更新的Docker配置
- `docker/Dockerfile.dynamic` - 支持多实例的Docker镜像

## 🚀 快速开始

### 1. 构建并启动服务

```bash
# 使用多实例模式启动服务
docker-compose up --build

# 或者重新启动现有服务
docker-compose down
docker-compose up --build
```

### 2. 验证部署

```bash
# 运行测试脚本
./test_multi_instance.sh
```

### 3. 监控GPU利用率

```bash
# 开始监控（默认监控1小时）
./monitor_gpu_usage.sh

# 自定义监控参数（每5秒监控一次，持续30分钟）
./monitor_gpu_usage.sh -i 5 -d 1800
```

### 4. 验证统计功能

```bash
# 测试统计功能
./test_stats.py

# 使用测试文件进行完整测试
./test_stats.py -f test_audio.mp3 -n 3

# 实时统计监控
./monitor_stats.sh

# 详细统计监控
./monitor_stats.sh -d
```

## 📊 预期效果

### 配置对比

| 指标 | 单实例模式 | 多实例模式（2个/GPU） |
|------|------------|---------------------|
| 每GPU实例数 | 1 | 2 |
| 总并发能力 | N | 2N |
| GPU利用率 | 30-40% | 70-90% |
| 吞吐量提升 | 基准 | 2-3倍 |
| 响应时间 | 基准 | 减少20-40% |

### 端口分配

| GPU | 实例 | 端口 | 说明 |
|-----|------|------|------|
| GPU 0 | 实例 0 | 5002 | 第一个实例 |
| GPU 0 | 实例 1 | 5003 | 第二个实例 |
| GPU 1 | 实例 0 | 5004 | 第三个实例 |
| GPU 1 | 实例 1 | 5005 | 第四个实例 |
| ... | ... | ... | 以此类推 |

## 🔧 配置参数

### 环境变量

在 `docker-compose.yml` 中可以调整以下参数：

```yaml
environment:
  - INSTANCES_PER_GPU=2        # 每GPU实例数
  - STARTUP_MODE=multi_instance # 启动模式
```

### 负载均衡器配置

```yaml
# 自动配置的环境变量
- BACKEND_SERVICES=http://localhost:5002,http://localhost:5003,...
- MAX_QUEUE_SIZE=100          # 最大队列大小
- REQUEST_TIMEOUT=1800        # 请求超时（秒）
- HEALTH_CHECK_INTERVAL=30    # 健康检查间隔（秒）
```

## 📈 监控和调试

### 实时监控命令

```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# 服务健康检查
curl http://localhost:5001/health

# 查看负载均衡器状态
curl http://localhost:5001/health | jq '.'

# 查看容器日志
docker logs faster-whisper-dynamic
```

#### 📊 详细统计数据

**获取所有实例的详细统计：**
```bash
# 获取负载均衡器汇总统计
curl http://localhost:5001/stats | jq '.'

# 获取单个实例统计
curl http://localhost:5002/stats | jq '.'
curl http://localhost:5003/stats | jq '.'
```

**实时监控统计信息：**
```bash
# 基本统计监控（简洁模式）
./monitor_stats.sh

# 详细统计监控
./monitor_stats.sh -d

# 自定义监控间隔（每5秒）
./monitor_stats.sh -i 5

# 指定负载均衡器URL
./monitor_stats.sh -l http://your-lb:5001
```

**统计功能测试：**
```bash
# 基本统计测试
./test_stats.py

# 使用测试文件进行完整测试
./test_stats.py -f test_audio.mp3 -n 3

# 自定义负载均衡器URL
./test_stats.py -l http://localhost:5001
```

#### 📋 统计数据说明

**实例统计指标：**
- **请求统计**: 总请求数、成功数、失败数、成功率、吞吐量
- **文件统计**: 处理文件数、总文件大小、chunk数量、平均文件大小
- **性能统计**: 总处理时间、上传统计、平均处理时间、处理速度
- **实例信息**: 实例ID、GPU设备、端口、运行时间、当前状态

**负载均衡器统计：**
- **状态信息**: 健康实例数、队列长度、活跃请求数
- **汇总数据**: 所有实例的统计汇总
- **实例详情**: 每个实例的详细性能数据
- **分布式处理**: 分布式任务的统计信息

### 常见问题排查

#### 1. 容器启动失败

```bash
# 检查容器状态
docker ps -a

# 查看启动日志
docker logs faster-whisper-dynamic

# 检查端口占用
netstat -tlnp | grep 5001
```

#### 2. GPU内存不足

```bash
# 检查GPU内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# 调整实例数
# 编辑 docker-compose.yml，减少 INSTANCES_PER_GPU 的值
```

#### 3. 服务无响应

```bash
# 检查健康状态
curl http://localhost:5001/health

# 重启服务
docker-compose restart faster-whisper-dynamic
```

## 🔄 切换回单实例模式

如需切换回原来的单实例模式：

```bash
# 修改 docker-compose.yml
# 将 STARTUP_MODE 改为 single_instance 或删除该行
# 将 INSTANCES_PER_GPU 改为 1

# 重新启动服务
docker-compose down
docker-compose up --build
```

## 📝 性能测试建议

### 1. 基准测试

启动多实例模式后，建议进行以下测试：

```bash
# 1. 并发请求测试（模拟高负载）
for i in {1..10}; do
  curl -X POST -H "X-API-Key: your-secret-api-key-here" \
       -F "file=@test_audio.mp3" \
       -F "response_format=srt" \
       http://localhost:5001/inference &
done
wait

# 2. 长时间稳定性测试
./monitor_gpu_usage.sh -d 7200  # 监控2小时
```

### 2. 性能指标

监控以下关键指标：

- **平均GPU利用率**：目标 > 70%
- **请求响应时间**：应该比单实例模式更快
- **并发处理能力**：理论上提升2倍
- **错误率**：应该保持在低水平

## 🎛️ 高级配置

### 自定义实例数

```bash
# 设置每GPU 3个实例
export INSTANCES_PER_GPU=3
docker-compose up
```

### GPU内存优化

如果遇到GPU内存不足，可以：

1. 减少每GPU实例数
2. 使用更小的模型（如 medium-v3）
3. 调整并发参数

### 网络优化

```bash
# 在 docker-compose.yml 中添加
environment:
  - UVICORN_WORKERS=2         # 每实例的worker数
  - UVICORN_BACKLOG=2048      # 连接队列大小
```

## 📞 故障排除

### 日志位置

- **容器日志**：`docker logs faster-whisper-dynamic`
- **监控日志**：`/tmp/gpu_monitor_*.log`
- **性能报告**：`/tmp/gpu_monitor_*_report.txt`

### 紧急恢复

```bash
# 停止所有服务
docker-compose down

# 清理Docker资源
docker system prune -f

# 重新启动
docker-compose up --build
```

## 📚 下一步优化

1. **智能调度**：根据文件大小和GPU负载智能分配任务
2. **动态扩缩容**：根据负载自动调整实例数
3. **模型热加载**：减少模型加载时间
4. **缓存优化**：优化音频处理缓存策略

---

## 📞 支持

如果遇到问题，请：

1. 查看相关日志文件
2. 运行 `./test_multi_instance.sh` 诊断
3. 检查GPU资源使用情况
4. 确认Docker服务正常运行