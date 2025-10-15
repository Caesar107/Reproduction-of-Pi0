# 数据格式转换文档

## 原始数据格式（NPZ）

### 文件结构
```
train_data/train/
├── 0001.npz  # 71个episode文件
├── 0002.npz
└── ...
```

### NPZ 内容（每个文件）
| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `actions` | `(N, 7)` | `float32` | 前6维=delta动作，第7维=夹爪绝对位置 |
| `images` | `(N, 256, 256, 3)` | `uint8` | 顶部相机图像（**HWC格式**） |
| `wrist_images` | `(N, 256, 256, 3)` | `uint8` | 手腕相机图像（**HWC格式**） |
| `jiont_states` | `(N, 8)` | `float32` | 关节状态 |
| `ee_states` | `(N, 8)` | `float32` | 末端执行器状态 |

---

## 转换后格式（LeRobot）

### 文件结构
```
~/.cache/huggingface/lerobot/caesar/my_robot_demo/
├── data/chunk-000/*.parquet     # 数据表（包含嵌入的PNG图像）
└── meta/                         # 元数据（episodes.jsonl, tasks.jsonl, info.json）
```

**注意**：使用 `--mode image` 时，图像以PNG bytes直接嵌入Parquet文件，无需单独的videos/文件夹

### 数据结构（HuggingFace Datasets）
**文件类型**: `Parquet` (表格数据) + `PNG` (图像)

LeRobot 使用 Apache Parquet 格式存储表格化数据，图像存储为独立文件。

#### Features（数据字段）

| 字段名 | 形状 | 数据类型 | 说明 |
|--------|------|----------|------|
| `observation.state` | `(8,)` | `float32` | 关节状态（来自原始的`jiont_states`） |
| `observation.end_effector_state` | `(8,)` | `float32` | 末端执行器状态（来自原始的`ee_states`） |
| `action` | `(7,)` | `float32` | 动作（来自原始的`actions`） |
| `observation.images.top` | `(3, 256, 256)` | `image` | 顶部相机图像（**CHW格式**，从PNG加载） |
| `observation.images.wrist` | `(3, 256, 256)` | `image` | 手腕相机图像（**CHW格式**） |
| `timestamp` | `(1,)` | `float32` | 时间戳（秒） |
### LeRobot 数据字段
| 字段 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `observation.state` | `(8,)` | `float32` | 关节状态（来自jiont_states） |
| `observation.end_effector_state` | `(8,)` | `float32` | 末端执行器状态 |
| `action` | `(7,)` | `float32` | 动作（单数命名） |
| `observation.images.top` | `(3, 256, 256)` | `image` | 顶部相机（**CHW格式**，PNG bytes） |
| `observation.images.wrist` | `(3, 256, 256)` | `image` | 手腕相机（**CHW格式**，PNG bytes） |
| `timestamp` | `(1,)` | `float32` | 时间戳 |
| `frame_index` | `(1,)` | `int64` | 帧索引 |
| `episode_index` | `(1,)` | `int64` | Episode索引 |
| `task` | - | `string` | 任务名称（"grab_bottle"） |

### 数据集统计
- **总episodes**: 71
- **总frames**: 3680
- **FPS**: 30

---

## 关键转换点

### 1. 字段映射
| 原始NPZ | LeRobot | 变化 |
|---------|---------|------|
| `jiont_states` | `observation.state` | 名称修正 |
| `ee_states` | `observation.end_effector_state` | 名称规范化 |
| `actions` | `action` | 复数→单数 |
| `images` | `observation.images.top` | HWC→CHW + PNG编码 |
| `wrist_images` | `observation.images.wrist` | HWC→CHW + PNG编码 |

### 2. 图像转换
- **格式**: HWC `(256,256,3)` → CHW `(3,256,256)`
- **存储**: NumPy数组 → PNG bytes（嵌入Parquet）
- **压缩**: 192KB → 81KB（~42%压缩率）
- **过程**: LeRobot的`add_frame()`自动将Tensor编码为PNG

### 3. 数据访问
```python
# 训练时加载
dataset = LeRobotDataset('caesar/my_robot_demo')
sample = dataset[0]
# 图像自动从PNG解码为Tensor (3, 256, 256)
    ```

---

## 训练时的数据流

```
LeRobot Dataset
    ↓
DataLoader (batch采样)
    ↓
RepackTransform (字段重命名 + 归一化)
    ↓
CustomRobotInputs (拼接: joint_state[8] + ee_state[8] → state[16])
    ↓
Pi0 Model
    ↓
输出: actions [batch, 10, 7]  # action_horizon=10
```

**归一化**: 存储在 `assets/pi0_custom_robot/caesar/my_robot_demo/norm_stats.safetensors`
- state[16]: z-score归一化 `(x - mean) / std`
- action[7]: 前6维delta + 第7维夹爪

---

## 使用方法

### 1. 转换数据
```bash
uv run python toollllllll/transfer \
    --raw-dir train_data/train \
    --repo-id caesar/my_robot_demo \
    --task "grab_bottle" \
    --robot-type "custom_robot" \
    --fps 30 \
    --mode image
```

### 2. 计算归一化统计
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_custom_robot
```

### 3. 开始训练
```bash
bash toollllllll/start_training.sh
```

---

## 4. 训练时的数据流

### 4.1 原始数据 → 模型输入

**转换流程**:
```
LeRobot Dataset
    ↓ (DataLoader采样)
batch = {
    'observation.state': [B, 8],
    'observation.end_effector_state': [B, 8],
    'observation.images.top': [B, 3, 256, 256],
    'observation.images.wrist': [B, 3, 256, 256],
    'action': [B, 7],
    'task': [B] (strings)
}
    ↓ (RepackTransform in DataConfig)
batch = {
    'joint_state': [B, 8],           # 重命名
    'ee_state': [B, 8],               # 重命名
    'top_image': [B, 3, 256, 256],    # 重命名
    'wrist_image': [B, 3, 256, 256],  # 重命名
    'action': [B, 7],
    'task': [B] (strings)
}
    ↓ (CustomRobotInputs.__call__)
model_input = {
    'state': [B, 16],                 # 拼接 joint_state + ee_state
    'images': {
        'top': [B, 3, 256, 256],
        'wrist': [B, 3, 256, 256]
    },
    'task': [B] (strings)
}
    ↓ (Pi0 Model)
model_output = {
    'actions': [B, action_horizon, 7]  # action_horizon=10
}
    ↓ (CustomRobotOutputs.__call__)
final_output = {
    'action': [B, action_horizon, 7]   # 格式转换回原始命名
}
```

### 4.2 归一化处理

**位置**: 在 `RepackTransform` 中应用归一化统计量

**归一化文件**: `/home/caesar/openpi/assets/pi0_custom_robot/caesar/my_robot_demo/`
```
norm_stats.json           # 人类可读格式
norm_stats.safetensors    # 二进制格式（训练时加载）
```

**归一化统计量** (从230个batch计算得出):
```json
{
  "state": {
    "mean": [16维向量],     // joint_state(8) + ee_state(8) 的均值
    "std": [16维向量]        // 标准差
  },
  "action": {
    "mean": [7维向量],       // 动作均值
    "std": [7维向量]         // 动作标准差
  }
}
```

**归一化公式**:
```python
# 状态归一化 (z-score)
state_normalized = (state - state_mean) / state_std

# 动作归一化（仅前6维：delta actions）
action[:6] = (action[:6] - action_mean[:6]) / action_std[:6]
action[6] = (action[6] - action_mean[6]) / action_std[6]  # 夹爪（绝对值）
```

---

## 5. 转换脚本使用

### 5.1 转换命令
```bash
cd /home/caesar/openpi

uv run python toollllllll/transfer \
    --raw-dir train_data/train \            # 原始NPZ文件目录
    --repo-id caesar/my_robot_demo \        # 数据集ID
    --task "grab_bottle" \                  # 任务名称
    --robot-type "custom_robot" \           # 机器人类型
    --fps 30 \                              # 采样帧率
    --mode image                            # 图像模式（PNG存储，避免FFmpeg依赖）
```

### 5.2 转换结果
```
输出:
- Dataset created successfully at /home/caesar/.cache/huggingface/lerobot/caesar/my_robot_demo
- Total episodes: 71
- Total frames: 3680

验证:
- 每个episode的帧数保持不变
- 数据完整性：所有字段都已正确转换
- 图像质量：PNG无损压缩，与原始一致
```

---

## 注意事项

- **图像格式**: HWC → CHW（LeRobot自动处理）
- **动作空间**: 前6维=delta（增量），第7维=夹爪（绝对）
- **PNG存储**: 图像以PNG bytes嵌入Parquet（`--mode image`）
- **压缩率**: 192KB → 81KB（~42%）

---

## 相关文件

- 转换脚本: `toollllllll/transfer`
- 训练配置: `src/openpi/training/config.py` (pi0_custom_robot)
- 数据转换: `src/openpi/policies/custom_robot_policy.py`
- 启动脚本: `toollllllll/start_training.sh`
