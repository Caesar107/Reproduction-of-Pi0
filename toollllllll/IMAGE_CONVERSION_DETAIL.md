# 图像转换说明：NumPy数组 → PNG bytes

## 转换流程

### 1. NPZ加载（原始数据）
```python
data = np.load('0001.npz')
images = data["images"]  # (52, 256, 256, 3) uint8, HWC格式
```

### 2. 转为Tensor（transfer脚本）
```python
# HWC → CHW + 归一化
images = torch.from_numpy(data["images"]).permute(0, 3, 1, 2).float() / 255.0
# (52, 3, 256, 256) float32, 值域[0, 1]
```

### 3. 添加到数据集
```python
for i in range(num_frames):
    dataset.add_frame({
        "observation.images.top": images[i],  # (3, 256, 256) float32
    })
```

### 4. LeRobot自动编码（内部处理）
```python
# add_frame() 内部自动执行：
# 1. 反归一化：[0,1] → [0,255] uint8
# 2. CHW → HWC（PIL需要）
# 3. 编码为PNG bytes
# 4. 存储到Parquet: {'bytes': PNG数据, 'path': None}
```

### 5. 存储结果
```
episode_000000.parquet
└── observation.images.top列
    ├── frame 0: {'bytes': b'\x89PNG...', 'path': None}  # 81KB
    ├── frame 1: {'bytes': b'\x89PNG...', 'path': None}
    └── ...
```

---

## 关键点

| 阶段 | 格式 | 大小 | 说明 |
|------|------|------|------|
| NPZ原始 | uint8 HWC | 192 KB/帧 | NumPy数组 |
| Tensor | float32 CHW | 768 KB/帧 | 归一化[0,1] |
| PNG存储 | bytes | 81 KB/帧 | 无损压缩（42%） |

**你不需要手动处理PNG转换**，LeRobot的`add_frame()`会自动完成编码，训练时`__getitem__()`自动解码回Tensor

**此时数据状态**：
- 类型：`bytes` (PNG格式)
- 大小：约 82KB（PNG压缩后）
- 内容：完整的PNG文件数据（包含PNG文件头 `\x89PNG\r\n\x1a\n`）

---

### 第5步：写入Parquet文件
```python
# LeRobot 在 save_episode() 时写入
dataset.save_episode()
# → 将所有帧数据写入 episode_000000.parquet

# Parquet 列存储结构
{
    'observation.images.top': [
        {'bytes': b'\x89PNG\r\n\x1a\n...', 'path': None},  # 第1帧
        {'bytes': b'\x89PNG\r\n\x1a\n...', 'path': None},  # 第2帧
        ...  # 52帧，每帧约82KB
    ],
    'action': [...],
    'observation.state': [...],
    ...
}
```

**最终存储状态**：
- 文件：`~/.cache/huggingface/lerobot/caesar/my_robot_demo/data/chunk-000/episode_000000.parquet`
- 图像存储方式：**PNG bytes 直接嵌入 Parquet**
- 每帧大小：约 82KB（PNG压缩）
- 总大小：52帧 × 82KB ≈ 4.3MB（每个episode）

---

## 为什么要转成PNG？

### 1. **压缩效率**
```
原始 uint8 NumPy数组：256 × 256 × 3 = 196,608 bytes ≈ 192KB（未压缩）
PNG压缩后：约 82KB（~58%压缩率）
```

### 2. **标准格式**
- PNG是无损压缩的标准图像格式
---

## 两种存储模式对比

| 模式 | 存储方式 | 优点 | 缺点 |
|------|---------|------|------|
| `--mode image` | PNG bytes嵌入Parquet | 单文件，无需FFmpeg | 文件较大 |
| `--mode video` | MP4文件分离存储 | 文件小，加载快 | 需要FFmpeg依赖 |

**你使用的是image模式**，图像直接嵌入Parquet，无需单独的videos/文件夹。

---

## 训练时自动解码

```python
# 训练加载数据
dataset = LeRobotDataset('caesar/my_robot_demo')
sample = dataset[0]

# LeRobot自动完成：
# Parquet PNG bytes → PIL → NumPy → Tensor CHW → 归一化[0,1]

image = sample['observation.images.top']  # torch.Tensor(3, 256, 256)
# 直接可用于模型输入
```

---

## 验证结果

实际测试结果（来自验证脚本）：
- 原始: 192.0 KB (NumPy uint8)
- PNG: 80.9 KB (压缩率42%)
- 解码后与原始**完全一致** ✅

**结论**：转换过程无损，LeRobot自动处理所有编码/解码
