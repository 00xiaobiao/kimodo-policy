# Episode Reader 使用指南

用于读取和可视化TWIST2数据记录的工具类。

## 功能特性

- ✅ 加载episode数据（支持front camera和world camera）
- ✅ 生成视频（front camera / world camera）
- ✅ 在world camera上可视化关节点
- ✅ 访问SMPLX数据、状态、动作等所有记录数据
- ✅ 支持部分帧的处理（指定start_frame和end_frame）

## 快速开始

### 1. 命令行使用

#### 查看episode信息
```bash
python3 tools/episode_reader.py /path/to/episode_0001 --info
```

#### 生成前置相机视频
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --create-video output/front.mp4 \
    --camera front
```

#### 生成世界相机视频
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --create-video output/world.mp4 \
    --camera world
```

#### 生成带关节点的世界相机视频
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --visualize-keypoints output/world_with_keypoints.mp4
```

#### 指定帧范围
```bash
python3 tools/episode_reader.py /path/to/episode_0001 \
    --visualize-keypoints output/partial.mp4 \
    --start-frame 0 \
    --end-frame 300
```

### 2. Python API使用

```python
from episode_reader import EpisodeReader

# 加载数据
reader = EpisodeReader('/path/to/episode_0001')

# 打印详细信息
reader.print_info()

# 生成视频
reader.create_video('front.mp4', camera='front')
reader.create_video('world.mp4', camera='world')

# 可视化关节点
reader.visualize_keypoints_on_world_cam('world_with_keypoints.mp4')

# 访问数据
frame_data = reader.get_frame(100)
front_img = reader.get_image(100, 'front')  # RGB格式
world_img = reader.get_image(100, 'world')
keypoints = reader.get_keypoints(100)
smplx = reader.get_smplx_data(100)
state = reader.get_state_body(100)
action = reader.get_action_body(100)
```

### 3. 运行示例

```bash
# 运行所有示例
python3 tools/example_usage.py --example all

# 运行特定示例
python3 tools/example_usage.py --example basic      # 基本使用
python3 tools/example_usage.py --example videos    # 生成视频
python3 tools/example_usage.py --example keypoints # 可视化关节点
python3 tools/example_usage.py --example data      # 访问数据
python3 tools/example_usage.py --example partial   # 部分视频
```

## 数据结构

Episode目录结构：
```
episode_0001/
├── data.json          # 元数据和所有非图像数据
├── front_rgb/         # 前置相机图像序列
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── world_rgb/         # 世界相机图像序列（可能不存在）
    ├── 000000.jpg
    ├── 000001.jpg
    └── ...
```

data.json包含的数据：
- `info`: 录制信息（版本、日期、fps等）
- `text`: 任务描述（goal、desc、steps）
- `data`: 帧数据列表，每帧包含：
  - `front_rgb`: 前置相机图像路径
  - `world_rgb`: 世界相机图像路径（可能为空）
  - `world_camera_joint_keypoints`: 关节点2D坐标
  - `smplx_data`: SMPLX人体姿态数据
  - `state_body`, `state_hand_left`, `state_hand_right`, `state_neck`: 状态数据
  - `action_body`, `action_hand_left`, `action_hand_right`, `action_neck`: 动作数据
  - 时间戳等

## API 文档

### EpisodeReader类

#### 初始化
```python
reader = EpisodeReader(episode_path: str)
```

#### 主要方法

**获取数据：**
- `get_frame(idx: int) -> Dict`: 获取指定帧的所有数据
- `get_image(idx: int, camera: str) -> np.ndarray`: 获取图像（RGB格式）
- `get_keypoints(idx: int) -> List`: 获取关节点坐标
- `get_smplx_data(idx: int) -> Dict`: 获取SMPLX数据
- `get_state_body(idx: int) -> List`: 获取身体状态
- `get_action_body(idx: int) -> List`: 获取身体动作
- `get_state_hand(idx: int, hand: str) -> List`: 获取手部状态
- `get_action_hand(idx: int, hand: str) -> List`: 获取手部动作

**生成视频：**
- `create_video(output_path, camera, fps, start_frame, end_frame) -> bool`: 生成视频
- `visualize_keypoints_on_world_cam(output_path, fps, ...) -> bool`: 生成带关节点的视频

**其他：**
- `print_info()`: 打印详细信息
- `__len__()`: 返回总帧数

#### 属性
- `frames`: 所有帧数据
- `info`: 录制信息
- `text`: 任务描述
- `has_front_cam`: 是否有前置相机
- `has_world_cam`: 是否有世界相机
- `fps`: 帧率

## 测试

测试数据：
```bash
/home/hcl4070-1/Desktop/taowen/projects/TWIST2/data/demo_20260114_222032/episode_0001
```

快速测试：
```bash
# 信息查看
python3 tools/episode_reader.py data/demo_20260114_222032/episode_0001 --info

# 生成测试视频（前30帧）
python3 tools/episode_reader.py data/demo_20260114_222032/episode_0001 \
    --visualize-keypoints test_output.mp4 \
    --start-frame 0 \
    --end-frame 30
```

## 注意事项

1. **World camera可能不存在**：代码已处理这种情况，会自动检测并提示
2. **图像格式**：`get_image()`返回RGB格式，OpenCV需要BGR格式时请转换
3. **关节点坐标**：某些关节点可能为None（表示不在视野内）
4. **视频编码**：默认使用mp4v编码，可通过codec参数修改

## 问题排查

**Q: 提示"World camera not available"**
A: 这个episode没有world camera数据，只能使用front camera相关功能

**Q: 生成的视频无法播放**
A: 尝试使用不同的codec参数，如'XVID', 'H264'等

**Q: 关节点没有显示**
A: 检查world_camera_joint_keypoints是否为None，某些帧可能没有关节点数据

## 相关文件

- `episode_reader.py`: 主要的数据读取类
- `example_usage.py`: 使用示例脚本
- `../deploy_real/server_data_record_from_shm.py`: 数据记录脚本
- `../deploy_real/data_utils/episode_writer.py`: 数据写入类
