# Ultimate Vocal Remover - Linux GPU 无界面执行指南

## 一、环境准备

### 1. 系统要求
- Linux 系统（推荐 Ubuntu 20.04+ 或 Debian 11+）
- NVIDIA GPU（至少 6GB 显存，推荐 8GB+）
- CUDA 11.7+ 和 cuDNN
- Python 3.9+

### 2. 安装依赖

#### 安装系统依赖
```bash
# Debian/Ubuntu 系统
sudo apt update && sudo apt upgrade
sudo apt-get install -y ffmpeg python3-pip python3-tk

# Arch 系统
sudo pacman -Syu
sudo pacman -S ffmpeg python-pip tk
```

#### 创建虚拟环境
```bash
cd /path/to/ultimatevocalremovergui
python3 -m venv venv
source venv/bin/activate
```

#### 安装 Python 依赖
```bash
pip install -r requirements.txt
```

#### 安装 CUDA 版本的 PyTorch（GPU 支持）
```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应 CUDA 版本的 PyTorch（以 CUDA 11.7 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 或者使用 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 验证 GPU 支持
```bash
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

## 二、模型下载

### 模型下载地址

**主要下载仓库：**
```
https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/
```

**更新补丁仓库：**
```
https://github.com/TRvlvr/model_repo/releases/download/uvr_update_patches/
```

### 模型存放目录

- **VR 模型**：`models/VR_Models/`
- **MDX-Net 模型**：`models/MDX_Net_Models/`
- **Demucs 模型**：`models/Demucs_Models/`

### 下载脚本示例

```bash
# 创建模型目录
mkdir -p models/VR_Models models/MDX_Net_Models models/Demucs_Models

# 下载 VR 模型示例（以 1_HP-UVR 为例）
cd models/VR_Models
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/1_HP-UVR.pth

# 下载 MDX-Net 模型示例
cd ../MDX_Net_Models
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_1.onnx

# 下载 Demucs 模型示例（需要下载多个文件）
cd ../Demucs_Models
wget https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_ft.yaml
```

## 三、模型特点说明

### VR Architecture 模型

#### VR Arch v5 模型特点：

1. **HP 系列（High Performance）**
   - `1_HP-UVR`, `2_HP-UVR`: 高性能人声分离模型，适合大多数音乐
   - `3_HP-Vocal-UVR`, `4_HP-Vocal-UVR`: 专门优化的人声提取模型
   - `5_HP-Karaoke-UVR`, `6_HP-Karaoke-UVR`: 卡拉OK专用，去除人声效果好
   - `7_HP2-UVR`, `8_HP2-UVR`, `9_HP2-UVR`: HP2 系列，性能进一步提升
   - `17_HP-Wind_Inst-UVR`: 专门处理管乐器伴奏

2. **SP 系列（Standard Performance）**
   - `10_SP-UVR-2B-32000-1`, `11_SP-UVR-2B-32000-2`: 2 频段，32kHz 采样率
   - `12_SP-UVR-3B-44100`: 3 频段，44.1kHz 采样率
   - `13_SP-UVR-4B-44100-1`, `14_SP-UVR-4B-44100-2`: 4 频段，44.1kHz 采样率
   - `15_SP-UVR-MID-44100-1`, `16_SP-UVR-MID-44100-2`: 中等性能，44.1kHz

3. **特殊功能模型（FoxJoy 制作）**
   - `UVR-DeNoise-Lite`: 轻量级降噪模型
   - `UVR-DeNoise`: 标准降噪模型
   - `UVR-De-Echo-Normal`: 正常模式去回声
   - `UVR-De-Echo-Aggressive`: 激进模式去回声
   - `UVR-DeEcho-DeReverb`: 去回声和混响

4. **其他模型**
   - `UVR-BVE-4B_SN-44100-1`: 4 频段特殊优化模型

#### VR Arch v4 模型（旧版）：
- `MGM_MAIN_v4`: 主模型
- `MGM_HIGHEND_v4`: 高频优化
- `MGM_LOWEND_A_v4`, `MGM_LOWEND_B_v4`: 低频优化

### MDX-Net 模型特点

1. **UVR 官方模型**
   - `UVR-MDX-NET Main`: 主模型，通用性强
   - `UVR-MDX-NET Inst Main`: 伴奏提取专用
   - `UVR-MDX-NET Inst HQ 1/2/3`: 高质量伴奏提取系列
   - `UVR-MDX-NET Inst 1/2/3`: 标准伴奏提取系列
   - `UVR-MDX-NET Voc FT`: 人声提取微调模型
   - `UVR-MDX-NET Karaoke`: 卡拉OK专用
   - `UVR-MDX-NET Karaoke 2`: 卡拉OK改进版

2. **Kim 系列模型**
   - `Kim_Vocal_1`, `Kim_Vocal_2`: 人声提取
   - `Kim_Inst`: 伴奏提取

3. **Kuielab 系列模型（多 stem 分离）**
   - `kuielab_a_vocals`, `kuielab_b_vocals`: 人声
   - `kuielab_a_other`, `kuielab_b_other`: 其他
   - `kuielab_a_bass`, `kuielab_b_bass`: 贝斯
   - `kuielab_a_drums`, `kuielab_b_drums`: 鼓

4. **特殊功能模型**
   - `Reverb_HQ_By_FoxJoy`: 高质量混响处理

5. **VIP 模型（需要特殊访问）**
   - `UVR-MDX-NET_Main_340/390/406/427/438`: 不同版本的主模型
   - `UVR-MDX-NET_Inst_82_beta/90_beta/187_beta`: 伴奏提取测试版
   - `UVR-MDX-NET-Inst_full_292`: 完整版伴奏提取

### Demucs 模型特点

#### Demucs v4 模型（最新，推荐）：
1. **htdemucs_ft**: 
   - 混合 Transformer 架构
   - 4 stem 分离（人声、鼓、贝斯、其他）
   - 需要下载 4 个 .th 文件 + 1 个 yaml 配置文件
   - 高质量分离效果

2. **htdemucs**: 
   - 标准混合 Transformer 模型
   - 4 stem 分离
   - 需要下载 1 个 .th 文件 + 1 个 yaml 配置文件

3. **hdemucs_mmi**: 
   - MMI 训练版本
   - 4 stem 分离

4. **htdemucs_6s**: 
   - 6 stem 分离（人声、鼓、贝斯、其他、吉他、钢琴）
   - 更精细的分离

#### Demucs v3 模型：
1. **mdx**: 标准 MDX 模型，4 stem
2. **mdx_q**: 量化版本，体积更小
3. **mdx_extra**: 扩展版本，性能更好
4. **mdx_extra**: 扩展扩展版本
5. **UVR Model**: UVR 定制版本
6. **repro_mdx_a**: 复现版本，包含完整版、时间版、混合版

#### Demucs v2 模型（旧版）：
- `demucs`: 标准模型
- `demucs_extra`: 扩展版
- `demucs48_hq`: 48kHz 高质量版本
- `tasnet`: TASNet 架构
- `tasnet_extra`: TASNet 扩展版

#### Demucs v1 模型（最旧）：
- `demucs`, `demucs_extra`: 基础版本
- `light`, `light_extra`: 轻量版本
- `tasnet`, `tasnet_extra`: TASNet 架构

### MDX23 模型
- `MDX23C_D1581`: MDX23C 架构，需要 .ckpt 文件和 yaml 配置文件

## 四、Linux GPU 无界面执行命令

### 方法一：使用命令行脚本直接处理 MP3/WAV（最推荐！）

项目已包含 `uvr_cli.py` 脚本，可以直接处理音频文件，无需 GUI！

#### 基本使用

```bash
# 激活虚拟环境
source venv/bin/activate

# 使用 VR 模型处理（推荐用于大多数人声分离任务）
python3 uvr_cli.py input.mp3 -m "1_HP-UVR" -t vr -o ./output

# 使用 MDX-Net 模型处理（高质量）
python3 uvr_cli.py input.mp3 -m "UVR-MDX-NET-Inst_HQ_1" -t mdx -o ./output

# 使用 Demucs 模型处理（最高质量，但较慢）
python3 uvr_cli.py input.mp3 -m "htdemucs_ft" -t demucs -o ./output
```

#### 完整参数说明

```bash
python3 uvr_cli.py <输入文件> [选项]

必需参数:
  input                  输入音频文件路径 (MP3/WAV)
  -m, --model MODEL      模型名称
  -t, --type TYPE        模型类型: vr, mdx, demucs
  -o, --output OUTPUT     输出目录 (默认: ./output)

可选参数:
  --gpu                   使用 GPU (默认启用)
  --cpu                   强制使用 CPU
  --primary-stem-only     只输出主干 (例如: 只输出人声)
  --secondary-stem-only   只输出次干 (例如: 只输出伴奏)

VR 模型参数:
  --aggression FLOAT     攻击性设置 (0.0-1.0, 默认: 0.5)
  --window-size INT      窗口大小 (默认: 512)
  --batch-size INT       批处理大小 (默认: 1)

MDX 模型参数:
  --mdx-segment-size INT 段大小 (默认: 256)
  --compensate FLOAT     补偿值 (默认: 1.035)

Demucs 模型参数:
  --shifts INT           移位次数 (默认: 1)
  --segment INT          段长度 (默认: 10)
```

#### 使用示例

```bash
# 示例 1: 使用 VR 模型分离人声和伴奏
python3 uvr_cli.py song.mp3 -m "1_HP-UVR" -t vr -o ./separated

# 示例 2: 只提取人声（不输出伴奏）
python3 uvr_cli.py song.mp3 -m "1_HP-UVR" -t vr -o ./vocals --primary-stem-only

# 示例 3: 只提取伴奏（不输出人声）
python3 uvr_cli.py song.mp3 -m "1_HP-UVR" -t vr -o ./instrumental --secondary-stem-only

# 示例 4: 使用 MDX-Net 高质量模型
python3 uvr_cli.py song.mp3 -m "UVR-MDX-NET-Inst_HQ_1" -t mdx -o ./output

# 示例 5: 使用 Demucs 4-stem 分离（人声、鼓、贝斯、其他）
python3 uvr_cli.py song.mp3 -m "htdemucs_ft" -t demucs -o ./stems

# 示例 6: 强制使用 CPU（如果没有 GPU）
python3 uvr_cli.py song.mp3 -m "1_HP-UVR" -t vr -o ./output --cpu

# 示例 7: 调整 VR 模型参数
python3 uvr_cli.py song.mp3 -m "1_HP-UVR" -t vr -o ./output --aggression 0.7 --window-size 1024
```

#### 批处理多个文件

创建批处理脚本 `batch_process.sh`:

```bash
#!/bin/bash
# 批处理脚本

INPUT_DIR="./input"
OUTPUT_DIR="./output"
MODEL="1_HP-UVR"
TYPE="vr"

for file in "$INPUT_DIR"/*.{mp3,wav}; do
    if [ -f "$file" ]; then
        echo "处理: $file"
        python3 uvr_cli.py "$file" -m "$MODEL" -t "$TYPE" -o "$OUTPUT_DIR"
    fi
done
```

使用:
```bash
chmod +x batch_process.sh
./batch_process.sh
```

#### 常用模型推荐

**VR 模型（速度快，质量好）:**
- `1_HP-UVR` - 高性能通用模型
- `3_HP-Vocal-UVR` - 人声提取专用
- `5_HP-Karaoke-UVR` - 卡拉OK专用

**MDX-Net 模型（质量高）:**
- `UVR-MDX-NET-Inst_HQ_1` - 高质量伴奏提取
- `UVR-MDX-NET Main` - 通用主模型
- `UVR-MDX-NET-Voc_FT` - 人声提取微调版

**Demucs 模型（质量最高，速度慢）:**
- `htdemucs_ft` - 4 stem 分离（推荐）
- `htdemucs_6s` - 6 stem 分离（更精细）
- `htdemucs` - 标准版本

### 方法二：使用环境变量运行 GUI（无界面模式，不推荐）

虽然 UVR 主要是 GUI 应用，但可以通过设置环境变量来优化无界面运行：

```bash
# 设置无显示模式（如果使用 SSH 连接）
export DISPLAY=:0

# 或者使用 Xvfb 虚拟显示
sudo apt-get install xvfb
xvfb-run -a python3 UVR.py
```

### 方法三：使用 Docker（推荐用于服务器环境）

```dockerfile
# Dockerfile 示例
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# 设置环境变量
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python3", "UVR.py"]
```

构建和运行：
```bash
docker build -t uvr-gpu .
docker run --gpus all -v /path/to/models:/app/models -v /path/to/input:/app/input -v /path/to/output:/app/output uvr-gpu
```

### 方法四：使用 systemd 服务（后台运行）

创建服务文件 `/etc/systemd/system/uvr.service`：

```ini
[Unit]
Description=Ultimate Vocal Remover GPU Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/ultimatevocalremovergui
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="DISPLAY=:0"
ExecStart=/path/to/ultimatevocalremovergui/venv/bin/python3 /path/to/ultimatevocalremovergui/UVR.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl enable uvr.service
sudo systemctl start uvr.service
```

## 五、常用执行命令示例

### 基本执行（需要 X 服务器）
```bash
cd /path/to/ultimatevocalremovergui
source venv/bin/activate
python3 UVR.py
```

### 使用虚拟显示执行
```bash
cd /path/to/ultimatevocalremovergui
source venv/bin/activate
xvfb-run -a python3 UVR.py
```

### 指定 GPU 设备
```bash
CUDA_VISIBLE_DEVICES=0 python3 UVR.py  # 使用第一个 GPU
CUDA_VISIBLE_DEVICES=1 python3 UVR.py  # 使用第二个 GPU
```

### 后台运行
```bash
nohup python3 UVR.py > uvr.log 2>&1 &
```

### 使用 screen 或 tmux（推荐）
```bash
# 使用 screen
screen -S uvr
source venv/bin/activate
python3 UVR.py
# 按 Ctrl+A 然后 D 来分离会话

# 使用 tmux
tmux new -s uvr
source venv/bin/activate
python3 UVR.py
# 按 Ctrl+B 然后 D 来分离会话
```

## 六、性能优化建议

1. **GPU 内存管理**
   - 如果遇到显存不足，降低 batch size 或 segment size
   - 使用 `clear_gpu_cache()` 函数清理显存

2. **模型选择**
   - VR 模型：适合大多数场景，速度快
   - MDX-Net 模型：质量高，但速度较慢
   - Demucs 模型：质量最高，但速度最慢，显存占用大

3. **批处理**
   - 对于多个文件，建议编写批处理脚本
   - 使用队列系统（如 Celery）处理大量文件

## 七、故障排除

### GPU 不可用
```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 PyTorch CUDA 支持
python3 -c "import torch; print(torch.cuda.is_available())"

# 重新安装 PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 显存不足
- 降低 batch size
- 使用较小的模型
- 关闭其他占用 GPU 的程序

### 模型文件缺失
- 检查模型文件是否下载完整
- 确认模型文件放在正确的目录
- 检查文件权限

## 八、参考资源

- 项目主页：https://github.com/Anjok07/ultimatevocalremovergui
- 模型仓库：https://github.com/TRvlvr/model_repo
- 问题反馈：https://github.com/Anjok07/ultimatevocalremovergui/issues

---

**注意**：由于 UVR 主要是 GUI 应用，完全无界面的命令行使用需要额外的脚本开发。建议在服务器环境中使用 Xvfb 或 Docker 来运行 GUI 应用。

