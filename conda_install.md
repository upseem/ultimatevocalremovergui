# Ultimate Vocal Remover - Conda 安装指南 (Ubuntu + RTX 4090)

本指南适用于在 Ubuntu 系统上使用 Conda 安装 UVR，并配置 NVIDIA RTX 4090 GPU 支持。

## 一、系统要求

- **操作系统**: Ubuntu 20.04+ 或 Ubuntu 22.04+ (推荐)
- **GPU**: NVIDIA RTX 4090
- **NVIDIA 驱动**: 525.60.13+ (推荐 535+)
- **CUDA**: 11.8+ 或 12.1+ (RTX 4090 支持 CUDA 11.8 和 12.x)
- **Python**: 3.9 或 3.10

## 二、前置准备

### 1. 检查 NVIDIA 驱动和 CUDA

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 CUDA 版本（如果已安装）
nvcc --version

# 如果没有安装 CUDA，需要先安装 NVIDIA 驱动
# Ubuntu 22.04:
sudo apt update
sudo apt install nvidia-driver-535  # 或更新版本
sudo reboot
```

### 2. 安装 Miniconda 或 Anaconda

```bash
# 下载 Miniconda (推荐，体积小)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 或下载 Anaconda (完整版)
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh
# 或
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# 初始化
source ~/.bashrc
# 或
source ~/.zshrc
```

## 三、创建 Conda 环境

### 方法一：使用 CUDA 11.8 (推荐，兼容性最好)

```bash
# 创建环境，Python 3.10
conda create -n uvr python=3.10 -y

# 激活环境
conda activate uvr

# 安装 CUDA 11.8 版本的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 验证 PyTorch CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 方法二：使用 CUDA 12.1 (最新，性能更好)

```bash
# 创建环境
conda create -n uvr python=3.10 -y
conda activate uvr

# 安装 CUDA 12.1 版本的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 四、安装项目依赖

### 1. 安装系统依赖

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y ffmpeg libsndfile1
```

### 2. 安装 Python 依赖

由于 `requirements.txt` 中的 `torch` 没有指定版本（已通过 conda 安装），我们需要调整安装顺序：

```bash
# 确保在 uvr 环境中
conda activate uvr

# 安装基础依赖（排除 torch，因为已通过 conda 安装）
pip install altgraph==0.17.3
pip install audioread==3.0.0
pip install certifi==2022.12.07
pip install cffi==1.15.1
pip install cryptography==3.4.6
pip install einops==0.6.0
pip install future==0.18.3
pip install julius==0.2.7
pip install kthread==0.2.3
pip install librosa==0.9.2
pip install llvmlite
pip install matchering==2.0.6
pip install ml_collections==0.1.1
pip install natsort==8.2.0
pip install omegaconf==2.2.3
pip install opencv-python==4.6.0.66
pip install Pillow==9.3.0
pip install psutil==5.9.4
pip install pydub==0.25.1
pip install pyglet==1.5.23
pip install pyperclip==1.8.2
pip install pyrubberband==0.3.0
pip install pytorch_lightning==2.0.0
pip install PyYAML==6.0
pip install resampy==0.4.2
pip install scipy==1.9.3
pip install soundstretch==1.2
pip install urllib3==1.26.12
pip install wget==3.2
pip install samplerate==0.1.0
pip install screeninfo==0.8.1
pip install diffq
pip install playsound
pip install onnx
pip install onnxruntime
pip install onnxruntime-gpu  # GPU 版本
pip install onnx2pytorch
pip install SoundFile==0.11.0  # Ubuntu 使用这个
pip install Dora==0.0.3
pip install numpy==1.23.5
```

### 3. 一键安装脚本

创建 `install_conda_deps.sh`:

```bash
#!/bin/bash
# 一键安装脚本

conda activate uvr

# 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg libsndfile1

# 安装 Python 依赖（排除 torch）
pip install -r <(grep -v "^torch" requirements.txt | grep -v "^onnxruntime$") 

# 单独安装 onnxruntime-gpu（覆盖 CPU 版本）
pip install onnxruntime-gpu --upgrade

echo "安装完成！"
```

使用:
```bash
chmod +x install_conda_deps.sh
./install_conda_deps.sh
```

## 五、验证安装

```bash
conda activate uvr

# 验证 PyTorch 和 CUDA
python -c "
import torch
print('=' * 50)
print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('GPU 数量:', torch.cuda.device_count())
    print('GPU 名称:', torch.cuda.get_device_name(0))
    print('GPU 显存:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
print('=' * 50)
"

# 验证 onnxruntime-gpu
python -c "
import onnxruntime as ort
print('ONNX Runtime 版本:', ort.__version__)
print('可用 providers:', ort.get_available_providers())
print('CUDA Execution Provider 可用:', 'CUDAExecutionProvider' in ort.get_available_providers())
"

# 验证其他关键库
python -c "
import librosa
import soundfile as sf
import numpy as np
print('librosa:', librosa.__version__)
print('soundfile:', sf.__version__)
print('numpy:', np.__version__)
print('所有关键库验证通过！')
"
```

## 六、requirements.txt 兼容性分析

### ✅ 完全兼容的包（Ubuntu + RTX 4090）

以下包都可以在 Ubuntu 上正常安装和使用：

- `altgraph`, `audioread`, `certifi`, `cffi`, `cryptography`
- `einops`, `future`, `julius`, `kthread`
- `librosa`, `llvmlite`, `matchering`, `ml_collections`
- `natsort`, `omegaconf`, `opencv-python`, `Pillow`
- `psutil`, `pydub`, `pyglet`, `pyperclip`, `pyrubberband`
- `pytorch_lightning`, `PyYAML`, `resampy`, `scipy`
- `soundstretch`, `urllib3`, `wget`, `samplerate`
- `screeninfo`, `diffq`, `playsound`
- `onnx`, `onnx2pytorch`
- `SoundFile` (Ubuntu 使用这个，不是 PySoundFile)
- `Dora`, `numpy`

### ⚠️ 需要特殊处理的包

1. **`torch`** (第 28 行)
   - ❌ requirements.txt 中没有指定版本
   - ✅ **解决方案**: 使用 conda 安装 CUDA 版本
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

2. **`onnxruntime` 和 `onnxruntime-gpu`** (第 36-37 行)
   - ⚠️ 同时安装会冲突
   - ✅ **解决方案**: 只安装 `onnxruntime-gpu`，它会包含 GPU 支持
   ```bash
   pip install onnxruntime-gpu
   ```

3. **`SoundFile` vs `PySoundFile`** (第 39-40 行)
   - ✅ Ubuntu 使用 `SoundFile==0.11.0`
   - ✅ macOS 使用 `PySoundFile==0.9.0.post1`
   - ✅ **解决方案**: pip 会根据系统自动选择

### 📝 推荐的 requirements.txt 修改建议

对于 Ubuntu GPU 环境，建议创建 `requirements-gpu.txt`:

```txt
# 基础依赖
altgraph==0.17.3
audioread==3.0.0
certifi==2022.12.07
cffi==1.15.1
cryptography==3.4.6
einops==0.6.0
future==0.18.3
julius==0.2.7
kthread==0.2.3
librosa==0.9.2
llvmlite
matchering==2.0.6
ml_collections==0.1.1
natsort==8.2.0
omegaconf==2.2.3
opencv-python==4.6.0.66
Pillow==9.3.0
psutil==5.9.4
pydub==0.25.1
pyglet==1.5.23
pyperclip==1.8.2
pyrubberband==0.3.0
pytorch_lightning==2.0.0
PyYAML==6.0
resampy==0.4.2
scipy==1.9.3
soundstretch==1.2
urllib3==1.26.12
wget==3.2
samplerate==0.1.0
screeninfo==0.8.1
diffq
playsound
onnx
# 注意: torch 需要通过 conda 安装 CUDA 版本
# torch  # 使用 conda: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# onnxruntime  # 不安装 CPU 版本
onnxruntime-gpu  # GPU 版本，包含 CPU 功能
onnx2pytorch
SoundFile==0.11.0; sys_platform != 'darwin'
PySoundFile==0.9.0.post1; sys_platform == 'darwin'
Dora==0.0.3
numpy==1.23.5
```

## 七、完整安装流程（推荐）

```bash
# 1. 创建并激活环境
conda create -n uvr python=3.10 -y
conda activate uvr

# 2. 安装 PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装系统依赖
sudo apt update
sudo apt install -y ffmpeg libsndfile1

# 4. 安装 Python 依赖（使用修改后的 requirements）
pip install -r requirements-gpu.txt

# 5. 验证
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

## 八、常见问题

### Q1: PyTorch 无法识别 GPU

```bash
# 检查 CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 检查驱动版本

# 重新安装匹配的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

### Q2: onnxruntime-gpu 无法使用 GPU

```bash
# 检查 providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# 重新安装
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

### Q3: 显存不足

RTX 4090 有 24GB 显存，通常足够。如果遇到问题：

```bash
# 降低 batch size
# 在 uvr_cli.py 中使用 --batch-size 1

# 或使用较小的模型
python3 uvr_cli.py input.mp3 -m "1_HP-UVR" -t vr -o ./output --batch-size 1
```

### Q4: 依赖冲突

```bash
# 清理并重新安装
conda deactivate
conda env remove -n uvr
conda create -n uvr python=3.10 -y
conda activate uvr
# 然后按照上述步骤重新安装
```

## 九、性能优化建议

1. **使用 CUDA 12.1** (如果驱动支持)
   - 更好的 RTX 4090 性能
   - 更新的 CUDA 特性

2. **调整批处理大小**
   - RTX 4090 可以支持更大的 batch size
   - 尝试 `--batch-size 2` 或 `4`

3. **使用混合精度**
   - PyTorch 自动使用，无需额外配置

## 十、快速参考

```bash
# 激活环境
conda activate uvr

# 运行 CLI
python3 uvr_cli.py input.mp3 -m "1_HP-UVR" -t vr -o ./output

# 检查 GPU 使用
watch -n 1 nvidia-smi
```

---

**总结**: requirements.txt 中的包基本都支持 Ubuntu + RTX 4090，但需要注意：
1. `torch` 需要通过 conda 安装 CUDA 版本
2. `onnxruntime-gpu` 替代 `onnxruntime`
3. 其他包都可以正常安装使用

