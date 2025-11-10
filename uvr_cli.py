#!/usr/bin/env python3
"""
Ultimate Vocal Remover - 命令行接口
直接处理 MP3/WAV 文件，无需 GUI
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 添加项目路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_PATH)

from separate import SeperateVR, SeperateMDX, SeperateMDXC, SeperateDemucs, clear_gpu_cache
from gui_data.constants import *
from lib_v5.vr_network.model_param_init import ModelParameters
import json
import yaml
from ml_collections import ConfigDict

# 模型目录
MODELS_DIR = os.path.join(BASE_PATH, 'models')
VR_MODELS_DIR = os.path.join(MODELS_DIR, 'VR_Models')
MDX_MODELS_DIR = os.path.join(MODELS_DIR, 'MDX_Net_Models')
DEMUCS_MODELS_DIR = os.path.join(MODELS_DIR, 'Demucs_Models')
DEMUCS_NEWER_REPO_DIR = os.path.join(DEMUCS_MODELS_DIR, 'v3_v4_repo')
VR_HASH_JSON = os.path.join(VR_MODELS_DIR, 'model_data', 'model_data.json')
MDX_HASH_JSON = os.path.join(MDX_MODELS_DIR, 'model_data', 'model_data.json')
MDX_C_CONFIG_PATH = os.path.join(MDX_MODELS_DIR, 'model_data', 'mdx_c_configs')
VR_PARAM_DIR = os.path.join(BASE_PATH, 'lib_v5', 'vr_network', 'modelparams')
DENOISER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeNoise-Lite.pth')
DEVERBER_MODEL_PATH = os.path.join(VR_MODELS_DIR, 'UVR-DeEcho-DeReverb.pth')

# 检查 GPU
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


class SimpleModelData:
    """简化的 ModelData 类，用于命令行处理"""
    def __init__(self, model_name, process_method, model_path=None, **kwargs):
        self.model_name = model_name
        self.model_basename = os.path.splitext(os.path.basename(model_path or model_name))[0]
        self.process_method = process_method
        self.model_path = model_path or self._get_model_path(process_method, model_name)
        
        # 默认设置
        self.is_gpu_conversion = 0 if cuda_available else -1
        self.device_set = "0"
        self.is_normalization = kwargs.get('normalize', True)
        self.wav_type_set = kwargs.get('wav_type', 'PCM_16')
        self.mp3_bit_set = kwargs.get('mp3_bitrate', '320k')
        self.save_format = kwargs.get('save_format', 'WAV')
        self.is_primary_stem_only = kwargs.get('primary_stem_only', False)
        self.is_secondary_stem_only = kwargs.get('secondary_stem_only', False)
        self.is_invert_spec = kwargs.get('invert_spec', False)
        
        # VR 模型设置
        if process_method == VR_ARCH_TYPE:
            self._init_vr_model(**kwargs)
        # MDX 模型设置
        elif process_method == MDX_ARCH_TYPE:
            self._init_mdx_model(**kwargs)
        # Demucs 模型设置
        elif process_method == DEMUCS_ARCH_TYPE:
            self._init_demucs_model(**kwargs)
        
        # 通用设置
        self.DENOISER_MODEL = DENOISER_MODEL_PATH if os.path.isfile(DENOISER_MODEL_PATH) else None
        self.DEVERBER_MODEL = DEVERBER_MODEL_PATH if os.path.isfile(DEVERBER_MODEL_PATH) else None
        self.is_deverb_vocals = kwargs.get('deverb', False)
        self.deverb_vocal_opt = kwargs.get('deverb_opt', 'Main Vocals Only')
        self.is_denoise_model = kwargs.get('denoise', False)
        self.is_denoise = kwargs.get('denoise', False)
        self.is_secondary_model_activated = False
        self.secondary_model = None
        self.is_vocal_split_model = False
        self.vocal_split_model = None
        self.is_ensemble_mode = False
        self.is_4_stem_ensemble = False
        self.is_multi_stem_ensemble = False
        self.is_pre_proc_model = False
        self.pre_proc_model = None
        self.pre_proc_model_activated = False
        self.is_secondary_model = False
        self.primary_model_primary_stem = None
        self.is_primary_model_primary_stem_only = False
        self.is_primary_model_secondary_stem_only = False
        self.mixer_path = os.path.join(BASE_PATH, 'lib_v5', 'mixer.ckpt')
        
    def _get_model_path(self, process_method, model_name):
        if process_method == VR_ARCH_TYPE:
            return os.path.join(VR_MODELS_DIR, f"{model_name}.pth")
        elif process_method == MDX_ARCH_TYPE:
            # 尝试 .onnx 和 .ckpt
            for ext in ['.onnx', '.ckpt']:
                path = os.path.join(MDX_MODELS_DIR, f"{model_name}{ext}")
                if os.path.isfile(path):
                    return path
            return os.path.join(MDX_MODELS_DIR, f"{model_name}.onnx")
        elif process_method == DEMUCS_ARCH_TYPE:
            # Demucs 模型路径更复杂，需要查找
            for ext in ['.yaml', '.th', '.gz']:
                path = os.path.join(DEMUCS_NEWER_REPO_DIR, f"{model_name}{ext}")
                if os.path.isfile(path):
                    return path
            return os.path.join(DEMUCS_NEWER_REPO_DIR, f"{model_name}.yaml")
        return None
    
    def _init_vr_model(self, **kwargs):
        """初始化 VR 模型参数"""
        self.aggression_setting = kwargs.get('aggression', 0.5)
        self.is_tta = kwargs.get('tta', False)
        self.is_post_process = kwargs.get('post_process', False)
        self.window_size = kwargs.get('window_size', 512)
        self.batch_size = kwargs.get('batch_size', 1)
        self.crop_size = kwargs.get('crop_size', 256)
        self.is_high_end_process = kwargs.get('high_end_process', 'mirroring')
        self.post_process_threshold = kwargs.get('post_process_threshold', 0.2)
        self.model_capacity = (32, 128)
        self.is_vr_51_model = False
        self.model_samplerate = 44100
        
        # 加载模型数据
        if os.path.isfile(VR_HASH_JSON):
            with open(VR_HASH_JSON, 'r') as f:
                vr_data = json.load(f)
                # 尝试通过文件名匹配
                model_hash = self._get_model_hash(self.model_path)
                if model_hash in vr_data:
                    model_info = vr_data[model_hash]
                    param_file = os.path.join(VR_PARAM_DIR, f"{model_info['vr_model_param']}.json")
                    if os.path.isfile(param_file):
                        self.vr_model_param = ModelParameters(param_file)
                        self.primary_stem = model_info.get('primary_stem', 'Instrumental')
                        self.secondary_stem = 'Vocals' if self.primary_stem == 'Instrumental' else 'Instrumental'
                        self.model_samplerate = self.vr_model_param.param['sr']
                        if 'nout' in model_info and 'nout_lstm' in model_info:
                            self.model_capacity = (model_info['nout'], model_info['nout_lstm'])
                            self.is_vr_51_model = True
                else:
                    # 使用默认参数
                    param_file = os.path.join(VR_PARAM_DIR, "4band_44100.json")
                    if os.path.isfile(param_file):
                        self.vr_model_param = ModelParameters(param_file)
                        self.primary_stem = kwargs.get('primary_stem', 'Instrumental')
                        self.secondary_stem = 'Vocals' if self.primary_stem == 'Instrumental' else 'Instrumental'
        else:
            # 使用默认参数
            param_file = os.path.join(VR_PARAM_DIR, "4band_44100.json")
            if os.path.isfile(param_file):
                self.vr_model_param = ModelParameters(param_file)
                self.primary_stem = kwargs.get('primary_stem', 'Instrumental')
                self.secondary_stem = 'Vocals' if self.primary_stem == 'Instrumental' else 'Instrumental'
        
        self.primary_stem_native = self.primary_stem
        self.is_karaoke = False
        self.is_bv_model = False
    
    def _init_mdx_model(self, **kwargs):
        """初始化 MDX 模型参数"""
        self.margin = kwargs.get('margin', 44100)
        self.chunks = 0
        self.mdx_segment_size = kwargs.get('mdx_segment_size', 256)
        self.mdx_batch_size = kwargs.get('mdx_batch_size', 1)
        self.compensate = kwargs.get('compensate', 1.035)
        self.is_mdx_c = False
        self.is_mdx_ckpt = False
        self.is_mdx_c_seg_def = kwargs.get('mdx_c_seg_def', True)
        self.is_mdx_combine_stems = kwargs.get('mdx_combine_stems', False)
        self.mdxnet_stem_select = kwargs.get('mdx_stem_select', 'Vocals')
        self.overlap_mdx = kwargs.get('overlap_mdx', 0.25)
        self.overlap_mdx23 = kwargs.get('overlap_mdx23', 8)
        self.mdx_c_configs = None
        
        # 检查是否是 MDX-C 模型
        if os.path.isfile(MDX_HASH_JSON):
            with open(MDX_HASH_JSON, 'r') as f:
                mdx_data = json.load(f)
                model_hash = self._get_model_hash(self.model_path)
                if model_hash in mdx_data:
                    model_info = mdx_data[model_hash]
                    if 'config_yaml' in model_info:
                        self.is_mdx_c = True
                        config_path = os.path.join(MDX_C_CONFIG_PATH, model_info['config_yaml'])
                        if os.path.isfile(config_path):
                            with open(config_path) as f:
                                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
                            self.mdx_c_configs = config
                            if self.mdx_c_configs.training.target_instrument:
                                self.primary_stem = self.mdx_c_configs.training.target_instrument
                            else:
                                self.primary_stem = self.mdx_c_configs.training.instruments[0]
                        else:
                            self.primary_stem = kwargs.get('primary_stem', 'Vocals')
                    else:
                        self.compensate = model_info.get('compensate', 1.035)
                        self.mdx_dim_f_set = model_info.get('mdx_dim_f_set', 2048)
                        self.mdx_dim_t_set = model_info.get('mdx_dim_t_set', 8)
                        self.mdx_n_fft_scale_set = model_info.get('mdx_n_fft_scale_set', 6144)
                        self.primary_stem = model_info.get('primary_stem', 'Vocals')
                else:
                    self.primary_stem = kwargs.get('primary_stem', 'Vocals')
        else:
            self.primary_stem = kwargs.get('primary_stem', 'Vocals')
        
        self.secondary_stem = 'Instrumental' if self.primary_stem == 'Vocals' else 'Vocals'
        self.primary_stem_native = self.primary_stem
        self.is_karaoke = False
        self.is_bv_model = False
        self.model_samplerate = 44100
    
    def _init_demucs_model(self, **kwargs):
        """初始化 Demucs 模型参数"""
        self.shifts = kwargs.get('shifts', 1)
        self.is_split_mode = kwargs.get('split_mode', True)
        self.segment = kwargs.get('segment', 10)
        self.is_chunk_demucs = kwargs.get('chunk_demucs', False)
        self.overlap = kwargs.get('overlap', 0.25)
        self.demucs_stems = kwargs.get('demucs_stems', 'Vocals')
        self.is_demucs_combine_stems = kwargs.get('demucs_combine_stems', False)
        self.demucs_version = kwargs.get('demucs_version', DEMUCS_V4)
        self.demucs_source_list = []
        self.demucs_stem_count = 4
        self.demucs_source_map = {}
        self.primary_stem = kwargs.get('primary_stem', 'Vocals')
        self.secondary_stem = 'Instrumental' if self.primary_stem == 'Vocals' else 'Vocals'
        self.primary_stem_native = self.primary_stem
        self.model_samplerate = 44100
        
        # 根据模型名称确定版本和源列表
        if 'htdemucs_ft' in self.model_name or 'htdemucs' in self.model_name:
            self.demucs_version = DEMUCS_V4
            self.demucs_source_list = ['drums', 'bass', 'other', 'vocals']
            self.demucs_stem_count = 4
        elif 'htdemucs_6s' in self.model_name:
            self.demucs_version = DEMUCS_V4
            self.demucs_source_list = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
            self.demucs_stem_count = 6
        elif 'mdx' in self.model_name.lower():
            self.demucs_version = DEMUCS_V3
            self.demucs_source_list = ['drums', 'bass', 'other', 'vocals']
            self.demucs_stem_count = 4
    
    def _get_model_hash(self, file_path):
        """计算模型文件的哈希值"""
        import hashlib
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        return None
    
    @property
    def mp(self):
        """返回 VR 模型参数"""
        return self.vr_model_param if hasattr(self, 'vr_model_param') else None


def process_audio(input_file, output_dir, model_name, process_method, **kwargs):
    """处理音频文件"""
    print(f"\n处理文件: {input_file}")
    print(f"模型: {model_name}")
    print(f"方法: {process_method}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型数据对象
    model_data = SimpleModelData(model_name, process_method, **kwargs)
    
    # 检查模型文件是否存在
    if not os.path.isfile(model_data.model_path):
        print(f"错误: 模型文件不存在: {model_data.model_path}")
        print(f"请确保模型已下载到正确的目录")
        return False
    
    # 准备处理数据
    audio_file_base = os.path.splitext(os.path.basename(input_file))[0]
    
    def set_progress_bar(step, inference_iterations=0):
        progress = step + inference_iterations
        print(f"\r进度: {progress*100:.1f}%", end='', flush=True)
    
    def write_to_console(text, base_text=''):
        print(f"\n{base_text}{text}")
    
    def process_iteration():
        pass
    
    def cached_source_callback(process_method, model_name=None):
        return None, None
    
    def cached_model_source_holder(process_method, sources, model_name=None):
        pass
    
    process_data = {
        'model_data': model_data,
        'export_path': output_dir,
        'audio_file_base': audio_file_base,
        'audio_file': input_file,
        'set_progress_bar': set_progress_bar,
        'write_to_console': write_to_console,
        'process_iteration': process_iteration,
        'cached_source_callback': cached_source_callback,
        'cached_model_source_holder': cached_model_source_holder,
        'list_all_models': [model_name],
        'is_ensemble_master': False,
        'is_4_stem_ensemble': False
    }
    
    try:
        # 根据处理方法选择分离器
        if process_method == VR_ARCH_TYPE:
            separator = SeperateVR(model_data, process_data)
        elif process_method == MDX_ARCH_TYPE:
            if model_data.is_mdx_c:
                separator = SeperateMDXC(model_data, process_data)
            else:
                separator = SeperateMDX(model_data, process_data)
        elif process_method == DEMUCS_ARCH_TYPE:
            separator = SeperateDemucs(model_data, process_data)
        else:
            print(f"错误: 不支持的处理方法: {process_method}")
            return False
        
        # 执行分离
        separator.seperate()
        clear_gpu_cache()
        
        print(f"\n完成! 输出目录: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Ultimate Vocal Remover - 命令行接口')
    parser.add_argument('input', help='输入音频文件路径 (MP3/WAV)')
    parser.add_argument('-o', '--output', default='./output', help='输出目录 (默认: ./output)')
    parser.add_argument('-m', '--model', required=True, help='模型名称 (例如: 1_HP-UVR, UVR-MDX-NET-Inst_HQ_1, htdemucs_ft)')
    parser.add_argument('-t', '--type', choices=['vr', 'mdx', 'demucs'], required=True,
                       help='模型类型: vr (VR Architecture), mdx (MDX-Net), demucs (Demucs)')
    parser.add_argument('--gpu', action='store_true', default=True, help='使用 GPU (默认: True)')
    parser.add_argument('--cpu', action='store_true', help='强制使用 CPU')
    parser.add_argument('--primary-stem-only', action='store_true', help='只输出主干 (例如: 只输出人声)')
    parser.add_argument('--secondary-stem-only', action='store_true', help='只输出次干 (例如: 只输出伴奏)')
    
    # VR 模型参数
    parser.add_argument('--aggression', type=float, default=0.5, help='VR 模型攻击性 (0.0-1.0, 默认: 0.5)')
    parser.add_argument('--window-size', type=int, default=512, help='VR 窗口大小 (默认: 512)')
    parser.add_argument('--batch-size', type=int, default=1, help='批处理大小 (默认: 1)')
    
    # MDX 模型参数
    parser.add_argument('--mdx-segment-size', type=int, default=256, help='MDX 段大小 (默认: 256)')
    parser.add_argument('--compensate', type=float, default=1.035, help='MDX 补偿值 (默认: 1.035)')
    
    # Demucs 模型参数
    parser.add_argument('--shifts', type=int, default=1, help='Demucs 移位次数 (默认: 1)')
    parser.add_argument('--segment', type=int, default=10, help='Demucs 段长度 (默认: 10)')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.isfile(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return 1
    
    # 确定处理方法
    method_map = {
        'vr': VR_ARCH_TYPE,
        'mdx': MDX_ARCH_TYPE,
        'demucs': DEMUCS_ARCH_TYPE
    }
    process_method = method_map[args.type]
    
    # GPU 设置
    use_gpu = args.gpu and not args.cpu
    
    # 准备参数
    kwargs = {
        'normalize': True,
        'wav_type': 'PCM_16',
        'mp3_bitrate': '320k',
        'save_format': 'WAV',
        'primary_stem_only': args.primary_stem_only,
        'secondary_stem_only': args.secondary_stem_only,
        'aggression': args.aggression,
        'window_size': args.window_size,
        'batch_size': args.batch_size,
        'mdx_segment_size': args.mdx_segment_size,
        'compensate': args.compensate,
        'shifts': args.shifts,
        'segment': args.segment,
    }
    
    # 处理文件
    success = process_audio(
        args.input,
        args.output,
        args.model,
        process_method,
        **kwargs
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

