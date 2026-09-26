from __future__ import annotations
import os
import sys
import time
import json
import ssl
import shutil
import tempfile
import requests
import gradio as gr
from typing import List, Dict

os.environ['SSL_CERT_FILE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['HF_HUB_DISABLE_VERIFICATION'] = '1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

ssl._create_default_https_context = ssl._create_unverified_context

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

try:
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
    from huggingface_hub.hf_api import HfApi
    from huggingface_hub.utils._http import hf_raise_for_status
    from huggingface_hub.utils import build_hf_headers
except ImportError as e:
    print(f"[ModelDownloader] 核心依赖缺失，请运行: pip install huggingface_hub")
    raise e

try:
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download
    from modelscope.hub.file_download import model_file_download
    from modelscope import HubApi
except ImportError as e:
    ms_snapshot_download = None
    model_file_download = None
    HubApi = None
    print(f"[ModelDownloader] ModelScope SDK 导入失败: {e}")

from modules import script_callbacks
from modules.shared import opts, cmd_opts
from modules.paths import models_path

MODEL_DOWNLOADER_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CACHE_DIR = os.path.join(MODEL_DOWNLOADER_DIR, "cache")
IMAGE_DIR = os.path.join(MODEL_DOWNLOADER_DIR, "image")
os.makedirs(CACHE_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 模型组合预设列表
# 所有文件均从魔搭社区（ModelScope）下载，逐文件下载，不使用 snapshot_download，
# 避免下载无关权重、示例图和文档。
# target_dir 是相对于 models_path 的子目录名（不含 models/ 前缀）。
# ════════════════════════════════════════════════════════════════════
MODEL_PRESETS = [
    # ── 1. Flux-Klein：人像美学写实感模型 ──
    {
        "id": "flux-klein",
        "name": "Flux-Klein",
        "description": "Flux2-Klein-9B-True-V3 完整模型组合",
        "role": "上下文编辑模型",
        "vram": "12 GB",
        "cover": "klein.png",
        "files": [
            {
                "repo_id": "wikeeyang/Flux2-Klein-9B-True-V3",
                "file_path": "Flux2-Klein-9B-True-V3-int8mixedrow.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "9.44 GB",
            },
            {
                "repo_id": "Comfy-Org/flux2-klein-9B",
                "file_path": "split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.66 GB",
            },
            {
                "repo_id": "Comfy-Org/flux2-klein-9B",
                "file_path": "split_files/vae/flux2-vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "336 MB",
            },
        ],
    },
    # ── 2. Zimage：人像美学写实感模型 ──
    {
        "id": "zimage",
        "name": "Zimage",
        "description": "Z-Image-Turbo 完整模型组合",
        "role": "人像美学写实感模型",
        "vram": "8 GB",
        "cover": "zimage.png",
        "files": [
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.20 GB",
            },
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/text_encoders/qwen_3_4b.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.04 GB",
            },
            {
                "repo_id": "Comfy-Org/z_image_turbo",
                "file_path": "split_files/vae/ae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "335 MB",
            },
        ],
    },
    # ── 3. Anima：二次元动漫模型 ──
    {
        "id": "anima",
        "name": "Anima",
        "description": "Anima 二次元动漫模型组合",
        "role": "二次元动漫模型",
        "vram": "8 GB",
        "cover": "anima2.9b.png",
        "files": [
            {
                "repo_id": "aa4a4a/Anima-2.9B",
                "file_path": "Anima-2.9B-preview-v1.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "5.84 GB",
            },
            {
                "repo_id": "circlestone-labs/Anima",
                "file_path": "split_files/text_encoders/qwen_3_06b_base.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "1.19 GB",
            },
            {
                "repo_id": "circlestone-labs/Anima",
                "file_path": "split_files/vae/qwen_image_vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "254 MB",
            },
        ],
    },
    # ── 4. Krea2：新一代审美向多风格文生图模型 ──
    {
        "id": "krea2",
        "name": "Krea2",
        "description": "Krea2-Turbo 完整模型组合",
        "role": "审美向多风格模型",
        "vram": "12 GB",
        "cover": "krea2.png",
        "files": [
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "13.14 GB",
            },
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "5.24 GB",
            },
            {
                "repo_id": "Comfy-Org/Krea-2",
                "file_path": "vae/qwen_image_vae.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "254 MB",
            },
        ],
    },
    # ── 5. Qwen-Image-2.1：通义千问图像生成模型 ──
    {
        "id": "qwen-image-2.1",
        "name": "Qwen-Image-2.1",
        "description": "Qwen-Image-2.1 int8 convrot 完整模型组合",
        "role": "通义千问图像生成模型",
        "vram": "12 GB",
        "cover": "qwen-image-2.1.png",
        "files": [
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.76 GB",
            },
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
                "target_dir": "text_encoder",
                "display_name": "文本编码器",
                "size": "8.71 GB",
            },
            {
                "repo_id": "Comfy-Org/Qwen-Image-2.1",
                "file_path": "vae/qwen_image_2.1_vae_bf16.safetensors",
                "target_dir": "vae",
                "display_name": "VAE",
                "size": "644 MB",
            },
        ],
    },
    # ── 6. xl-Illustrious：SDXL 二次元动漫单文件模型 ──
    {
        "id": "xl-illustrious",
        "name": "xl-Illustrious",
        "description": "xl-Illustrious 4.0 单文件 checkpoint",
        "role": "初代二次元动漫模型",
        "vram": "6 GB",
        "cover": "xl-Illustrious.png",
        "files": [
            {
                "repo_id": "yangzxo/xl-Illustrious_4.0",
                "file_path": "xl-Illustrious_4.0.safetensors",
                "target_dir": "Stable-diffusion",
                "display_name": "主模型",
                "size": "6.46 GB",
            },
        ],
    },
]

def get_preset(preset_id: str):
    """按 ID 查找预设，找不到时返回 None。"""
    for p in MODEL_PRESETS:
        if p["id"] == preset_id:
            return p
    return None

# ════════════════════════════════════════════════════════════════════
# LoRA 模型下载预设
# 所有文件均从魔搭社区（ModelScope）逐文件下载。
# target_dir 是相对于 models_path 的子目录名（LoRA）。
# ════════════════════════════════════════════════════════════════════
MODEL_LORAS = [
    {
        "id": "klein-9b-anime",
        "name": "Klein-9B 二次元动画",
        "role": "Klein-9B 动漫风格 LoRA",
        "repo_id": "yangzxo/klein-9b-anime",
        "file_path": "klein-9b二次元动画.safetensors",
        "target_dir": "Lora",
        "size": "158 MB",
    },
    {
        "id": "krea2-sticker",
        "name": "Krea2 贴图风格",
        "role": "Krea2 贴图风格 LoRA",
        "repo_id": "yangzxo/krea2-Sticker",
        "file_path": "krea2-贴图风格.safetensors",
        "target_dir": "Lora",
        "size": "218 MB",
    },
    {
        "id": "anima-character-design",
        "name": "Anima 角色设计",
        "role": "Anima 角色设计 LoRA",
        "repo_id": "yangzxo/anima_character_design",
        "file_path": "anima_lora.safetensors",
        "target_dir": "Lora",
        "size": "66 MB",
    },
    {
        "id": "krea2-scene-concept-art",
        "name": "Krea2 场景概念艺术",
        "role": "Krea2 场景概念艺术 LoRA",
        "repo_id": "yangzxo/krea2-sceneconceptart",
        "file_path": "krea2场景概念艺术.safetensors",
        "target_dir": "Lora",
        "size": "218 MB",
    },
    {
        "id": "xl-comic-illustration",
        "name": "XL 漫画人设",
        "role": "SDXL 漫画人设 LoRA",
        "repo_id": "yangzxo/XL-comic-illustration",
        "file_path": "XL漫画人设.safetensors",
        "target_dir": "Lora",
        "size": "870 MB",
    },
]


def get_lora(lora_id: str):
    """按 ID 查找 LoRA 预设，找不到时返回 None。"""
    for l in MODEL_LORAS:
        if l["id"] == lora_id:
            return l
    return None

# ════════════════════════════════════════════════════════════════════
# 高清放大上采样模型（Real-ESRGAN）预设
# Forge 的 ESRGAN 上采样器目录由 --esrgan-models-path 指定，默认 models/ESRGAN。
# 文件下载放入该目录后会被自动扫描，出现在上采样器下拉列表，无需额外配置。
# target_dir 是相对于 models_path 的子目录名（ESRGAN）。
# ════════════════════════════════════════════════════════════════════
MODEL_UPSCALERS = [
    {
        "id": "4x-ultrasharp",
        "name": "4x-UltraSharp",
        "role": "通用高清放大",
        "repo_id": "XiangZL0/4x-UltraSharp",
        "file_path": "4x-UltraSharp.pth",
        "target_dir": "ESRGAN",
        "size": "63.87 MB",
    },
    {
        "id": "realesrgan-x4plus-anime-6b",
        "name": "realesrgan-x4plus-anime-6b",
        "role": "二次元动漫放大",
        "repo_id": "amd/realesrgan-x4plus-anime-6b",
        "file_path": "RealESRGAN_x4plus_anime_6B.pth",
        "target_dir": "ESRGAN",
        "size": "17.11 MB",
    },
]

def get_upscaler(upscaler_id: str):
    """按 ID 查找上采样模型预设，找不到时返回 None。"""
    for u in MODEL_UPSCALERS:
        if u["id"] == upscaler_id:
            return u
    return None

# ── SeedVR2 高清放大模型（sd-webui-forge-neo-seedvr2 插件依赖）──
# 下载放入 models/SEEDVR2，只取 3B fp8 主模型 + VAE 两个文件。
SEEDVR2_BUNDLE = {
    "id": "seedvr2",
    "name": "SeedVR2",
    "role": "AI 图像/视频高清放大",
    "repo_id": "numz/SeedVR2_comfyUI",
    "target_dir": "SEEDVR2",
    "files": [
        {
            "file_path": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
            "display_name": "主模型",
            "size": "3.16 GB",
        },
        {
            "file_path": "ema_vae_fp16.safetensors",
            "display_name": "VAE",
            "size": "478.09 MB",
        },
    ],
}

class ModelDownloader:
    def __init__(self):
        self.downloading = False
        self.current_task = None
        self.download_history = []
        self.history_file = os.path.join(CACHE_DIR, "download_history.json")
        self._load_history()

    def _load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.download_history = json.load(f)
            except Exception:
                self.download_history = []

    def _save_history(self):
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.download_history[:100], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def list_model_files(self, model_name: str, source: str) -> List[str]:
        try:
            if source == "huggingface":
                session = requests.Session()
                session.verify = False
                session.headers.update(build_hf_headers())
                url = f"https://huggingface.co/api/models/{model_name}/tree/main?recursive=True&expand=False"
                response = session.get(url)
                hf_raise_for_status(response)
                data = response.json()
                files = []
                self._extract_files(data, files)
                return files
            else:
                if HubApi:
                    api = HubApi()
                    info = api.model_info(model_name)
                    files = []
                    if hasattr(info, 'siblings') and info.siblings:
                        for item in info.siblings:
                            if hasattr(item, 'rfilename'):
                                files.append(item.rfilename)
                    return files
                return []
        except Exception as e:
            print(f"[ModelDownloader] 获取文件列表失败: {e}")
            return []
    
    def _extract_files(self, data, files):
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if item.get('type') == 'file':
                        files.append(item.get('path', ''))
                    elif item.get('type') == 'dir':
                        self._extract_files(item.get('children', []), files)

    def download_preset(self, preset_id: str, progress_callback=None) -> str:
        """从魔搭社区（ModelScope）逐文件下载指定预设组合，放入 Forge 对应目录。

        每个文件独立下载，绝不使用 snapshot_download，避免下载无关权重、示例和文档。
        支持来自不同仓库的文件。progress_callback(i, total, desc) 可选用于 UI 进度反馈。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        preset = get_preset(preset_id)
        if not preset:
            return f"找不到预设 ID: {preset_id}"

        self.downloading = True
        files = preset["files"]
        total = len(files)
        copied = []
        temp_dir = tempfile.mkdtemp(prefix=f"preset-{preset_id}-", dir=CACHE_DIR)

        try:
            for i, fspec in enumerate(files):
                repo_id = fspec["repo_id"]
                file_path = fspec["file_path"]
                target_dir = fspec["target_dir"]
                display_name = fspec["display_name"]
                filename = os.path.basename(file_path)

                if progress_callback:
                    progress_callback(i, total, f"正在下载 {display_name} ({i+1}/{total}) ...")

                print(f"[ModelDownloader] 下载 {display_name}: {repo_id}/{file_path}")

                # 逐文件下载到临时目录
                result = model_file_download(
                    model_id=repo_id,
                    file_path=file_path,
                    cache_dir=temp_dir,
                )

                # 定位下载后的实际文件路径
                source_file = result if isinstance(result, str) and os.path.isfile(result) else None
                if not source_file:
                    candidates = []
                    for root, _, names in os.walk(temp_dir):
                        if filename in names:
                            candidates.append(os.path.join(root, filename))
                    if not candidates:
                        raise FileNotFoundError(f"下载后找不到文件: {filename}")
                    source_file = candidates[0]

                # 复制到 Forge models_path 下的对应子目录
                destination_dir = os.path.join(models_path, target_dir)
                os.makedirs(destination_dir, exist_ok=True)
                destination = os.path.join(destination_dir, filename)
                shutil.copy2(source_file, destination)
                copied.append(destination)
                print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(total, total, "全部下载完成")

            self.download_history.append({
                "model_name": preset["name"],
                "source": "modelscope",
                "save_path": models_path,
                "filename": ", ".join(os.path.basename(x) for x in copied),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": f"{preset['name']} 组合",
            })
            self._save_history()
            return f"✅ {preset['name']} 组合下载完成，共 {len(copied)} 个文件:\n" + "\n".join(f"  • {x}" for x in copied)
        except Exception as e:
            print(f"[ModelDownloader] {preset['name']} 组合下载错误: {e}")
            return f"❌ {preset['name']} 组合下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_upscaler(self, upscaler_id: str, progress_callback=None) -> str:
        """从魔搭社区（ModelScope）下载指定上采样模型，放入 Forge 的 models/ESRGAN 目录。

        单文件下载。下载完成后文件会被 Forge 自动扫描，出现在上采样器下拉列表，可直接使用。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        upscaler = get_upscaler(upscaler_id)
        if not upscaler:
            return f"找不到上采样模型 ID: {upscaler_id}"

        repo_id = upscaler["repo_id"]
        file_path = upscaler["file_path"]
        target_dir = upscaler["target_dir"]
        name = upscaler["name"]
        filename = os.path.basename(file_path)

        self.downloading = True
        temp_dir = tempfile.mkdtemp(prefix=f"upscaler-{upscaler_id}-", dir=CACHE_DIR)
        try:
            if progress_callback:
                progress_callback(0, 1, f"正在下载 {name} ...")
            print(f"[ModelDownloader] 下载上采样模型 {name}: {repo_id}/{file_path}")

            result = model_file_download(
                model_id=repo_id,
                file_path=file_path,
                cache_dir=temp_dir,
            )

            source_file = result if isinstance(result, str) and os.path.isfile(result) else None
            if not source_file:
                candidates = []
                for root, _, names in os.walk(temp_dir):
                    if filename in names:
                        candidates.append(os.path.join(root, filename))
                if not candidates:
                    raise FileNotFoundError(f"下载后找不到文件: {filename}")
                source_file = candidates[0]

            # 尊重用户自定义的 --esrgan-models-path，缺省回退到 models/ESRGAN
            destination_dir = getattr(cmd_opts, "esrgan_models_path", None) or os.path.join(models_path, target_dir)
            os.makedirs(destination_dir, exist_ok=True)
            destination = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination)
            print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(1, 1, "下载完成")

            self.download_history.append({
                "model_name": name,
                "source": "modelscope",
                "save_path": destination_dir,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "上采样模型",
            })
            self._save_history()
            return f"✅ 上采样模型 {name} 下载完成：\n  • {destination}\n重启 WebUI 后会自动出现在上采样器列表，可直接使用。"
        except Exception as e:
            print(f"[ModelDownloader] 上采样模型 {name} 下载错误: {e}")
            return f"❌ 上采样模型 {name} 下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_lora(self, lora_id: str, progress_callback=None) -> str:
        """从魔搭社区（ModelScope）下载指定 LoRA 模型，放入 Forge 的 models/Lora 目录。

        单文件下载。下载完成后文件会被 Forge 自动扫描，出现在 LoRA 下拉列表，可直接使用。
        """
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        lora = get_lora(lora_id)
        if not lora:
            return f"找不到 LoRA 模型 ID: {lora_id}"

        repo_id = lora["repo_id"]
        file_path = lora["file_path"]
        target_dir = lora["target_dir"]
        name = lora["name"]
        filename = os.path.basename(file_path)

        self.downloading = True
        temp_dir = tempfile.mkdtemp(prefix=f"lora-{lora_id}-", dir=CACHE_DIR)
        try:
            if progress_callback:
                progress_callback(0, 1, f"正在下载 {name} ...")
            print(f"[ModelDownloader] 下载 LoRA 模型 {name}: {repo_id}/{file_path}")

            result = model_file_download(
                model_id=repo_id,
                file_path=file_path,
                cache_dir=temp_dir,
            )

            source_file = result if isinstance(result, str) and os.path.isfile(result) else None
            if not source_file:
                candidates = []
                for root, _, names in os.walk(temp_dir):
                    if filename in names:
                        candidates.append(os.path.join(root, filename))
                if not candidates:
                    raise FileNotFoundError(f"下载后找不到文件: {filename}")
                source_file = candidates[0]

            destination_dir = os.path.join(models_path, target_dir)
            os.makedirs(destination_dir, exist_ok=True)
            destination = os.path.join(destination_dir, filename)
            shutil.copy2(source_file, destination)
            print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(1, 1, "下载完成")

            self.download_history.append({
                "model_name": name,
                "source": "modelscope",
                "save_path": destination_dir,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "LoRA 模型",
            })
            self._save_history()
            return f"✅ LoRA 模型 {name} 下载完成：\n  • {destination}\n重启 WebUI 后会自动出现在 LoRA 列表，可直接使用。"
        except Exception as e:
            print(f"[ModelDownloader] LoRA 模型 {name} 下载错误: {e}")
            return f"❌ LoRA 模型 {name} 下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_seedvr2(self, progress_callback=None) -> str:
        """从魔搭社区下载 SeedVR2 所需的两个模型文件，放入 models/SEEDVR2 目录。"""
        if self.downloading:
            return "已有下载任务正在进行，请等待当前任务完成"
        if not model_file_download:
            return "ModelScope SDK 不可用，请先运行: pip install modelscope"

        name = SEEDVR2_BUNDLE["name"]
        repo_id = SEEDVR2_BUNDLE["repo_id"]
        files = SEEDVR2_BUNDLE["files"]
        destination_dir = os.path.join(models_path, SEEDVR2_BUNDLE["target_dir"])

        self.downloading = True
        temp_dir = tempfile.mkdtemp(prefix="seedvr2-", dir=CACHE_DIR)
        saved = []
        try:
            os.makedirs(destination_dir, exist_ok=True)
            total = len(files)
            for idx, fspec in enumerate(files):
                filename = os.path.basename(fspec["file_path"])
                if progress_callback:
                    progress_callback(idx, total, f"正在下载 {filename} ...")
                print(f"[ModelDownloader] 下载 SeedVR2 文件 {filename}: {repo_id}/{fspec['file_path']}")

                result = model_file_download(
                    model_id=repo_id,
                    file_path=fspec["file_path"],
                    cache_dir=temp_dir,
                )

                source_file = result if isinstance(result, str) and os.path.isfile(result) else None
                if not source_file:
                    candidates = []
                    for root, _, names in os.walk(temp_dir):
                        if filename in names:
                            candidates.append(os.path.join(root, filename))
                    if not candidates:
                        raise FileNotFoundError(f"下载后找不到文件: {filename}")
                    source_file = candidates[0]

                destination = os.path.join(destination_dir, filename)
                shutil.copy2(source_file, destination)
                saved.append(destination)
                print(f"[ModelDownloader] 已保存: {destination}")

            if progress_callback:
                progress_callback(total, total, "下载完成")

            for d in saved:
                self.download_history.append({
                    "model_name": name,
                    "source": "modelscope",
                    "save_path": destination_dir,
                    "filename": os.path.basename(d),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "上采样模型",
                })
            self._save_history()
            lines = "\n".join(f"  • {d}" for d in saved)
            return f"✅ SeedVR2 模型下载完成：\n{lines}\n在 seedvr2 插件设置中选择对应模型即可使用。"
        except Exception as e:
            print(f"[ModelDownloader] SeedVR2 下载错误: {e}")
            return f"❌ SeedVR2 下载失败: {e}"
        finally:
            self.downloading = False
            shutil.rmtree(temp_dir, ignore_errors=True)

    def download_model(self, model_name: str, source: str, save_path: str, filename: str = "") -> bool:
        try:
            if self.downloading:
                return False
            self.downloading = True
            
            model_name = model_name.strip()
            if not model_name:
                raise ValueError("模型名称不能为空")
                
            if len(model_name.split('/')) < 2:
                raise ValueError(f"模型 ID 格式错误。正确格式: '用户名/模型名'\n例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct")
                
            self.current_task = f"Downloading {model_name}"
            os.makedirs(save_path, exist_ok=True)

            if source == "huggingface":
                print(f"[ModelDownloader] 从 HuggingFace 下载: {model_name}")
                session = requests.Session()
                session.verify = False
                if filename:
                    hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        token=None,
                        session=session,
                    )
                else:
                    ignore_8bit = not getattr(cmd_opts, 'load_in_8bit', False)
                    ignore_patterns = ["*.bin", "*.h5"] if ignore_8bit else []
                    snapshot_download(
                        repo_id=model_name,
                        local_dir=save_path,
                        local_dir_use_symlinks=False,
                        ignore_patterns=ignore_patterns,
                        token=None,
                        session=session,
                    )
            else:
                print(f"[ModelDownloader] 从 ModelScope 下载: {model_name}")
                if filename:
                    if model_file_download:
                        model_file_download(
                            model_id=model_name,
                            file_path=filename,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 单文件下载功能不可用，请尝试完整模型下载")
                else:
                    if ms_snapshot_download:
                        ms_snapshot_download(
                            model_id=model_name,
                            cache_dir=save_path,
                        )
                    else:
                        raise RuntimeError("ModelScope SDK 完整模型下载功能不可用")

            self.download_history.append({
                "model_name": model_name, 
                "source": source, 
                "save_path": save_path,
                "filename": filename,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), 
                "type": "single" if filename else "full",
            })
            self._save_history()
            return True
        except Exception as e:
            print(f"[ModelDownloader] 下载错误: {str(e)}")
            return False
        finally:
            self.downloading = False
            self.current_task = None

model_downloader = ModelDownloader()

def _preset_file_markdown(preset_id: str):
    """从指定预设生成简洁的文件清单 Markdown（每个文件一行）。"""
    preset = get_preset(preset_id)
    if not preset:
        return ""
    lines = []
    for fspec in preset["files"]:
        filename = os.path.basename(fspec["file_path"])
        lines.append(f"- **{fspec['display_name']}**：{filename}（{fspec['size']}）")
    return "\n".join(lines)


def create_ui():
    # 注意：gr.Blocks(css=...) 的样式在嵌套渲染进主 WebUI Blocks 时会被 Gradio 5
    # 的 Blocks.render() 直接丢弃（只合并子块和事件，不合并 css），所以这里改用
    # gr.HTML + <style> 注入封面样式，确保能到达浏览器。
    # .image-container / .image-frame 是 gr.Image 内部预览层类名（Gradio 5.49.1）。
    cover_style = """
    <style>
    .model-cover {
        width: 360px !important;
        max-width: 100% !important;
        margin: 0 auto 8px;
    }
    .model-cover .image-container {
        min-width: 0 !important;
        height: auto !important;
    }
    .model-cover .image-container button,
    .model-cover .image-frame {
        width: 100% !important;
        height: auto !important;
    }
    .model-cover img {
        width: 100% !important;
        height: auto !important;
        border-radius: 8px;
    }
    /* gr.Row/gr.Column 是 flex 布局，列宽由容器分配（默认各占 50%），
       只改内部组件宽度无效，必须固定列本身的宽度 */
    .preset-row {
        justify-content: flex-start !important;
        gap: 16px !important;
    }
    .preset-col {
        flex: 0 1 360px !important;
        min-width: 0 !important;
    }
    </style>
    """
    with gr.Blocks(
        title="开源社区模型下载器",
        analytics_enabled=False,
    ) as model_downloader_tab:
        gr.HTML(cover_style)
        with gr.Tabs():
            # ── 标签页 1：一键下载模型组合 ──
            with gr.TabItem("一键下载模型组合"):
                # 为每个预设生成独立的下载区块，5 个块并排一行（列宽 240px）
                with gr.Row(elem_classes=["preset-row"]):
                    for preset in MODEL_PRESETS:
                        pid = preset["id"]
                        with gr.Column(elem_classes=["preset-col"]):
                            cover_path = os.path.join(IMAGE_DIR, preset["cover"])
                            gr.Image(
                                value=cover_path,
                                label=f"{preset['name']} · {preset['role']}",
                                show_label=True,
                                show_download_button=False,
                                show_fullscreen_button=True,
                                interactive=False,
                                elem_classes=["model-cover"],
                                min_width=0,
                            )
                            gr.Markdown(f"显存要求：**≥ {preset['vram']}**")
                            with gr.Accordion(f"{preset['name']} — {preset['role']}", open=False):
                                gr.Markdown(f"**{preset['description']}**")
                                gr.Markdown(_preset_file_markdown(pid))

                            # 一键下载按钮放在折叠块下方
                            btn = gr.Button(
                                f"⚡ 一键下载 {preset['name']}",
                                variant="primary",
                                size="lg",
                            )
                            status_box = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                            def make_download_fn(current_pid):
                                def _download():
                                    prog = gr.Progress()
                                    def progress_callback(i, total, desc):
                                        prog(i / total if total > 0 else 0, desc=desc)
                                    return model_downloader.download_preset(current_pid, progress_callback=progress_callback)
                                return _download

                            btn.click(make_download_fn(pid), outputs=status_box)

                # ── LoRA 模型下载区（主模型区下方）──
                gr.Markdown("---")
                gr.Markdown("## 🎨 LoRA 模型下载区")
                gr.Markdown(
                    "LoRA 微调模型。从魔搭社区下载，自动放入 Forge 的 LoRA 目录"
                    "（默认 `models/Lora`）；下载后重启 WebUI，即可在 LoRA 下拉列表中直接使用。"
                )
                with gr.Row(elem_classes=["preset-row"]):
                    for lora in MODEL_LORAS:
                        lid = lora["id"]
                        with gr.Column(elem_classes=["preset-col"]):
                            gr.Markdown(
                                f"**{lora['name']}**\n"
                                f"- 类型：{lora['role']}\n"
                                f"- 文件：{os.path.basename(lora['file_path'])}\n"
                                f"- 大小：{lora['size']}\n"
                                f"- 来源仓库：{lora['repo_id']}"
                            )
                            lora_btn = gr.Button(
                                f"⚡ 一键下载 {lora['name']}",
                                variant="primary",
                                size="lg",
                            )
                            lora_status = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                            def make_lora_fn(current_lid):
                                def _download():
                                    prog = gr.Progress()
                                    def progress_callback(i, total, desc):
                                        prog(i / total if total > 0 else 0, desc=desc)
                                    return model_downloader.download_lora(current_lid, progress_callback=progress_callback)
                                return _download

                            lora_btn.click(make_lora_fn(lid), outputs=lora_status)

            # ── 标签页：高清放大算法 ──
            with gr.TabItem("高清放大算法"):
                gr.Markdown(
                    "高清放大上采样模型（Real-ESRGAN）。从魔搭社区下载，自动放入 Forge 的 ESRGAN 上采样器目录"
                    "（默认 `models/ESRGAN`）；下载后重启 WebUI，即可在上采样器下拉列表中直接使用，无需额外配置。"
                )
                with gr.Row(elem_classes=["preset-row"]):
                    for up in MODEL_UPSCALERS:
                        uid = up["id"]
                        with gr.Column(elem_classes=["preset-col"]):
                            gr.Markdown(
                                f"**{up['name']}**\n"
                                f"- 类型：{up['role']}\n"
                                f"- 文件：{os.path.basename(up['file_path'])}\n"
                                f"- 大小：{up['size']}\n"
                                f"- 来源仓库：{up['repo_id']}"
                            )
                            up_btn = gr.Button(
                                f"⚡ 一键下载 {up['name']}",
                                variant="primary",
                                size="lg",
                            )
                            up_status = gr.Textbox(
                                label="下载状态",
                                interactive=False,
                                lines=3,
                            )

                            def make_up_fn(current_uid):
                                def _download():
                                    prog = gr.Progress()
                                    def progress_callback(i, total, desc):
                                        prog(i / total if total > 0 else 0, desc=desc)
                                    return model_downloader.download_upscaler(current_uid, progress_callback=progress_callback)
                                return _download

                            up_btn.click(make_up_fn(uid), outputs=up_status)

                # ── SeedVR2（sd-webui-forge-neo-seedvr2 插件）──
                with gr.Column(elem_classes=["preset-col"]):
                    gr.Markdown("---")
                    gr.Markdown(
                        f"**{SEEDVR2_BUNDLE['name']}** — {SEEDVR2_BUNDLE['role']}\n"
                        + "\n".join(
                            f"- {f['display_name']}：{os.path.basename(f['file_path'])}（{f['size']}）"
                            for f in SEEDVR2_BUNDLE["files"]
                        )
                        + f"\n- 来源仓库：{SEEDVR2_BUNDLE['repo_id']}\n"
                        f"- 保存目录：models\\{SEEDVR2_BUNDLE['target_dir']}"
                    )
                    seed_btn = gr.Button(
                        "⚡ 一键下载 SeedVR2（共 2 个文件，约 3.63 GB）",
                        variant="primary",
                        size="lg",
                    )
                    seed_status = gr.Textbox(
                        label="下载状态",
                        interactive=False,
                        lines=3,
                    )

                    def download_seedvr2_fn():
                        prog = gr.Progress()
                        def progress_callback(i, total, desc):
                            prog(i / total if total > 0 else 0, desc=desc)
                        return model_downloader.download_seedvr2(progress_callback=progress_callback)

                    seed_btn.click(download_seedvr2_fn, outputs=seed_status)

            # ── 标签页 2：自定义模型下载 ──
            with gr.TabItem("自定义模型下载"):
                with gr.Row():
                    with gr.Column(scale=1):
                        source = gr.Radio(
                            choices=["huggingface", "modelscope"],
                            value="modelscope",
                            label="模型源",
                            interactive=True,
                        )
                    with gr.Column(scale=3):
                        model_input = gr.Textbox(
                            label="模型名称/仓库ID",
                            placeholder="例如: Comfy-Org/Krea-2 或 qwen/Qwen2.5-7B-Instruct",
                            interactive=True,
                        )
                    with gr.Column(scale=1):
                        fetch_files_btn = gr.Button("获取文件列表")
                        download_full_btn = gr.Button("下载完整模型", variant="primary")

                with gr.Row():
                    with gr.Column(scale=3):
                        file_dropdown = gr.Dropdown(
                            choices=[],
                            label="选择要下载的文件",
                            interactive=True,
                            multiselect=True,
                        )
                    with gr.Column(scale=1):
                        download_single_btn = gr.Button("下载选中文件")

                save_path = gr.Textbox(
                    label="保存路径",
                    value=os.path.join(models_path, "Stable-diffusion"),
                    interactive=True,
                )

                custom_status = gr.Textbox(label="下载状态", interactive=False)

                def fetch_files(model_name, src):
                    if not model_name:
                        return gr.update(choices=[], value=None), "请输入模型名称"
                    if len(model_name.split('/')) < 2:
                        return gr.update(choices=[], value=None), "模型 ID 格式错误，正确格式: 用户名/模型名"
                    files = model_downloader.list_model_files(model_name, src)
                    if not files:
                        return gr.update(choices=[], value=None), f"未能获取 {model_name} 的文件列表"
                    return gr.update(choices=files, value=None), f"获取到 {len(files)} 个文件"

                def download_full(model_name, src, path):
                    if not model_name: return "请输入模型名称"
                    try:
                        prog = gr.Progress()
                        prog(0, desc="开始下载...")
                        success = model_downloader.download_model(model_name, src, path)
                        prog(1, desc="下载完成" if success else "下载失败")
                        return f"模型 {model_name} 下载完成，保存到: {path}" if success else "下载失败，请查看控制台日志"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                def download_single(model_name, src, path, filenames):
                    if not model_name: return "请输入模型名称"
                    if not filenames or len(filenames) == 0: return "请先选择要下载的文件"
                    try:
                        prog = gr.Progress()
                        prog(0, desc=f"开始下载 {len(filenames)} 个文件...")
                        for i, filename in enumerate(filenames):
                            success = model_downloader.download_model(model_name, src, path, filename)
                            prog((i+1)/len(filenames), desc=f"正在下载 {filename}...")
                            if not success:
                                return f"文件 {filename} 下载失败，请查看控制台日志"
                        return f"已成功下载 {len(filenames)} 个文件到: {path}"
                    except Exception as e:
                        return f"下载异常: {str(e)}"

                fetch_files_btn.click(fetch_files, inputs=[model_input, source], outputs=[file_dropdown, custom_status])
                download_full_btn.click(download_full, inputs=[model_input, source, save_path], outputs=custom_status)
                download_single_btn.click(download_single, inputs=[model_input, source, save_path, file_dropdown], outputs=custom_status)

            # ── 标签页 3：下载历史 ──
            with gr.TabItem("下载历史"):
                def load_history_data():
                    return [[h["time"], h["model_name"], h["source"], h["type"], h["save_path"]] for h in model_downloader.download_history]

                history_table = gr.Dataframe(
                    headers=["时间", "模型名称", "来源", "类型", "路径"],
                    datatype=["str", "str", "str", "str", "str"],
                    label="下载历史",
                    interactive=False,
                    value=load_history_data(),
                )

                def clear_history():
                    model_downloader.download_history.clear()
                    model_downloader._save_history()
                    return []

                clear_history_btn = gr.Button("清空历史")
                clear_history_btn.click(clear_history, outputs=history_table)

    return [(model_downloader_tab, "开源社区模型下载器", "model-downloader")]

def on_ui_tabs():
    return create_ui()

script_callbacks.on_ui_tabs(on_ui_tabs)
