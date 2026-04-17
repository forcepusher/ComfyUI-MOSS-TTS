import hashlib
import os
import re
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoProcessor

import comfy.model_management as mm
import folder_paths

from ..utils.constants import MODEL_VARIANTS, DEFAULT_CODEC_PATH, SAMPLE_RATE
from ..utils import backend

MODEL_ID_TTSD = "OpenMOSS-Team/MOSS-TTSD-v1.0"
MODEL_ID_VOICE_GENERATOR = "OpenMOSS-Team/MOSS-VoiceGenerator"

# Track loaded model so we can free VRAM before loading a new one
_current_model = None
_current_processor = None

# Store models in ComfyUI's models directory under moss-tts/
MOSS_MODELS_DIR = os.path.join(folder_paths.models_dir, "moss-tts")
os.makedirs(MOSS_MODELS_DIR, exist_ok=True)

# Redirect HuggingFace custom-module cache to stay within ComfyUI's model directory
# instead of the user's ~/.cache/huggingface/modules
_hf_modules_cache = os.path.join(MOSS_MODELS_DIR, ".hf_modules_cache")
os.makedirs(_hf_modules_cache, exist_ok=True)
os.environ.setdefault("HF_MODULES_CACHE", _hf_modules_cache)


def _resolve_local_dir(repo_id_or_path):
    """If repo_id_or_path is a HF repo ID, download it into ComfyUI's
    models/moss-tts/ directory and return the local path.
    If it's already a local path, return it as-is.
    Works around MOSS processor bug: Path(repo_id) mangles / to \\ on Windows."""
    if os.path.isdir(repo_id_or_path):
        local = repo_id_or_path
    else:
        # e.g. "OpenMOSS-Team/MOSS-TTS" → "OpenMOSS-Team--MOSS-TTS"
        safe_name = repo_id_or_path.replace("/", "--")
        local_dir = os.path.join(MOSS_MODELS_DIR, safe_name)
        local = snapshot_download(repo_id_or_path, local_dir=local_dir)

    _patch_remote_code(local)
    return local


# --- Remote-code compatibility patch for transformers >= 5.0 -----------------
#
# transformers 5.x rewrites every `PreTrainedConfig` subclass with
# `@dataclass(repr=False)` inside `__init_subclass__`. MOSS's remote
# `configuration_moss_audio_tokenizer.py` declares bare class-level
# annotations (e.g. `sampling_rate: int`) without defaults. These become
# no-default dataclass fields *after* default fields inherited from the
# parent (`problem_type`, `id2label`, ...), which raises:
#
#     TypeError: non-default argument 'sampling_rate' follows default
#                argument 'problem_type'
#
# during class creation, before our code ever runs. The annotations are
# redundant: the hand-written `__init__` already supplies defaults for all
# of them, so we rewrite each bare `name: type` into an assignment
# (`name: type = None`) on the downloaded snapshot.

_BARE_ANNOTATION_RE = re.compile(
    r"^( {4})([A-Za-z_]\w*)[ \t]*:[ \t]*([^=\n]+?)[ \t]*$",
    re.MULTILINE,
)

_CONFIG_FILES_TO_PATCH = (
    "configuration_moss_audio_tokenizer.py",
)


def _patch_remote_code(local_dir):
    """Strip bare class-level annotations from MOSS config files and invalidate
    any stale copies in the Hugging Face dynamic module cache so the patched
    source is re-imported."""
    patched_any = False
    for fname in _CONFIG_FILES_TO_PATCH:
        path = os.path.join(local_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            continue

        new_source = _BARE_ANNOTATION_RE.sub(r"\1\2: \3 = None", source)
        if new_source == source:
            continue

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_source)
            patched_any = True
        except OSError:
            pass

    if patched_any:
        _invalidate_hf_module_cache(_CONFIG_FILES_TO_PATCH)


def _invalidate_hf_module_cache(filenames):
    """Delete cached copies of the given files from HF_MODULES_CACHE so that
    transformers re-imports them from the patched snapshot."""
    try:
        from transformers.utils import HF_MODULES_CACHE
    except Exception:
        return

    modules_root = os.path.join(HF_MODULES_CACHE, "transformers_modules")
    if not os.path.isdir(modules_root):
        return

    for root, _dirs, files in os.walk(modules_root):
        for fname in filenames:
            if fname in files:
                try:
                    os.remove(os.path.join(root, fname))
                except OSError:
                    pass


class MossTTSModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_variant": (list(MODEL_VARIANTS.keys()),),
                "local_model_path": ("STRING", {"default": ""}),
                "codec_local_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MOSS_TTS_PIPE",)
    FUNCTION = "load_model"
    CATEGORY = "audio/MOSS-TTS"

    def load_model(self, model_variant, local_model_path, codec_local_path):
        global _current_model, _current_processor

        device = mm.get_torch_device()
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        attn_implementation = backend.resolve_attn_implementation(device, dtype)

        mm.unload_all_models()

        # Free the previous MOSS model from VRAM
        if _current_processor is not None:
            if hasattr(_current_processor, "audio_tokenizer"):
                _current_processor.audio_tokenizer.cpu()
            _current_processor = None
        if _current_model is not None:
            _current_model.cpu()
            del _current_model
            _current_model = None
        torch.cuda.empty_cache()

        model_id = MODEL_VARIANTS[model_variant]
        model_path = local_model_path.strip() if local_model_path.strip() else model_id

        # Pre-resolve to local directory to avoid MOSS processor's
        # Path(repo_id) bug on Windows (converts / to \)
        local_dir = _resolve_local_dir(model_path)

        codec_path = codec_local_path.strip() or DEFAULT_CODEC_PATH
        processor_kwargs = {
            "trust_remote_code": True,
            "codec_path": _resolve_local_dir(codec_path),
        }
        if model_id == MODEL_ID_VOICE_GENERATOR:
            processor_kwargs["normalize_inputs"] = True

        processor = AutoProcessor.from_pretrained(local_dir, **processor_kwargs)
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

        model = AutoModel.from_pretrained(
            local_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        _current_model = model
        _current_processor = processor

        return ((model, processor, SAMPLE_RATE, device, model_id),)

    @classmethod
    def IS_CHANGED(cls, model_variant, local_model_path, codec_local_path):
        h = hashlib.md5()
        h.update(model_variant.encode())
        h.update(local_model_path.encode())
        h.update(codec_local_path.encode())
        return h.hexdigest()
