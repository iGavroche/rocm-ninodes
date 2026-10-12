"""
Seed-VC zero-shot voice conversion for ROCm.

Converts the timbre of a source recording to a reference voice without changing
what was said or sung. Two checkpoints back the node: a whisper-small/wavenet
model for speech, and a 44 kHz F0-conditioned model for singing.

Optimized for gfx1151 (Strix Halo):
- fp16 BigVGAN vocoder (the vocoder dominates runtime; fp32 is ~6x slower)
- the model set is cached across executions instead of reloaded per run
- KV/rope caches sized to the actual sequence length instead of 8192
"""

import os
from typing import Optional

import numpy as np
import torch
import torchaudio
import yaml
import folder_paths
from transformers import AutoFeatureExtractor, WhisperModel

from ..seedvc.modules.commons import build_model, load_checkpoint, recursive_munch
from ..seedvc.modules.campplus.DTDNN import CAMPPlus
from ..seedvc.modules.bigvgan import bigvgan
from ..seedvc.modules.audio import mel_spectrogram
from ..seedvc.modules.rmvpe import RMVPE

from ..utils.memory import gentle_memory_cleanup

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seedvc", "configs")


def _resolve_model_root() -> str:
    """Seed-VC checkpoints live in models/TTS/Seed-VC, shared with other packs."""
    return os.path.join(folder_paths.models_dir, "TTS", "Seed-VC")


def _load_stage(config_path: str, checkpoint_path: str, device: torch.device):
    config = yaml.safe_load(open(config_path, "r"))
    model = build_model(recursive_munch(config["model_params"]), stage="DiT")
    model, _, _, _ = load_checkpoint(
        model, None, checkpoint_path,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)

    spect = config["preprocess_params"]["spect_params"]
    mel_args = {
        "n_fft": spect["n_fft"],
        "win_size": spect["win_length"],
        "hop_size": spect["hop_length"],
        "num_mels": spect["n_mels"],
        "sampling_rate": config["preprocess_params"]["sr"],
        "fmin": 0,
        "fmax": None,
        "center": False,
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_args)
    return model, to_mel


class SeedVCWrapper:
    """All Seed-VC submodels for one conversion pipeline."""

    overlap_frame_len = 16

    def __init__(self, device: Optional[torch.device] = None, half_vocoder: bool = True):
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        model_path = _resolve_model_root()
        whisper_path = os.path.join(folder_paths.models_dir, "TTS", "whisper-small")

        # Speech model: 22.05 kHz, no F0 conditioning.
        self.model, self.to_mel = _load_stage(
            os.path.join(CONFIG_DIR, "config_dit_mel_seed_uvit_whisper_small_wavenet.yml"),
            os.path.join(model_path, "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth"),
            self.device,
        )
        # Singing model: 44.1 kHz, F0 conditioned.
        self.model_f0, self.to_mel_f0 = _load_stage(
            os.path.join(CONFIG_DIR, "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml"),
            os.path.join(model_path, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"),
            self.device,
        )

        # The estimator only ever sees prompt + one 30 s window, so the rope/KV
        # caches do not need the upstream 8192-frame reservation.
        for stage in (self.model, self.model_f0):
            stage.cfm.estimator.setup_caches(max_batch_size=2, max_seq_length=2048)

        self.whisper_model = WhisperModel.from_pretrained(whisper_path, torch_dtype=torch.float16).to(self.device)
        del self.whisper_model.decoder
        self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_path)

        campplus_path = os.path.join(model_path, "campplus_cn_common.bin")
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        self.campplus_model.load_state_dict(torch.load(campplus_path, map_location="cpu"))
        self.campplus_model.eval().to(self.device)

        # BigVGAN is the dominant cost of a conversion and is pure conv/activation
        # work, so it runs in fp16 alongside the DiT autocast.
        vocoder_dtype = torch.float16 if (half_vocoder and self.device.type == "cuda") else torch.float32
        self.bigvgan_model = self._load_bigvgan(
            os.path.join(folder_paths.models_dir, "TTS", "bigvgan_v2_22khz_80band_256x"),
            vocoder_dtype, self.device)
        self.bigvgan_44k_model = self._load_bigvgan(
            os.path.join(folder_paths.models_dir, "TTS", "bigvgan_v2_44khz_128band_512x"),
            vocoder_dtype, self.device)

        self.rmvpe = RMVPE(os.path.join(model_path, "rmvpe.pt"), is_half=False, device=self.device)

    @staticmethod
    def _load_bigvgan(path: str, dtype: torch.dtype, device: torch.device):
        model = bigvgan.BigVGAN.from_pretrained(path, use_cuda_kernel=False)
        model.remove_weight_norm()
        return model.eval().to(device=device, dtype=dtype)

    def _whisper_features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Whisper encoder features for a (1, n) 16 kHz waveform."""
        def encode(wave):
            inputs = self.whisper_feature_extractor(
                [wave.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000,
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            out = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True,
            )
            return out.last_hidden_state.to(torch.float32)

        if audio_16k.size(-1) <= 16000 * 30:
            return encode(audio_16k)[:, :audio_16k.size(-1) // 320 + 1]

        # Long audio: 30 s windows with a 5 s overlap, stitched frame-wise.
        overlap_s = 5
        features_list = []
        buffer = None
        traversed = 0
        while traversed < audio_16k.size(-1):
            if buffer is None:
                chunk = audio_16k[:, traversed:traversed + 16000 * 30]
            else:
                chunk = torch.cat([buffer, audio_16k[:, traversed:traversed + 16000 * (30 - overlap_s)]], dim=-1)
            chunk_features = encode(chunk)[:, :chunk.size(-1) // 320 + 1]
            features_list.append(chunk_features if traversed == 0 else chunk_features[:, 50 * overlap_s:])
            buffer = chunk[:, -16000 * overlap_s:]
            traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlap_s
        return torch.cat(features_list, dim=1)

    @staticmethod
    def _crossfade(chunk1, chunk2, overlap):
        fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
        fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
        if len(chunk2) < overlap:
            chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
        else:
            chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
        return chunk2

    @torch.no_grad()
    def convert(self, source_wave, source_sr, ref_wave, ref_sr,
                diffusion_steps=30, length_adjust=1.0, inference_cfg_rate=0.7,
                f0_condition=False, auto_f0_adjust=True, pitch_shift=0):
        """Convert source timbre to the reference voice. Returns (float32 numpy, sr)."""
        inference_module = self.model_f0 if f0_condition else self.model
        mel_fn = self.to_mel_f0 if f0_condition else self.to_mel
        bigvgan_fn = self.bigvgan_44k_model if f0_condition else self.bigvgan_model
        sr = 44100 if f0_condition else 22050
        hop_length = 512 if f0_condition else 256
        max_context_window = sr // hop_length * 30
        overlap_wave_len = self.overlap_frame_len * hop_length

        def to_model_sr(wave, native_sr):
            wave = wave.reshape(1, -1).float().to(self.device)
            if native_sr != sr:
                wave = torchaudio.functional.resample(wave, native_sr, sr)
            return wave

        source_audio = to_model_sr(source_wave, source_sr)
        ref_audio = to_model_sr(ref_wave[:ref_sr * 25], ref_sr)

        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        S_alt = self._whisper_features(converted_waves_16k)
        S_ori = self._whisper_features(ref_waves_16k)

        mel = mel_fn(source_audio)
        mel2 = mel_fn(ref_audio)

        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        if f0_condition:
            F0_ori = torch.from_numpy(self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)).to(self.device)[None]
            F0_alt = torch.from_numpy(self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)).to(self.device)[None]

            voiced_mask = F0_alt > 1
            shifted_f0_alt = F0_alt.clone()
            if auto_f0_adjust:
                log_ori = torch.log(F0_ori[F0_ori > 1] + 1e-5)
                log_alt = torch.log(F0_alt[voiced_mask] + 1e-5)
                shifted_f0_alt[voiced_mask] = torch.exp(log_alt - log_alt.median() + log_ori.median())
            if pitch_shift != 0:
                shifted_f0_alt[voiced_mask] *= 2 ** (pitch_shift / 12)
        else:
            F0_ori = shifted_f0_alt = None

        cond, _, _, _, _ = inference_module.length_regulator(
            S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt)
        prompt_condition, _, _, _, _ = inference_module.length_regulator(
            S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori)

        vocoder_dtype = next(bigvgan_fn.parameters()).dtype
        max_source_window = max_context_window - mel2.size(2)
        processed_frames = 0
        generated = []
        previous_chunk = None

        while processed_frames < cond.size(1):
            chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
            is_last_chunk = processed_frames + max_source_window >= cond.size(1)
            cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                vc_target = inference_module.cfm.inference(
                    cat_condition,
                    torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                    mel2, style2, None, diffusion_steps,
                    inference_cfg_rate=inference_cfg_rate,
                )
                vc_target = vc_target[:, :, mel2.size(-1):]

            vc_wave = bigvgan_fn(vc_target.to(vocoder_dtype))[0].float()

            wave_np = vc_wave[0].cpu().numpy()
            if processed_frames == 0:
                if is_last_chunk:
                    generated.append(wave_np)
                    break
                generated.append(wave_np[:-overlap_wave_len])
                previous_chunk = vc_wave[0, -overlap_wave_len:]
            elif is_last_chunk:
                generated.append(self._crossfade(previous_chunk.cpu().numpy(), wave_np, overlap_wave_len))
            else:
                generated.append(self._crossfade(previous_chunk.cpu().numpy(), wave_np[:-overlap_wave_len],
                                                overlap_wave_len))
                previous_chunk = vc_wave[0, -overlap_wave_len:]

            processed_frames += vc_target.size(2) - self.overlap_frame_len

        return np.concatenate(generated), sr


_wrappers = {}


def get_wrapper(device: torch.device, half_vocoder: bool) -> SeedVCWrapper:
    key = (device.type, half_vocoder)
    if key not in _wrappers:
        _wrappers[key] = SeedVCWrapper(device, half_vocoder)
    return _wrappers[key]


class ROCmSeedVoiceConversion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_audio": ("AUDIO", {"tooltip": "Recording whose content and rhythm are kept."}),
                "reference_audio": ("AUDIO", {"tooltip": "Voice to adopt (3-25 s; longer is truncated to 25 s)."}),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1,
                                  "tooltip": "Flow-matching steps. Below ~16 starts smearing detail."}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05,
                                    "tooltip": "Length adjust factor on the output duration."}),
                "inference_cfg_rate": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                                 "tooltip": "Classifier-free guidance rate. 0 disables it and halves sampling cost."}),
                "f0_condition": ("BOOLEAN", {"default": False,
                                             "tooltip": "Required for singing voice conversion; uses the 44.1 kHz F0 model."}),
                "auto_f0_adjust": ("BOOLEAN", {"default": True,
                                               "tooltip": "Match the source pitch level to the reference. F0 model only."}),
                "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1,
                                        "tooltip": "Extra semitone shift on the source pitch. F0 model only."}),
                "precision": (["fp16", "fp32"], {"default": "fp16",
                                                 "tooltip": "fp16 runs the BigVGAN vocoder at half precision; ~6x faster on gfx1151."}),
                "unload_model": ("BOOLEAN", {"default": False,
                                             "tooltip": "Free the loaded models after this run. Keep off: reloading costs ~30 s."}),
            },
        }

    CATEGORY = "🎛 ROCm/audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "convert"
    DESCRIPTION = "Zero-shot voice conversion (Seed-VC) for speech or singing."

    def convert(self, source_audio, reference_audio, steps, speed, inference_cfg_rate,
                f0_condition, auto_f0_adjust, pitch_shift, precision, unload_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half_vocoder = precision == "fp16" and device.type == "cuda"

        try:
            wrapper = get_wrapper(device, half_vocoder)

            def unpack(audio):
                wave = audio["waveform"].squeeze(0)
                if wave.ndim > 1:
                    wave = wave.mean(dim=0)
                return wave, int(audio["sample_rate"])

            source_wave, source_sr = unpack(source_audio)
            ref_wave, ref_sr = unpack(reference_audio)

            gentle_memory_cleanup()
            wave, sr = wrapper.convert(
                source_wave, source_sr, ref_wave, ref_sr,
                diffusion_steps=steps, length_adjust=speed,
                inference_cfg_rate=inference_cfg_rate, f0_condition=f0_condition,
                auto_f0_adjust=auto_f0_adjust, pitch_shift=pitch_shift,
            )
        finally:
            if unload_model:
                _wrappers.clear()
                gentle_memory_cleanup()

        waveform = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0).unsqueeze(0)
        return ({"waveform": waveform, "sample_rate": sr},)
