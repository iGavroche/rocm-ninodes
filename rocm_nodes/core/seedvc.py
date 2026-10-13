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

import logging
import os
from typing import Optional

import numpy as np
import torch
import torchaudio
import yaml
from scipy.ndimage import median_filter, map_coordinates, uniform_filter1d
from scipy.signal import find_peaks
import comfy.utils
import folder_paths
from transformers import AutoFeatureExtractor, WhisperModel

from ..seedvc.modules.commons import build_model, load_checkpoint, recursive_munch
from ..seedvc.modules.campplus.DTDNN import CAMPPlus
from ..seedvc.modules.bigvgan import bigvgan
from ..seedvc.modules.audio import mel_spectrogram
from ..seedvc.modules.rmvpe import RMVPE

from ..utils.memory import gentle_memory_cleanup

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "seedvc", "configs")

KARAOKE_MODEL = "mel_band_roformer_karaoke_becruily.ckpt"

_separator = None


def _get_separator():
    """Mel-band roformer karaoke separator, cached across executions."""
    global _separator
    if _separator is None:
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            raise RuntimeError(
                "separate_source needs the audio-separator package: pip install audio-separator"
            ) from e
        model_dir = os.path.join(folder_paths.models_dir, "TTS", "Seed-VC", "separator")
        os.makedirs(model_dir, exist_ok=True)
        sep = Separator(model_file_dir=model_dir, log_level=logging.ERROR)
        sep.load_model(KARAOKE_MODEL)
        _separator = sep
    return _separator


def separate_mix(wave: torch.Tensor, sr: int):
    """Split a mix into (vocal, vocal_sr, instrumental, instrumental_sr) float32 waves."""
    import tempfile
    import scipy.io.wavfile as wavio

    sep = _get_separator()
    with tempfile.TemporaryDirectory() as tmp:
        pcm = (wave.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
        if pcm.ndim == 2:
            pcm = pcm.T  # [channels, samples] -> [samples, channels]
        wavio.write(os.path.join(tmp, "source.wav"), sr, pcm)
        sep.output_dir = tmp
        if sep.model_instance:
            sep.model_instance.output_dir = tmp
        out_files = sep.separate(os.path.join(tmp, "source.wav"))

        def read_stem(tag):
            name = next(f for f in out_files if tag in f)
            rate, data = wavio.read(name if os.path.isabs(name) else os.path.join(tmp, name))
            t = torch.from_numpy(data.astype(np.float32) / 32768.0)
            if t.ndim == 2:
                t = t.T  # -> [channels, samples]
            return t, rate

        vocal, vocal_sr = read_stem("(Vocals)")
        instrumental, inst_sr = read_stem("(Instrumental)")

    vocal = vocal.mean(dim=0) if vocal.ndim > 1 else vocal
    return vocal, vocal_sr, instrumental, inst_sr


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

F0_GAP_BRIDGE_FRAMES = 10


def _bridge_f0_gaps(f0: torch.Tensor, max_gap: int) -> torch.Tensor:
    """Linear-interpolate short unvoiced dropouts so sustained notes stay continuous."""
    flat = f0.reshape(-1)
    voiced = flat > 1
    if voiced.sum() < 2:
        return f0
    idx = np.arange(flat.numel())
    v = voiced.cpu().numpy()
    filled = torch.from_numpy(np.interp(idx, idx[v], flat.cpu().numpy()[v])).to(flat.device)
    out = flat.clone()
    change = torch.diff(voiced.int())
    starts = (change == -1).nonzero(as_tuple=True)[0].tolist()
    ends = (change == 1).nonzero(as_tuple=True)[0].tolist()
    for s in starts:
        after = [e for e in ends if e > s]
        e = after[0] if after else flat.numel() - 1
        if 0 < e - s <= max_gap:
            out[s + 1:e + 1] = filled[s + 1:e + 1]
    return out.reshape(f0.shape)


MICRO_PITCH_SKELETON_FRAMES = 31


def _f0_residual(f0: torch.Tensor) -> Optional[np.ndarray]:
    """Log-pitch deviation from the smoothed note skeleton; 0 on unvoiced frames."""
    flat = f0.reshape(-1).float().cpu().numpy()
    voiced = flat > 1
    if voiced.sum() < 2:
        return None
    filled = np.interp(np.arange(flat.size), np.flatnonzero(voiced), flat[voiced])
    skeleton = median_filter(np.log(filled), size=MICRO_PITCH_SKELETON_FRAMES, mode="nearest")
    resid = np.log(np.maximum(flat, 1.0)) - skeleton
    resid[~voiced] = 0.0
    return resid


def _micro_pitch_ratio(src_f0: torch.Tensor, out_f0: torch.Tensor, strength: float) -> Optional[np.ndarray]:
    """Per-frame pitch multiplier transferring the source's vibrato/portamento onto the output.

    The transfer fades out inside fast note runs and ramps in after each note
    onset: natural voices step between close notes instead of carrying wobble
    through them, and vibrato develops after the attack, not during it.
    """
    rs = _f0_residual(src_f0)
    ro = _f0_residual(out_f0)
    if rs is None or ro is None:
        return None
    n = min(rs.size, ro.size)
    delta = torch.from_numpy(rs[:n] - ro[:n])
    _, _, fast_w, attack_w = _note_onset_density(src_f0)
    if fast_w is not None:
        delta = delta * torch.from_numpy(fast_w[:n] * attack_w[:n])
    delta = torch.nn.functional.avg_pool1d(delta[None, None], 5, 1, 2)[0, 0]
    return (torch.exp(strength * delta).clamp(0.943, 1.060)).numpy()


def _apply_micro_pitch(wave: np.ndarray, ratio: np.ndarray, frame_len: int) -> np.ndarray:
    """Time-warp resampling y(t) = x(phi(t)) with phi' = ratio; duration preserved."""
    pos = np.interp(np.arange(wave.size), np.arange(ratio.size) * frame_len + frame_len / 2, ratio)
    pos /= pos.mean()
    phase = np.cumsum(pos) - 1
    phase *= (wave.size - 1) / phase[-1]
    return map_coordinates(wave, [phase], order=3, mode="nearest").astype(np.float32)


FAST_RUN_FADE_HZ = (2.5, 5.0)
ATTACK_RAMP_FRAMES = 10


def _note_onset_density(f0: torch.Tensor):
    """(semitone contour, stepped contour, 0..1 slow-ness weight, 0..1 attack ramp) or (None,)*4.

    Note onsets are prominence peaks of the semitone contour after a median
    filter removes vibrato, so they mark note changes rather than wobble.
    The weight is 1 where notes are sustained and 0 where they arrive faster
    than FAST_RUN_FADE_HZ. The attack ramp is 0 at each onset, reaching 1
    ATTACK_RAMP_FRAMES later.
    """
    flat = f0.reshape(-1).float().cpu().numpy()
    v = flat > 1
    if v.sum() < 100:
        return None, None, None, None
    st = 12 * np.log2(np.maximum(flat, 1.0))
    st = np.where(v, st, np.interp(np.arange(st.size), np.flatnonzero(v), st[v]))
    sm = median_filter(st, size=15, mode="nearest")
    ev = np.zeros(st.size, bool)
    for sig in (sm, -sm):
        pk, _ = find_peaks(sig, prominence=1.0, distance=8)
        ev[pk] = True
    dens = uniform_filter1d(ev.astype(float), size=100) * 100.0
    lo, hi = FAST_RUN_FADE_HZ
    weight = uniform_filter1d(np.clip((hi - dens) / (hi - lo), 0.0, 1.0), size=51)
    last = np.maximum.accumulate(np.where(ev, np.arange(ev.size), -1))
    attack = np.clip((np.arange(ev.size) - last) / float(ATTACK_RAMP_FRAMES), 0.0, 1.0)
    return st, sm, weight, attack


def _quantize_fast_runs(f0: torch.Tensor) -> torch.Tensor:
    """Step-ify the pitch contour inside fast note runs, before conditioning.

    The model sings whatever contour it is given, so a glissando source comes
    out as a glissando; natural voices step between close notes. Where the
    note-onset rate is high the contour is replaced by its median-filtered
    (stepped) version.
    """
    st, sm, slow_w, _ = _note_onset_density(f0)
    if st is None:
        return f0
    flat = f0.reshape(-1).float().cpu().numpy()
    stepped = slow_w * st + (1.0 - slow_w) * sm
    out = np.where(flat > 1, np.power(2.0, stepped / 12.0), flat)
    return torch.from_numpy(out).reshape(f0.shape).to(device=f0.device, dtype=f0.dtype)



class SeedVCWrapper:
    """All Seed-VC submodels for one conversion pipeline."""

    overlap_frame_len = 64

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
            os.path.join(model_path, "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"),
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
                f0_condition=False, auto_f0_adjust=True, pitch_shift=0, prompt_length=10.0,
                vibrato_transfer=1.0, progress_cb=None):
        """Convert source timbre to the reference voice. Returns (float32 numpy, sr).

        progress_cb(done_frames, total_frames) is called once per chunk window.
        """
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
        # Only the in-context prompt is truncated; timbre and the pitch-level
        # match still see the whole reference.
        prompt_audio = ref_audio[:, :int(sr * prompt_length)]

        ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
        prompt_waves_16k = torchaudio.functional.resample(prompt_audio, sr, 16000)
        converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)

        S_alt = self._whisper_features(converted_waves_16k)
        S_ori = self._whisper_features(prompt_waves_16k)

        mel = mel_fn(source_audio)
        mel2 = mel_fn(prompt_audio)
        target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

        feat2 = torchaudio.compliance.kaldi.fbank(
            ref_waves_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.campplus_model(feat2.unsqueeze(0))

        if f0_condition:
            F0_ref = torch.from_numpy(self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)).to(self.device)[None]
            F0_ori = F0_ref[:, :prompt_waves_16k.size(-1) // 160 + 1]
            F0_alt = torch.from_numpy(self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)).to(self.device)[None]

            voiced_mask = F0_alt > 1
            shifted_f0_alt = F0_alt.clone()
            if auto_f0_adjust:
                log_ori = torch.log(F0_ref[F0_ref > 1] + 1e-5)
                log_alt = torch.log(F0_alt[voiced_mask] + 1e-5)
                shifted_f0_alt[voiced_mask] = torch.exp(log_alt - log_alt.median() + log_ori.median())
            if pitch_shift != 0:
                shifted_f0_alt[voiced_mask] *= 2 ** (pitch_shift / 12)
            shifted_f0_alt = _bridge_f0_gaps(shifted_f0_alt, F0_GAP_BRIDGE_FRAMES)
            shifted_f0_alt = _quantize_fast_runs(shifted_f0_alt)
            F0_ori = _bridge_f0_gaps(F0_ori, F0_GAP_BRIDGE_FRAMES)
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

        total_frames = cond.size(1)
        while processed_frames < total_frames:
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
                break
            else:
                generated.append(self._crossfade(previous_chunk.cpu().numpy(), wave_np[:-overlap_wave_len],
                                                overlap_wave_len))
                previous_chunk = vc_wave[0, -overlap_wave_len:]

            processed_frames += vc_target.size(2) - self.overlap_frame_len
            if progress_cb:
                progress_cb(processed_frames, total_frames)

        if progress_cb:
            progress_cb(total_frames, total_frames)
        wave = np.concatenate(generated)
        if f0_condition and vibrato_transfer > 0:
            out16k = torchaudio.functional.resample(
                torch.from_numpy(wave).unsqueeze(0).to(self.device), sr, 16000)
            out_f0 = torch.from_numpy(self.rmvpe.infer_from_audio(out16k[0], thred=0.03))[None]
            ratio = _micro_pitch_ratio(F0_alt, out_f0, vibrato_transfer)
            if ratio is not None:
                wave = _apply_micro_pitch(wave, ratio, sr // 100)
        return wave, sr


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
                "prompt_length": ("FLOAT", {"default": 10.0, "min": 3.0, "max": 25.0, "step": 0.5,
                                            "tooltip": "Seconds of reference used as the in-context prompt. Shorter = fewer chunk seams and faster; timbre and pitch matching still use the full reference."}),
                "f0_condition": ("BOOLEAN", {"default": False,
                                             "tooltip": "Required for singing voice conversion; uses the 44.1 kHz F0 model."}),
                "auto_f0_adjust": ("BOOLEAN", {"default": True,
                                               "tooltip": "Match the source pitch level to the reference. F0 model only."}),
                "pitch_shift": ("INT", {"default": 0, "min": -24, "max": 24, "step": 1,
                                        "tooltip": "Extra semitone shift on the source pitch. F0 model only."}),
                "vibrato_transfer": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                               "tooltip": "Copy the source's vibrato and note glides onto the converted vocal. 0 keeps the raw model output; lower for a more literal pitch. F0 model only."}),
                "precision": (["fp16", "fp32"], {"default": "fp16",
                                                 "tooltip": "fp16 runs the BigVGAN vocoder at half precision; ~6x faster on gfx1151."}),
                "unload_model": ("BOOLEAN", {"default": False,
                                             "tooltip": "Free the loaded models after this run. Keep off: reloading costs ~30 s."}),
                "separate_source": ("BOOLEAN", {"default": False,
                                                 "tooltip": "Split the source into vocal + instrumental first; converts the vocal and outputs the instrumental on the accompaniment socket. Needs audio-separator."}),
            },
        }

    CATEGORY = "🎛 ROCm/audio"
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio", "accompaniment")
    FUNCTION = "convert"
    DESCRIPTION = "Zero-shot voice conversion (Seed-VC) for speech or singing."

    def convert(self, source_audio, reference_audio, steps, speed, inference_cfg_rate,
                f0_condition, auto_f0_adjust, pitch_shift, precision, unload_model, separate_source,
                prompt_length=10.0, vibrato_transfer=1.0):
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

            accompaniment = None
            if separate_source:
                vocal, vocal_sr, instrumental, inst_sr = separate_mix(source_wave, source_sr)
                source_wave, source_sr = vocal, vocal_sr
                accompaniment = {"waveform": instrumental.unsqueeze(0), "sample_rate": inst_sr}

            pbar = comfy.utils.ProgressBar(1000)

            def progress(done, total):
                pbar.update_absolute(int(1000 * done / total))

            gentle_memory_cleanup()
            wave, sr = wrapper.convert(
                source_wave, source_sr, ref_wave, ref_sr,
                diffusion_steps=steps, length_adjust=speed,
                inference_cfg_rate=inference_cfg_rate, f0_condition=f0_condition,
                auto_f0_adjust=auto_f0_adjust, pitch_shift=pitch_shift,
                prompt_length=prompt_length, vibrato_transfer=vibrato_transfer,
                progress_cb=progress,
            )
        finally:
            if unload_model:
                _wrappers.clear()
                gentle_memory_cleanup()

        waveform = torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0).unsqueeze(0)
        if accompaniment is None:
            accompaniment = {"waveform": torch.zeros_like(waveform), "sample_rate": sr}
        return ({"waveform": waveform, "sample_rate": sr}, accompaniment)
