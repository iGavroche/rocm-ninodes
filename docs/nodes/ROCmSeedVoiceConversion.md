# ROCmSeedVoiceConversion

Display name: `ROCm Seed Voice Conversion` · Category: `🎛 ROCm/audio` · Introduced: 2.3.15

Source: `rocm_nodes/core/seedvc.py`, vendored model code in `rocm_nodes/seedvc/`

## What it does

Zero-shot voice conversion (Seed-VC): re-voices a source recording in a reference voice while keeping the words, rhythm, and — with `f0_condition` — the melody. Two DiT flow-matching stages back the node:

- **Speech** (`f0_condition=False`): whisper-small encoder + wavenet vocoder, 22.05 kHz output.
- **Singing** (`f0_condition=True`): 44.1 kHz F0-conditioned model (RMVPE pitch extraction), 44.1 kHz output.

Per run: whisper features + CAM++ speaker embedding (+ RMVPE F0 in singing mode) condition the flow-matching estimator, then BigVGAN vocodes the converted mel. Long inputs are processed in overlapping 30 s windows.

## Required models

Download once into `models/` (shared with other Seed-VC packs):

| Path | Source |
|---|---|
| `models/TTS/Seed-VC/DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth` | Plachta/Seed-VC |
| `models/TTS/Seed-VC/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth` | Plachta/Seed-VC |
| `models/TTS/Seed-VC/campplus_cn_common.bin` | Plachta/Seed-VC |
| `models/TTS/Seed-VC/rmvpe.pt` | Plachta/Seed-VC |
| `models/TTS/whisper-small/` | openai/whisper-small |
| `models/TTS/bigvgan_v2_22khz_80band_256x/` | nvidia/bigvgan_v2_22khz_80band_256x |
| `models/TTS/bigvgan_v2_44khz_128band_512x/` | nvidia/bigvgan_v2_44khz_128band_512x |

The BigVGAN folders must be present on disk; a missing path is treated as an HF repo id and fails. Fetch with `hf download nvidia/bigvgan_v2_22khz_80band_256x --local-dir models/TTS/bigvgan_v2_22khz_80band_256x` (same for the 44 kHz folder).

With `separate_source=True` the node also needs the `audio-separator` package and the mel-band roformer karaoke checkpoint (~1.7 GB) in `models/TTS/Seed-VC/separator/`; audio-separator downloads it there on first use.

## Inputs

| Name | Type | Default | Range / options | Notes |
|---|---|---|---|---|
| `source_audio` | AUDIO | — | required | Content to keep. Downmixed to mono; long input is chunked. |
| `reference_audio` | AUDIO | — | required | Voice to adopt; truncated to 25 s. |
| `steps` | INT | `30` | 1 … 200 | Flow-matching steps. Below ~16 smears detail. |
| `speed` | FLOAT | `1.0` | 0.25 … 4.0 | Output length factor. |
| `inference_cfg_rate` | FLOAT | `0.7` | 0.0 … 1.0 | Classifier-free guidance; 0 disables it and halves sampling cost. |
| `f0_condition` | BOOLEAN | `False` | — | Singing mode: 44.1 kHz F0 model + RMVPE. |
| `auto_f0_adjust` | BOOLEAN | `True` | — | Median-match source pitch to the reference. F0 model only. |
| `pitch_shift` | INT | `0` | -24 … 24 | Extra semitone shift on voiced frames. F0 model only. |
| `precision` | combo | `fp16` | `fp16`, `fp32` | fp16 runs BigVGAN at half precision — ~6x faster vocoding, ~2x faster end to end on gfx1151. Falls back to fp32 on non-CUDA devices. |
| `unload_model` | BOOLEAN | `False` | — | Frees the model set after the run; reloading costs ~30 s. |
| `separate_source` | BOOLEAN | `False` | — | Split the source into vocal + instrumental first (mel-band roformer karaoke). Converts the vocal; the instrumental appears on `accompaniment`. Adds ~13 s per 30 s of audio. |

## Outputs

| Name | Type | Notes |
|---|---|---|
| `audio` | AUDIO | `[1, 1, N]` float waveform at 22.05 kHz (speech) or 44.1 kHz (singing). |
| `accompaniment` | AUDIO | Instrumental of the source, stereo 44.1 kHz. Only when `separate_source=True`, otherwise `None`. Merge with `audio` using the core AudioMerge node. |

## gfx1151 notes

- **fp16 BigVGAN is the main optimization.** On a 12 s singing clip (30 steps) the vocoder is ~91% of upstream runtime in fp32; isolated BigVGAN benchmark: 20.5 s fp32 vs 3.3 s fp16 (6.2x). End to end through the node: 19.2 s fp32 vs 9.9 s fp16 (1.94x), peak 3.7–5.5 GB.
- The DiT estimator runs under fp16 autocast (upstream behavior); the vocoder dtype follows `precision`.
- The full model set (~3.2 GB of checkpoints) is cached in a module-level `_wrappers` dict keyed by `(device, half_vocoder)` and reused across executions; `unload_model=True` clears it.
- `setup_caches(max_batch_size=2, max_seq_length=2048)` shrinks the estimator's rope/KV caches from the upstream 8192-frame reservation.
- Startup warning `Skipped loading some keys due to shape mismatch` is benign: `estimator.input_pos` is a fixed-length positional buffer and the f0 checkpoint carries wavenet-final-layer weights the mlp-final config does not build.
- Vendored code drops the `einops`, `munch`, and `audiotools` dependencies of upstream Seed-VC.

## Typical wiring

```
LoadAudio(source) ──> [source_audio]
                          ROCmSeedVoiceConversion ──> SaveAudio
LoadAudio(reference) ─> [reference_audio]
```

For singing, set `f0_condition=True`; leave it off for speech.

## Troubleshooting

- Node missing at startup: console prints `ROCmSeedVoiceConversion unavailable: …` when a vendored import fails (missing `transformers`, `torchaudio`, or `yaml`).
- `FileNotFoundError` under `models/TTS/Seed-VC`: the four Seed-VC checkpoints are not present — see Required models.
- Output sample rate changes with `f0_condition` (22.05 vs 44.1 kHz); downstream nodes that pin a rate need to follow it.
- `precision=fp16` has no effect on CPU/MPS; the vocoder stays fp32 there.
