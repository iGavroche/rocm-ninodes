# Checkpoint cache for repeated API runs

When you run the same ComfyUI workflow multiple times via the API (e.g. same prompt, same checkpoint), each run normally re-executes the whole graph and reloads the checkpoint from disk. That can cause high NVMe usage and RAM growth until the system becomes unstable.

## Solution: use the ROCm Checkpoint Loader with cache

1. **Use ROCm Checkpoint Loader** instead of ComfyUI’s built-in **CheckpointLoaderSimple** in your workflow (or API template).
2. Leave **use_cache** enabled (default: True) so that if the same checkpoint is already loaded, the node returns the cached (model, clip, vae) and does **not** reload from disk.
3. Use **force_reload** only when you need a fresh load (e.g. after replacing the checkpoint file).

## Behavior

- **First run** (or first run after loading a different checkpoint): checkpoint is loaded from disk and cached.
- **Later runs** with the same checkpoint: cache hit; no disk load, same (model, clip, vae) returned.
- **Switching checkpoint**: loading a different `ckpt_name` evicts the previous cache (single-slot), runs a gentle memory cleanup, then loads the new checkpoint. This avoids unbounded RAM on gfx1151.

## Inputs

| Input           | Default | Description |
|----------------|---------|-------------|
| `ckpt_name`    | (required) | Checkpoint filename. |
| `use_cache`    | True    | If True and this checkpoint is already cached, return cached tuple (no reload). |
| `force_reload` | False   | If True, load from disk and update cache (ignores cache for this call). |
| `compatibility_mode` | False | Extra validation for quantized/unusual models. |

## When to use

- Workflows executed repeatedly via API with the same checkpoint.
- Avoiding NVMe and RAM growth from repeated full checkpoint loads.
- gfx1151 (and other architectures) when you want at most one checkpoint in memory at a time (single-slot eviction).

## See also

- [LOADER_NODES_GUIDE.md](LOADER_NODES_GUIDE.md) – ROCm Checkpoint Loader overview and features.
