"""
ROCm H3 EasyCache — step-caching tuned for 4-step MiniMax H3 workflows.

Wraps ComfyUI's core EasyCache with defaults optimized for the fast H3
pipeline (4 inference steps, Spectrum H3 forecasting, res_multistep sampler).
"""

import logging

import torch
import comfy.patcher_extension
import comfy.model_patcher
from comfy_api.latest import io

from comfy_extras.nodes_easycache import (
    EasyCacheHolder,
    easycache_forward_wrapper,
    easycache_calc_cond_batch_wrapper,
    easycache_sample_wrapper,
)

LOG = logging.getLogger(__name__)


class ROCmH3EasyCache(io.ComfyNode):
    """EasyCache pre-tuned for 4-step MiniMax H3 on ROCm.

    Uses the core EasyCache defaults (threshold 0.2, 15-95% range) which are
    safe for short 4-step schedules. With 4 steps, EasyCache can skip 1 of 3
    eligible steps for a ~25% speedup when the denoising trajectory is smooth.
    Place between the model loader/LoRA chain and the sampler.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ROCmH3EasyCache",
            display_name="ROCm H3 EasyCache",
            category="ROCm Ninodes/Optimization",
            description=(
                "Step-caching for 4-step MiniMax H3. Skips transformer calls "
                "when denoising change is below threshold. Tuned for H3 + "
                "Spectrum on ROCm."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 model (after loader + LoRA + Spectrum)",
                ),
                io.Float.Input(
                    "reuse_threshold",
                    min=0.0, max=3.0, default=0.2, step=0.01,
                    tooltip=(
                        "How aggressive to skip steps. Higher = skip more. "
                        "0.2 is the safe default for 4-step H3. "
                        "Raise to 0.3+ for more speed if quality holds."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    min=0.0, max=1.0, default=0.15, step=0.01, advanced=True,
                    tooltip=(
                        "Relative step to begin caching. 0.15 = start at step 2 "
                        "of 4 (skip step 1 to build cache history)."
                    ),
                ),
                io.Float.Input(
                    "end_percent",
                    min=0.0, max=1.0, default=0.95, step=0.01, advanced=True,
                    tooltip=(
                        "Relative step to stop caching. 0.95 covers steps 2-3 "
                        "of 4 (always run the last step for quality)."
                    ),
                ),
                io.Boolean.Input(
                    "verbose",
                    default=False,
                    advanced=True,
                    tooltip="Log per-step cache decisions.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    tooltip="Model with EasyCache applied",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, reuse_threshold=0.2, start_percent=0.15,
                end_percent=0.95, verbose=False) -> io.NodeOutput:
        model = model.clone()
        model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
            reuse_threshold,
            start_percent,
            end_percent,
            subsample_factor=8,
            offload_cache_diff=False,
            verbose=verbose,
            output_channels=model.model.latent_format.latent_channels,
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "easycache", easycache_sample_wrapper,
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
            "easycache", easycache_calc_cond_batch_wrapper,
        )
        model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            "easycache", easycache_forward_wrapper,
        )
        LOG.info(
            "[H3-EasyCache] threshold=%.2f start=%.0f%% end=%.0f%%",
            reuse_threshold, start_percent * 100, end_percent * 100,
        )
        return io.NodeOutput(model)


__all__ = ["ROCmH3EasyCache"]
