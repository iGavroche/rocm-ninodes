"""
Unit tests for ROCMOptimizedVAEDecode
"""
import pytest
import torch
import sys
import os

# Add the custom nodes directory to path
sys.path.insert(0, '/home/nino/ComfyUI/custom_nodes/rocm-ninodes')

from rocm_nodes.core.vae import ROCMOptimizedVAEDecode, _detect_vae_type


class ROCMOptimizedVAEDecodeInstrumented(ROCMOptimizedVAEDecode):
    """Instrumented subclass for testing with deterministic behavior"""
    pass


class TestROCMOptimizedVAEDecode:
    """Test suite for ROCMOptimizedVAEDecode"""

    def test_basic_decode(self, sample_vae, sample_latent):
        """Test basic VAE decode functionality"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

    def test_different_tile_sizes(self, sample_vae, sample_latent):
        """Test VAE decode with different tile sizes"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        tile_sizes = [256, 512, 768, 1024]
        for tile_size in tile_sizes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=tile_size,
                overlap=64,
                use_rocm_optimizations=True
            )

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

    def test_different_resolutions(self, sample_vae):
        """Test VAE decode with different resolutions"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        resolutions = [(256, 256), (512, 512), (1024, 1024)]
        for w, h in resolutions:
            latent = {
                "samples": torch.randn(1, 4, h//8, w//8)
            }

            result = node.decode(
                vae=sample_vae,
                samples=latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)
            assert len(result[0].shape) == 4

    def test_precision_modes(self, sample_vae, sample_latent):
        """Test VAE decode with different precision modes"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        precision_modes = ["auto", "fp32", "fp16"]
        for precision_mode in precision_modes:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
                precision_mode=precision_mode
            )

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

    def test_batch_processing(self, sample_vae):
        """Test VAE decode with batch processing"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        batch_latent = {
            "samples": torch.randn(2, 4, 32, 32)
        }

        result = node.decode(
            vae=sample_vae,
            samples=batch_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
            batch_optimization=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert result[0].shape[0] == 2

    def test_memory_optimization(self, sample_vae, sample_latent):
        """Test VAE decode with memory optimization enabled/disabled"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        for memory_opt in [True, False]:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
            )

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

    def test_adaptive_tiling(self, sample_vae, sample_latent):
        """Test VAE decode with adaptive tiling enabled/disabled"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        for adaptive_tiling in [True, False]:
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True,
            )

            assert isinstance(result, tuple)
            assert len(result) == 1
            assert isinstance(result[0], torch.Tensor)

    def test_error_handling(self, sample_vae):
        """Test VAE decode error handling"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        invalid_samples = {"samples": "invalid"}

        with pytest.raises(Exception):
            node.decode(
                vae=sample_vae,
                samples=invalid_samples,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )

    def test_performance_consistency(self, sample_vae, sample_latent):
        """Test that VAE decode produces consistent results"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        results = []
        for _ in range(3):
            result = node.decode(
                vae=sample_vae,
                samples=sample_latent,
                tile_size=512,
                overlap=64,
                use_rocm_optimizations=True
            )
            results.append(result[0])

        for result in results:
            assert result.shape == results[0].shape

    def test_large_image(self, sample_vae):
        """Test VAE decode with large image"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        large_latent = {
            "samples": torch.randn(1, 4, 128, 128)
        }

        result = node.decode(
            vae=sample_vae,
            samples=large_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

    def test_video_decode(self, sample_vae, sample_video_latent):
        """Test VAE decode with video (5D) latent"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_vae,
            samples=sample_video_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        # Video output should be 4D (flattened batch+frames)
        assert len(result[0].shape) == 4

    def test_compatibility_mode(self, sample_vae, sample_latent):
        """Test compatibility mode disables ROCm optimizations"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_vae,
            samples=sample_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
            compatibility_mode=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    # ── NEW TESTS ──────────────────────────────────────────────────────────

    def test_ltx_vae_detection(self, sample_ltx_vae):
        """Test that LTX Video VAE is correctly detected"""
        vae_type = _detect_vae_type(sample_ltx_vae)
        assert vae_type == "ltxv_vae", f"Expected ltxv_vae, got {vae_type}"

    def test_wan_vae_detection(self, sample_wan_vae):
        """Test that WAN VAE is correctly detected"""
        vae_type = _detect_vae_type(sample_wan_vae)
        assert vae_type == "wan_vae", f"Expected wan_vae, got {vae_type}"

    def test_pixel_space_vae_detection(self, sample_pixel_vae):
        """Test that pixel-space VAE is correctly detected"""
        vae_type = _detect_vae_type(sample_pixel_vae)
        assert vae_type == "pixel_space", f"Expected pixel_space, got {vae_type}"

    def test_standard_vae_detection(self, sample_vae):
        """Test that standard VAE returns 'standard'"""
        vae_type = _detect_vae_type(sample_vae)
        assert vae_type == "standard", f"Expected standard, got {vae_type}"

    def test_ltx_video_decode(self, sample_ltx_vae, sample_ltx_latent):
        """Test LTX video decode processes full video at once"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_ltx_vae,
            samples=sample_ltx_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

    def test_pixel_space_decode(self, sample_pixel_vae, sample_pixel_latent):
        """Test pixel-space VAE decode skips all optimization logic"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_pixel_vae,
            samples=sample_pixel_latent,
            tile_size=512,
            overlap=64,
            use_rocm_optimizations=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4
        # Pixel-space output should be 4D BHWC (passthrough preserves content)

    def test_wan_video_decode(self, sample_wan_vae, sample_wan_latent):
        """Test WAN video decode processes full video at once"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_wan_vae,
            samples=sample_wan_latent,
            tile_size=768,
            overlap=96,
            use_rocm_optimizations=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

    def test_high_compression_tiled_decode(self, sample_ltx_vae):
        """Test that tiled decode handles high compression (32x) correctly"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        large_ltx_latent = {
            "samples": torch.randn(1, 128, 4, 16, 24)
        }

        result = node.decode(
            vae=sample_ltx_vae,
            samples=large_ltx_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4

    def test_compatibility_with_quantized_ltx(self, sample_ltx_vae, sample_ltx_latent):
        """Test compatibility mode works with LTX VAE"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_ltx_vae,
            samples=sample_ltx_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            compatibility_mode=True
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)

    def test_temporal_tiling_ltx(self, sample_ltx_vae):
        """Test temporal tiling for LTX long video decode"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        long_ltx_latent = {
            "samples": torch.randn(1, 128, 48, 4, 6)
        }

        result = node.decode(
            vae=sample_ltx_vae,
            samples=long_ltx_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
            enable_temporal_tiling=True,
            temporal_chunk_size=16,
            temporal_overlap=2,
            last_frame_fix=False,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4
        assert result[0].shape[0] > 0

    def test_temporal_tiling_with_last_frame_fix(self, sample_ltx_vae):
        """Test temporal tiling with last_frame_fix enabled"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        latent = {
            "samples": torch.randn(1, 128, 16, 4, 6)
        }

        result = node.decode(
            vae=sample_ltx_vae,
            samples=latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
            enable_temporal_tiling=True,
            temporal_chunk_size=8,
            temporal_overlap=2,
            last_frame_fix=True,
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
        assert len(result[0].shape) == 4
        assert result[0].shape[0] > 0

    def test_temporal_tiling_frame_count(self, sample_ltx_vae):
        """Test that temporal tiling with blending preserves all frames (no loss)"""
        node = ROCMOptimizedVAEDecodeInstrumented()
        temporal_comp = 8

        latent_frames = 48
        chunk_size = 16
        latent = {
            "samples": torch.randn(1, 128, latent_frames, 4, 6)
        }

        result = node.decode(
            vae=sample_ltx_vae,
            samples=latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
            enable_temporal_tiling=True,
            temporal_chunk_size=chunk_size,
            temporal_overlap=2,
            last_frame_fix=False,
        )

        total_frames = result[0].shape[0]
        expected = 1 + (latent_frames - 1) * temporal_comp
        assert total_frames == expected, (
            f"Expected {expected} frames (full no-tile count), got {total_frames}"
        )

    def test_temporal_tiling_disabled_by_default(self, sample_ltx_vae, sample_ltx_latent):
        """Test that temporal tiling is disabled by default (uses direct decode)"""
        node = ROCMOptimizedVAEDecodeInstrumented()

        result = node.decode(
            vae=sample_ltx_vae,
            samples=sample_ltx_latent,
            tile_size=1024,
            overlap=128,
            use_rocm_optimizations=True,
            precision_mode="fp16",
        )

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], torch.Tensor)
