"""
Integration tests for ROCm Text Generate LTX2 Prompt node.

Verifies the node is exported from core and runnable with a mock clip.
Full registry check (NODE_CLASS_MAPPINGS) would require loading all nodes
and ComfyUI mocks; output parity with ComfyUI TextGenerateLTX2Prompt on a
real clip/seed would require a full ComfyUI + Gemma environment and is not run in CI.
"""
import pytest
import sys
from unittest.mock import Mock

sys.path.insert(0, "/home/nino/ComfyUI/custom_nodes/rocm-ninodes")


class MockClip:
    def tokenize(self, text, image=None, skip_template=False, min_length=1):
        return [[(1, 1.0), (2, 1.0)]]

    def generate(
        self, tokens, do_sample, max_length, temperature, top_k, top_p, min_p, repetition_penalty, seed
    ):
        return [106]

    def decode(self, generated_ids, skip_special_tokens=True):
        return "Style: cinematic. Integration test output."


class TestTextgenLTX2Integration:
    """Integration tests for ROCm LTX2 prompt node."""

    def test_node_exported_from_core(self):
        from rocm_nodes.core import __all__
        assert "ROCmTextGenerateLTX2Prompt" in __all__
        from rocm_nodes.core.textgen_ltx2 import ROCmTextGenerateLTX2Prompt
        assert ROCmTextGenerateLTX2Prompt is not None
        assert hasattr(ROCmTextGenerateLTX2Prompt, "INPUT_TYPES")
        assert hasattr(ROCmTextGenerateLTX2Prompt, "generate")

    def test_node_instantiation_and_generate(self):
        from rocm_nodes.core.textgen_ltx2 import ROCmTextGenerateLTX2Prompt
        node = ROCmTextGenerateLTX2Prompt()
        mock_clip = MockClip()
        result = node.generate(
            clip=mock_clip,
            prompt="a cat",
            max_length=32,
            sampling_mode="on",
            seed=123,
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert "Integration test output" in result[0]
