"""
Unit tests for ROCm Text Generate LTX2 Prompt node.
"""
import pytest
import sys
from unittest.mock import Mock

# Ensure ComfyUI and rocm-ninodes are on path (conftest does ComfyUI; add rocm-ninodes)
sys.path.insert(0, "/home/nino/ComfyUI/custom_nodes/rocm-ninodes")

from rocm_nodes.core.textgen_ltx2 import (
    ROCmTextGenerateLTX2Prompt,
    LTX2_T2V_SYSTEM_PROMPT,
    LTX2_I2V_SYSTEM_PROMPT,
)


class MockClip:
    """Mock CLIP that implements tokenize, generate, decode for LTX2-style nodes."""

    def tokenize(self, text, image=None, skip_template=False, min_length=1):
        return [[(1, 1.0), (2, 1.0)]]  # minimal token list

    def generate(
        self,
        tokens,
        do_sample,
        max_length,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        seed,
    ):
        # Return a short list of token ids (e.g. stop token 106)
        return [106]

    def decode(self, generated_ids, skip_special_tokens=True):
        return "Style: realistic. Mock generated prompt."


@pytest.fixture
def mock_clip():
    return MockClip()


@pytest.fixture
def node():
    return ROCmTextGenerateLTX2Prompt()


class TestROCmTextGenerateLTX2PromptSchema:
    """Test node schema (INPUT_TYPES, RETURN_TYPES, etc.)."""

    def test_input_types_structure(self, node):
        inp = node.INPUT_TYPES()
        assert "required" in inp
        assert "optional" in inp
        assert "clip" in inp["required"]
        assert "prompt" in inp["required"]
        assert "max_length" in inp["required"]
        assert "image" in inp["optional"]
        assert "sampling_mode" in inp["optional"]
        assert "seed" in inp["optional"]

    def test_return_types(self, node):
        assert node.RETURN_TYPES == ("STRING",)
        assert node.RETURN_NAMES == ("generated_text",)
        assert node.FUNCTION == "generate"
        assert "ROCm" in node.CATEGORY
        assert "generative" in node.CATEGORY.lower()


class TestLTX2PromptFormatting:
    """Test that LTX2 prompt formatting matches ComfyUI (T2V vs I2V)."""

    def test_t2v_formatted_contains_system_prompt(self):
        # Same logic as node: no image -> T2V
        formatted = (
            f"<start_of_turn>system\n{LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
            "<start_of_turn>user\nUser Raw Input Prompt: hello.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert "Creative Assistant" in formatted
        assert "User Raw Input Prompt: hello." in formatted
        assert "<start_of_turn>model" in formatted
        assert "<image_soft_token>" not in formatted

    def test_i2v_formatted_contains_image_token(self):
        formatted = (
            f"<start_of_turn>system\n{LTX2_I2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
            "<start_of_turn>user\n\n<image_soft_token>\n\n"
            "User Raw Input Prompt: run.<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        assert "<image_soft_token>" in formatted
        assert "User Raw Input Prompt: run." in formatted


class TestROCmTextGenerateLTX2PromptGenerate:
    """Test generate() with mock clip."""

    def test_generate_returns_tuple_of_string(self, node, mock_clip):
        result = node.generate(
            clip=mock_clip,
            prompt="test prompt",
            max_length=10,
            image=None,
            sampling_mode="on",
            temperature=0.7,
            top_k=64,
            top_p=0.95,
            min_p=0.05,
            repetition_penalty=1.05,
            seed=42,
        )
        assert isinstance(result, tuple)
        assert len(result) == 1
        assert isinstance(result[0], str)
        assert "Mock generated prompt" in result[0]

    def test_generate_sampling_off(self, node, mock_clip):
        result = node.generate(
            clip=mock_clip,
            prompt="another",
            max_length=5,
            sampling_mode="off",
            seed=0,
        )
        assert isinstance(result, tuple)
        assert result[0] == "Style: realistic. Mock generated prompt."

    def test_generate_with_image_uses_i2v_path(self, node, mock_clip):
        # Pass a dummy image tensor so the node uses I2V formatting; mock clip ignores it
        import torch
        fake_image = torch.zeros(1, 64, 64, 3)  # (B, H, W, C) ComfyUI image
        result = node.generate(
            clip=mock_clip,
            prompt="with image",
            max_length=20,
            image=fake_image,
        )
        assert isinstance(result[0], str)
        assert len(result[0]) > 0
        assert "Mock generated prompt" in result[0]
