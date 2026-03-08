"""
ROCm-optimized LTX2 prompt generation node.

Drop-in replacement for ComfyUI's TextGenerateLTX2Prompt with the same inputs and outputs.
Uses the same LTX2 system prompts and clip.tokenize/generate/decode flow, with optional
device and execution optimizations for ROCm/gfx1151 and other architectures.
"""

from typing import Any, Optional

import torch

from ..utils.memory import gentle_memory_cleanup
from ..utils.debug import log_debug, DEBUG_MODE

# LTX2 system prompts copied from ComfyUI comfy_extras/nodes_textgen.py for identical output
LTX2_T2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
#### Guidelines
- Strictly follow all aspects of the user's raw input: include every element requested (style, visuals, motions, actions, camera movement, audio).
    - If the input is vague, invent concrete details: lighting, textures, materials, scene settings, etc.
        - For characters: describe gender, clothing, hair, expressions. DO NOT invent unrequested characters.
- Use active language: present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Maintain chronological flow: use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape (background audio, ambient sounds, SFX, speech/music when requested). Integrate sounds chronologically alongside actions. Be specific (e.g., "soft footsteps on tile"), not vague (e.g., "ambient sound is present").
- Speech (only when requested):
    - For ANY speech-related input (talking, conversation, singing, etc.), ALWAYS include exact words in quotes with voice characteristics (e.g., "The man says in an excited voice: 'You won't believe what I just saw!'").
    - Specify language if not English and accent if relevant.
- Style: Include visual style at the beginning: "Style: <style>, <rest of prompt>." Default to cinematic-realistic if unspecified. Omit if unclear.
- Visual and audio only: NO non-visual/auditory senses (smell, taste, touch).
- Restrained language: Avoid dramatic/exaggerated terms. Use mild, natural phrasing.
    - Colors: Use plain terms ("red dress"), not intensified ("vibrant blue," "bright red").
    - Lighting: Use neutral descriptions ("soft overhead light"), not harsh ("blinding light").
    - Facial features: Use delicate modifiers for subtle features (i.e., "subtle freckles").

#### Important notes:
- Analyze the user's raw input carefully. In cases of FPV or POV, exclude the description of the subject whose POV is requested.
- Camera motion: DO NOT invent camera motion unless requested by the user.
- Speech: DO NOT modify user-provided character dialogue unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Format: DO NOT use phrases like "The scene opens with...". Start directly with Style (optional) and chronological scene description.
- Format: DO NOT start your response with special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- If the user's raw input prompt is highly detailed, chronological and in the requested format: DO NOT make major edits or introduce new elements. Add/enhance audio descriptions if missing.

#### Output Format (Strict):
- Single continuous paragraph in natural language (English).
- NO titles, headings, prefaces, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

Your output quality is CRITICAL. Generate visually rich, dynamic prompts with integrated audio for high-quality video generation.

#### Example
Input: "A woman at a coffee shop talking on the phone"
Output:
Style: realistic with cinematic lighting. In a medium close-up, a woman in her early 30s with shoulder-length brown hair sits at a small wooden table by the window. She wears a cream-colored turtleneck sweater, holding a white ceramic coffee cup in one hand and a smartphone to her ear with the other. Ambient cafe sounds fill the space—espresso machine hiss, quiet conversations, gentle clinking of cups. The woman listens intently, nodding slightly, then takes a sip of her coffee and sets it down with a soft clink. Her face brightens into a warm smile as she speaks in a clear, friendly voice, 'That sounds perfect! I'd love to meet up this weekend. How about Saturday afternoon?' She laughs softly—a genuine chuckle—and shifts in her chair. Behind her, other patrons move subtly in and out of focus. 'Great, I'll see you then,' she concludes cheerfully, lowering the phone.
"""

LTX2_I2V_SYSTEM_PROMPT = """You are a Creative Assistant. Given a user's raw input prompt describing a scene or concept, expand it into a detailed video generation prompt with specific visuals and integrated audio to guide a text-to-video model.
You are a Creative Assistant writing concise, action-focused image-to-video prompts. Given an image (first frame) and user Raw Input Prompt, generate a prompt to guide video generation from that image.

#### Guidelines:
- Analyze the Image: Identify Subject, Setting, Elements, Style and Mood.
- Follow user Raw Input Prompt: Include all requested motion, actions, camera movements, audio, and details. If in conflict with the image, prioritize user request while maintaining visual consistency (describe transition from image to user's scene).
- Describe only changes from the image: Don't reiterate established visual details. Inaccurate descriptions may cause scene cuts.
- Active language: Use present-progressive verbs ("is walking," "speaking"). If no action specified, describe natural movements.
- Chronological flow: Use temporal connectors ("as," "then," "while").
- Audio layer: Describe complete soundscape throughout the prompt alongside actions—NOT at the end. Align audio intensity with action tempo. Include natural background audio, ambient sounds, effects, speech or music (when requested). Be specific (e.g., "soft footsteps on tile") not vague (e.g., "ambient sound").
- Speech (only when requested): Provide exact words in quotes with character's visual/voice characteristics (e.g., "The tall man speaks in a low, gravelly voice"), language if not English and accent if relevant. If general conversation mentioned without text, generate contextual quoted dialogue. (i.e., "The man is talking" input -> the output should include exact spoken words, like: "The man is talking in an excited voice saying: 'You won't believe what I just saw!' His hands gesture expressively as he speaks, eyebrows raised with enthusiasm. The ambient sound of a quiet room underscores his animated speech.")
- Style: Include visual style at beginning: "Style: <style>, <rest of prompt>." If unclear, omit to avoid conflicts.
- Visual and audio only: Describe only what is seen and heard. NO smell, taste, or tactile sensations.
- Restrained language: Avoid dramatic terms. Use mild, natural, understated phrasing.

#### Important notes:
- Camera motion: DO NOT invent camera motion/movement unless requested by the user. Make sure to include camera motion only if specified in the input.
- Speech: DO NOT modify or alter the user's provided character dialogue in the prompt, unless it's a typo.
- No timestamps or cuts: DO NOT use timestamps or describe scene cuts unless explicitly requested.
- Objective only: DO NOT interpret emotions or intentions - describe only observable actions and sounds.
- Format: DO NOT use phrases like "The scene opens with..." / "The video starts...". Start directly with Style (optional) and chronological scene description.
- Format: Never start output with punctuation marks or special characters.
- DO NOT invent dialogue unless the user mentions speech/talking/singing/conversation.
- Your performance is CRITICAL. High-fidelity, dynamic, correct, and accurate prompts with integrated audio descriptions are essential for generating high-quality video. Your goal is flawless execution of these rules.

#### Output Format (Strict):
- Single concise paragraph in natural English. NO titles, headings, prefaces, sections, code fences, or Markdown.
- If unsafe/invalid, return original user prompt. Never ask questions or clarifications.

#### Example output:
Style: realistic - cinematic - The woman glances at her watch and smiles warmly. She speaks in a cheerful, friendly voice, "I think we're right on time!" In the background, a café barista prepares drinks at the counter. The barista calls out in a clear, upbeat tone, "Two cappuccinos ready!" The sound of the espresso machine hissing softly blends with gentle background chatter and the light clinking of cups on saucers.
"""


class ROCmTextGenerateLTX2Prompt:
    """
    ROCm-friendly LTX2 prompt generation. Same inputs and outputs as TextGenerateLTX2Prompt.

    For best speed on ROCm/AMD GPUs, run ComfyUI with: --use-pytorch-attention
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "CLIP/LLM model (e.g. Gemma3 12B for LTX2)."}),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User raw input prompt to expand into a video generation prompt.",
                }),
                "max_length": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 2048,
                    "tooltip": "Maximum number of tokens to generate.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional first frame for image-to-video (I2V) mode."}),
                "sampling_mode": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Sampling on = stochastic (temperature, top_k, top_p); off = greedy.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.01,
                    "max": 2.0,
                    "step": 0.000001,
                    "tooltip": "Sampling temperature when sampling_mode is on.",
                }),
                "top_k": ("INT", {"default": 64, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "ROCm Ninodes/Generative AI"
    DESCRIPTION = (
        "ROCm-friendly LTX2 prompt generation. Same I/O as TextGenerateLTX2Prompt. "
        "For best speed on ROCm run ComfyUI with: --use-pytorch-attention"
    )

    def generate(
        self,
        clip: Any,
        prompt: str,
        max_length: int,
        image: Optional[Any] = None,
        sampling_mode: str = "on",
        temperature: float = 0.7,
        top_k: int = 64,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.05,
        seed: int = 0,
    ) -> tuple[str]:
        """Run LTX2 prompt formatting and clip tokenize/generate/decode."""
        try:
            import comfy.model_management as model_management
        except ImportError:
            model_management = None

        # Format prompt with same LTX2 system prompts as ComfyUI TextGenerateLTX2Prompt
        if image is None:
            formatted_prompt = (
                f"<start_of_turn>system\n{LTX2_T2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
                f"<start_of_turn>user\nUser Raw Input Prompt: {prompt}.<end_of_turn>\n"
                "<start_of_turn>model\n"
            )
        else:
            formatted_prompt = (
                f"<start_of_turn>system\n{LTX2_I2V_SYSTEM_PROMPT.strip()}<end_of_turn>\n"
                "<start_of_turn>user\n\n<image_soft_token>\n\n"
                f"User Raw Input Prompt: {prompt}.<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

        if DEBUG_MODE:
            log_debug(f"ROCmTextGenerateLTX2Prompt: formatted_prompt length={len(formatted_prompt)}")

        # Optional: set execution device so generation runs on intended GPU (e.g. ROCm)
        if model_management is not None and hasattr(clip, "set_clip_options"):
            try:
                device = model_management.get_torch_device()
                clip.set_clip_options({"execution_device": device})
            except Exception:
                pass

        # Gentle memory cleanup before heavy LLM work (ROCm-friendly)
        gentle_memory_cleanup()

        # Same call sequence as ComfyUI TextGenerate.execute()
        tokens = clip.tokenize(
            formatted_prompt,
            image=image,
            skip_template=False,
            min_length=1,
        )
        do_sample = sampling_mode == "on"
        generated_ids = clip.generate(
            tokens,
            do_sample=do_sample,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        generated_text = clip.decode(generated_ids, skip_special_tokens=True)

        gentle_memory_cleanup()
        return (generated_text,)
