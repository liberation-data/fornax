#!/usr/bin/env bash
# Downloads the models Fornax's integration tests use into the Hugging Face cache, with the hf CLI. Without them those
# tests skip. All of them fit a 32 GB machine: a 4B and a 0.6B language model, 0.6B embedding and reranker models,
# EmbeddingGemma 2, Qwen3-Reranker 4B, a 4B vision model, Kokoro, Whisper base.en, HTDemucs, Real-ESRGAN and FLUX.2
# klein 4B (about 24 GB on disk).
# FLUX.2 klein is gated: accept its licence on huggingface.co and `hf auth login`.
set -euo pipefail
hf=$(command -v hf || echo "$HOME/.local/bin/hf")
"$hf" download unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
"$hf" download Qwen/Qwen3-0.6B-GGUF Qwen3-0.6B-Q8_0.gguf
"$hf" download Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf
"$hf" download unsloth/embeddinggemma-2-GGUF embeddinggemma-2-Q8_0.gguf
"$hf" download ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf
"$hf" download Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp Qwen3-Reranker-4B.Q8_0.gguf
"$hf" download Qwen/Qwen3-VL-4B-Instruct-GGUF Qwen3VL-4B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
"$hf" download ggml-org/Kokoro-82M-GGUF Kokoro-82M-F32.gguf mmproj-Kokoro-82M-F16.gguf
"$hf" download ggerganov/whisper.cpp ggml-base.en.bin
"$hf" download thepatch/htdemucs-GGUF htdemucs-42M-v1.0-F32.gguf
"$hf" download Comfy-Org/Real-ESRGAN RealESRGAN_x4plus.safetensors
"$hf" download Comfy-Org/flux2-dev split_files/vae/flux2-vae.safetensors
"$hf" download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors
"$hf" download black-forest-labs/FLUX.2-klein-base-4b-fp8 flux-2-klein-base-4b-fp8.safetensors
