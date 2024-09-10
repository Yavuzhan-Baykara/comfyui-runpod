# Stage 1: Base image with common dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    aria2c

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN git clone https://github.com/SeargeDP/ComfyUI_Searge_LLM /comfyui/custom_nodes/ComfyUI_Searge_LLM
RUN git clone https://github.com/giriss/comfy-image-saver /comfyui/custom_nodes/comfy-image-saver
RUN git clone https://github.com/AlekPet/ComfyUI_Custom_Nodes_AlekPet /comfyui/custom_nodes/ComfyUI_Custom_Nodes_AlekPet
RUN git clone https://github.com/ltdrdata/ComfyUI-Manager /comfyui/custom_nodes/ComfyUI-Manager
# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --upgrade -r requirements.txt

#----------------------------------------------------------------------
# Change working directory to ComfyUI_Searge_LLM
WORKDIR /comfyui/custom_nodes/ComfyUI_Searge_LLM
# Install ComfyUI dependencies
RUN pip3 install --upgrade -r requirements.txt
# Change working directory to comfy-image-saver
WORKDIR /comfyui/custom_nodes/comfy-image-saver
# Install ComfyUI dependencies
RUN pip3 install --upgrade -r requirements.txt
# Change working directory to comfy-image-saver
WORKDIR /comfyui/custom_nodes/ComfyUI_Custom_Nodes_AlekPet
# Install ComfyUI dependencies
RUN pip3 install --upgrade -r requirements.txt
# Change working directory to comfy-image-saver
WORKDIR /comfyui/custom_nodes/ComfyUI-Manager
# Install ComfyUI dependencies
RUN pip3 install --upgrade -r requirements.txt
#----------------------------------------------------------------------
# Install runpod
RUN pip3 install runpod requests

# Support for the network volume
ADD src/extra_model_paths.yaml ./

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Stage 2: Download models
FROM base AS downloader

ARG HUGGINGFACE_ACCESS_TOKEN
ARG MODEL_TYPE

# Change working directory to ComfyUI
WORKDIR /comfyui

# Download checkpoints/vae/LoRA to include in image based on model type
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/unet/flux1-dev.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/flux1-dev.safetensors
RUN wget -O models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
RUN wget -O models/clip/t5xxl_fp8_e4m3fn.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
RUN wget --header="Authorization: Bearer ${HUGGINGFACE_ACCESS_TOKEN}" -O models/vae/ae.safetensors https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors


# LLM models
RUN mkdir -p /models/llm_gguf/ && \
    wget -O /models/llm_gguf/Mistral-7B-Instruct-v0.3.IQ1_M.gguf https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3.IQ1_M.gguf


# Loras
RUN wget -O models/loras/Hyper-FLUX.1-dev-16steps-lora.safetensors https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-16steps-lora.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/747534?token=3f3b111c6e2d0dc793cd5bccc635c06e -d models/loras -o Cyberpunk_Anime_Style.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/755549?token=3f3b111c6e2d0dc793cd5bccc635c06e -d models/loras -o Sinfully_Stylish.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/740927?token=3f3b111c6e2d0dc793cd5bccc635c06e -d models/loras -o Neon_Cyberpunk_Splash_Art_FLUX.safetensors
RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://civitai.com/api/download/models/727461?token=3f3b111c6e2d0dc793cd5bccc635c06e -d models/loras -o real-lora.safetensors


# Stage 3: Final image
FROM base AS final

# Copy models from stage 2 to the final image
COPY --from=downloader /comfyui/models /comfyui/models
#COPY --from=downloader /models/llm_gguf /models/llm_gguf

# Start the container
CMD /start.sh
