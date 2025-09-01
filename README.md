# MyAI.Bot Discord Bot

A comprehensive AI Discord bot that integrates with Ollama and ComfyUI to provide text responses, image generation, video generation, music generation, and image analysis capabilities. Features configurable AI models with admin controls.

## Features

- 🤖 AI-powered text responses via Ollama with configurable models
- 🎨 AI image generation using ComfyUI with CLIP Text Encode
- 🎬 AI video generation using ComfyUI with advanced video workflows
- 🎵 AI music generation using ComfyUI with ACE Step audio models
- 🖼️ AI image analysis using vision models (llava, bakllava, etc.)
- 🔧 Dynamic model switching (admin-only)
- 📋 List available Ollama models
- 🎯 Channel-specific responses (responds in "myai-bot" channel and direct messages)
- 🔒 Admin-only shutdown and model management commands
- 📱 Beautiful Discord embeds with image, video, and audio attachment support
- ⚡ Async processing with typing indicators
- 🛡️ Comprehensive error handling with fallback mechanisms

## Setup

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
- ComfyUI installed and running (for image, video, and music generation)
- Discord bot token

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Ollama:**
   ```bash
   # Pull your desired models (default is gemma3:12b)
   ollama pull gemma3:12b
   ollama pull llama3.2  # or any other model you prefer
   # For image analysis, pull a vision model
   ollama pull llava
   ollama serve
   ```

3. **Set up ComfyUI (optional, for image, video, and music generation):**
   ```bash
   # Install ComfyUI and start the server
   # Make sure it's running on http://localhost:8188
   # Ensure you have required models loaded:
   # - sd_xl_base_1.0.safetensors (for images)
   # - wan2.1_t2v_1.3B_fp16.safetensors, umt5_xxl_fp8_e4m3fn_scaled.safetensors, wan_2.1_vae.safetensors (for videos)
   # - ace_step_v1_3.5b.safetensors (for music)
   ```

4. **Configure Discord Bot:**
   - Go to https://discord.com/developers/applications
   - Create a new application named "MyAI.Bot"
   - Go to Bot section and create a bot
   - Copy the bot token
   - Enable "Message Content Intent" under Privileged Gateway Intents

5. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Discord token and API URLs
   ```
   
   Example .env file:
   ```bash
   DISCORD_TOKEN=your_discord_bot_token_here
   OLLAMA_API_URL=http://localhost:11434
   COMFYUI_API_URL=http://localhost:8188
   ```

6. **Create "myai-bot" channel:**
   Create a channel named "myai-bot" in your Discord server (bot also responds to direct messages)

7. **Run the bot:**
   ```bash
   python bot.py
   ```

## Commands

**Note:** All commands work in both the "myai-bot" channel and direct messages for private requests.

### General Commands
- `!ask me <question>` or `!askme <question>` - Ask the AI anything
- `!help` or `!myai help` - Show help message

### AI Media Features
- `!create <description>` - Generate images using ComfyUI (e.g., `!create a cat in space`)
- `!video <description>` - Generate videos using ComfyUI (e.g., `!video a fox running in the forest`)
- `!music <description>` - Generate music using ComfyUI (e.g., `!music upbeat electronic dance music`)
- `!image <question>` - Analyze uploaded images (attach image + `!image what's in this photo?`)

### Model Management (Admin Only)
- `!models` or `!model list` - List all available Ollama models
- `!llm <model_name>` - Change the AI model (e.g., `!llm llava` for image analysis)

### Admin Commands
- `!shutdown` or `!myai shutdown` - Shutdown bot (admin only)

## Bot Permissions Needed

When inviting the bot to your server, make sure to grant these permissions:
- Send Messages
- Use Slash Commands  
- Read Message History
- Add Reactions
- Embed Links
- Attach Files (for image uploads and downloads)

## Troubleshooting

- **Bot not responding**: Check if Ollama is running and the "myai-bot" channel exists
- **API errors**: Verify Ollama is accessible at the configured URL
- **Image generation not working**: Ensure ComfyUI is running on port 8188 with required models
- **Video generation not working**: Ensure ComfyUI has video models loaded (wan2.1_t2v, umt5_xxl, wan_2.1_vae)
- **Music generation not working**: Ensure ComfyUI has audio models loaded (ace_step_v1_3.5b.safetensors)
- **Image analysis not working**: Use a vision model like `!llm llava`
- **Permission errors**: Ensure the bot has proper permissions in the channel
- **Large files failing**: Keep images under 5MB and audio/video under 8MB for optimal processing

## License

MIT License - feel free to modify and distribute!
