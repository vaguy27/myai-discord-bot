# MyAI.Bot Discord Bot

A Discord bot that integrates with Ollama to answer questions in your Discord server. Features configurable AI models with admin controls.

## Features

- 🤖 AI-powered responses via Ollama with configurable models
- 🔧 Dynamic model switching (admin-only)
- 📋 List available Ollama models
- 🎯 Channel-specific responses (only responds in "myai-bot" channel)
- 🔒 Admin-only shutdown and model management commands
- 📱 Beautiful Discord embeds
- ⚡ Async processing with typing indicators
- 🛡️ Comprehensive error handling

## Setup

### Prerequisites
- Python 3.8 or higher
- Ollama installed and running
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
   ollama serve
   ```

3. **Configure Discord Bot:**
   - Go to https://discord.com/developers/applications
   - Create a new application named "MyAI.Bot"
   - Go to Bot section and create a bot
   - Copy the bot token
   - Enable "Message Content Intent" under Privileged Gateway Intents

4. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add your Discord token
   ```

5. **Create "myai-bot" channel:**
   Create a channel named "myai-bot" in your Discord server

6. **Run the bot:**
   ```bash
   python bot.py
   ```

## Commands

### General Commands
- `!ask me <question>` or `!askme <question>` - Ask the AI anything
- `!help` or `!myai help` - Show help message

### Model Management (Admin Only)
- `!models` or `!model list` - List all available Ollama models
- `!llm <model_name>` - Change the AI model (e.g., `!llm llama3.2`)

### Admin Commands
- `!shutdown` or `!myai shutdown` - Shutdown bot (admin only)

## Bot Permissions Needed

When inviting the bot to your server, make sure to grant these permissions:
- Send Messages
- Use Slash Commands  
- Read Message History
- Add Reactions
- Embed Links

## Troubleshooting

- **Bot not responding**: Check if Ollama is running and the "myai-bot" channel exists
- **API errors**: Verify Ollama is accessible at the configured URL
- **Permission errors**: Ensure the bot has proper permissions in the channel

## License

MIT License - feel free to modify and distribute!
