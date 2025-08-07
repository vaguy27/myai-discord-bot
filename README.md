# MyAI.Bot Discord Bot

A Discord bot that integrates with Ollama's Llama 3.2 model to answer questions in your Discord server.

## Features

- 🤖 AI-powered responses using Llama 3.2 via Ollama
- 🎯 Channel-specific responses (only responds in "myai-bot" channel)
- 🔒 Admin-only shutdown command
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
   ollama pull llama3.2
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

- `!ask me <question>` or `!askme <question>` - Ask the AI anything
- `!shutdown` or `!myai shutdown` - Shutdown bot (admin only)  
- `!help` or `!myai help` - Show help message

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
