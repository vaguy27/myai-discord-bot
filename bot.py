import discord
from discord.ext import commands
import aiohttp
import asyncio
import os
import json
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
# Global variable for model name (can be changed with !llm command)
current_model = 'gemma3:12b'
TARGET_CHANNEL = 'myai-bot'

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot setup with intents
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix='!', intents=intents, help_command=None)

class MyAIBot:
    def __init__(self):
        self.session = None

    async def create_session(self):
        """Create aiohttp session for API calls"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    async def call_ollama(self, prompt):
        """Call Ollama API with the given prompt"""
        await self.create_session()
        
        payload = {
            "model": current_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 500
            }
        }

        try:
            async with self.session.post(f"{OLLAMA_API_URL}/api/generate", 
                                       json=payload,
                                       headers={'Content-Type': 'application/json'}) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', 'No response received')
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error {response.status}: {error_text}")
                    raise Exception(f"API returned status {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling Ollama API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise Exception(f"Failed to get response from Ollama API: {str(e)}")

    async def get_available_models(self):
        """Get list of available Ollama models"""
        await self.create_session()
        
        try:
            async with self.session.get(f"{OLLAMA_API_URL}/api/tags", 
                                      headers={'Content-Type': 'application/json'}) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return data.get('models', [])
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama models API error {response.status}: {error_text}")
                    raise Exception(f"API returned status {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling Ollama models API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling Ollama models API: {str(e)}")
            raise Exception(f"Failed to get models from Ollama API: {str(e)}")

# Create bot instance
myai = MyAIBot()

@bot.event
async def on_ready():
    """Event triggered when bot is ready"""
    print(f'✅ {bot.user.name} is online and ready!')
    print(f'🎯 Looking for channel: {TARGET_CHANNEL}')
    
    # Set bot status
    activity = discord.Activity(type=discord.ActivityType.watching, name="for questions...")
    await bot.change_presence(activity=activity)

@bot.event
async def on_message(message):
    """Handle incoming messages"""
    # Ignore messages from bots
    if message.author.bot:
        return
    
    # Only respond in the target channel
    if message.channel.name != TARGET_CHANNEL:
        return

    content = message.content.strip().lower()

    # Handle "ask me" command
    if content.startswith('!ask me ') or content.startswith('!askme '):
        query_start = 8 if content.startswith('!ask me ') else 7
        query = message.content.strip()[query_start:].strip()
        
        if not query:
            await message.reply('❌ Please provide a question after the command. Example: `!ask me What is the weather like?`')
            return

        # Show typing indicator
        async with message.channel.typing():
            try:
                logger.info(f'📝 Processing query from {message.author.name}: "{query}"')
                
                response = await myai.call_ollama(query)
                
                # Create embed for better formatting
                embed = discord.Embed(
                    title="🤖 MyAI.Bot Response",
                    description=response,
                    color=0x0099FF,
                    timestamp=message.created_at
                )
                embed.set_footer(
                    text=f"Asked by {message.author.display_name}",
                    icon_url=message.author.display_avatar.url
                )

                # Handle long responses
                if len(response) > 4096:
                    # Split into chunks if too long for embed
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.reply(f"🤖 **MyAI.Bot Response:**\n{chunks[0]}")
                    for chunk in chunks[1:]:
                        await message.channel.send(chunk)
                else:
                    await message.reply(embed=embed)

            except Exception as e:
                logger.error(f'Error processing ask me command: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Error",
                    description="Sorry, I encountered an error while processing your request. Please try again later.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• Ollama server might be down\n• Model not available\n• Network connectivity issues",
                    inline=False
                )
                
                await message.reply(embed=error_embed)

    # Handle shutdown command (restrict to admins)
    elif content in ['!shutdown', '!myai shutdown']:
        # Check if user has administrator permissions
        if not message.author.guild_permissions.administrator:
            await message.reply('❌ You need administrator permissions to use this command.')
            return

        shutdown_embed = discord.Embed(
            title="🔌 Shutting Down",
            description="MyAI.Bot is shutting down. Goodbye!",
            color=0xFF6B6B,
            timestamp=message.created_at
        )
        shutdown_embed.set_footer(text=f"Shutdown initiated by {message.author.display_name}")

        await message.reply(embed=shutdown_embed)
        
        logger.info(f'🔌 Shutdown command received from {message.author.name}')
        
        # Graceful shutdown
        await asyncio.sleep(2)
        await myai.close_session()
        await bot.close()

    # LLM model configuration command (Admin only)
    elif content.startswith('!llm '):
        global current_model  # Declare global variable at the start
        
        # Check if user has administrator permissions
        if not message.author.guild_permissions.administrator:
            await message.reply('❌ You need administrator permissions to change the model.')
            return
        
        new_model = message.content.strip()[5:].strip()  # Remove "!llm " prefix
        
        if not new_model:
            await message.reply(f'❌ Please specify a model name. Example: `!llm gemma3:12b`\n**Current model:** {current_model}')
            return
        
        # Update global model variable
        old_model = current_model
        current_model = new_model
        
        # Create confirmation embed
        model_embed = discord.Embed(
            title="🔄 Model Changed",
            description=f"Successfully changed model from **{old_model}** to **{new_model}**",
            color=0x00FF00,
            timestamp=message.created_at
        )
        model_embed.set_footer(text=f"Changed by {message.author.display_name}")
        
        await message.reply(embed=model_embed)
        logger.info(f'🔄 Model changed from {old_model} to {new_model} by {message.author.name}')

    # List available models command
    elif content in ['!models', '!model list', '!list models']:
        async with message.channel.typing():
            try:
                logger.info(f'📋 Fetching available models for {message.author.name}')
                
                models = await myai.get_available_models()
                
                if not models:
                    no_models_embed = discord.Embed(
                        title="❌ No Models Found",
                        description="No Ollama models are currently available.",
                        color=0xFF6B6B,
                        timestamp=message.created_at
                    )
                    await message.reply(embed=no_models_embed)
                    return
                
                # Create models list embed
                models_embed = discord.Embed(
                    title="🤖 Available Ollama Models",
                    description=f"Found {len(models)} available model(s):",
                    color=0x0099FF,
                    timestamp=message.created_at
                )
                
                # Format models list
                model_list = []
                for i, model in enumerate(models[:10]):  # Limit to 10 models to avoid embed limits
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    
                    # Format size in a readable way
                    if size > 0:
                        if size >= 1024**3:  # GB
                            size_str = f"{size / (1024**3):.1f}GB"
                        elif size >= 1024**2:  # MB
                            size_str = f"{size / (1024**2):.1f}MB"
                        else:
                            size_str = f"{size}B"
                    else:
                        size_str = "Unknown size"
                    
                    # Mark current model
                    current_marker = " **(current)**" if name == current_model else ""
                    model_list.append(f"`{name}`{current_marker} - {size_str}")
                
                models_embed.add_field(
                    name="📦 Models",
                    value="\n".join(model_list),
                    inline=False
                )
                
                if len(models) > 10:
                    models_embed.add_field(
                        name="ℹ️ Note",
                        value=f"Showing first 10 of {len(models)} models",
                        inline=False
                    )
                
                models_embed.add_field(
                    name="🔧 Usage",
                    value=f"Use `!llm <model_name>` to switch models\nCurrent: **{current_model}**",
                    inline=False
                )
                
                models_embed.set_footer(text=f"Requested by {message.author.display_name}")
                
                await message.reply(embed=models_embed)
                
            except Exception as e:
                logger.error(f'Error fetching models: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Error Fetching Models",
                    description="Sorry, I couldn't retrieve the available models.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• Ollama server might be down\n• Network connectivity issues\n• API endpoint unavailable",
                    inline=False
                )
                
                await message.reply(embed=error_embed)

    # Help command
    elif content in ['!help', '!myai help']:
        help_embed = discord.Embed(
            title="🤖 MyAI.Bot Commands",
            description="Here are the available commands:",
            color=0x00FF00,
            timestamp=message.created_at
        )
        help_embed.add_field(
            name="💬 Ask Questions",
            value="`!ask me <your question>` or `!askme <your question>`\nAsk me anything and I'll use " + current_model + " to help you!",
            inline=False
        )
        help_embed.add_field(
            name="📋 List Models",
            value="`!models` or `!model list`\nShow all available Ollama models",
            inline=False
        )
        help_embed.add_field(
            name="🤖 Change Model (Admin Only)",
            value=f"`!llm <model_name>`\nChange the AI model (currently: {current_model})",
            inline=False
        )
        help_embed.add_field(
            name="🔌 Shutdown (Admin Only)",
            value="`!shutdown` or `!myai shutdown`\nShuts down the bot (requires admin permissions)",
            inline=False
        )
        help_embed.add_field(
            name="❓ Help",
            value="`!help` or `!myai help`\nShows this help message",
            inline=False
        )
        help_embed.set_footer(text="Powered by Ollama & " + current_model)
        
        await message.reply(embed=help_embed)

    # Process commands
    await bot.process_commands(message)

@bot.event
async def on_error(event, *args, **kwargs):
    """Handle bot errors"""
    logger.error(f'Bot error in {event}: {args} {kwargs}')

async def main():
    """Main function to run the bot"""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not found in environment variables")
        return
    
    try:
        await bot.start(DISCORD_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
    finally:
        await myai.close_session()
        if not bot.is_closed():
            await bot.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 MyAI.Bot shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
