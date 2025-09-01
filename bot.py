import discord
from discord.ext import commands
import aiohttp
import asyncio
import os
import json
from dotenv import load_dotenv
import logging
import uuid
import io
import base64

# Load environment variables
load_dotenv()

# Configuration
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
COMFYUI_API_URL = os.getenv('COMFYUI_API_URL', 'http://localhost:8188')
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
                "temperature": 0.2,
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

    async def create_image_with_comfyui(self, prompt):
        """Generate image using ComfyUI with CLIP Text Encode"""
        await self.create_session()
        
        # ComfyUI workflow for text-to-image generation from ComfyUIwf.json
        workflow = {
            "3": {
                "inputs": {
                    "seed": 629982222946916,
                    "steps": 20,
                    "cfg": 8,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler",
                "_meta": {
                    "title": "KSampler"
                }
            },
            "4": {
                "inputs": {
                    "ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {
                    "title": "CheckpointLoaderSimple"
                }
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage",
                "_meta": {
                    "title": "EmptyLatentImage"
                }
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIPTextEncode"
                }
            },
            "7": {
                "inputs": {
                    "text": "",
                    "clip": ["4", 1]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {
                    "title": "CLIPTextEncode"
                }
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2]
                },
                "class_type": "VAEDecode",
                "_meta": {
                    "title": "VAEDecode"
                }
            },
            "9": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "images": ["8", 0]
                },
                "class_type": "SaveImage",
                "_meta": {
                    "title": "SaveImage"
                }
            }
        }
        
        try:
            # Generate a unique prompt ID
            prompt_id = str(uuid.uuid4())
            
            # Queue the prompt
            queue_payload = {
                "prompt": workflow,
                "client_id": prompt_id
            }
            
            async with self.session.post(f"{COMFYUI_API_URL}/prompt", 
                                       json=queue_payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ComfyUI queue error {response.status}: {error_text}")
                    raise Exception(f"Failed to queue prompt: {response.status}")
                
                queue_result = await response.json()
                prompt_id = queue_result.get("prompt_id")
                
            if not prompt_id:
                raise Exception("No prompt_id returned from ComfyUI")
            
            # Poll for completion
            max_attempts = 60  # 5 minutes max wait
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between checks
                
                async with self.session.get(f"{COMFYUI_API_URL}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            # Get the output images
                            outputs = history[prompt_id].get("outputs", {})
                            if "9" in outputs and "images" in outputs["9"]:
                                image_info = outputs["9"]["images"][0]
                                filename = image_info["filename"]
                                subfolder = image_info.get("subfolder", "")
                                
                                # Download the image
                                view_url = f"{COMFYUI_API_URL}/view"
                                params = {"filename": filename}
                                if subfolder:
                                    params["subfolder"] = subfolder
                                    
                                async with self.session.get(view_url, params=params) as img_response:
                                    if img_response.status == 200:
                                        image_data = await img_response.read()
                                        return image_data
                                    else:
                                        raise Exception(f"Failed to download image: {img_response.status}")
                            break
                    
            raise Exception("Image generation timed out")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling ComfyUI API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling ComfyUI API: {str(e)}")
            raise Exception(f"Failed to generate image: {str(e)}")

    async def create_video_with_comfyui(self, prompt):
        """Generate video using ComfyUI with video workflow"""
        await self.create_session()
        
        # Embedded video workflow
        workflow = {
            "3": {
                "inputs": {
                    "seed": 181197454961476,
                    "steps": 30,
                    "cfg": 6,
                    "sampler_name": "uni_pc",
                    "scheduler": "simple",
                    "denoise": 1,
                    "model": ["48", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["40", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "6": {
                "inputs": {
                    "text": prompt,
                    "clip": ["38", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
            },
            "7": {
                "inputs": {
                    "text": "Vibrant colors, overexposure, static, blurred details, subtitles, style, artwork, painting, still image, overall grayness, worst quality, low quality, JPEG compression residue, ugly, mutilated, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, malformed limbs, fused fingers, still image, cluttered background, three legs, crowded background, walking backwards",
                    "clip": ["38", 0]
                },
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "CLIP Text Encode (Negative Prompt)"}
            },
            "8": {
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["39", 0]
                },
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode"}
            },
            "37": {
                "inputs": {
                    "unet_name": "wan2.1_t2v_1.3B_fp16.safetensors",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {"title": "Load Diffusion Model"}
            },
            "38": {
                "inputs": {
                    "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "type": "wan",
                    "device": "default"
                },
                "class_type": "CLIPLoader",
                "_meta": {"title": "Load CLIP"}
            },
            "39": {
                "inputs": {
                    "vae_name": "wan_2.1_vae.safetensors"
                },
                "class_type": "VAELoader",
                "_meta": {"title": "Load VAE"}
            },
            "40": {
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": 33,
                    "batch_size": 1
                },
                "class_type": "EmptyHunyuanLatentVideo",
                "_meta": {"title": "EmptyHunyuanLatentVideo"}
            },
            "48": {
                "inputs": {
                    "shift": 8,
                    "model": ["37", 0]
                },
                "class_type": "ModelSamplingSD3",
                "_meta": {"title": "ModelSamplingSD3"}
            },
            "49": {
                "inputs": {
                    "fps": 8,
                    #"fps": 16,
                    "images": ["8", 0]
                },
                "class_type": "CreateVideo",
                "_meta": {"title": "Create Video"}
            },
            "50": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "format": "mp4",
                    "codec": "h264",
                    "video-preview": "",
                    "video": ["49", 0]
                },
                "class_type": "SaveVideo",
                "_meta": {"title": "Save Video"}
            }
        }
        
        try:
            # Generate a unique prompt ID
            prompt_id = str(uuid.uuid4())
            
            # Queue the prompt
            queue_payload = {
                "prompt": workflow,
                "client_id": prompt_id
            }
            
            async with self.session.post(f"{COMFYUI_API_URL}/prompt", 
                                       json=queue_payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ComfyUI video queue error {response.status}: {error_text}")
                    raise Exception(f"Failed to queue video prompt: {response.status}")
                
                queue_result = await response.json()
                prompt_id = queue_result.get("prompt_id")
                
            if not prompt_id:
                raise Exception("No prompt_id returned from ComfyUI")
            
            # Poll for completion (videos take longer than images)
            max_attempts = 120  # 10 minutes max wait
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between checks
                
                async with self.session.get(f"{COMFYUI_API_URL}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            # Get the output video
                            outputs = history[prompt_id].get("outputs", {})
                            logger.info(f"ComfyUI outputs: {list(outputs.keys())}")
                            
                            if "50" in outputs:
                                save_video_output = outputs["50"]
                                logger.info(f"SaveVideo output: {save_video_output}")
                                
                                # Try different possible keys for video output
                                video_info = None
                                filename = None
                                subfolder = ""
                                
                                # Check for various possible output structures
                                if "videos" in save_video_output and save_video_output["videos"]:
                                    video_info = save_video_output["videos"][0]
                                    filename = video_info.get("filename")
                                    subfolder = video_info.get("subfolder", "")
                                elif "gifs" in save_video_output and save_video_output["gifs"]:
                                    video_info = save_video_output["gifs"][0]
                                    filename = video_info.get("filename")
                                    subfolder = video_info.get("subfolder", "")
                                elif isinstance(save_video_output, dict):
                                    # Check if the output directly contains filename info
                                    for key, value in save_video_output.items():
                                        if isinstance(value, list) and value:
                                            if isinstance(value[0], dict) and "filename" in value[0]:
                                                video_info = value[0]
                                                filename = video_info.get("filename")
                                                subfolder = video_info.get("subfolder", "")
                                                logger.info(f"Found video in key '{key}': {filename}")
                                                break
                                
                                if filename:
                                    logger.info(f"Video file: {filename}, subfolder: {subfolder}")
                                    
                                    # Download the video
                                    view_url = f"{COMFYUI_API_URL}/view"
                                    params = {"filename": filename}
                                    if subfolder:
                                        params["subfolder"] = subfolder
                                        
                                    logger.info(f"Downloading from: {view_url} with params: {params}")
                                    
                                    async with self.session.get(view_url, params=params) as vid_response:
                                        if vid_response.status == 200:
                                            video_data = await vid_response.read()
                                            logger.info(f"Downloaded video: {len(video_data)} bytes")
                                            return video_data, filename
                                        else:
                                            logger.error(f"Video download failed via /view: {vid_response.status}")
                                            
                                            # Try alternative download method using direct file path
                                            try:
                                                # Try the output directory path
                                                file_path = f"output/{subfolder}/{filename}" if subfolder else f"output/{filename}"
                                                alt_url = f"{COMFYUI_API_URL}/view"
                                                alt_params = {"filename": file_path}
                                                
                                                logger.info(f"Trying alternative download: {alt_url} with filename={file_path}")
                                                
                                                async with self.session.get(alt_url, params=alt_params) as alt_response:
                                                    if alt_response.status == 200:
                                                        video_data = await alt_response.read()
                                                        logger.info(f"Downloaded video via alternative method: {len(video_data)} bytes")
                                                        return video_data, filename
                                            except Exception as alt_e:
                                                logger.error(f"Alternative download also failed: {alt_e}")
                                            
                                            error_text = await vid_response.text()
                                            logger.error(f"Video download error: {error_text}")
                                            raise Exception(f"Failed to download video: {vid_response.status}")
                                else:
                                    logger.error(f"No video file found in SaveVideo output: {save_video_output}")
                                    raise Exception("No video file found in ComfyUI output")
                            else:
                                logger.error(f"SaveVideo node (50) not found in outputs: {list(outputs.keys())}")
                                raise Exception("SaveVideo node output not found")
                            break
                    
            raise Exception("Video generation timed out")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling ComfyUI video API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling ComfyUI video API: {str(e)}")
            raise Exception(f"Failed to generate video: {str(e)}")

    async def create_music_with_comfyui(self, prompt):
        """Generate music/song using ComfyUI with audio workflow"""
        await self.create_session()
        
        # Embedded audio workflow for ACE Step v1.3.5b model
        workflow = {
            "14": {
                "inputs": {
                    "tags": "anime, soft female vocals, kawaii pop, j-pop, childish, piano, guitar, synthesizer, fast, happy, cheerful, lighthearted",
                    "lyrics": prompt,
                    "lyrics_strength": 0.99,
                    "clip": ["40", 1]
                },
                "class_type": "TextEncodeAceStepAudio",
                "_meta": {"title": "TextEncodeAceStepAudio"}
            },
            "17": {
                "inputs": {
                    "seconds": 30,
                    "batch_size": 1
                },
                "class_type": "EmptyAceStepLatentAudio",
                "_meta": {"title": "EmptyAceStepLatentAudio"}
            },
            "18": {
                "inputs": {
                    "samples": ["52", 0],
                    "vae": ["40", 2]
                },
                "class_type": "VAEDecodeAudio",
                "_meta": {"title": "VAEDecodeAudio"}
            },
            "40": {
                "inputs": {
                    "ckpt_name": "ace_step_v1_3.5b.safetensors"
                },
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": "Load Checkpoint"}
            },
            "44": {
                "inputs": {
                    "conditioning": ["14", 0]
                },
                "class_type": "ConditioningZeroOut",
                "_meta": {"title": "ConditioningZeroOut"}
            },
            "49": {
                "inputs": {
                    "model": ["51", 0],
                    "operation": ["50", 0]
                },
                "class_type": "LatentApplyOperationCFG",
                "_meta": {"title": "LatentApplyOperationCFG"}
            },
            "50": {
                "inputs": {
                    "multiplier": 1.0
                },
                "class_type": "LatentOperationTonemapReinhard",
                "_meta": {"title": "LatentOperationTonemapReinhard"}
            },
            "51": {
                "inputs": {
                    "shift": 5.0,
                    "model": ["40", 0]
                },
                "class_type": "ModelSamplingSD3",
                "_meta": {"title": "ModelSamplingSD3"}
            },
            "52": {
                "inputs": {
                    "seed": 962231370012320,
                    "steps": 50,
                    "cfg": 5,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1,
                    "model": ["49", 0],
                    "positive": ["14", 0],
                    "negative": ["44", 0],
                    "latent_image": ["17", 0]
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "59": {
                "inputs": {
                    "filename_prefix": "ComfyUI",
                    "quality": "V0",
                    "audioUI": "",
                    "audio": ["18", 0]
                },
                "class_type": "SaveAudioMP3",
                "_meta": {"title": "Save Audio (MP3)"}
            }
        }
        
        try:
            # Generate a unique prompt ID
            prompt_id = str(uuid.uuid4())
            
            # Queue the prompt
            queue_payload = {
                "prompt": workflow,
                "client_id": prompt_id
            }
            
            async with self.session.post(f"{COMFYUI_API_URL}/prompt", 
                                       json=queue_payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"ComfyUI audio queue error {response.status}: {error_text}")
                    raise Exception(f"Failed to queue audio prompt: {response.status}")
                
                queue_result = await response.json()
                prompt_id = queue_result.get("prompt_id")
                
            if not prompt_id:
                raise Exception("No prompt_id returned from ComfyUI")
            
            # Poll for completion (audio generation can take a while)
            max_attempts = 120  # 10 minutes max wait
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Wait 5 seconds between checks
                
                async with self.session.get(f"{COMFYUI_API_URL}/history/{prompt_id}") as response:
                    if response.status == 200:
                        history = await response.json()
                        if prompt_id in history:
                            # Get the output audio
                            outputs = history[prompt_id].get("outputs", {})
                            logger.info(f"ComfyUI audio outputs: {list(outputs.keys())}")
                            
                            if "59" in outputs:
                                save_audio_output = outputs["59"]
                                logger.info(f"SaveAudio output: {save_audio_output}")
                                
                                # Try different possible keys for audio output
                                audio_info = None
                                filename = None
                                subfolder = ""
                                
                                # Check for various possible output structures
                                if "audio" in save_audio_output and save_audio_output["audio"]:
                                    audio_info = save_audio_output["audio"][0]
                                    filename = audio_info.get("filename")
                                    subfolder = audio_info.get("subfolder", "")
                                elif isinstance(save_audio_output, dict):
                                    # Check if the output directly contains filename info
                                    for key, value in save_audio_output.items():
                                        if isinstance(value, list) and value:
                                            if isinstance(value[0], dict) and "filename" in value[0]:
                                                audio_info = value[0]
                                                filename = audio_info.get("filename")
                                                subfolder = audio_info.get("subfolder", "")
                                                logger.info(f"Found audio in key '{key}': {filename}")
                                                break
                                
                                if filename:
                                    logger.info(f"Audio file: {filename}, subfolder: {subfolder}")
                                    
                                    # Download the audio
                                    view_url = f"{COMFYUI_API_URL}/view"
                                    params = {"filename": filename}
                                    if subfolder:
                                        params["subfolder"] = subfolder
                                        
                                    logger.info(f"Downloading audio from: {view_url} with params: {params}")
                                    
                                    async with self.session.get(view_url, params=params) as audio_response:
                                        if audio_response.status == 200:
                                            audio_data = await audio_response.read()
                                            logger.info(f"Downloaded audio: {len(audio_data)} bytes")
                                            return audio_data, filename
                                        else:
                                            logger.error(f"Audio download failed via /view: {audio_response.status}")
                                            
                                            # Try alternative download method using direct file path
                                            try:
                                                # Try the output directory path
                                                file_path = f"output/{subfolder}/{filename}" if subfolder else f"output/{filename}"
                                                alt_url = f"{COMFYUI_API_URL}/view"
                                                alt_params = {"filename": file_path}
                                                
                                                logger.info(f"Trying alternative audio download: {alt_url} with filename={file_path}")
                                                
                                                async with self.session.get(alt_url, params=alt_params) as alt_response:
                                                    if alt_response.status == 200:
                                                        audio_data = await alt_response.read()
                                                        logger.info(f"Downloaded audio via alternative method: {len(audio_data)} bytes")
                                                        return audio_data, filename
                                            except Exception as alt_e:
                                                logger.error(f"Alternative audio download also failed: {alt_e}")
                                            
                                            error_text = await audio_response.text()
                                            logger.error(f"Audio download error: {error_text}")
                                            raise Exception(f"Failed to download audio: {audio_response.status}")
                                else:
                                    logger.error(f"No audio file found in SaveAudio output: {save_audio_output}")
                                    raise Exception("No audio file found in ComfyUI output")
                            else:
                                logger.error(f"SaveAudio node (59) not found in outputs: {list(outputs.keys())}")
                                raise Exception("SaveAudio node output not found")
                            break
                    
            raise Exception("Audio generation timed out")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling ComfyUI audio API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling ComfyUI audio API: {str(e)}")
            raise Exception(f"Failed to generate audio: {str(e)}")

    async def analyze_image_with_ollama(self, prompt, image_data):
        """Analyze image using Ollama with vision model"""
        await self.create_session()
        
        # Convert image data to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        payload = {
            "model": current_model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {
                "temperature": 0.2,
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
                    logger.error(f"Ollama vision API error {response.status}: {error_text}")
                    raise Exception(f"API returned status {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error("Timeout calling Ollama vision API")
            raise Exception("Request timed out")
        except Exception as e:
            logger.error(f"Error calling Ollama vision API: {str(e)}")
            raise Exception(f"Failed to get response from Ollama vision API: {str(e)}")

    async def download_discord_image(self, attachment):
        """Download image from Discord attachment"""
        await self.create_session()
        
        try:
            async with self.session.get(attachment.url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Failed to download image: {response.status}")
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise Exception(f"Failed to download image: {str(e)}")

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
                for i, model in enumerate(models):  # Show all models
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

    # Create image command using ComfyUI
    elif content.startswith('!create '):
        prompt = message.content.strip()[8:].strip()  # Remove "!create " prefix
        
        if not prompt:
            await message.reply('❌ Please provide a description for the image. Example: `!create a beautiful sunset over mountains`')
            return

        async with message.channel.typing():
            try:
                logger.info(f'🎨 Processing image generation from {message.author.name}: "{prompt}"')
                
                # Generate image with ComfyUI
                image_data = await myai.create_image_with_comfyui(prompt)
                
                # Create a file-like object from the image data
                image_file = discord.File(io.BytesIO(image_data), filename="generated_image.png")
                
                # Create embed with image
                embed = discord.Embed(
                    title="🎨 Generated Image",
                    description=f"**Prompt:** {prompt}",
                    color=0xFF6B9D,
                    timestamp=message.created_at
                )
                embed.set_image(url="attachment://generated_image.png")
                embed.set_footer(
                    text=f"Created by {message.author.display_name}",
                    icon_url=message.author.display_avatar.url
                )
                
                await message.reply(embed=embed, file=image_file)
                
            except Exception as e:
                logger.error(f'Error generating image: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Image Generation Error",
                    description="Sorry, I couldn't generate the image. Please try again later.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• ComfyUI server might be down\n• Network connectivity issues\n• Model not loaded in ComfyUI\n• Queue might be full",
                    inline=False
                )
                
                await message.reply(embed=error_embed)

    # Create video command using ComfyUI
    elif content.startswith('!video '):
        prompt = message.content.strip()[7:].strip()  # Remove "!video " prefix
        
        if not prompt:
            await message.reply('❌ Please provide a description for the video. Example: `!video a fox moving quickly in a beautiful winter scenery`')
            return

        async with message.channel.typing():
            try:
                logger.info(f'🎬 Processing video generation from {message.author.name}: "{prompt}"')
                
                # Generate video with ComfyUI
                video_data, filename = await myai.create_video_with_comfyui(prompt)
                
                logger.info(f'Video generated successfully: {len(video_data)} bytes, filename: {filename}')
                
                # Check file size (Discord limit is 8MB for regular users, 50MB for Nitro)
                if len(video_data) > 8 * 1024 * 1024:  # 8MB limit
                    await message.reply(f'❌ Generated video is too large ({len(video_data)/(1024*1024):.1f}MB). Discord limit is 8MB for regular users.')
                    return
                
                # Create a file-like object from the video data with proper extension
                video_io = io.BytesIO(video_data)
                video_io.seek(0)  # Reset pointer to beginning
                video_file = discord.File(video_io, filename=f"generated_video_{message.id}.mp4")
                
                logger.info(f'Created Discord file object with {len(video_data)} bytes, attempting to send...')
                
                # Create embed (without trying to reference the video file)
                embed = discord.Embed(
                    title="🎬 Generated Video",
                    description=f"**Prompt:** {prompt}",
                    color=0xFF9500,
                    timestamp=message.created_at
                )
                embed.add_field(
                    name="📹 Video Info",
                    value=f"Resolution: 832x480\nFPS: 16\nLength: ~2 seconds\nSize: {len(video_data)/(1024*1024):.1f}MB",
                    inline=False
                )
                embed.set_footer(
                    text=f"Created by {message.author.display_name}",
                    icon_url=message.author.display_avatar.url
                )
                
                # Send the video file as attachment with embed
                try:
                    await message.reply(file=video_file, embed=embed)
                    logger.info('Video sent successfully to Discord')
                except discord.HTTPException as http_err:
                    logger.error(f'Discord HTTP error sending video: {http_err}')
                    # Try sending just the file without embed if embed fails
                    video_io.seek(0)  # Reset for retry
                    video_file = discord.File(video_io, filename=f"generated_video_{message.id}.mp4")
                    await message.reply(f"🎬 **Generated Video** (Prompt: {prompt})", file=video_file)
                    logger.info('Video sent successfully without embed')
                
            except Exception as e:
                logger.error(f'Error generating video: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Video Generation Error",
                    description="Sorry, I couldn't generate the video. Please try again later.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• ComfyUI server might be down\n• Video models not loaded in ComfyUI\n• Network connectivity issues\n• Queue might be full\n• Video generation timeout",
                    inline=False
                )
                
                await message.reply(embed=error_embed)

    # Create music command using ComfyUI
    elif content.startswith('!music '):
        prompt = message.content.strip()[7:].strip()  # Remove "!music " prefix
        
        if not prompt:
            await message.reply('❌ Please provide a description for the music. Example: `!music upbeat electronic dance music with synthesizers`')
            return

        async with message.channel.typing():
            try:
                logger.info(f'🎵 Processing music generation from {message.author.name}: "{prompt}"')
                
                # Generate music with ComfyUI
                audio_data, filename = await myai.create_music_with_comfyui(prompt)
                
                logger.info(f'Music generated successfully: {len(audio_data)} bytes, filename: {filename}')
                
                # Check file size (Discord limit is 8MB for regular users, 50MB for Nitro)
                if len(audio_data) > 8 * 1024 * 1024:  # 8MB limit
                    await message.reply(f'❌ Generated music file is too large ({len(audio_data)/(1024*1024):.1f}MB). Discord limit is 8MB for regular users.')
                    return
                
                # Create audio file attachment
                audio_file = discord.File(io.BytesIO(audio_data), filename=f"generated_music_{message.id}.mp3")
                
                logger.info(f'Created audio file attachment, sending {len(audio_data)} bytes...')
                
                # Send the music file with simple message
                await message.reply(
                    f"🎵 **Generated Music**\n"
                    f"**Prompt:** {prompt}\n"
                    f"**Duration:** 30 seconds | **Size:** {len(audio_data)/(1024*1024):.1f}MB", 
                    file=audio_file
                )
                logger.info('Music file sent successfully to Discord')
                
            except Exception as e:
                logger.error(f'Error generating music: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Music Generation Error",
                    description="Sorry, I couldn't generate the music. Please try again later.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• ComfyUI server might be down\n• Audio models not loaded in ComfyUI\n• Network connectivity issues\n• Queue might be full\n• Audio generation timeout",
                    inline=False
                )
                
                await message.reply(embed=error_embed)

    # Image analysis command using Ollama vision models
    elif content.startswith('!image '):
        # Check if message has image attachments
        if not message.attachments:
            await message.reply('❌ Please attach an image to analyze. Example: Upload an image and type `!image describe this photo`')
            return
        
        # Get the first image attachment
        attachment = message.attachments[0]
        
        # Check if it's an image
        if not attachment.content_type or not attachment.content_type.startswith('image/'):
            await message.reply('❌ Please attach a valid image file (PNG, JPG, GIF, etc.)')
            return
        
        # Check file size (Discord max is 8MB, but let's limit to 5MB for processing)
        if attachment.size > 5 * 1024 * 1024:  # 5MB
            await message.reply('❌ Image file is too large. Please use an image smaller than 5MB.')
            return
        
        prompt = message.content.strip()[7:].strip()  # Remove "!image " prefix
        
        if not prompt:
            prompt = "Describe this image in detail."
        
        async with message.channel.typing():
            try:
                logger.info(f'🖼️ Processing image analysis from {message.author.name}: "{prompt}"')
                
                # Download the image
                image_data = await myai.download_discord_image(attachment)
                
                # Analyze image with Ollama
                response = await myai.analyze_image_with_ollama(prompt, image_data)
                
                # Create embed for better formatting
                embed = discord.Embed(
                    title="🖼️ Image Analysis",
                    description=response,
                    color=0x9B59B6,
                    timestamp=message.created_at
                )
                embed.add_field(
                    name="📝 Query",
                    value=prompt,
                    inline=False
                )
                embed.add_field(
                    name="🤖 Model",
                    value=current_model,
                    inline=True
                )
                embed.set_footer(
                    text=f"Analyzed by {message.author.display_name}",
                    icon_url=message.author.display_avatar.url
                )
                
                # Set the analyzed image as thumbnail
                embed.set_thumbnail(url=attachment.url)

                # Handle long responses
                if len(response) > 4096:
                    # Split into chunks if too long for embed
                    chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
                    await message.reply(f"🖼️ **Image Analysis:**\n**Query:** {prompt}\n**Model:** {current_model}\n\n{chunks[0]}")
                    for chunk in chunks[1:]:
                        await message.channel.send(chunk)
                else:
                    await message.reply(embed=embed)

            except Exception as e:
                logger.error(f'Error analyzing image: {str(e)}')
                
                error_embed = discord.Embed(
                    title="❌ Image Analysis Error",
                    description="Sorry, I couldn't analyze the image. Please try again later.",
                    color=0xFF0000,
                    timestamp=message.created_at
                )
                error_embed.add_field(
                    name="Possible Issues",
                    value="• Model doesn't support vision (try a vision model like llava)\n• Image format not supported\n• Ollama server issues\n• Network connectivity problems",
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
            name="🎨 Generate Images",
            value="`!create <description>`\nGenerate images using ComfyUI (e.g., `!create a cat in space`)",
            inline=False
        )
        help_embed.add_field(
            name="🎬 Generate Videos",
            value="`!video <description>`\nGenerate videos using ComfyUI (e.g., `!video a fox running in the forest`)",
            inline=False
        )
        help_embed.add_field(
            name="🎵 Generate Music",
            value="`!music <description>`\nGenerate music using ComfyUI (e.g., `!music upbeat electronic dance music`)",
            inline=False
        )
        help_embed.add_field(
            name="🖼️ Analyze Images",
            value="`!image <question>`\nAnalyze uploaded images (attach image + `!image what's in this photo?`)",
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
