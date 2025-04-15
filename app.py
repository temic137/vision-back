# app.py
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
from PIL import Image
import io
import base64
from gtts import gTTS
import groq
import os
from base64 import b64encode
import time
import json
from functools import lru_cache
import tempfile
import requests
import queue
import threading
from datetime import datetime
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})  # Enable CORS for all routes with SSE support

# Create a message queue for each client
clients = {}

# Function to generate SSE data format
def format_sse(data, event=None):
    msg = f"data: {json.dumps(data)}\n\n"
    if event is not None:
        msg = f"event: {event}\n{msg}"
    return msg

# Import dotenv for environment variables
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client with API key from environment variable
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please create a .env file with your API key.")

groq_client = groq.Groq(api_key=groq_api_key)

# Cache for image descriptions to avoid redundant API calls
# This helps with real-time processing by caching similar frames
@lru_cache(maxsize=20)
def get_cached_description(image_hash):
    return None  # Initially cache is empty

def encode_image_to_base64(image_buffer):
    return b64encode(image_buffer.getvalue()).decode('utf-8')

# Simple image hashing function to identify similar frames
def get_image_hash(image):
    # Resize to small dimensions for faster comparison
    small_img = image.resize((32, 32), Image.LANCZOS)
    # Convert to grayscale
    small_img = small_img.convert('L')
    # Get pixel data
    pixels = list(small_img.getdata())
    # Create a simple hash
    avg = sum(pixels) / len(pixels)
    # Return a binary hash
    return ''.join('1' if pixel > avg else '0' for pixel in pixels)

# Simple in-memory cache for image descriptions
image_cache = {}

@app.route('/describe-image', methods=['POST'])
def describe_image():
    # Get parameters from request
    file = request.files['image']
    mode = request.form.get('mode', 'normal')  # 'normal', 'quick', 'detailed', or 'camera'
    is_camera = request.form.get('is_camera', 'false').lower() == 'true'
    context = request.form.get('context', '')  # 'indoor', 'outdoor', or ''
    voice_type = request.form.get('voice_type', 'female')  # 'female', 'male', or 'default'

    # Open and process the image
    image = Image.open(file.stream)

    # Resize image to reduce size while maintaining quality
    max_size = 800 if not is_camera else 640  # Smaller size for camera mode
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    # Prepare image for Groq
    buffer = io.BytesIO()
    # Lower quality for camera mode to reduce size and processing time
    quality = 75 if is_camera else 85

    # Convert RGBA images to RGB mode (remove alpha channel) before saving as JPEG
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the background using alpha channel
        background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
        image = background
    elif image.mode != 'RGB':
        # Convert any other mode to RGB
        image = image.convert('RGB')

    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # Generate image hash for caching
    image_hash = get_image_hash(image)

    # Check if we have this image in cache
    if image_hash in image_cache and time.time() - image_cache[image_hash].get('timestamp', 0) < 300:  # 5 minute cache
        return jsonify(image_cache[image_hash])

    # Convert image to base64
    base64_image = encode_image_to_base64(buffer)

    # Prepare the prompt based on mode and whether it's from camera
    if is_camera or mode == 'camera':
        prompt = """You are assisting a blind person using their phone camera. Describe what's in this camera view in ONE SHORT SENTENCE.
        Focus ONLY on the most important elements. Be extremely brief but informative.

        PRIORITIZE IN THIS ORDER:
        1) Immediate hazards/obstacles that could cause harm (stairs, traffic, objects in path)
        2) Spatial orientation (doorway ahead, open space to the right, narrow hallway)
        3) People and their activities (especially if interacting with the user)
        4) Important objects relevant to navigation or context
        5) Text that provides critical information (signs, labels, displays)

        IMPORTANT RULES:
        - Use 15 words or less
        - Start with the most relevant noun or verb
        - Be direct and factual
        - Include approximate distances when relevant ("Chair 3 feet ahead")
        - Use directional terms (left, right, ahead, behind) from the user's perspective
        - Mention changes from previous view if significant
        - For example: "Stairs descending 5 feet ahead" or "Open doorway to your right"
        - For crowds: "Busy sidewalk with pedestrians moving left to right"""
        model = "meta-llama/llama-4-scout-17b-16e-instruct"  # Updated model for vision capabilities
        temperature = 0.1  # Lower temperature for more consistent, focused responses
        max_tokens = 50  # Very limited tokens for faster responses
    elif mode == 'quick':
        prompt = """Describe this image for a blind person in 1-2 short sentences. Focus on what would be most helpful for them to understand.

        PRIORITIZE IN THIS ORDER:
        1) The main scene or environment (indoor/outdoor, type of room/location)
        2) Key objects and their spatial relationships
        3) People, animals, or moving elements
        4) Colors and lighting only if particularly important
        5) Text that provides context or information

        IMPORTANT RULES:
        - Be direct and concise (30 words max)
        - Start with the most relevant information
        - Use spatial terms (above, below, left, right, center)
        - Mention approximate sizes when helpful
        - For example: "Kitchen with island in center, stainless steel appliances along back wall"
        - Or: "Park with children playing on swings, walking path to the right"""
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        temperature = 0.2
        max_tokens = 100
    else:  # detailed or normal mode
        prompt = """Provide a comprehensive description of this image for a blind person in 3-4 sentences. Your description should create a clear mental image.

        INCLUDE IN THIS ORDER:
        1. Overall scene/setting and atmosphere (indoor/outdoor, time of day, weather if relevant)
        2. Main subjects, their positions, and spatial relationships (use clock positions if helpful)
        3. Important details about appearance, colors, and actions
        4. Background elements that provide context
        5. Any text visible in the image (signs, labels, etc.)
        6. Emotional tone or mood of the scene if apparent

        IMPORTANT RULES:
        - Be specific and descriptive but concise (80 words max)
        - Use vivid, sensory language that helps create a mental picture
        - Organize information from general to specific
        - Use precise spatial terms (foreground, background, left, right, above, below)
        - Mention colors and textures when they add important context
        - For example: "A sunlit kitchen with white cabinets and marble countertops. A woman in a blue apron stands at the island, chopping vegetables on a wooden cutting board. Copper pots hang from a rack above, and through the window behind her, a garden with flowering plants is visible."""
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        temperature = 0.3
        max_tokens = 150

    # Call Groq API with selected model
    start_time = time.time()
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    api_time = time.time() - start_time

    description = chat_completion.choices[0].message.content

    # For camera mode, optimize the description further
    if is_camera or mode == 'camera':
        # Remove any unnecessary text like "I see" or "In this image"
        for phrase in ["I see ", "In this image, ", "The image shows ", "This is ", "There is ", "There are ",
                      "I can see ", "It appears to be ", "It looks like ", "This image contains ", "This picture shows ",
                      "This photo shows ", "This photograph shows ", "This photograph contains ", "This picture contains ",
                      "This image depicts ", "This photo depicts ", "This photograph depicts ", "Visible in the image ",
                      "The photo shows ", "The picture shows ", "The photograph shows ", "The image depicts ",
                      "The photo depicts ", "The picture depicts ", "The photograph depicts ",
                      "The camera view shows ", "Camera view shows ", "The camera shows ", "Camera shows ",
                      "The view shows ", "View shows ", "The scene shows ", "Scene shows ", "The camera captures ",
                      "Camera captures ", "The view captures ", "View captures ", "The scene captures ", "Scene captures "]:
            description = description.replace(phrase, "")

        # Also remove these phrases when they start with a capital letter
        for phrase in ["I see", "In this image", "The image shows", "This is", "There is", "There are",
                      "The camera view shows", "Camera view shows", "The camera shows", "Camera shows"]:
            description = description.replace(phrase, "")

        # Capitalize first letter if needed
        if description and description[0].islower():
            description = description[0].upper() + description[1:]

        # Ensure the description ends with a period
        if description and not description.endswith("."):
            description = description + "."

    # Convert to speech using our enhanced TTS endpoint
    tts_start = time.time()

    # Use the voice_type parameter from the request
    # If not provided, use female voice for camera/blind assistant mode and default for other modes
    if 'voice_type' not in request.form:
        voice_type = "female" if is_camera or mode == 'camera' else "default"

    print(f"Using voice type: {voice_type}")

    # Call our enhanced TTS function with Hugging Face models
    audio_b64 = generate_tts_audio(description, voice_type)

    tts_time = time.time() - tts_start

    # Prepare response
    response = {
        'description': description,
        'audio': audio_b64,
        'timestamp': time.time(),
        'processing_time': {
            'api': round(api_time, 2),
            'tts': round(tts_time, 2),
            'total': round(api_time + tts_time, 2)
        }
    }

    # Cache the result (limit cache size)
    if len(image_cache) >= 50:  # Reduced from 100 to save memory
        # Remove oldest entry
        oldest_key = min(image_cache.keys(), key=lambda k: image_cache[k].get('timestamp', 0))
        image_cache.pop(oldest_key)

    image_cache[image_hash] = response

    return jsonify(response)

# if __name__ == '__main__':
#     print("Starting Flask server...")
#     app.run(debug=True)
#     print("Server is running on http://localhost:5000")


# Helper function to generate TTS audio and return base64 encoded audio
def generate_tts_audio(text, voice_type="default"):
    # Use Groq's PlayAI TTS model for high-quality text-to-speech
    print(f"Generating audio with Groq PlayAI TTS for text: {text[:50]}...")

    try:
        # Determine which PlayAI TTS model to use based on voice_type
        model = "playai-tts"  # Default model for English

        # For Arabic language support, uncomment this:
        # if voice_type == "arabic":
        #     model = "playai-tts-arabic"

        # Map voice types to Groq PlayAI voices
        groq_voice_mapping = {
            "default": "Fritz-PlayAI",
            "nova": "Celeste-PlayAI",
            "shimmer": "Cheyenne-PlayAI",
            "female": "Celeste-PlayAI",
            "male": "Cillian-PlayAI",
            "onyx": "Cillian-PlayAI",
            "echo": "Basil-PlayAI",
            "alloy": "Atlas-PlayAI",
            "fable": "Indigo-PlayAI"
        }

        # Get the appropriate voice for Groq PlayAI TTS
        groq_voice = groq_voice_mapping.get(voice_type, "Fritz-PlayAI")

        # Direct API request to Groq's TTS endpoint
        url = "https://api.groq.com/openai/v1/audio/speech"

        # Get the API key from the existing groq_client
        api_key = groq_client.api_key

        # Set up headers with the API key
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Prepare the payload
        data = {
            "model": model,
            "voice": groq_voice,
            "input": text,
            "response_format": "wav"
        }

        # Make the API request
        response = requests.post(url, headers=headers, json=data)

        # Check if request was successful
        if response.status_code == 200:
            # The response is the audio data directly
            audio_data = response.content
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            print(f"Successfully generated audio with Groq PlayAI TTS, size: {len(audio_b64)} bytes")
            return audio_b64
        else:
            print(f"Error from Groq PlayAI TTS API: {response.status_code}, {response.text}")
            # Check if the error is about terms acceptance
            if "terms acceptance" in response.text or "model_terms_required" in response.text:
                print("Groq PlayAI TTS requires terms acceptance. Falling back to Together AI TTS.")
                return try_together_tts(text, voice_type)
            else:
                # Fall back to Together AI TTS for other errors
                return try_together_tts(text, voice_type)
    except Exception as e:
        print(f"Error using Groq PlayAI TTS: {str(e)}")
        # Fall back to Together AI TTS
        return try_together_tts(text, voice_type)

# Try enhanced Hugging Face TTS models as first fallback
def try_together_tts(text, voice_type="default"):
    print(f"Trying enhanced Hugging Face TTS for text: {text[:50]}...")
    try:
        # Select a high-quality TTS model based on voice type
        if voice_type in ["nova", "shimmer", "female"]:
            # Female voice - VITS model with high quality
            API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
        elif voice_type in ["onyx", "male"]:
            # Male voice - Another high quality model
            API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
        else:
            # Default/neutral voice
            API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan"

        # Prepare the payload
        payload = {"inputs": text}

        # Set up headers for Hugging Face API
        hf_headers = {
            "Content-Type": "application/json"
        }

        # Get Hugging Face API key from environment if available
        hf_api_key = os.environ.get("HF_API_KEY")
        if hf_api_key:
            hf_headers["Authorization"] = f"Bearer {hf_api_key}"

        # Make the API request
        response = requests.post(API_URL, headers=hf_headers, json=payload)

        if response.status_code == 200:
            # Convert to base64
            audio_data = response.content
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            print(f"Successfully generated audio with enhanced Hugging Face TTS, size: {len(audio_b64)} bytes")
            return audio_b64
        else:
            print(f"Error from enhanced Hugging Face TTS API: {response.status_code}, {response.text}")
            # Fall back to standard Hugging Face TTS models
            return try_huggingface_tts(text, voice_type)
    except Exception as e:
        print(f"Error using enhanced Hugging Face TTS: {str(e)}")
        # Fall back to standard Hugging Face TTS
        return try_huggingface_tts(text, voice_type)

# Try Hugging Face TTS models as a second fallback
def try_huggingface_tts(text, voice_type="default"):
    print(f"Trying Hugging Face TTS for text: {text[:50]}...")
    try:
        # Select the appropriate Hugging Face model based on voice type
        if voice_type in ["nova", "shimmer", "female"]:
            # Female voice - Microsoft SpeechT5
            API_URL = "https://api-inference.huggingface.co/models/microsoft/speecht5_tts"
        elif voice_type in ["onyx", "male"]:
            # Male voice - Facebook MMS TTS
            API_URL = "https://api-inference.huggingface.co/models/facebook/mms-tts-eng"
        else:
            # Default/neutral voice
            API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

        # Prepare the payload
        payload = {"inputs": text}

        # Set up headers for Hugging Face API
        hf_headers = {
            "Content-Type": "application/json"
        }

        # Get Hugging Face API key from environment if available
        hf_api_key = os.environ.get("HF_API_KEY")
        if hf_api_key:
            hf_headers["Authorization"] = f"Bearer {hf_api_key}"

        # Make the API request
        response = requests.post(API_URL, headers=hf_headers, json=payload)

        if response.status_code == 200:
            # Convert to base64
            audio_data = response.content
            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
            print(f"Successfully generated audio with Hugging Face TTS, size: {len(audio_b64)} bytes")
            return audio_b64
        else:
            print(f"Error from Hugging Face TTS API: {response.status_code}, {response.text}")
            # Fall back to gTTS if both Together AI and Hugging Face APIs fail
            return fallback_gtts(text)
    except Exception as e:
        print(f"Error using Hugging Face TTS: {str(e)}")
        # Fall back to gTTS
        return fallback_gtts(text)

# Fallback to gTTS if Groq API fails
def fallback_gtts(text):
    print(f"Falling back to gTTS for text: {text[:50]}...")
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)

        # Save to buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        # Convert to base64
        audio_data = audio_buffer.read()
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        print(f"Successfully generated audio with gTTS fallback, size: {len(audio_b64)} bytes")
        return audio_b64
    except Exception as e:
        print(f"Error using gTTS fallback: {str(e)}")
        # Return an empty string as a last resort
        return ""

@app.route("/tts/", methods=["POST"])
def tts():
    text = request.form.get("text", "")
    voice_type = request.form.get("voice", "default")

    # Map voice types to our internal voice types
    # This allows the frontend to use simple voice names
    voice_mapping = {
        "default": "default",  # Default voice (Fritz-PlayAI)
        "female": "female",   # Female voice (Celeste-PlayAI)
        "male": "male",      # Male voice (Cillian-PlayAI)
        "sexy": "shimmer",   # Bright voice (Cheyenne-PlayAI)
        "better": "echo",    # Balanced voice (Basil-PlayAI)
        "fable": "fable",    # Expressive voice (Indigo-PlayAI)
        "alloy": "alloy"     # Neutral voice (Atlas-PlayAI)
    }

    # Map the requested voice to our internal voice types
    internal_voice = voice_mapping.get(voice_type, "default")

    # Generate audio using our helper function with the mapped voice type
    audio_b64 = generate_tts_audio(text, internal_voice)

    # Convert base64 back to binary
    audio = base64.b64decode(audio_b64)

    # Determine file extension and mimetype based on content
    # Microsoft SpeechT5 returns audio in WAV format
    # The alternative model returns audio in WAV format as well
    suffix = ".wav"
    mimetype = "audio/wav"

    # Check if the audio data has MP3 header (unlikely but possible with some models)
    if len(audio) > 2 and audio[0] == 0xFF and (audio[1] & 0xE0) == 0xE0:
        suffix = ".mp3"
        mimetype = "audio/mpeg"

    # Return the audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as out:
        out.write(audio)
        return send_file(out.name, mimetype=mimetype, as_attachment=True, download_name=f"output{suffix}")

@app.route('/process-voice-command', methods=['POST'])
def process_voice_command():
    # Get audio data from request
    if 'audio' not in request.files and 'command' not in request.form:
        return jsonify({'error': 'No audio file or command text provided'}), 400

    command_text = ""

    # If audio file is provided, transcribe it
    if 'audio' in request.files:
        audio_file = request.files['audio']
        # Save the audio file temporarily
        temp_audio_path = os.path.join(tempfile.gettempdir(), 'voice_command.webm')
        audio_file.save(temp_audio_path)

        try:
            # Use Groq API to transcribe the audio
            # For now, we'll use a simple approach and assume the command is already transcribed
            # In a production app, you would use a proper speech-to-text API here
            command_text = "Unknown command"

            # Clean up the temp file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    else:
        # If command text is provided directly (for testing or fallback)
        command_text = request.form.get('command', '').lower()

    # Process the command
    response = {
        'command': command_text,
        'action': 'none',
        'message': 'Command not recognized'
    }

    # Check for camera control commands
    if any(phrase in command_text for phrase in ['turn on camera', 'start camera', 'open camera', 'activate camera']):
        response['action'] = 'camera_on'
        response['message'] = 'Camera activated'
    elif any(phrase in command_text for phrase in ['turn off camera', 'stop camera', 'close camera', 'deactivate camera']):
        response['action'] = 'camera_off'
        response['message'] = 'Camera deactivated'

    # Check for blind assistant commands - add more variations and debug logging
    elif any(phrase in command_text for phrase in [
        'turn on blind assistant', 'start blind assistant', 'activate blind assistant', 'enable blind mode',
        'blind assistant on', 'enable blind assistant', 'start blind mode', 'blind mode on',
        'turn on blind mode', 'activate blind mode'
    ]):
        print(f"Recognized blind assistant ON command: '{command_text}'")
        response['action'] = 'blind_assistant_on'
        response['message'] = 'Blind assistant mode activated'
    elif any(phrase in command_text for phrase in [
        'turn off blind assistant', 'stop blind assistant', 'deactivate blind assistant', 'disable blind mode',
        'blind assistant off', 'disable blind assistant', 'stop blind mode', 'blind mode off',
        'turn off blind mode', 'deactivate blind mode'
    ]):
        print(f"Recognized blind assistant OFF command: '{command_text}'")
        response['action'] = 'blind_assistant_off'
        response['message'] = 'Blind assistant mode deactivated'

    # Generate audio response
    response['audio'] = generate_tts_audio(response['message'], 'female')

    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)