# Vision Backend

This is the backend for the Vision application, which provides image description services for blind users.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables:
   - Copy the `.env.example` file to a new file named `.env`
   - Replace `your_groq_api_key_here` with your actual Groq API key
   - Optionally, add your Hugging Face API key if you want to use Hugging Face models as fallback

   ```
   cp .env.example .env
   # Then edit the .env file with your API keys
   ```

3. Run the application:
   ```
   python app.py
   ```

## API Endpoints

- `/describe-image`: Analyzes an image and returns a description with audio
- `/tts`: Converts text to speech
- `/process-voice-command`: Processes voice commands for controlling the application

## Environment Variables

- `GROQ_API_KEY`: Your Groq API key (required)
- `HF_API_KEY`: Your Hugging Face API key (optional)
- `PORT`: The port to run the Flask server on (default: 5000)

## Voice Commands

The application supports the following voice commands:
- "Turn on camera"
- "Turn off camera"
- "Turn on blind assistant"
- "Turn off blind assistant"
"# vision-back" 
