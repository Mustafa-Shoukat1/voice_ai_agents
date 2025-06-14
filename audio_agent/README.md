# Audio Agent - Intelligent Voice Assistant

## Overview

The Audio Agent is a sophisticated real-time speech recognition and response system that enables natural voice conversations with AI. It combines cutting-edge speech-to-text, natural language processing, and text-to-speech technologies to create a seamless voice interaction experience.

## Features

### 🎤 **Advanced Speech Recognition**
- Real-time audio capture from microphone
- High-accuracy speech-to-text using OpenAI Whisper
- Ambient noise adjustment and optimization
- Configurable timeout and phrase limits

### 🧠 **Intelligent Response Generation**
- Context-aware conversations using OpenAI GPT-3.5-turbo
- Conversation history tracking
- Customizable response length and style
- Smart text cleaning for better speech synthesis

### 🔊 **Natural Text-to-Speech**
- Optimized voice synthesis with adjustable speed and volume
- Text preprocessing for better pronunciation
- Abbreviation expansion for clearer speech
- Cross-platform TTS engine support

### 💬 **Conversation Management**
- Continuous conversation loops
- Graceful exit commands
- Interaction counting and limits
- Comprehensive logging and error handling

## Prerequisites

### System Requirements
- Python 3.7 or higher
- Microphone access
- Internet connection for OpenAI API
- Audio output device (speakers/headphones)

### Required Dependencies
```
speech_recognition
pyttsx3
openai
pyaudio (for microphone support)
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mustafa-Shoukat1/voice_ai_agents.git
   cd voice_ai_agents/audio_agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**
   - Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Replace `"your-api-key-here"` in `agent.py` with your actual API key
   - Or set as environment variable: `export OPENAI_API_KEY="your-key"`

## Usage

### Basic Usage
```python
from agent import AudioAgent

# Initialize the agent
agent = AudioAgent(
    api_key="your-openai-api-key",
    voice_rate=150,    # Words per minute
    voice_volume=0.9   # Volume level (0.0-1.0)
)

# Start conversation
agent.run_conversation_loop()
```

### Advanced Configuration
```python
# Custom settings for different use cases
agent = AudioAgent(
    api_key="your-api-key",
    voice_rate=180,      # Faster speech for quick interactions
    voice_volume=0.7     # Lower volume for quiet environments
)

# Limited conversation with custom timeout
agent.run_conversation_loop(max_interactions=10)
```

### Individual Component Usage
```python
# Just speech recognition
user_input = agent.listen_for_speech(timeout=10)

# Just response generation
response = agent.generate_response("Hello, how are you?")

# Just text-to-speech
agent.speak_response("Hello! How can I help you today?")
```

## Configuration Options

### AudioAgent Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | OpenAI API key |
| `voice_rate` | int | 150 | Speech rate (words per minute) |
| `voice_volume` | float | 0.9 | Voice volume (0.0 to 1.0) |

### Conversation Settings
| Method | Parameters | Description |
|--------|------------|-------------|
| `listen_for_speech()` | `timeout=5`, `phrase_time_limit=10` | Audio capture settings |
| `run_conversation_loop()` | `max_interactions=50` | Conversation limits |

## Voice Commands

### Starting Conversation
- Simply speak after running the agent
- The system will automatically detect speech

### Ending Conversation
- Say "goodbye", "exit", "quit", or "stop"
- Press Ctrl+C to force quit

## Logging and Monitoring

The agent provides comprehensive logging for:
- Speech recognition attempts and results
- Response generation processes
- TTS operations
- Error handling and debugging

Logs are saved to `audio_agent.log` and displayed in console.

## Troubleshooting

### Common Issues

**Microphone Not Detected**
```bash
# Install PyAudio properly
pip uninstall pyaudio
pip install pyaudio
```

**API Key Errors**
- Verify your OpenAI API key is valid
- Check your account has sufficient credits
- Ensure internet connection is stable

**Speech Recognition Issues**
- Check microphone permissions
- Reduce background noise
- Adjust microphone volume in system settings

**TTS Not Working**
- Verify audio drivers are installed
- Check speaker/headphone connections
- Try different TTS engines if available

### Platform-Specific Notes

**Windows**
- May require Visual C++ redistributables for PyAudio
- Windows Defender might block microphone access

**macOS**
- Grant microphone permissions in System Preferences
- Install Xcode command line tools if needed

**Linux**
- Install ALSA development packages
- Configure PulseAudio for microphone access

## Performance Optimization

### For Better Recognition
- Use a good quality microphone
- Minimize background noise
- Speak clearly and at moderate pace
- Position microphone 6-12 inches from mouth

### For Faster Response
- Use faster internet connection
- Reduce `max_tokens` in GPT calls
- Optimize system prompt length
- Consider using GPT-3.5-turbo-instruct for simpler tasks

## Security Considerations

- API keys should be stored securely (environment variables)
- Audio data is sent to OpenAI servers for processing
- Consider local alternatives for sensitive applications
- Implement rate limiting for production use

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper comments
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review OpenAI API documentation

## Changelog

### v1.0.0
- Initial release with core functionality
- Speech recognition using OpenAI Whisper
- Response generation with GPT-3.5-turbo
- Text-to-speech with pyttsx3
- Conversation loop management
- Comprehensive logging system

---

**Author:** Mustafa Shoukat  
**Version:** 1.0.0  
**Last Updated:** 2024