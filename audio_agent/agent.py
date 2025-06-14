"""
Audio Agent - Real-time Speech Recognition and Response System
============================================================

This module implements an intelligent audio agent that can:
- Capture real-time audio from microphone
- Convert speech to text using OpenAI Whisper
- Generate intelligent responses using OpenAI GPT
- Convert responses back to speech using text-to-speech

Author: Mustafa Shoukat
Version: 1.0.0
"""

import speech_recognition as sr
import pyttsx3
import openai
import logging
from typing import Optional, Dict, Any
import time

# Configure logging for better debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AudioAgent:
    """
    A comprehensive audio agent for speech-to-speech conversation.
    
    This class handles the complete pipeline from audio input to audio output:
    1. Speech Recognition (STT)
    2. Natural Language Processing 
    3. Text-to-Speech (TTS)
    """
    
    def __init__(self, api_key: str, voice_rate: int = 150, voice_volume: float = 0.9):
        """
        Initialize the Audio Agent with required configurations.
        
        Args:
            api_key (str): OpenAI API key for GPT and Whisper services
            voice_rate (int): Speech rate for TTS (words per minute)
            voice_volume (float): Voice volume level (0.0 to 1.0)
        """
        # Initialize OpenAI client with API key
        self.client = openai.OpenAI(api_key=api_key)
        
        # Configure text-to-speech engine with optimized settings
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', voice_rate)  # Adjust speaking speed
        self.tts_engine.setProperty('volume', voice_volume)  # Set volume level
        
        # Initialize speech recognition with enhanced error handling
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Optimize microphone settings for better recognition
        with self.microphone as source:
            logger.info("Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
        logger.info("Audio Agent initialized successfully")
    
    def listen_for_speech(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Capture and convert speech to text using advanced speech recognition.
        
        Args:
            timeout (int): Maximum time to wait for speech input
            phrase_time_limit (int): Maximum time for a single phrase
            
        Returns:
            Optional[str]: Transcribed text or None if recognition fails
        """
        try:
            logger.info("Listening for speech input...")
            
            # Capture audio with timeout and phrase limits
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=phrase_time_limit
                )
            
            logger.info("Processing speech recognition...")
            
            # Use OpenAI Whisper for high-accuracy transcription
            audio_data = audio.get_wav_data()
            
            # Call Whisper API for transcription
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_data,
                response_format="text"
            )
            
            transcribed_text = response['text'].strip()
            logger.info(f"Transcription successful: {transcribed_text}")
            return transcribed_text
            
        except sr.WaitTimeoutError:
            logger.warning("No speech detected within timeout period")
            return None
        except Exception as e:
            logger.error(f"Speech recognition failed: {str(e)}")
            return None
    
    def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """
        Generate intelligent response using OpenAI GPT with context awareness.
        
        Args:
            user_input (str): User's spoken input text
            context (Dict[str, Any]): Conversation context and metadata
            
        Returns:
            str: Generated response text
        """
        try:
            logger.info(f"Generating response for: {user_input[:50]}...")
            
            # Prepare system prompt with context
            system_prompt = """You are a helpful and intelligent audio assistant. 
            Provide clear, concise, and contextually appropriate responses. 
            Keep responses conversational and under 100 words for optimal speech delivery."""
            
            # Include conversation context if available
            if context:
                system_prompt += f"\nContext: {context}"
            
            # Call GPT API for response generation
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            generated_text = response.choices[0].message.content.strip()
            logger.info("Response generated successfully")
            return generated_text
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now."
    
    def speak_response(self, text: str) -> bool:
        """
        Convert text to speech with error handling and optimization.
        
        Args:
            text (str): Text to be spoken
            
        Returns:
            bool: True if speech was successful, False otherwise
        """
        try:
            logger.info(f"Speaking response: {text[:30]}...")
            
            # Clean and optimize text for better speech
            cleaned_text = self._clean_text_for_speech(text)
            
            # Generate speech with the TTS engine
            self.tts_engine.say(cleaned_text)
            self.tts_engine.runAndWait()
            
            logger.info("Speech output completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Text-to-speech failed: {str(e)}")
            return False
    
    def _clean_text_for_speech(self, text: str) -> str:
        """
        Clean and optimize text for better speech synthesis.
        
        Args:
            text (str): Raw text to be cleaned
            
        Returns:
            str: Cleaned and optimized text
        """
        # Remove markdown formatting, URLs, and other non-speech elements
        text = text.replace('\n', ' ').replace('\r', '').strip()
        
        # Replace abbreviations with full words for better pronunciation
        replacements = {
            "AI": "Artificial Intelligence",
            "API": "Application Programming Interface",
            "URL": "web address",
            # Add more as needed
        }
        
        for abbrev, full_form in replacements.items():
            text = text.replace(abbrev, full_form)
        
        return text.strip()
    
    def run_conversation_loop(self, max_interactions: int = 50) -> None:
        """
        Main conversation loop for continuous audio interaction.
        
        Args:
            max_interactions (int): Maximum number of conversation turns
        """
        logger.info("Starting audio conversation loop...")
        print("Audio Agent is ready! Say something to start the conversation.")
        print("Say 'goodbye' or 'exit' to end the conversation.")
        
        interaction_count = 0
        conversation_context = {"start_time": time.time(), "interactions": []}
        
        try:
            while interaction_count < max_interactions:
                # Listen for user input
                user_speech = self.listen_for_speech()
                
                if user_speech is None:
                    continue
                
                # Check for exit commands
                if any(exit_word in user_speech.lower() for exit_word in ['goodbye', 'exit', 'quit', 'stop']):
                    self.speak_response("Goodbye! Have a great day!")
                    break
                
                # Store interaction in context
                conversation_context["interactions"].append({
                    "user": user_speech,
                    "timestamp": time.time()
                })
                
                # Generate and speak response
                response = self.generate_response(user_speech, conversation_context)
                conversation_context["interactions"][-1]["assistant"] = response
                
                self.speak_response(response)
                
                interaction_count += 1
                
        except KeyboardInterrupt:
            logger.info("Conversation interrupted by user")
            self.speak_response("Conversation ended. Goodbye!")
        except Exception as e:
            logger.error(f"Conversation loop error: {str(e)}")
        finally:
            logger.info(f"Conversation ended after {interaction_count} interactions")

# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Audio Agent.
    Replace 'your-api-key-here' with your actual OpenAI API key.
    """
    
    # Configuration
    API_KEY = "your-api-key-here"  # Replace with your OpenAI API key
    
    try:
        # Initialize the audio agent
        agent = AudioAgent(
            api_key=API_KEY,
            voice_rate=160,  # Slightly faster speech
            voice_volume=0.8  # Comfortable volume level
        )
        
        # Start the conversation loop
        agent.run_conversation_loop(max_interactions=30)
        
    except Exception as e:
        logger.error(f"Failed to start Audio Agent: {str(e)}")
        print("Please ensure you have set up your OpenAI API key correctly.")