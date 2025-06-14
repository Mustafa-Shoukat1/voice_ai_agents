"""
Customer Support Voice Agent - Advanced AI-Powered Support System
================================================================

This module implements an intelligent customer support voice agent that can:
- Handle customer inquiries through voice interaction
- Provide contextual support responses
- Manage customer data and conversation history
- Escalate complex issues to human agents
- Generate support tickets and follow-ups

Author: Mustafa Shoukat
Version: 1.0.0
"""

import speech_recognition as sr
import pyttsx3
import openai
import json
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any
import re
import uuid

# Configure logging for customer support operations
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_support.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomerSupportAgent:
    """
    Advanced Customer Support Voice Agent with comprehensive support capabilities.
    
    This class provides:
    - Voice-based customer interaction
    - Context-aware support responses
    - Customer data management
    - Issue categorization and escalation
    - Support ticket generation
    """
    
    def __init__(self, api_key: str, company_name: str = "Your Company"):
        """
        Initialize the Customer Support Agent with company-specific settings.
        
        Args:
            api_key (str): OpenAI API key for AI services
            company_name (str): Name of the company for personalized responses
        """
        # Core configuration
        self.company_name = company_name
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize speech components with customer service optimizations
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Configure TTS for professional customer service voice
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 140)  # Slower, clearer speech
        self.tts_engine.setProperty('volume', 0.9)
        
        # Set professional voice (typically female for customer service)
        voices = self.tts_engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            female_voice = next((v for v in voices if 'female' in v.name.lower()), voices[0])
            self.tts_engine.setProperty('voice', female_voice.id)
        
        # Customer support knowledge base and categories
        self.support_categories = {
            'technical': ['bug', 'error', 'not working', 'broken', 'issue', 'problem'],
            'billing': ['payment', 'charge', 'bill', 'invoice', 'refund', 'pricing'],
            'account': ['login', 'password', 'access', 'account', 'profile', 'settings'],
            'product': ['feature', 'how to', 'tutorial', 'guide', 'usage', 'function'],
            'general': ['question', 'help', 'support', 'information', 'inquiry']
        }
        
        # Customer interaction tracking
        self.current_session = {
            'session_id': str(uuid.uuid4()),
            'start_time': datetime.now(),
            'customer_data': {},
            'interactions': [],
            'issue_category': None,
            'escalated': False,
            'resolved': False
        }
        
        # Optimize microphone for customer calls
        with self.microphone as source:
            logger.info("Calibrating microphone for customer support...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
            
        logger.info(f"Customer Support Agent initialized for {company_name}")
    
    def greet_customer(self) -> None:
        """
        Provide professional greeting to start customer interaction.
        """
        greeting = f"""Hello! Thank you for contacting {self.company_name} customer support. 
        My name is Alex, and I'm your AI support assistant. 
        I'm here to help you with any questions or issues you may have today. 
        How can I assist you?"""
        
        logger.info("Greeting customer")
        self.speak_response(greeting)
        
        # Log session start
        self.current_session['interactions'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'greeting',
            'agent_response': greeting
        })
    
    def listen_for_customer_input(self, timeout: int = 8) -> Optional[str]:
        """
        Listen for customer speech with enhanced recognition for support calls.
        
        Args:
            timeout (int): Maximum wait time for customer input
            
        Returns:
            Optional[str]: Customer's speech as text or None
        """
        try:
            logger.info("Listening for customer input...")
            
            with self.microphone as source:
                # Longer timeout for customers who might need time to think
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=15  # Allow longer customer explanations
                )
            
            # Use Whisper for accurate transcription of customer speech
            logger.info("Transcribing customer speech...")
            
            # Convert audio to text using OpenAI Whisper
            audio_data = audio.get_wav_data()
            
            # Save audio temporarily for Whisper processing
            with open("temp_customer_audio.wav", "wb") as f:
                f.write(audio_data)
            
            with open("temp_customer_audio.wav", "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            customer_input = transcription.text.strip()
            logger.info(f"Customer said: {customer_input}")
            
            # Log customer interaction
            self.current_session['interactions'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'customer_input',
                'content': customer_input
            })
            
            return customer_input
            
        except sr.WaitTimeoutError:
            logger.warning("Customer input timeout")
            return None
        except Exception as e:
            logger.error(f"Customer speech recognition failed: {str(e)}")
            return None
    
    def categorize_customer_issue(self, customer_input: str) -> str:
        """
        Automatically categorize customer issue for appropriate routing.
        
        Args:
            customer_input (str): Customer's description of their issue
            
        Returns:
            str: Issue category (technical, billing, account, product, general)
        """
        customer_input_lower = customer_input.lower()
        
        # Check for category keywords
        for category, keywords in self.support_categories.items():
            if any(keyword in customer_input_lower for keyword in keywords):
                logger.info(f"Issue categorized as: {category}")
                return category
        
        # Default to general if no specific category found
        logger.info("Issue categorized as: general")
        return 'general'
    
    def generate_support_response(self, customer_input: str, category: str) -> str:
        """
        Generate contextual support response based on customer issue and category.
        
        Args:
            customer_input (str): Customer's input text
            category (str): Categorized issue type
            
        Returns:
            str: Professional support response
        """
        try:
            # Build context-aware system prompt
            system_prompt = f"""You are Alex, a professional customer support representative for {self.company_name}.

GUIDELINES:
- Be empathetic, helpful, and solution-oriented
- Use professional but friendly language
- Keep responses clear and concise (under 100 words)
- Offer specific steps when possible
- Acknowledge customer frustration when appropriate
- Ask clarifying questions if needed

ISSUE CATEGORY: {category}
COMPANY: {self.company_name}

Previous interactions in this session: {len(self.current_session['interactions'])}

Provide a helpful response to resolve the customer's {category} issue."""

            # Generate response using GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": customer_input}
                ],
                max_tokens=150,
                temperature=0.3  # Lower temperature for more consistent support responses
            )
            
            support_response = response.choices[0].message.content.strip()
            
            # Log the response
            self.current_session['interactions'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'agent_response',
                'category': category,
                'content': support_response
            })
            
            logger.info("Support response generated successfully")
            return support_response
            
        except Exception as e:
            logger.error(f"Support response generation failed: {str(e)}")
            
            # Fallback response for errors
            fallback = f"""I apologize, but I'm experiencing a technical issue right now. 
            Let me connect you with a human agent who can better assist you with your {category} concern. 
            Please hold for just a moment."""
            
            return fallback
    
    def check_escalation_needed(self, customer_input: str, interaction_count: int) -> bool:
        """
        Determine if the issue needs escalation to human agent.
        
        Args:
            customer_input (str): Customer's latest input
            interaction_count (int): Number of interactions in session
            
        Returns:
            bool: True if escalation is needed
        """
        escalation_triggers = [
            'human agent', 'supervisor', 'manager', 'escalate',
            'not satisfied', 'frustrated', 'angry', 'complaint',
            'cancel account', 'legal action', 'lawyer'
        ]
        
        # Check for explicit escalation requests
        customer_lower = customer_input.lower()
        if any(trigger in customer_lower for trigger in escalation_triggers):
            logger.info("Escalation triggered by customer request")
            return True
        
        # Auto-escalate after many interactions without resolution
        if interaction_count > 6 and not self.current_session['resolved']:
            logger.info("Escalation triggered by interaction count")
            return True
        
        return False
    
    def escalate_to_human(self) -> str:
        """
        Handle escalation to human agent with professional transition.
        
        Returns:
            str: Escalation message for customer
        """
        self.current_session['escalated'] = True
        
        escalation_message = f"""I understand your concern, and I want to make sure you receive the best possible assistance. 
        I'm now connecting you with one of our specialized human agents who will be better equipped to resolve this issue for you. 
        
        Your case reference number is {self.current_session['session_id'][:8].upper()}. 
        The human agent will have access to our entire conversation. 
        
        Please stay on the line, and someone will be with you shortly. Thank you for your patience."""
        
        logger.info(f"Escalating session {self.current_session['session_id']} to human agent")
        
        # Log escalation
        self.current_session['interactions'].append({
            'timestamp': datetime.now().isoformat(),
            'type': 'escalation',
            'reason': 'Customer request or interaction limit reached'
        })
        
        return escalation_message
    
    def speak_response(self, text: str) -> bool:
        """
        Deliver response with professional customer service tone.
        
        Args:
            text (str): Response text to speak
            
        Returns:
            bool: Success status
        """
        try:
            # Clean text for better speech delivery
            cleaned_text = self._optimize_text_for_speech(text)
            
            logger.info(f"Speaking response: {cleaned_text[:50]}...")
            self.tts_engine.say(cleaned_text)
            self.tts_engine.runAndWait()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            return False
    
    def _optimize_text_for_speech(self, text: str) -> str:
        """
        Optimize text for clear customer service speech delivery.
        
        Args:
            text (str): Raw response text
            
        Returns:
            str: Optimized text for speech
        """
        # Add pauses for better clarity
        text = text.replace('. ', '. ')  # Ensure pauses after sentences
        text = text.replace('?', '? ')   # Pause after questions
        text = text.replace('!', '! ')   # Pause after exclamations
        
        # Expand common abbreviations for clarity
        replacements = {
            'AI': 'A.I.',
            'FAQ': 'Frequently Asked Questions',
            'URL': 'web address',
            'app': 'application',
            'etc.': 'and so on'
        }
        
        for abbrev, expansion in replacements.items():
            text = text.replace(abbrev, expansion)
        
        return text.strip()
    
    def end_session_summary(self) -> str:
        """
        Generate session summary and closing message.
        
        Returns:
            str: Professional closing message
        """
        session_duration = datetime.now() - self.current_session['start_time']
        
        if self.current_session['escalated']:
            closing = """Thank you for contacting our support team today. 
            Your case has been escalated to our specialized team, and you should hear back within 24 hours. 
            Have a great day!"""
        else:
            closing = f"""Thank you for contacting {self.company_name} support today. 
            I hope I was able to resolve your issue satisfactorily. 
            If you need any further assistance, please don't hesitate to contact us again. 
            Have a wonderful day!"""
        
        # Log session end
        self.current_session['end_time'] = datetime.now()
        self.current_session['duration_minutes'] = session_duration.total_seconds() / 60
        
        logger.info(f"Session ended: {self.current_session['session_id']}")
        
        return closing
    
    def run_customer_support_session(self, max_interactions: int = 20) -> None:
        """
        Main customer support session loop.
        
        Args:
            max_interactions (int): Maximum number of interaction rounds
        """
        logger.info("Starting customer support session...")
        
        # Welcome customer
        self.greet_customer()
        
        interaction_count = 0
        
        try:
            while interaction_count < max_interactions:
                # Listen for customer input
                customer_input = self.listen_for_customer_input()
                
                if customer_input is None:
                    # Handle silence with helpful prompt
                    prompt = "I'm here to help. Please let me know how I can assist you today."
                    self.speak_response(prompt)
                    continue
                
                # Check for session end requests
                if any(end_phrase in customer_input.lower() for end_phrase in 
                       ['goodbye', 'thank you', 'that\'s all', 'no more questions', 'resolved']):
                    self.current_session['resolved'] = True
                    closing_message = self.end_session_summary()
                    self.speak_response(closing_message)
                    break
                
                # Categorize the issue
                if not self.current_session['issue_category']:
                    self.current_session['issue_category'] = self.categorize_customer_issue(customer_input)
                
                # Check if escalation is needed
                if self.check_escalation_needed(customer_input, interaction_count):
                    escalation_message = self.escalate_to_human()
                    self.speak_response(escalation_message)
                    break
                
                # Generate and deliver support response
                category = self.current_session['issue_category']
                response = self.generate_support_response(customer_input, category)
                self.speak_response(response)
                
                interaction_count += 1
                
        except KeyboardInterrupt:
            logger.info("Support session interrupted")
            self.speak_response("Thank you for contacting support. Have a great day!")
        except Exception as e:
            logger.error(f"Support session error: {str(e)}")
            self.speak_response("I apologize for the technical difficulty. Please try calling our support line.")
        finally:
            logger.info(f"Support session completed after {interaction_count} interactions")
    
    def save_session_data(self, filename: str = None) -> str:
        """
        Save session data for record keeping and analysis.
        
        Args:
            filename (str): Optional custom filename
            
        Returns:
            str: Path to saved session file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"support_session_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_session, f, indent=2, default=str)
            
            logger.info(f"Session data saved to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save session data: {str(e)}")
            return ""

# Example usage and demonstration
if __name__ == "__main__":
    """
    Example usage of the Customer Support Voice Agent.
    """
    
    # Configuration
    API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key
    COMPANY_NAME = "TechCorp Solutions"  # Replace with your company name
    
    try:
        # Initialize customer support agent
        support_agent = CustomerSupportAgent(
            api_key=API_KEY,
            company_name=COMPANY_NAME
        )
        
        # Start customer support session
        support_agent.run_customer_support_session(max_interactions=15)
        
        # Save session data for records
        session_file = support_agent.save_session_data()
        print(f"Session data saved to: {session_file}")
        
    except Exception as e:
        logger.error(f"Failed to start Customer Support Agent: {str(e)}")
        print("Please ensure you have set up your OpenAI API key correctly.")