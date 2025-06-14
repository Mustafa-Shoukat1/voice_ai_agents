"""
Voice RAG Agent - Advanced Retrieval-Augmented Generation with Voice Interface
==============================================================================

This module implements an intelligent Voice RAG (Retrieval-Augmented Generation) agent that:
- Processes documents and creates searchable knowledge bases
- Accepts voice queries and provides spoken responses
- Uses vector embeddings for semantic document search
- Combines retrieved information with AI generation for accurate answers
- Maintains conversation context and provides source citations

Author: Mustafa Shoukat
Version: 1.0.0
"""

import speech_recognition as sr
import pyttsx3
import openai
import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import pickle
import hashlib

# Vector database and embeddings
try:
    import faiss
except ImportError:
    faiss = None
    logging.warning("FAISS not installed. Using basic similarity search.")

# Document processing
try:
    import PyPDF2
    from docx import Document as DocxDocument
except ImportError:
    PyPDF2 = None
    DocxDocument = None
    logging.warning("Document processing libraries not installed.")

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_rag_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VoiceRAGAgent:
    """
    Advanced Voice RAG Agent with document processing and semantic search capabilities.
    
    This agent provides:
    - Voice-based question answering using RAG
    - Document ingestion and processing
    - Vector-based semantic search
    - Context-aware response generation
    - Source citation and verification
    """
    
    def __init__(self, api_key: str, knowledge_base_path: str = "knowledge_base"):
        """
        Initialize the Voice RAG Agent with document processing capabilities.
        
        Args:
            api_key (str): OpenAI API key for embeddings and generation
            knowledge_base_path (str): Path to store knowledge base files
        """
        # Core configuration
        self.client = openai.OpenAI(api_key=api_key)
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(exist_ok=True)
        
        # Initialize speech components optimized for Q&A
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Configure TTS for clear information delivery
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)  # Moderate speed for information
        self.tts_engine.setProperty('volume', 0.9)
        
        # Document storage and retrieval system
        self.documents = []  # List of processed documents
        self.embeddings = []  # Document embeddings for semantic search
        self.chunks = []  # Text chunks with metadata
        
        # Vector database for efficient similarity search
        self.vector_index = None
        self.embedding_dimension = 1536  # OpenAI text-embedding-ada-002 dimension
        
        # Conversation context for multi-turn Q&A
        self.conversation_history = []
        self.current_session = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'queries': [],
            'documents_used': set(),
            'confidence_scores': []
        }
        
        # Load existing knowledge base if available
        self._load_knowledge_base()
        
        # Optimize microphone for question recognition
        with self.microphone as source:
            logger.info("Calibrating microphone for voice queries...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            
        logger.info("Voice RAG Agent initialized successfully")
    
    def add_document(self, file_path: str, document_type: str = "auto") -> bool:
        """
        Add a document to the knowledge base with automatic processing.
        
        Args:
            file_path (str): Path to the document file
            document_type (str): Type of document (pdf, txt, docx, auto)
            
        Returns:
            bool: Success status of document addition
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.error(f"Document not found: {file_path}")
                return False
            
            # Auto-detect document type if not specified
            if document_type == "auto":
                document_type = file_path.suffix.lower().lstrip('.')
            
            logger.info(f"Processing document: {file_path.name} (type: {document_type})")
            
            # Extract text based on document type
            text_content = self._extract_text_from_document(file_path, document_type)
            
            if not text_content:
                logger.error(f"Failed to extract text from {file_path}")
                return False
            
            # Create document metadata
            document_metadata = {
                'filename': file_path.name,
                'file_path': str(file_path),
                'type': document_type,
                'size': file_path.stat().st_size,
                'added_date': datetime.now().isoformat(),
                'content_hash': hashlib.md5(text_content.encode()).hexdigest()
            }
            
            # Split document into chunks for better retrieval
            chunks = self._split_text_into_chunks(text_content, document_metadata)
            
            # Generate embeddings for each chunk
            chunk_embeddings = self._generate_embeddings(chunks)
            
            # Add to knowledge base
            self.chunks.extend(chunks)
            self.embeddings.extend(chunk_embeddings)
            self.documents.append(document_metadata)
            
            # Update vector index
            self._update_vector_index()
            
            # Save updated knowledge base
            self._save_knowledge_base()
            
            logger.info(f"Successfully added document: {file_path.name} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {file_path}: {str(e)}")
            return False
    
    def _extract_text_from_document(self, file_path: Path, document_type: str) -> str:
        """
        Extract text content from various document formats.
        
        Args:
            file_path (Path): Path to the document
            document_type (str): Document type (pdf, txt, docx)
            
        Returns:
            str: Extracted text content
        """
        try:
            if document_type == "txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            elif document_type == "pdf" and PyPDF2:
                text = ""
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                return text
            
            elif document_type == "docx" and DocxDocument:
                doc = DocxDocument(file_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text
            
            else:
                logger.warning(f"Unsupported document type: {document_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {str(e)}")
            return ""
    
    def _split_text_into_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into manageable chunks for embedding and retrieval.
        
        Args:
            text (str): Full document text
            metadata (Dict): Document metadata
            
        Returns:
            List[Dict]: List of text chunks with metadata
        """
        # Configuration for chunking
        chunk_size = 1000  # Characters per chunk
        chunk_overlap = 200  # Overlap between chunks
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            # Calculate chunk boundaries
            end = start + chunk_size
            
            # Try to break at sentence boundaries for better context
            if end < len(text):
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = {
                    'text': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end,
                    'document_metadata': metadata.copy(),
                    'word_count': len(chunk_text.split()),
                    'char_count': len(chunk_text)
                }
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next chunk with overlap
            start = end - chunk_overlap
            
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def _generate_embeddings(self, chunks: List[Dict]) -> List[np.ndarray]:
        """
        Generate vector embeddings for text chunks using OpenAI embeddings.
        
        Args:
            chunks (List[Dict]): Text chunks to embed
            
        Returns:
            List[np.ndarray]: Vector embeddings
        """
        try:
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            
            embeddings = []
            batch_size = 20  # Process in batches to avoid rate limits
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk['text'] for chunk in batch_chunks]
                
                # Generate embeddings using OpenAI
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch_texts
                )
                
                # Extract embeddings and convert to numpy arrays
                batch_embeddings = [np.array(embedding.embedding) for embedding in response.data]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1}")
            
            logger.info("Embedding generation completed")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            return []
    
    def _update_vector_index(self) -> None:
        """
        Update the vector index for efficient similarity search.
        """
        try:
            if not self.embeddings:
                return
            
            # Convert embeddings to numpy array
            embedding_matrix = np.vstack(self.embeddings).astype('float32')
            
            if faiss:
                # Use FAISS for efficient similarity search
                self.vector_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product
                faiss.normalize_L2(embedding_matrix)  # Normalize for cosine similarity
                self.vector_index.add(embedding_matrix)
                logger.info(f"Updated FAISS index with {len(self.embeddings)} vectors")
            else:
                # Store embeddings for basic similarity search
                self.vector_index = embedding_matrix
                logger.info(f"Updated basic index with {len(self.embeddings)} vectors")
                
        except Exception as e:
            logger.error(f"Vector index update failed: {str(e)}")
    
    def listen_for_query(self, timeout: int = 10) -> Optional[str]:
        """
        Listen for voice query with enhanced recognition for questions.
        
        Args:
            timeout (int): Maximum wait time for input
            
        Returns:
            Optional[str]: User's query as text
        """
        try:
            logger.info("Listening for voice query...")
            
            with self.microphone as source:
                # Allow longer queries for complex questions
                audio = self.recognizer.listen(
                    source, 
                    timeout=timeout, 
                    phrase_time_limit=20
                )
            
            logger.info("Processing voice query...")
            
            # Use Whisper for accurate transcription
            audio_data = audio.get_wav_data()
            
            with open("temp_query_audio.wav", "wb") as f:
                f.write(audio_data)
            
            with open("temp_query_audio.wav", "rb") as audio_file:
                transcription = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            
            query = transcription.text.strip()
            logger.info(f"Voice query: {query}")
            
            # Add to session tracking
            self.current_session['queries'].append({
                'timestamp': datetime.now().isoformat(),
                'query': query
            })
            
            return query
            
        except sr.WaitTimeoutError:
            logger.warning("Voice query timeout")
            return None
        except Exception as e:
            logger.error(f"Voice query recognition failed: {str(e)}")
            return None
    
    def search_knowledge_base(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search the knowledge base for relevant information using semantic similarity.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Dict, float]]: List of (chunk, similarity_score) tuples
        """
        try:
            if not self.chunks or not self.embeddings:
                logger.warning("Knowledge base is empty")
                return []
            
            logger.info(f"Searching knowledge base for: {query[:50]}...")
            
            # Generate embedding for the query
            query_response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query]
            )
            query_embedding = np.array(query_response.data[0].embedding)
            
            # Perform similarity search
            if faiss and self.vector_index:
                # Use FAISS for efficient search
                query_embedding = query_embedding.reshape(1, -1).astype('float32')
                faiss.normalize_L2(query_embedding)
                
                scores, indices = self.vector_index.search(query_embedding, top_k)
                
                results = []
                for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                    if idx < len(self.chunks):
                        results.append((self.chunks[idx], float(score)))
                
            else:
                # Basic similarity search using numpy
                embedding_matrix = np.vstack(self.embeddings)
                
                # Calculate cosine similarity
                similarities = np.dot(embedding_matrix, query_embedding) / (
                    np.linalg.norm(embedding_matrix, axis=1) * np.linalg.norm(query_embedding)
                )
                
                # Get top k results
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                results = [(self.chunks[idx], similarities[idx]) for idx in top_indices]
            
            logger.info(f"Found {len(results)} relevant chunks")
            
            # Track which documents were used
            for chunk, _ in results:
                self.current_session['documents_used'].add(
                    chunk['document_metadata']['filename']
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Knowledge base search failed: {str(e)}")
            return []
    
    def generate_rag_response(self, query: str, search_results: List[Tuple[Dict, float]]) -> str:
        """
        Generate response using retrieved information and AI generation.
        
        Args:
            query (str): User's query
            search_results (List[Tuple[Dict, float]]): Retrieved chunks with scores
            
        Returns:
            str: Generated response with source information
        """
        try:
            if not search_results:
                return "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
            
            # Prepare context from retrieved chunks
            context_chunks = []
            sources = []
            
            for chunk, score in search_results:
                context_chunks.append(chunk['text'])
                source_info = f"{chunk['document_metadata']['filename']} (confidence: {score:.2f})"
                if source_info not in sources:
                    sources.append(source_info)
            
            context = "\n\n".join(context_chunks)
            
            # Build conversation history for context
            recent_history = []
            if len(self.conversation_history) > 0:
                recent_history = self.conversation_history[-3:]  # Last 3 exchanges
            
            # Create system prompt for RAG
            system_prompt = f"""You are a knowledgeable assistant that answers questions based on provided context. 

INSTRUCTIONS:
- Answer the question using ONLY the information provided in the context
- Be precise and informative
- If the context doesn't contain enough information, say so honestly
- Keep responses concise but complete (under 150 words)
- Mention relevant source documents when appropriate
- Maintain conversational tone for voice delivery

CONTEXT FROM KNOWLEDGE BASE:
{context}

RECENT CONVERSATION:
{chr(10).join([f"Q: {h['query']}" + chr(10) + f"A: {h['response']}" for h in recent_history])}

Please answer based on the provided context."""

            # Generate response using GPT
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=200,
                temperature=0.3  # Lower temperature for factual responses
            )
            
            generated_response = response.choices[0].message.content.strip()
            
            # Add source citation if multiple sources
            if len(sources) > 1:
                source_text = f"\n\nThis information comes from {len(sources)} sources in your knowledge base."
            else:
                source_text = f"\n\nSource: {sources[0].split(' (')[0]}" if sources else ""
            
            final_response = generated_response + source_text
            
            # Store in conversation history
            self.conversation_history.append({
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': generated_response,
                'sources': sources,
                'confidence': sum(score for _, score in search_results) / len(search_results)
            })
            
            # Track confidence scores
            avg_confidence = sum(score for _, score in search_results) / len(search_results)
            self.current_session['confidence_scores'].append(avg_confidence)
            
            logger.info("RAG response generated successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"RAG response generation failed: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def speak_response(self, text: str) -> bool:
        """
        Deliver response with clear pronunciation for information delivery.
        
        Args:
            text (str): Response text to speak
            
        Returns:
            bool: Success status
        """
        try:
            # Optimize text for clear information delivery
            optimized_text = self._optimize_text_for_information_speech(text)
            
            logger.info(f"Speaking response: {optimized_text[:50]}...")
            self.tts_engine.say(optimized_text)
            self.tts_engine.runAndWait()
            
            return True
            
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")
            return False
    
    def _optimize_text_for_information_speech(self, text: str) -> str:
        """
        Optimize text for clear delivery of information.
        
        Args:
            text (str): Raw response text
            
        Returns:
            str: Optimized text for speech
        """
        # Add pauses for better comprehension
        text = text.replace('. ', '. ')
        text = text.replace('? ', '? ')
        text = text.replace(': ', ': ')
        text = text.replace(', ', ', ')
        
        # Expand technical abbreviations
        replacements = {
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning',
            'API': 'A.P.I.',
            'URL': 'U.R.L.',
            'PDF': 'P.D.F.',
            'CSV': 'C.S.V.',
            'JSON': 'J.S.O.N.',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'and so on'
        }
        
        for abbrev, expansion in replacements.items():
            text = text.replace(abbrev, expansion)
        
        return text.strip()
    
    def run_rag_session(self, max_queries: int = 20) -> None:
        """
        Main RAG session loop for continuous voice Q&A.
        
        Args:
            max_queries (int): Maximum number of queries in session
        """
        logger.info("Starting Voice RAG session...")
        
        # Welcome message
        welcome = f"""Hello! I'm your Voice RAG assistant. I have access to {len(self.documents)} documents 
        in my knowledge base. You can ask me questions about the content, and I'll search for relevant 
        information to provide accurate answers. What would you like to know?"""
        
        self.speak_response(welcome)
        
        query_count = 0
        
        try:
            while query_count < max_queries:
                # Listen for user query
                user_query = self.listen_for_query()
                
                if user_query is None:
                    prompt = "I'm ready for your question. Please ask me anything about the documents."
                    self.speak_response(prompt)
                    continue
                
                # Check for session end
                if any(end_word in user_query.lower() for end_word in 
                       ['goodbye', 'exit', 'quit', 'thank you', 'that\'s all']):
                    closing = "Thank you for using the Voice RAG system. Have a great day!"
                    self.speak_response(closing)
                    break
                
                # Search knowledge base
                search_results = self.search_knowledge_base(user_query, top_k=5)
                
                # Generate and deliver response
                response = self.generate_rag_response(user_query, search_results)
                self.speak_response(response)
                
                query_count += 1
                
        except KeyboardInterrupt:
            logger.info("RAG session interrupted by user")
            self.speak_response("Session ended. Goodbye!")
        except Exception as e:
            logger.error(f"RAG session error: {str(e)}")
        finally:
            self._save_session_data()
            logger.info(f"RAG session completed after {query_count} queries")
    
    def _save_knowledge_base(self) -> None:
        """Save knowledge base to persistent storage."""
        try:
            kb_file = self.knowledge_base_path / "knowledge_base.pkl"
            
            knowledge_base = {
                'documents': self.documents,
                'chunks': self.chunks,
                'embeddings': self.embeddings,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(kb_file, 'wb') as f:
                pickle.dump(knowledge_base, f)
            
            logger.info("Knowledge base saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {str(e)}")
    
    def _load_knowledge_base(self) -> None:
        """Load existing knowledge base from storage."""
        try:
            kb_file = self.knowledge_base_path / "knowledge_base.pkl"
            
            if kb_file.exists():
                with open(kb_file, 'rb') as f:
                    knowledge_base = pickle.load(f)
                
                self.documents = knowledge_base.get('documents', [])
                self.chunks = knowledge_base.get('chunks', [])
                self.embeddings = knowledge_base.get('embeddings', [])
                
                # Rebuild vector index
                if self.embeddings:
                    self._update_vector_index()
                
                logger.info(f"Loaded knowledge base with {len(self.documents)} documents")
            else:
                logger.info("No existing knowledge base found")
                
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
    
    def _save_session_data(self) -> None:
        """Save session data for analysis."""
        try:
            session_file = self.knowledge_base_path / f"session_{self.current_session['session_id']}.json"
            
            session_data = {
                **self.current_session,
                'conversation_history': self.conversation_history,
                'total_queries': len(self.current_session['queries']),
                'documents_accessed': list(self.current_session['documents_used']),
                'average_confidence': np.mean(self.current_session['confidence_scores']) if self.current_session['confidence_scores'] else 0
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session data saved: {session_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {str(e)}")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current knowledge base.
        
        Returns:
            Dict: Knowledge base statistics
        """
        total_chunks = len(self.chunks)
        total_words = sum(chunk['word_count'] for chunk in self.chunks)
        total_chars = sum(chunk['char_count'] for chunk in self.chunks)
        
        stats = {
            'total_documents': len(self.documents),
            'total_chunks': total_chunks,
            'total_words': total_words,
            'total_characters': total_chars,
            'average_chunk_size': total_words / total_chunks if total_chunks > 0 else 0,
            'document_types': list(set(doc['type'] for doc in self.documents)),
            'knowledge_base_size_mb': sum(doc['size'] for doc in self.documents) / (1024 * 1024)
        }
        
        return stats

# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the Voice RAG Agent.
    """
    
    # Configuration
    API_KEY = "your-openai-api-key"  # Replace with your OpenAI API key
    KNOWLEDGE_BASE_PATH = "my_knowledge_base"
    
    try:
        # Initialize Voice RAG Agent
        rag_agent = VoiceRAGAgent(
            api_key=API_KEY,
            knowledge_base_path=KNOWLEDGE_BASE_PATH
        )
        
        # Add sample documents (replace with your document paths)
        sample_docs = [
            "sample_document.pdf",
            "manual.txt",
            "faq.docx"
        ]
        
        for doc_path in sample_docs:
            if os.path.exists(doc_path):
                success = rag_agent.add_document(doc_path)
                if success:
                    print(f"Added document: {doc_path}")
        
        # Display knowledge base statistics
        stats = rag_agent.get_knowledge_base_stats()
        print(f"Knowledge Base Stats: {json.dumps(stats, indent=2)}")
        
        # Start Voice RAG session
        rag_agent.run_rag_session(max_queries=15)
        
    except Exception as e:
        logger.error(f"Failed to start Voice RAG Agent: {str(e)}")
        print("Please ensure you have set up your OpenAI API key and added documents to the knowledge base.")