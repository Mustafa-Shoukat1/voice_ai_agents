# Voice RAG Agent - Intelligent Document Q&A System

## Overview

The Voice RAG (Retrieval-Augmented Generation) Agent is an advanced AI system that combines document processing, semantic search, and voice interaction to provide intelligent question-answering capabilities. It allows users to ask questions about their documents using natural speech and receive accurate, contextual answers with source citations.

## Key Features

### 🎙️ **Voice-First Interface**
- Natural speech recognition for questions
- Clear text-to-speech responses
- Optimized for information delivery
- Support for complex, multi-part queries

### 📚 **Advanced Document Processing**
- Support for PDF, DOCX, and TXT files
- Intelligent text chunking for optimal retrieval
- Automatic metadata extraction
- Document versioning and tracking

### 🔍 **Semantic Search & Retrieval**
- Vector embeddings using OpenAI's text-embedding-ada-002
- FAISS integration for efficient similarity search
- Context-aware chunk retrieval
- Relevance scoring and ranking

### 🧠 **Intelligent Response Generation**
- RAG-based answer generation using GPT-3.5-turbo
- Source citation and verification
- Conversation history tracking
- Confidence scoring for responses

### 💾 **Knowledge Base Management**
- Persistent storage of processed documents
- Incremental document addition
- Knowledge base statistics and analytics
- Session tracking and analysis

## Architecture

```
Voice Input → Speech Recognition → Query Processing → 
Semantic Search → Document Retrieval → Response Generation → 
Text-to-Speech → Voice Output
```

### Core Components

1. **Document Processor**: Extracts and chunks text from various formats
2. **Embedding Engine**: Generates vector representations using OpenAI
3. **Vector Database**: Stores and indexes embeddings (FAISS or numpy)
4. **Retrieval System**: Finds relevant document chunks
5. **Generation Engine**: Creates contextual responses
6. **Voice Interface**: Handles speech input/output

## Installation & Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Microphone and speakers
- Documents to add to knowledge base

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
```
openai>=1.0.0
speech-recognition==3.10.0
pyttsx3==2.90
numpy>=1.21.0
faiss-cpu>=1.7.0  # or faiss-gpu for GPU acceleration
PyPDF2>=3.0.0
python-docx>=0.8.11
```

### Optional Dependencies
```bash
# For GPU acceleration (if supported)
pip install faiss-gpu

# For advanced document processing
pip install pymupdf  # Better PDF processing
pip install python-magic  # File type detection
```

## Quick Start

### 1. Initialize the Agent
```python
from voice_rag_agent import VoiceRAGAgent

# Initialize with your OpenAI API key
agent = VoiceRAGAgent(
    api_key="your-openai-api-key",
    knowledge_base_path="my_documents"
)
```

### 2. Add Documents
```python
# Add individual documents
agent.add_document("manual.pdf")
agent.add_document("faq.docx")
agent.add_document("guidelines.txt")

# Check knowledge base stats
stats = agent.get_knowledge_base_stats()
print(f"Added {stats['total_documents']} documents")
```

### 3. Start Voice Q&A Session
```python
# Begin interactive voice session
agent.run_rag_session(max_queries=20)
```

## Advanced Usage

### Batch Document Processing
```python
import os

# Add all documents from a directory
doc_directory = "documents/"
for filename in os.listdir(doc_directory):
    if filename.endswith(('.pdf', '.txt', '.docx')):
        file_path = os.path.join(doc_directory, filename)
        success = agent.add_document(file_path)
        print(f"{'✓' if success else '✗'} {filename}")
```

### Custom Search and Response
```python
# Manual query processing
query = "What are the system requirements?"
search_results = agent.search_knowledge_base(query, top_k=3)
response = agent.generate_rag_response(query, search_results)
print(response)
```

### Knowledge Base Analysis
```python
# Get detailed statistics
stats = agent.get_knowledge_base_stats()
print(f"""
Knowledge Base Summary:
- Documents: {stats['total_documents']}
- Text Chunks: {stats['total_chunks']}
- Total Words: {stats['total_words']:,}
- Average Chunk Size: {stats['average_chunk_size']:.0f} words
- Total Size: {stats['knowledge_base_size_mb']:.1f} MB
- Document Types: {', '.join(stats['document_types'])}
""")
```

## Configuration Options

### VoiceRAGAgent Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | Required | OpenAI API key |
| `knowledge_base_path` | str | "knowledge_base" | Storage path for KB |

### Document Processing Settings
| Setting | Default | Description |
|---------|---------|-------------|
| Chunk Size | 1000 chars | Text chunk size for embedding |
| Chunk Overlap | 200 chars | Overlap between chunks |
| Embedding Model | text-embedding-ada-002 | OpenAI embedding model |

### Search Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 5 | Number of chunks to retrieve |
| Similarity Threshold | None | Minimum similarity score |
| Context Window | 3 | Recent conversation history |

## Voice Commands

### Starting Session
- Agent will greet and explain capabilities
- Simply ask your question after the greeting

### During Conversation
- Ask follow-up questions for clarification
- Reference previous responses in new queries
- Request specific document sources

### Ending Session
- Say "goodbye", "exit", "quit", or "thank you"
- Session data will be automatically saved

## Supported Document Formats

### PDF Files (.pdf)
- Text extraction using PyPDF2
- Preserves document structure
- Handles multi-page documents
- Extracts metadata automatically

### Word Documents (.docx)
- Full text extraction from paragraphs
- Maintains formatting context
- Handles tables and lists
- Preserves document metadata

### Text Files (.txt)
- Direct text processing
- UTF-8 encoding support
- Large file handling
- Custom encoding detection

### Future Format Support
- HTML/Web pages
- PowerPoint presentations (.pptx)
- Excel spreadsheets (.xlsx)
- Markdown files (.md)

## Performance Optimization

### For Large Document Collections
```python
# Use FAISS for faster search
# Ensure FAISS is installed: pip install faiss-cpu

# For GPU acceleration (if available)
# pip install faiss-gpu
```

### Memory Management
- Process documents in batches
- Use appropriate chunk sizes
- Clear old embeddings when updating
- Monitor system memory usage

### Response Speed
- Cache frequently accessed documents
- Optimize chunk size for your use case
- Use faster embedding models if available
- Implement parallel processing for multiple queries

## Error Handling & Troubleshooting

### Common Issues

**Document Processing Failures**
```python
# Check file format support
supported_formats = ['.pdf', '.txt', '.docx']
if file_extension not in supported_formats:
    print(f"Unsupported format: {file_extension}")
```

**Embedding Generation Errors**
```python
# Monitor API rate limits
# Implement retry logic with exponential backoff
# Check OpenAI API key validity
```

**Voice Recognition Issues**
```python
# Adjust microphone sensitivity
# Reduce background noise
# Check audio device permissions
```

### Debugging Tools

**Logging Configuration**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Session Analysis**
```python
# Review saved session data
with open('session_data.json', 'r') as f:
    session = json.load(f)
    print(f"Queries: {len(session['queries'])}")
    print(f"Average confidence: {session['average_confidence']}")
```

## Security & Privacy

### Data Protection
- Documents processed locally before embedding
- Embeddings stored securely in knowledge base
- Session data encrypted at rest
- API communications over HTTPS

### Privacy Considerations
- OpenAI processes text for embeddings and generation
- Consider local alternatives for sensitive documents
- Implement data retention policies
- Regular security audits recommended

### Access Control
```python
# Implement user authentication
def authenticate_user(username, password):
    # Your authentication logic
    pass

# Restrict document access
def check_document_permissions(user, document):
    # Your authorization logic
    pass
```

## Integration Examples

### Web API Integration
```python
from fastapi import FastAPI
from voice_rag_agent import VoiceRAGAgent

app = FastAPI()
rag_agent = VoiceRAGAgent(api_key="your-key")

@app.post("/query")
async def process_query(query: str):
    results = rag_agent.search_knowledge_base(query)
    response = rag_agent.generate_rag_response(query, results)
    return {"response": response}
```

### Slack Bot Integration
```python
from slack_bolt import App
from voice_rag_agent import VoiceRAGAgent

app = App(token="your-slack-token")
rag_agent = VoiceRAGAgent(api_key="your-openai-key")

@app.message("ask")
def handle_question(message, say):
    query = message['text'].replace("ask", "").strip()
    # Process with RAG agent
    response = rag_agent.process_text_query(query)
    say(response)
```

### Discord Bot Integration
```python
import discord
from voice_rag_agent import VoiceRAGAgent

client = discord.Client()
rag_agent = VoiceRAGAgent(api_key="your-openai-key")

@client.event
async def on_message(message):
    if message.content.startswith('!ask'):
        query = message.content[4:].strip()
        response = rag_agent.process_text_query(query)
        await message.channel.send(response)
```

## Performance Metrics

### Knowledge Base Metrics
- Document processing time
- Embedding generation speed
- Index build/update time
- Storage efficiency

### Search Performance
- Query processing time
- Retrieval accuracy
- Response relevance scores
- User satisfaction ratings

### System Metrics
- Memory usage optimization
- CPU utilization
- API call efficiency
- Error rates and handling

## Best Practices

### Document Preparation
1. **Clean Text**: Remove unnecessary formatting
2. **Consistent Structure**: Use standard document layouts
3. **Metadata**: Include relevant document information
4. **Size Optimization**: Balance detail with processing efficiency

### Query Optimization
1. **Clear Questions**: Use specific, well-formed questions
2. **Context**: Provide background information when needed
3. **Follow-up**: Build on previous responses
4. **Specificity**: Ask for particular types of information

### Knowledge Base Management
1. **Regular Updates**: Keep documents current
2. **Quality Control**: Review and validate content
3. **Organization**: Maintain logical document structure
4. **Backup**: Regular knowledge base backups

## Contributing

### Development Setup
```bash
git clone https://github.com/Mustafa-Shoukat1/voice_ai_agents.git
cd voice_ai_agents/voice_rag_agent
pip install -r requirements-dev.txt
```

### Testing
```bash
pytest tests/
pytest --cov=voice_rag_agent tests/
```

### Code Quality
```bash
black voice_rag_agent.py
flake8 voice_rag_agent.py
mypy voice_rag_agent.py
```

## Roadmap

### Upcoming Features
- [ ] Multi-modal document support (images, tables)
- [ ] Real-time document updates
- [ ] Advanced conversation memory
- [ ] Custom embedding models
- [ ] Distributed knowledge base
- [ ] Multi-language support

### Performance Improvements
- [ ] Async document processing
- [ ] Streaming responses
- [ ] Caching optimizations
- [ ] Parallel search execution

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

### Documentation
- API Reference: `/docs/api.md`
- Examples: `/examples/`
- Tutorials: `/tutorials/`

### Community
- GitHub Issues: Bug reports and feature requests
- Discussions: Community support and questions
- Wiki: Additional documentation and guides

---

**Author:** Mustafa Shoukat  
**Version:** 1.0.0  
**Last Updated:** 2024  
**License:** MIT