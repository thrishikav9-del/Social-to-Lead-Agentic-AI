# Social-to-Lead Agentic AI

**GenAI-Powered Conversational Agent for Intelligent Lead Qualification**

Social-to-Lead Agentic AI is a conversational AI system that combines Retrieval-Augmented Generation (RAG), intent classification, and LangGraph-based state management to provide intelligent product assistance while automatically identifying and qualifying potential customer leads.

The system maintains multi-turn conversations, retrieves relevant information from a knowledge base, and collects lead information only when high purchase intent is detected, enabling a structured and context-aware customer engagement workflow.

---

## Overview

Modern conversational AI systems must do more than answer user queries. They need to understand user intent, maintain conversational context, retrieve domain-specific knowledge, and identify potential customers for business workflows.

Social-to-Lead Agentic AI addresses these challenges by integrating Retrieval-Augmented Generation (RAG), structured intent detection, entity extraction, and workflow orchestration into a unified conversational agent. The system dynamically routes users through different conversation paths based on their intent while preserving conversation state across multiple interactions.

---

## Key Features

- Retrieval-Augmented Generation (RAG) for product-specific question answering
- Intent classification using structured LLM outputs
- LangGraph-based multi-turn conversation management
- Automated lead qualification workflow
- Context-aware entity extraction
- Conditional backend tool execution
- Local knowledge base retrieval using vector search
- Extensible architecture for business automation

---

## System Architecture

The conversational workflow follows a state-driven architecture powered by LangGraph.

```text
                    User Query
                         │
                         ▼
               Intent Classification
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
   Greeting        Product Inquiry    High Intent
                         │                │
                         ▼                ▼
                 RAG Retrieval      Lead Collection
                         │                │
                         └──────┬─────────┘
                                ▼
                      Tool Execution
                                │
                                ▼
                     Response Generation
```

The workflow ensures that backend actions are executed only after all required lead information has been collected and validated.

---

## Core Components

### Intent Classification

The intent classification module analyzes the user's message together with the conversation history and categorizes each interaction into one of the supported intents:

- Greeting
- Product Inquiry
- High Purchase Intent

This enables intelligent routing throughout the conversation.

---

### Retrieval-Augmented Generation (RAG)

The RAG pipeline retrieves relevant information from a local knowledge base using vector embeddings before generating responses.

Components include:

- HuggingFace Embeddings
- FAISS Vector Store
- Local JSON Knowledge Base

This approach improves factual consistency and reduces hallucinations.

---

### Lead Qualification

When purchase intent is detected, the system automatically begins collecting essential lead information, including:

- Name
- Email Address
- Creator Platform

Missing information is requested conversationally while maintaining context across multiple interactions.

---

### State Management

LangGraph manages the entire conversational workflow through a shared state object containing:

- Conversation history
- Detected intent
- Lead information
- Tool execution status

This enables reliable multi-turn conversations without losing context.

---

### Tool Execution

Backend tools are executed only after all required lead information has been successfully collected and validated.

This conditional execution ensures workflow consistency and prevents incomplete lead submissions.

---

## Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Large Language Model | Gemini 1.5 Flash |
| AI Framework | LangChain |
| Workflow Orchestration | LangGraph |
| Vector Database | FAISS |
| Embeddings | HuggingFace Embeddings |
| Knowledge Base | JSON |
| Backend (Deployment) | FastAPI (Planned) |

---

## Project Structure

```text
Social-to-Lead-Agentic-AI/
│
├── main.py
├── intent_classifier.py
├── rag_pipeline.py
├── lead_handler.py
├── tools.py
├── knowledge_base.json
├── requirements.txt
├── output_demo/
├── README.md
└── LICENSE
```

---

## Installation

### Clone the repository

```bash
git clone https://github.com/your-username/Social-to-Lead-Agentic-AI.git

cd Social-to-Lead-Agentic-AI
```

### Create a virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure API Key

Windows

```bash
set GOOGLE_API_KEY=your_api_key
```

Linux / macOS

```bash
export GOOGLE_API_KEY=your_api_key
```

### Run the application

```bash
python main.py
```

---

## Example Conversation

**User**

> Hi, what is AutoStream?

**Assistant**

> AutoStream is an AI-powered platform designed to help video creators streamline content production and publishing.

---

**User**

> I'm interested in using it.

**Assistant**

> Great! May I know your name?

---

**User**

> John

**Assistant**

> Thanks, John. Could you also share your email address?

---

**User**

> john@email.com

**Assistant**

> Perfect. Finally, which creator platform do you primarily use?

---

After all required information is collected, the workflow automatically executes the backend lead processing tool.

---

## Future Deployment

The system can be extended into a production-ready customer support platform by integrating:

- FastAPI REST backend
- WhatsApp Business API
- Redis or MongoDB for conversation persistence
- Cloud deployment on AWS, Azure, or Google Cloud
- CRM integration for automated lead management

---

## Future Enhancements

- Multi-agent collaboration
- Memory optimization
- Voice-based conversations
- CRM integration
- Analytics dashboard
- Multi-language support
- Human-in-the-loop escalation
- Real-time monitoring

---

## Applications

- AI Customer Support
- Intelligent Lead Qualification
- Sales Automation
- Product Assistance
- Business Process Automation
- Conversational AI Research

---

## License

This project is licensed under the MIT License.

---

## Disclaimer

This project was developed for educational and research purposes to demonstrate conversational AI, Retrieval-Augmented Generation (RAG), workflow orchestration using LangGraph, and intelligent lead qualification.

---

## Author

**Thrishika**

B.Tech Computer Science and Engineering (Artificial Intelligence)

Amrita Vishwa Vidyapeetham
