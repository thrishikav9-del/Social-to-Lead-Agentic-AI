# Social-to-Lead Agentic Workflow (AutoStream AI Agent)

## Overview

This project implements a GenAI-powered conversational AI agent for a fictional SaaS product AutoStream, designed to convert user conversations into qualified business leads.

Unlike traditional chatbots, this system integrates:

* Intent Detection
* RAG (Retrieval-Augmented Generation)
* Stateful Conversation Flow
* Tool Execution for Lead Capture

The agent simulates a real-world AI sales assistant that interacts with users, answers product queries, and captures leads when high intent is detected.

---

## Features

### 1. Intent Detection

Classifies user input into:

* Greeting
* Product Inquiry
* High Intent (Ready to purchase)

---

### 2. RAG-Based Knowledge Retrieval

The agent answers queries using a local knowledge base instead of hardcoded responses.

Includes:

* Pricing plans
* Features
* Company policies

---

### 3. Lead Qualification System

When a user shows high intent, the agent:

* Collects:

  * Name
  * Email
  * Creator Platform
* Maintains conversation memory across multiple turns

---

### 4. Tool Execution

After collecting all required details:

```python
def mock_lead_capture(name, email, platform):
    print(f"Lead captured successfully: {name}, {email}, {platform}")
```

Ensures:

* No premature execution
* Proper validation
* Single trigger

---

## Tech Stack

* Language: Python 3.9+
* Framework: LangChain / LangGraph
* LLM: Google Gemini / OpenAI (configurable)
* Embeddings: HuggingFace (all-MiniLM-L6-v2)
* Vector Store: FAISS / In-memory
* State Management: Conversation memory buffers

---

## Project Structure

```
AutoStream-Agentic-Workflow/
│── main.py                  # Entry point
│── rag_pipeline.py          # RAG logic
│── intent_classifier.py     # Intent detection
│── lead_handler.py          # Lead flow handling
│── tools.py                 # Tool execution
│── knowledge_base.json      # Local data source
│── requirements.txt         # Dependencies
│── README.md                # Documentation
│── demo_outputs/            # Screenshots of outputs
```

---

## How to Run Locally

### 1. Clone Repository

```
git clone https://github.com/yourusername/AutoStream-Agentic-Workflow.git
cd AutoStream-Agentic-Workflow
```

---

### 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Set API Key

For Gemini:

```
set GOOGLE_API_KEY=your_api_key
```

OR for OpenAI:

```
set OPENAI_API_KEY=your_api_key
```

---

### 5. Run the Agent

```
python main.py
```

---

## Example Conversation Flow

```
User: What is your pricing?
Agent: AutoStream offers Basic ($29) and Pro ($79) plans.

User: I want to try the Pro plan
Agent: Great! Can I have your name?

User: Thrishika
Agent: Please share your email.

User: thrishika@gmail.com
Agent: Which platform do you create content on?

User: YouTube

Agent: Lead captured successfully: Thrishika, thrishika@gmail.com, YouTube
```

---
