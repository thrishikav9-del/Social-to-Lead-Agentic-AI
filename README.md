# Social-to-Lead Agentic Workflow for AutoStream

This repository contains a production-ready conversational AI agent built for AutoStream, a fictional SaaS product for video creators. The agent uses RAG to answer product questions and LangGraph state management to classify intents and conditionally execute a mock backend tool only after collecting high-intent user details (Name, Email, Creator Platform).

## 🚀 Setup & Run Instructions

**1. Clone the repository and navigate to the directory**
Navigate to this folder in your terminal.

**2. Create a virtual environment**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set your API Key**
```bash
# Set your Gemini API key (since we use gemini-1.5-flash)
# On Windows
set GOOGLE_API_KEY=your_key_here
# On Mac/Linux
export GOOGLE_API_KEY=your_key_here
```

**5. Run the application**
```bash
python main.py
```

## 🧠 Architecture Explanation

This agent is built using **LangGraph** because it provides robust, cyclic state management essential for complex multi-turn conversations, unlike linear chains or standard autonomous agents which can get stuck in loops or lose critical memory context. 

The architecture centers around a shared `AgentState` containing the conversation history, detected intent, and extracted lead data. 
- The **Entry Node** classification engine uses structured LLM output to accurately parse the user's latest message against the full history, outputting `greeting`, `product_inquiry`, or `high_intent`. 
- **Routing** dynamically shifts the user: if product inquiry, the flow moves to the RAG Node (FAISS + HuggingFace Embeddings reading the local JSON). 
- If high intent is detected, it hits the **Lead Collection Node**. Here, structured LLM extraction pulls existing entities (Name, Email, Platform) from memory. If missing, it instructs the LLM to conversationally prompt the user for the missing fields, terminating the turn. 
- Only once all three variables are populated does the graph route to the **Tool Execution Node**, validating and executing the mock backend callback, guaranteeing state safety.

## 📡 WhatsApp Webhooks Integration

To integrate this agent with WhatsApp, we would deploy this LangGraph application via FastAPI. 

1. **Webhook Endpoint:** Expose a `POST /webhook` endpoint that Meta's WhatsApp API can call when an inbound user message is received.
2. **Session Persistence:** Store the LangGraph `AgentState` in a fast key-value store (like Redis or MongoDB) keyed by the user's WhatsApp phone number.
3. **Graph Execution:** Upon receiving a webhook, load the state from the external store, append the new `HumanMessage`, and execute `app.invoke(state)`. 
4. **Sending Replies:** Extract the resulting `AIMessage` from the updated state, save the state back, and trigger Meta's Outbound Messaging API to send the text sequentially back to the WhatsApp user.
