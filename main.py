import os
from typing import Annotated, Optional, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load project modules
from intent_classifier import IntentClassifier
from rag_pipeline import RAGPipeline
from lead_handler import LeadHandler
from tools import mock_lead_capture

# Load environment variables (e.g., GOOGLE_API_KEY)
load_dotenv()

# We still instantiate the LLM even if the key is missing so we don't crash instantly on import, 
# but it will fail on invoke if GOOGLE_API_KEY is not in env or python environment.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

intent_classifier = IntentClassifier(llm)
rag_pipeline = RAGPipeline("knowledge_base.json")
lead_handler = LeadHandler(llm)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    intent: str
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    tool_executed: bool

def format_history(messages: list[AnyMessage]) -> str:
    history = ""
    for msg in messages:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        history += f"{role}: {msg.content}\n"
    return history

def classify_intent_node(state: AgentState):
    history = format_history(state["messages"])
    latest_msg = state["messages"][-1].content
    intent = intent_classifier.classify(latest_msg, history)
    return {"intent": intent}

def greeting_node(state: AgentState):
    history = format_history(state["messages"])
    prompt = PromptTemplate.from_template(
        "You are a friendly agent for AutoStream. Provide a brief, conversational greeting.\n"
        "History: {history}"
    )
    chain = prompt | llm
    response = chain.invoke({"history": history})
    return {"messages": [AIMessage(content=response.content)]}

def rag_qa_node(state: AgentState):
    latest_msg = state["messages"][-1].content
    context = rag_pipeline.retrieve(latest_msg)
    
    prompt = PromptTemplate.from_template(
        "You are a helpful sales assistant for AutoStream.\n"
        "Answer the user's question using ONLY the provided context. If the answer is not in the context, politely say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "User Question: {query}\n"
    )
    chain = prompt | llm
    response = chain.invoke({"context": context, "query": latest_msg})
    return {"messages": [AIMessage(content=response.content)]}

def collect_lead_info_node(state: AgentState):
    history = format_history(state["messages"])
    lead_info = lead_handler.extract_info(history)
    
    missing_fields = []
    if not lead_info.name: missing_fields.append("Name")
    if not lead_info.email: missing_fields.append("Email")
    if not lead_info.platform: missing_fields.append("Creator Platform")
    
    updates = {
        "lead_name": lead_info.name,
        "lead_email": lead_info.email,
        "lead_platform": lead_info.platform
    }
    
    if len(missing_fields) > 0:
        response = lead_handler.ask_for_missing_info(missing_fields, history)
        updates["messages"] = [AIMessage(content=response)]
    return updates

def execute_tool_node(state: AgentState):
    mock_lead_capture(state["lead_name"], state["lead_email"], state["lead_platform"])
    response = "Thank you! We've captured your details and successfully initiated the setup for your chosen plan."
    return {"tool_executed": True, "messages": [AIMessage(content=response)]}

def route_intent(state: AgentState):
    intent = state.get("intent", "greeting")
    if intent == "greeting":
        return "greeting"
    elif intent == "product_inquiry":
        return "rag_qa"
    elif intent == "high_intent":
        return "collect_lead_info"
    return "greeting"

def route_lead_collection(state: AgentState):
    # Check if we have all necessary fields
    if state.get("lead_name") and state.get("lead_email") and state.get("lead_platform"):
        if not state.get("tool_executed"):
            return "execute_tool"
    return END

# Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("rag_qa", rag_qa_node)
workflow.add_node("collect_lead_info", collect_lead_info_node)
workflow.add_node("execute_tool", execute_tool_node)

workflow.set_entry_point("classify_intent")

workflow.add_conditional_edges("classify_intent", route_intent)
workflow.add_edge("greeting", END)
workflow.add_edge("rag_qa", END)

workflow.add_conditional_edges("collect_lead_info", route_lead_collection)
workflow.add_edge("execute_tool", END)

app = workflow.compile()

def start_chat():
    num_stars = 50
    print("\n" + "*" * num_stars)
    print("Welcome to AutoStream Sales Agent!")
    print("Type 'quit' or 'exit' to stop.")
    print("*" * num_stars + "\n")
    
    state = {
        "messages": [],
        "intent": "",
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "tool_executed": False
    }
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            state["messages"].append(HumanMessage(content=user_input))
            
            result = app.invoke(state)
            
            ai_response = result["messages"][-1].content
            print(f"Agent: {ai_response}")
            print("-" * num_stars)
            
            # Update local dict state for next iteration
            state = result
            
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    start_chat()
