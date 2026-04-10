from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

class IntentClassification(BaseModel):
    intent: str = Field(description="The classified intent: 'greeting', 'product_inquiry', or 'high_intent'")

class IntentClassifier:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(IntentClassification)
        self.prompt = PromptTemplate.from_template(
            "You are an intent classification system for a SaaS product named AutoStream.\n"
            "Analyze the latest user message and conversation history to determine the intent.\n\n"
            "Categories:\n"
            "- greeting: Casual hellos, greetings, etc.\n"
            "- product_inquiry: Questions about pricing, features, 'tell me about your product', refund policies, support policies.\n"
            "- high_intent: The user clearly shows interest in signing up, buying, trying a plan, or states they want it for their specific platform (e.g. YouTube, Instagram).\n\n"
            "Conversation History:\n{history}\n\n"
            "Latest User Message: {query}\n\n"
            "Classify the intent strictly into one of the three categories."
        )
        self.chain = self.prompt | self.structured_llm

    def classify(self, query: str, history: str) -> str:
        res = self.chain.invoke({"query": query, "history": history})
        return res.intent
