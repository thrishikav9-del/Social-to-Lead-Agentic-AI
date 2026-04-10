from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class LeadInformation(BaseModel):
    name: Optional[str] = Field(description="The user's first and/or last name.", default=None)
    email: Optional[str] = Field(description="The user's email address.", default=None)
    platform: Optional[str] = Field(description="The content creator platform the user uses (e.g., YouTube, Instagram, TikTok).", default=None)

class LeadHandler:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        self.structured_llm = self.llm.with_structured_output(LeadInformation)
        self.extraction_prompt = PromptTemplate.from_template(
            "Extract lead details from the entire conversation history.\n\n"
            "Conversation History:\n{history}\n\n"
            "If any field (name, email, platform) is mentioned by the user currently or previously, extract it. Otherwise, leave it null."
        )
        self.extraction_chain = self.extraction_prompt | self.structured_llm
        
        self.response_prompt = PromptTemplate.from_template(
            "You are a helpful sales assistant for AutoStream. The user has shown high intent to use our product.\n"
            "We need to collect their Name, Email, and Creator Platform.\n"
            "We currently still need the following information from the user: {missing_fields}.\n\n"
            "Conversation History:\n{history}\n\n"
            "Write a polite, conversational and very brief message asking the user for the missing fields."
        )
        self.response_chain = self.response_prompt | self.llm
        
    def extract_info(self, history: str) -> LeadInformation:
        return self.extraction_chain.invoke({"history": history})
        
    def ask_for_missing_info(self, missing_fields: list[str], history: str) -> str:
        missing_str = ", ".join(missing_fields)
        res = self.response_chain.invoke({"missing_fields": missing_str, "history": history})
        return res.content
