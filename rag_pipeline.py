import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self, kb_path: str = "knowledge_base.json"):
        self.kb_path = kb_path
        # Using a small, fast local embedding model suitable for CPU
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.retriever = None
        self._initialize_kb()

    def _initialize_kb(self):
        # Load JSON knowledge base
        if not os.path.exists(self.kb_path):
            raise FileNotFoundError(f"Knowledge base file {self.kb_path} not found.")
        
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        documents = []
        
        # Parse Pricing
        for plan, details in data.get("pricing", {}).items():
            content = f"Plan: {plan}\nPrice: {details['price']}\nFeatures: {', '.join(details['features'])}"
            documents.append(Document(page_content=content, metadata={"source": "pricing", "plan": plan}))
            
        # Parse Policies
        for policy_name, policy_desc in data.get("policies", {}).items():
            content = f"Policy ({policy_name.replace('_', ' ').title()}): {policy_desc}"
            documents.append(Document(page_content=content, metadata={"source": "policy"}))
            
        # Load into FAISS vector store
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

    def retrieve(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
