import os
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# Load your Hugging Face token
HF_TOKEN = os.environ.get("HF_TOKEN")  # Make sure this env variable is set
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Replace with your model

def load_llm(huggingface_repo_id):
    client = InferenceClient(token=HF_TOKEN)
    llm = HuggingFaceEndpoint(
        client=client,
        model=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512  # ✅ Passed directly, not in model_kwargs
    )
    return llm


# Custom prompt template
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Do not provide anything outside the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load FAISS vector database
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Get user input and run the chain
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

# Display result and sources
print("\nRESULT:\n", response["result"])
print("\nSOURCE DOCUMENTS:\n", response["source_documents"])
