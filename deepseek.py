import streamlit as st
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.title("🎓 PDF Summary Generator")
st.markdown("Know your Doc better than ever before! 🚀")
st.markdown("---")
output_space = st.empty()
uploader = st.file_uploader("📤 Upload Your PDF Below", type=['pdf'])

prompt = st.text_input("Enter your prompt", max_chars=200, help="Enter the prompt here", autocomplete="on")
btn = st.button("Generate")

PDF_STORAGE_PATH = "G:\CSproject\RAG\document_store"
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

embedding_model = OllamaEmbeddings(model="deepseek-r1:1.5b")
document_embedding_db = InMemoryVectorStore(embedding_model)
language_model = OllamaLLM(model="deepseek-r1:1.5b")

Prompt_Template = """You are an expert research assistant.
Use the provided context to answer the query.
If unsure, state that you need more information. Be concise and to the point (Max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    return loader.load()

def chunk_documents(raw_documents):
    processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return processor.split_documents(raw_documents)

def document_embedding(documents):
    document_embedding_db.add_documents(documents)

def similarity_search(query):
    return document_embedding_db.similarity_search(query)

def generate_summary(user_query, context_documents):
    context_text = "\n\n".join(doc.page_content for doc in context_documents)
    prompt_template = ChatPromptTemplate.from_template(Prompt_Template)
    response_chain = prompt_template | language_model
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

if btn:
    output_space.empty()
    if uploader:
        file_path = save_uploaded_file(uploader)
        raw_documents = load_pdf(file_path)
        
        if raw_documents:
            context_documents = chunk_documents(raw_documents)
            document_embedding(context_documents)

            if prompt:
                summary = generate_summary(prompt, context_documents)
                output_space.write("### 📄 Summary Output")
                output_space.write(summary)
            else:
                output_space.write("Prompt daalo.")
                st.toast("🔔Prompt nahi daala.")
        else:
            output_space.error("❌ Error loading the PDF file. Please try again.")
    else:
        st.toast("Chote Acche se upload kr", icon="😑")
        st.error("Upload nahi hua.")