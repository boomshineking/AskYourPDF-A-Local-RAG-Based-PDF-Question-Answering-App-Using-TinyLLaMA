import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers

# Load PDF and process
def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Check if the PDF is loaded correctly
    if not pages:
        raise ValueError("The PDF is empty or couldn't be loaded properly.")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pages)
    
    # Check if documents were created after splitting
    if not documents:
        raise ValueError("No text chunks found after splitting the document.")
    
    return documents

# Embed and store for retrieval
def setup_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="models/all-MiniLM-L6-v2", model_kwargs={"local_files_only": True})
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Check if the FAISS vector store is empty or not created properly
    if not vectorstore.index:
        raise ValueError("The FAISS vector store is empty or wasn't created properly.")
    
    retriever = vectorstore.as_retriever()
    return retriever

# QA with local TinyLLaMA model (or any GGML llama-compatible model)
def get_pdf_answer(query, retriever):
    if not query.strip():
        raise ValueError("Please enter a valid question.")
    
    print(f"Query: {query}")
    print(f"Retriever: {retriever}")
    
    llm = CTransformers(
        model='models/tinyllama-1.1b-chat-v1.0.Q6_K.gguf',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.1}
    )
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)

# Streamlit UI
st.set_page_config(page_title="Ask PDF 🤖", page_icon="📄", layout="centered")

st.title("📄 Ask Your PDF using TinyLLaMA")

uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
user_question = st.text_input("Ask a question based on the PDF")

if uploaded_file and user_question:
    try:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        docs = load_pdf("temp.pdf")
        retriever = setup_retriever(docs)
        response = get_pdf_answer(user_question, retriever)

        st.markdown("### 📌 Answer")
        st.write(response)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
