import streamlit as st
import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro HR Help Desk", page_icon="🏢")
st.title("🏢 Zyro Dynamics HR Help Desk")

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

@st.cache_resource(show_spinner="Indexing Policy Documents...")
def init_rag():
    loader = PyPDFDirectoryLoader("./hr_policies")
    docs = loader.load()
    if not docs: return None, None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # 25-chunk memory grabber to ensure it catches abbreviations like "SL"
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 25})
    
    # Increased token limit to prevent mid-sentence cut-offs
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.1, max_output_tokens=1024)
    
    # The strict, no-fluff guardrail prompt
    prompt = ChatPromptTemplate.from_template('''You are the official Zyro Dynamics HR Help Desk AI.
    
    Context:
    {context}
    
    Question:
    {question}
    
    INSTRUCTIONS:
    1. Find the exact answer and state it directly in ONE short sentence. 
    2. DO NOT use fluffy intros like "According to the policy..." or "Based on the documents...". Just state the fact.
    3. If the answer is NOT in the Context, or if the question is NOT about HR policies, you MUST output exactly:
    "I can only answer HR-related questions from Zyro Dynamics policy documents."
    
    Answer:''')
    
    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
        
    chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain, retriever

rag_chain, retriever = init_rag()

if rag_chain is None:
    st.error("Missing 'hr_policies' folder. Please upload the PDFs to your GitHub repository.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Ask a question regarding HR policies..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        response = rag_chain.invoke(user_query)
        st.write(response)
        with st.expander("📚 Source Documents"):
            docs = retriever.invoke(user_query)
            for d in docs:
                source_name = d.metadata.get("source", "Unknown").split("/")[-1]
                st.markdown(f"**{source_name}**")
                st.caption(d.page_content)
                
    st.session_state.messages.append({"role": "assistant", "content": response})
