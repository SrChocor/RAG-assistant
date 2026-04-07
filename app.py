from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import langchain_core.tools
import streamlit as st
from dotenv import load_dotenv
import langchain_huggingface
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os
import tempfile


load_dotenv()

st.title("RAG Assistant - Personal project")

#initialize the embedding model
@st.cache_resource
def initialize_embeddings():
     with st.spinner("Loading embedding model, please wait..."):
        return langchain_huggingface.HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_lm():
    return ChatOpenAI(model="stepfun/step-3.5-flash:free", 
                      openai_api_key=os.getenv("OPENAI_API_KEY"),
                      openai_api_base="https://openrouter.ai/api/v1",
                      temperature=0.3,
                      streaming=False)

embeddings = initialize_embeddings()
llm = get_lm()

#Session state to store the conversation history and the vector store
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
#To upload a PDF file
uploaded_files = st.file_uploader("Upload your PDF file please", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        os.unlink(tmp_path) 
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        all_chunks.extend(chunks)

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(all_chunks, embeddings)
    else:
        st.session_state.vectorstore.add_documents(all_chunks)
        

    st.success(f"PDFs uploaded — {len(all_chunks)} total chunks indexed.")
    
#To ask questions to the assistant

if st.session_state.vectorstore:
    
    retriever_tool = langchain_core.tools.create_retriever_tool(
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
        name="pdf_retriever",
        description="Searches and returns information from the uploaded PDF documents. Use this first for any question. If the answer is not in the PDF, inform that you don't know and you can use the DuckDuckGoSearchRun tool to search the web for the answer."
    )
    
    web_search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description="Searches the web for information. Use this tool if the answer is not in the PDF documents. Always try to use the pdf_retriever tool first before using this one."
    )
    
    tools = [retriever_tool, web_search_tool]
    
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. You have access to two tools:
                1. pdf_retriever: searches the user's uploaded PDF documents
                2. web_search: searches the web for additional information
                If the answer is not in the PDF documents, say the answer is not in the PDF and that you need to use the web_search tool.
                Always try pdf_retriever first. Only use web_search if the answer is not in the PDFs.
                If you don't know the answer after both tools, say you don't know."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)
    
    
    #Display the conversation history
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    question = st.chat_input("Ask a question")
    
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
            
        window = st.session_state.chat_history[-7:-1]
        
        langchain_history = []
        
        for msg in window:
            if msg["role"] == "user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            else:
                langchain_history.append(AIMessage(content=msg["content"]))
        
        with st.spinner("Thinking..."):
            
            try:
                response = agent_executor.invoke({"input": question, "chat_history": langchain_history})
                answer = response["output"]
            except Exception as e:
                st.error(f"An error occurred: {e}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)        

else: 
    st.info("Please upload a PDF file to start the conversation.")
    
