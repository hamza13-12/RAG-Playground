import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import time

# Import custom utilities
from utils import (
    get_text_splitter,
    get_available_splitters,
    get_retriever,
    get_available_retriever_types,
    get_available_search_types
)

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG Playground")
st.write("Explore different chunking and retrieval strategies using LangChain")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'results_comparison' not in st.session_state:
    st.session_state.results_comparison = []

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Document Upload
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True, type=["pdf", "txt"])
    
    # Chunking Strategy
    st.subheader("2. Chunking Strategy")
    chunking_strategy = st.selectbox(
        "Select chunking strategy",
        get_available_splitters()
    )
    
    chunk_size = st.slider("Chunk Size", 100, 2000, 500, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 50, 10)
    
    # Vector Store
    st.subheader("3. Vector Store")
    vector_store_type = st.selectbox("Select vector store", ["FAISS", "Chroma"])
    
    # Retrieval Strategy
    st.subheader("4. Retrieval Strategy")
    retriever_type = st.selectbox(
        "Select retriever type",
        get_available_retriever_types()
    )
    
    search_type = st.selectbox(
        "Select search type",
        get_available_search_types()
    )
    
    k_value = st.slider("Number of documents to retrieve (k)", 1, 10, 4)
    
    # Process Button
    process_button = st.button("Process Documents")

# Main functionality
if uploaded_files and process_button:
    with st.spinner("Processing documents..."):
        try:
            # Process documents
            documents = []
            for file in uploaded_files:
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(file.read())
                temp_file.close()
                
                if file.name.endswith(".pdf"):
                    loader = PyPDFLoader(temp_file.name)
                    documents.extend(loader.load())
                elif file.name.endswith(".txt"):
                    loader = TextLoader(temp_file.name)
                    documents.extend(loader.load())
                
                os.unlink(temp_file.name)
            
            # Apply selected chunking strategy
            text_splitter = get_text_splitter(
                chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
                
            chunks = text_splitter.split_documents(documents)
            
            # Display chunk statistics
            st.write(f"Created {len(chunks)} chunks from {len(documents)} document(s)")
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Create vector store
            if vector_store_type == "FAISS":
                db = FAISS.from_documents(chunks, embeddings)
            elif vector_store_type == "Chroma":
                db = Chroma.from_documents(chunks, embeddings)
            
            # Configure retriever
            retriever = get_retriever(
                db,
                search_type=search_type,
                search_kwargs={"k": k_value},
                retriever_type=retriever_type
            )
            
            # Set up LLM and QA chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            # Create the RAG prompt
            template = """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            
            # Create the retrieval chain
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            retrieval_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            # Store in session state for later use
            st.session_state.retriever = retriever
            st.session_state.retrieval_chain = retrieval_chain
            st.session_state.chunks = chunks
            st.session_state.config = {
                "chunking_strategy": chunking_strategy,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "vector_store": vector_store_type,
                "retriever_type": retriever_type,
                "search_type": search_type,
                "k_value": k_value
            }
            
            st.success("Documents processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Question Answering", "Chunk Explorer", "Comparison"])

# Tab 1: Question Answering
with tab1:
    st.header("Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query and 'retrieval_chain' in st.session_state:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            # Get retrieved documents 
            retrieved_docs = st.session_state.retriever.get_relevant_documents(query)
            # Get answer
            result = st.session_state.retrieval_chain.invoke(query)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Save result to history
            history_item = {
                "query": query,
                "result": result,
                "source_documents": retrieved_docs,
                "config": st.session_state.config,
                "response_time": response_time
            }
            st.session_state.history.append(history_item)
            
            st.subheader("Answer")
            st.write(result)
            
            st.subheader("Retrieved Documents")
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"Document {i+1}"):
                    st.write(doc.page_content)
                    st.write(f"Source: {doc.metadata}")
            
            st.write(f"Response time: {response_time:.2f} seconds")
    elif query:
        st.warning("Please process documents first.")

# Tab 2: Chunk Explorer
with tab2:
    if 'chunks' in st.session_state:
        st.header("Chunk Explorer")
        
        # Summary statistics
        st.subheader("Chunk Statistics")
        chunk_lengths = [len(chunk.page_content) for chunk in st.session_state.chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        
        st.write(f"Total chunks: {len(st.session_state.chunks)}")
        st.write(f"Average chunk length: {avg_length:.2f} characters")
        st.write(f"Min chunk length: {min(chunk_lengths)} characters")
        st.write(f"Max chunk length: {max(chunk_lengths)} characters")
        
        # Histogram of chunk lengths
        fig, ax = plt.subplots()
        ax.hist(chunk_lengths, bins=20)
        ax.set_xlabel('Chunk Length (characters)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Chunk Lengths')
        st.pyplot(fig)
        
        # Individual chunk viewer
        st.subheader("View Individual Chunks")
        chunk_index = st.slider("Chunk Index", 0, len(st.session_state.chunks) - 1, 0)
        
        selected_chunk = st.session_state.chunks[chunk_index]
        st.write(f"**Chunk {chunk_index+1}** (Length: {len(selected_chunk.page_content)} characters)")
        st.write(selected_chunk.page_content)
        st.write("Metadata:", selected_chunk.metadata)
    else:
        st.info("Process documents to view chunks.")

# Tab 3: Comparison
with tab3:
    st.header("Compare Results")
    
    if st.session_state.history:
        # Show query history
        st.subheader("Query History")
        for i, item in enumerate(st.session_state.history):
            if st.button(f"Add to comparison: '{item['query']}'", key=f"add_{i}"):
                if item not in st.session_state.results_comparison:
                    st.session_state.results_comparison.append(item)
        
        # Show comparison
        if st.session_state.results_comparison:
            st.subheader("Comparison Table")
            
            # Create a DataFrame for comparison
            comparison_data = []
            for item in st.session_state.results_comparison:
                row = {
                    "Query": item["query"],
                    "Answer": item["result"][:100] + "...",
                    "Chunking": f"{item['config']['chunking_strategy']} (size: {item['config']['chunk_size']}, overlap: {item['config']['chunk_overlap']})",
                    "Retrieval": f"{item['config']['retriever_type']} with {item['config']['search_type']} search (k={item['config']['k_value']})",
                    "Response Time": f"{item['response_time']:.2f}s"
                }
                comparison_data.append(row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            if st.button("Clear Comparison"):
                st.session_state.results_comparison = []
    else:
        st.info("Ask questions to build up comparison data.") 