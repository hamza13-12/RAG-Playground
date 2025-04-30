from langchain_community.vectorstores import FAISS, Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

def get_retriever(vector_store, search_type="similarity", search_kwargs=None, **kwargs):
    """
    Create a retriever based on the vector store and search parameters.
    
    Args:
        vector_store: The vector store instance to use
        search_type (str): The type of search to perform (similarity, mmr)
        search_kwargs (dict): Additional search parameters
        **kwargs: Additional parameters for specialized retrievers
        
    Returns:
        Retriever: A retriever instance
    """
    if search_kwargs is None:
        search_kwargs = {"k": 4}
        
    # Basic retriever
    base_retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
    
    retriever_type = kwargs.get("retriever_type", "basic")
    
    if retriever_type == "basic":
        return base_retriever
    
    elif retriever_type == "contextual_compression":
        # Create an LLM for compression
        llm = ChatOpenAI(temperature=0)
        
        # Create a compressor
        compressor = LLMChainExtractor.from_llm(llm)
        
        # Create and return a contextual compression retriever
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

def get_available_retriever_types():
    """Returns a list of available retriever types"""
    return [
        "basic",
        "contextual_compression"
    ]

def get_available_search_types():
    """Returns a list of available search types"""
    return [
        "similarity",
        "mmr"
    ] 