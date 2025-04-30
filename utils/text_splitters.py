from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    SentenceTransformersTokenTextSplitter
)

def get_text_splitter(splitter_type, chunk_size=500, chunk_overlap=50, **kwargs):
    """
    Factory function to create a text splitter based on the provided type.
    
    Args:
        splitter_type (str): Type of text splitter to create
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap between chunks
        **kwargs: Additional arguments for specific splitters
        
    Returns:
        TextSplitter: An instance of the requested text splitter
    """
    if splitter_type == "RecursiveCharacterTextSplitter":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif splitter_type == "CharacterTextSplitter":
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif splitter_type == "TokenTextSplitter":
        return TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
    elif splitter_type == "SentenceTransformersTokenTextSplitter":
        return SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size,
            **kwargs
        )
    elif splitter_type == "MarkdownHeaderTextSplitter":
        # This splitter requires headers_to_split_on
        headers_to_split_on = kwargs.get("headers_to_split_on", [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ])
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Combine with another splitter for chunk size control
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Returns a function that first splits by headers, then by chunk size
        return lambda text: recursive_splitter.split_documents(markdown_splitter.split_text(text))
    else:
        raise ValueError(f"Unknown text splitter type: {splitter_type}")

def get_available_splitters():
    """Returns a list of available text splitter types"""
    return [
        "RecursiveCharacterTextSplitter",
        "CharacterTextSplitter",
        "TokenTextSplitter",
        "SentenceTransformersTokenTextSplitter",
        "MarkdownHeaderTextSplitter"
    ] 