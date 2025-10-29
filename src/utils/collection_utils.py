


def create_chroma_collection(chroma_client, name, embedding = None):
    """
    Create (or fetch) a Chroma collection by name.
    chroma_client       : chromadb.HttpClient       - An initialized ChromaDB client.
    name                : str                       - Target collection name.
    embedding           : Optional[Callable]        - Optional embedding function passed to Chroma. If provided, Chroma will
                                                      call this function to generate vectors on `.add(...)` when embeddings
                                                      are not explicitly supplied.
    """
    # Create or get the collection
    collection = chroma_client.create_collection(name, get_or_create=True, embedding_function=embedding)

    return collection