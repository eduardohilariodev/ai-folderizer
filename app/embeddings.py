import getpass
import os
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def create_vector_store(documents: List[Document]) -> InMemoryVectorStore:
    """Create an InMemoryVectorStore with the given documents.

    Args:
        documents (List[Document]): A list of Document objects to add to the vector store.

    Returns:
        InMemoryVectorStore: The created vector store containing the documents.
    """
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = embeddings_model.embed_documents(documents)
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store


def main() -> None:
    """Main function to create vector store and perform similarity search."""
    document_1 = Document(id="1", page_content="foo", metadata={"baz": "bar"})
    document_2 = Document(id="2", page_content="thud", metadata={"bar": "baz"})
    document_3 = Document(id="3", page_content="i will be deleted :(")

    documents = [document_1, document_2, document_3]
    vector_store = create_vector_store(documents)

    results = vector_store.similarity_search(query="thud", k=1)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")


if __name__ == "__main__":
    main()
