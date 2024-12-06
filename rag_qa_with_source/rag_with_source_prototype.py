import pprint
# from genflow import CustomComponent
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.schema import Document
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os, random
import pathlib as Path
from urllib.parse import quote


def load_data(datapath: str) -> List[Document]:
    """
    params:
    datapath: path to data directory containing PDF files
    returns: a list of LangChain Documents

    Loads data from all PDF files in the directory and returns a list of LangChain Documents.
    """
    try:
        langchain_docs = []
        for filename in tqdm(os.listdir(datapath), desc="Loading PDFs"):
            if filename.endswith(".pdf"):
                file_path = os.path.join(datapath, filename)
                pdf_loader = PyPDFLoader(file_path)
                documents = pdf_loader.load()
                langchain_docs.extend(documents)
        return langchain_docs
    except Exception as e:
        print(e)
        raise



def chunk(data: List[Document]) -> List[Document]:
    """
    params:
    data: list of documents
    returns: list of documents

    Chunk the documents into smaller documents.
    """

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128, length_function=len)
        chunks = splitter.split_documents(data)
        return chunks
    except Exception as e:
        print(e)
        raise


import uuid  # to generate unique chunk IDs


def store(chunks: List[Document], persist_directory="database") -> Chroma:
    """
    Store chunks in a Chroma vector store, skipping if it already exists.
    """
    tags = ["HR", "Data Science", "Software"]
    try:
        persist_path = Path.Path(persist_directory)
        if persist_path.exists():
            print("Vector store already exists. Skipping embedding and storing.")
            embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            db = Chroma(collection_name="sample_rag", embedding_function=embed_model,
                        persist_directory=str(persist_path))
        else:
            embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            db = Chroma(collection_name="sample_rag", embedding_function=embed_model,
                        persist_directory=str(persist_path))

            # Add chunk_id to metadata and then store documents
            for chunk in chunks:
                chunk.metadata['chunk_id'] = str(uuid.uuid4())  # Add a unique chunk_id
                chunk.metadata['tag'] = random.choice(tags)
            db.add_documents(chunks)
            db.persist()  # Save the embeddings and documents to the directory

        return db
    except Exception as e:
        print(e)
        raise



# Main execution
if __name__ == "__main__":
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    persist_path = Path.Path("database")

    if not persist_path.exists():
        data = load_data("sample_data")
        chunks = chunk(data)
        vector_store = store(chunks)
    else:
        vector_store = Chroma(collection_name="sample_rag", embedding_function=embed_model, persist_directory=str(persist_path))

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":3, "filter": {"tag":"HR"}})

    documents = retriever.get_relevant_documents("Best practices in Machine Learning")
    print(documents)

    result = []
    for doc in documents:
        # Create a clickable file URL link to the document with page number
        source_path = Path.Path(doc.metadata['source'])
        page_number = doc.metadata.get('page', 1)
        id = doc.metadata.get('chunk_id', None)
        tag = doc.metadata.get('tag', None)

        result.append({
            "content": doc.page_content,
            "page": page_number,
            "chunk_id" : id,
            "tag" : tag
        })

    pprint.pp(result, compact=True)
