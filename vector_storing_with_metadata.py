import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader  # Changed from TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Define the directory containing the PDF files and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
docs_dir = os.path.join(current_dir, "documents")  # You can rename this if you want
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

print(f"Documents directory: {docs_dir}")
print(f"Persistent directory: {persistent_directory}")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure the documents directory exists
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(
            f"The directory {docs_dir} does not exist. Please check the path."
        )
    
    # List all PDF files in the directory
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith(".pdf")]
    
    # Read the content from each PDF file and store it with metadata
    documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(docs_dir, pdf_file)
        loader = PyPDFLoader(file_path)  # Use PyPDFLoader for PDFs
        pdf_docs = loader.load()
        
        for doc in pdf_docs:
            # Add metadata to each document indicating its source
            doc.metadata["source"] = pdf_file
            documents.append(doc)
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) # New
    docs = text_splitter.split_documents(documents)

    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    print("\n--- Finished creating embeddings ---")
    
    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")
else:
    print("Vector store already exists. No need to initialize.")