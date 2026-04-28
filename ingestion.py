import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()

if __name__ == '__main__':
  print("Ingesting...")
  loader = TextLoader("/Users/sebastianolko/Documents/langchain-course/langchain-course/mediumblog1.txt")
  encoding = "UTF-8"
  documents = loader.load()

  print("splitting...")
  text_spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_spliter.split_documents(documents)
  print(f"Found {len(texts)} chunks")

  embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

  print("ingesting...")
  PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
  print("finish")