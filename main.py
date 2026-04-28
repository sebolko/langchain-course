import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2", temperature=0)

vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
  """Answer the question based only on the following context: 
  
  {context}

  Question: {question}

  Provide a detailed answer:"""
)

def format_docs(docs):
  """Format retrieved documents into a single string."""
  return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):

  """
  Simple retrieval chain without Langchain Expressions Language (LCEL)

  Limitations:
  - manual step-by-step execution
  - no built-in streaming support
  - no async support without additional code
  - harder to compose with other chains
  - more verbose and error-prone
  """

  # Step 1: Retrieve relevant documents
  docs = retriever.invoke(query)
  context = format_docs(docs)

  # Step 2: Format documents into context string
  context = format_docs(docs)

  # Step 3: format the prompt with context and question
  messages = prompt_template.format(context=context, question=query)

  # Step 4: invoke the LLM with the formatted messages
  result = llm.invoke(messages)

  # Step 5: return the content
  return result.content


# =====================================================
# IMPLEMENTATION 2: With LCEL (Langchain Expressions Language) - Better approach
# -----------------------------------------------------
def create_retrieval_chain_with_lcel():
  """
  Create a retrieval chain with LCEL
  Return a chain that can be invoked with a {question: ...}

  Advantages over non_LCEL approach:
  - declarative, and composable: easy to chain operations with pipe operator (|)
  - built-in streaming support: chain.stream() works out of the box
  - built-in async: chain.ainvoke() and  chain.astream() available
  - batch processing: chain.batch() for multiple inputs
  - type safety: better integration with lanchain's type system
  - less code: more concise and readable
  - reusable: chain can be saved, shared, and composed with other chains
  - better debugging: LangChain providees better observability and debugging tools
  """
  retrieval_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
  )
  return retrieval_chain

if __name__ == "__main__":
  print("Retrieving...")
  
  # Query
  query = "what is Pinecone in machine learning?"
  # -----------------------------------------------------
  # Option 0: Raw invocation without RAG
  # -----------------------------------------------------
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 0: Raw invocation without RAG")
  print("=" * 70)
  result_raw = llm.invoke([HumanMessage(content=query)])
  print("\nAnswer:")
  print(result_raw.content)

  # =====================================================
  # Option 1: Use implementation without LCEL
  # -----------------------------------------------------
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 1: Without LCEL")
  print("=" * 70)
  result_without_lcel = retrieval_chain_without_lcel(query)
  print("\nAnswer:")
  print(result_without_lcel)

  # =====================================================
  # Option 2: Use implementation with LCEL
  # -----------------------------------------------------
  print("\n" + "=" * 70)
  print("IMPLEMENTATION 2: With LCEL")
  print("=" * 70)

  chain_with_lcel = create_retrieval_chain_with_lcel()
  result_with_lcel = chain_with_lcel.invoke({"question": query})
  print("\nAnswer:")
  print(result_with_lcel)