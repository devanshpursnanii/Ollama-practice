from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader

doc="./data.pdf"
model="llama3.2"

#pdf ingestion
# if you have a URL, you can use OnlinePDFLoader
if doc:
    loader=PyPDFLoader(file_path=doc)
    data=loader.load()
    print("document loaded")
else:
    print("No document provided")

content=data[0].page_content
print(content[:10])

#pdf extraction and chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks=text_splitter.split_documents(data)

#creating vector db
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

ollama.pull("nomic-embed-text")

vector_db=Chroma.from_documents(
            documents=chunks,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
            collection_name="pdf-rag"
)

print("vector db ready")

#retrieval
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm=ChatOllama(model=model)

PROMPT=PromptTemplate(input_variables=["question"],
                      template="""You are an AI language model assistant. Your task is to generate five
    different versions of the given user question to retrieve relevant documents from
    a vector database. By generating multiple perspectives on the user question, your
    goal is to help the user overcome some of the limitations of the distance-based
    similarity search. Provide these alternative questions separated by newlines.
    Original question: {question}""",)


retriever=MultiQueryRetriever.from_llm(vector_db.as_retriever(),
                                       llm=llm,
                                       prompt=PROMPT
                                       )

# RAG prompt
template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

prompt=ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res=chain.invoke(input={"question": "What is the main topic of the document?"})
print(res)
