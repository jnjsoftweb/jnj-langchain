from dotenv import load_dotenv
import os
import sys
sys.path.append("..")

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from models.gen_model import ai_model
from langchain.retrievers import MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()

# Loader
pdf_filepath = 'LuckyDay.pdf'
loader = PyPDFLoader(pdf_filepath)
pages = loader.load()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, 
    chunk_overlap=30,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)

# Embeddings
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# load it into chroma db
db = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory="db",
)

# question
question = "김첨지의 직업은 무엇인가요?"
# question = "아내가 먹고 싶어하는 음식은 무엇인가요?"
# llm = ai_model(platform="openai", options={"temperature": 0})
# # llm = ai_model(platform="gemini", options={"temperature": 0})

# retriever_from_llm = MultiQueryRetriever.from_llm(
#     retriever=db.as_retriever(),
#     llm=llm,
# )

# result = retriever_from_llm.get_relevant_documents(question)
# print(result)

platform = "openai"
# platform = "gemini"

llm = ai_model(platform=platform, options={"temperature": 0.5})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    # chain_type="stuff",
    retriever=db.as_retriever(),
    # return_source_documents=True,
)

result = qa_chain.invoke({"query": question})
print(result)