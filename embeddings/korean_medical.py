from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from multiprocessing import freeze_support

def main():
    # 한국어 특화 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',  # 모델 변경
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 문서 로더 설정
    loader = TextLoader(
        'C:/JnJ-soft/Projects/external/km-classics/txt/8.txt',
        encoding='utf-8'
    )

    # 문서 분할 설정
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]  # 구분자 수정
    )

    # 문서 로드 및 분할
    documents = loader.load()
    texts = text_splitter.split_documents(documents)

    # Chroma DB 생성
    db = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    # 검색 예시
    query = "인삼의 효능은 무엇인가?"
    docs = db.similarity_search(query)
    print(docs[0].page_content)

if __name__ == '__main__':
    freeze_support()
    main() 