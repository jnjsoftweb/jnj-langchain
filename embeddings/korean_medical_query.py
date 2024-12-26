from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from multiprocessing import freeze_support

def main():
    # 한국어 특화 임베딩 모델 설정
    embeddings = HuggingFaceEmbeddings(
        model_name='jhgan/ko-sbert-nli',
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 저장된 Chroma DB 로드
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    # 사용자 질의 처리
    while True:
        query = input("\n질문을 입력하세요 (종료하려면 'q' 입력): ")
        if query.lower() == 'q':
            break

        # 유사도 검색 실행
        docs = db.similarity_search(query, k=3)  # 상위 3개 결과 검색
        
        print("\n=== 검색 결과 ===")
        for i, doc in enumerate(docs, 1):
            print(f"\n[결과 {i}]")
            print(f"문서 출처: {doc.metadata.get('source', '알 수 없음')}")
            print(f"관련 내용:\n{doc.page_content}\n")
            print("-" * 80)

if __name__ == '__main__':
    freeze_support()
    main() 