from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# 임베딩 모델 초기화
embeddings = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sroberta-multitask',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 저장된 Chroma DB 로드
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 프롬프트 템플릿 설정
template = """당신은 한의학 전문가입니다. 주어진 문맥을 바탕으로 질문에 답변해주세요.

문맥:
{context}

질문: {question}

답변:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# RAG 체인 설정
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": QA_PROMPT}
)

# 질의응답 실행
def ask_medical_question(question: str) -> str:
    return qa_chain.run(question)

# 사용 예시
questions = [
    "인삼의 효능은 무엇인가?",
    "감기에 좋은 한약재는?",
    "두통이 있을 때 어떤 치료법이 있나요?"
]

for q in questions:
    print(f"\n질문: {q}")
    print(f"답변: {ask_medical_question(q)}") 