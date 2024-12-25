# https://python.langchain.com/docs/integrations/chat/perplexity/

from dotenv import load_dotenv
import os, sys
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional

load_dotenv()

# 재귀 제한 증가
sys.setrecursionlimit(5000)

def create_perplexity_model(
    model_name: str = "llama-3.1-sonar-small-128k-online",
    options: dict = None
) :
    """
    Perplexity 모델을 초기화하고 반환하는 함수
    
    Args:
        model_name (str): 사용할 모델 이름 (기본값: "pplx-7b-chat")
        options (dict): 모델 설정 옵션
    
    Returns:
        Optional[ChatPerplexity]: 초기화된 Perplexity 모델 인스턴스 또는 None
    """
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not found in environment variables")
        return None
    
    try:
        api_key = "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"
        llm = ChatPerplexity(temperature=0, pplx_api_key=api_key, model=model_name)
        return llm
    except Exception as e:
        print(f"Error creating Perplexity model: {e}")
        return None


def test_chat(llm: ChatPerplexity, query: str) -> None:
    """
    채팅 모델을 테스트하는 함수
    
    Args:
        llm (ChatPerplexity): Perplexity 채팅 모델
        query (str): 테스트할 질문
    """
    try:
        system = "You are a helpful assistant."
        human = "{input}"
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", human)
        ])

        chain = prompt | llm
        response = chain.invoke({"input": query})
        print("\nQuery:", query)
        print("\nResponse:", response.content)
    except Exception as e:
        print(f"Error during chat: {e}")


if __name__ == "__main__":
    api_key = os.getenv("PERPLEXITY_API_KEY")
    print(f"API Key available: {'Yes' if api_key else 'No'}")
    
    llm = create_perplexity_model()
    if llm:
        print("Perplexity model created successfully")
        print(f"Model configuration: {llm}")
        
        # 테스트 실행
        test_chat(llm, "Why is the Higgs Boson important?")