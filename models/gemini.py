from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional

load_dotenv()

def create_gemini_model(
    model_name: str = "gemini-1.5-flash",
    options: dict = None
) -> ChatGoogleGenerativeAI:
    """
    Gemini 모델을 초기화하고 반환하는 함수
    
    Args:
        model_name (str): 사용할 Gemini 모델 이름 (기본값: "gemini-pro")
        temperature (float): 생성 텍스트의 무작위성 정도 (0.0 ~ 1.0)
        api_key (Optional[str]): Google API 키. None인 경우 환경 변수에서 가져옴
        streaming (bool): 스트리밍 응답 사용 여부
    
    Returns:
        ChatGoogleGenerativeAI: 초기화된 Gemini 모델 인스턴스
    """
    
    default_options = {
        "temperature": 0.7,
        "max_tokens": 4095,
        "api_key": os.getenv("GEMINI_API_KEY"),
        "streaming": True
    }
    
    if options is None:
        options = default_options
    else:
        options = {**default_options, **options}
    
    llm = ChatGoogleGenerativeAI(
        google_api_key=options["api_key"],
        model=model_name,
        max_output_tokens=options["max_tokens"],
        temperature=options["temperature"],
        streaming=options["streaming"],
        convert_system_message_to_human=True
    )
    
    return llm


if __name__ == "__main__":
    model = "gemini-1.5-flash"
    llm = create_gemini_model(model)
    print(llm)