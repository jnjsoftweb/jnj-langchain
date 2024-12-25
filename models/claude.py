from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from typing import Optional

load_dotenv()

def create_claude_model(
    model_name: str = "claude-3-opus-20240229",
    options: dict = None
) -> Optional[ChatAnthropic]:
    """
    Claude 모델을 초기화하고 반환하는 함수
    
    Args:
        model_name (str): 사용할 Claude 모델 이름 (기본값: "claude-3-opus-20240229")
        options (dict): 모델 설정 옵션
    
    Returns:
        Optional[ChatAnthropic]: 초기화된 Claude 모델 인스턴스 또는 None
    """
    
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in environment variables")
        return None
    
    default_options = {
        "temperature": 0.7,
        "max_tokens": 4096,
        "api_key": api_key,
        "streaming": True
    }
    
    if options is None:
        options = default_options
    else:
        options = {**default_options, **options}
    
    try:
        llm = ChatAnthropic(
            anthropic_api_key=options["api_key"],
            model=model_name,
            max_tokens=options["max_tokens"],
            temperature=options["temperature"],
            streaming=options["streaming"]
        )
        return llm
    except Exception as e:
        print(f"Error creating Claude model: {e}")
        return None


if __name__ == "__main__":
    model = "claude-3-opus-20240229"
    api_key = os.getenv("CLAUDE_API_KEY")
    print(f"API Key available: {'Yes' if api_key else 'No'}")
    
    llm = create_claude_model(model)
    if llm:
        print("Claude model created successfully")
        print(f"Model configuration: {llm}")
