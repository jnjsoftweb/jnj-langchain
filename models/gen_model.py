from models.openai import create_openai_model
from models.gemini import create_gemini_model

def ai_model(platform="openai", model=None, options={}):
    if platform == "openai":
        model = model or "gpt-3.5-turbo-instruct"
        return create_openai_model(model, options)
    elif platform == "gemini":
        model = model or "gemini-pro"  # Gemini의 기본 모델
        return create_gemini_model(model, options)
    else:
        raise ValueError(f"지원하지 않는 플랫폼: {platform}")