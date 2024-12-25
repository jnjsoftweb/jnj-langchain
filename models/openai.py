from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-instruct"
)

def create_openai_model(model="gpt-3.5-turbo", options=None):
    """
    OpenAI 모델을 초기화하고 반환하는 함수
    
    Args:
        model (str): 사용할 OpenAI 모델 이름 (기본값: "gpt-3.5-turbo-instruct")
        options (dict): 모델 설정 옵션
    
    Returns:
        OpenAI: 초기화된 OpenAI 모델 인스턴스
    """
    
    default_options = {
        "temperature": 0.7,
        "max_tokens": 4095,
        "streaming": True,
        "api_key": os.getenv("OPENAI_API_KEY")
    }
    
    if options is None:
        options = default_options
    else:
        options = {**default_options, **options}
        
    return OpenAI(
        openai_api_key=options["api_key"],
        model=model,
        max_tokens=options["max_tokens"],
        temperature=options["temperature"],
        streaming=options["streaming"]
    )