from models.gemini import create_gemini_model
from dotenv import load_dotenv
import os

load_dotenv()

# 환경변수에서 API 키 가져오기
llm = create_gemini_model(
    api_key=os.getenv("GEMINI_API_KEY")  # .env 파일에서 키를 가져옴
)

response = llm.invoke("안녕하세요!")
print(response)
