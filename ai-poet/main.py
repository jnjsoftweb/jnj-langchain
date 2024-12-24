from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-instruct"
)

def generate_poem(theme= "코딩", prompt = "{theme}에 대한 시를 써줘"):
    return llm.invoke(prompt.format(theme=theme))

st.title("인공지능 시인")

theme = st.text_input("시의 주제를 입력하세요")


if st.button("시 생성", type="primary"):
    with st.spinner("시가 써집니다..."):
        # st.write("_Streamlit_ is :blue[cool] :sunglasses:")
        st.write(generate_poem(theme))  # 

