from langchain.prompts import PromptTemplate

template = """당신은 한의학 전문가입니다. 주어진 문맥을 바탕으로 질문에 답변해주세요.

문맥:
{context}

질문: {question}

답변:"""

QA_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
) 