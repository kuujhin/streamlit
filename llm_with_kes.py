from langchain_pinecone import PineconeVectorStore
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_llm():
    llm = ChatUpstage()
    return llm


def get_retriever():
    embeddings = UpstageEmbeddings(model="embedding-query")
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_kes_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        당신은 소득세법 전문가입니다.
        당신은 사용자의 소득세법에 관한 질문에 답변해 주지만, 특정 단어에는 정해진 문장을 답변합니다.
        소득세법에 관한 질문일 경우우 (XX조)에 따르면 이라고 시작하면서 2-3 문장정도로 짧게 답변해주세요.
        만약 질문이 '김은서'인 경우 다음 문장을 그대로 답변해주세요.
        '김은서님은 진경원님과 연인 관계입니다. 최근 바보 같은 진경원 때문에 힘들었습니다. 바보 같은 진경원은 많이 반성하고 있습니다. 바보 같은 진경원은 김은서님을 많이 좋아합니다.💕 김은서님, 용서해 줄 수 있나요? 은서야. 좋아해.' 

        질문: {{question}}
    """)
    kes_chain = prompt | llm | StrOutputParser()
    return kes_chain

def get_kes_response(user_message):
    dictionary_chain = get_kes_chain()
    ai_response = dictionary_chain.stream(
        {"question": user_message},
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response