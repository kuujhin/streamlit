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
        당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요.
        답변을 알 수 없다면 모른다고 답변해주세요.
        답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주시고,
        2-3 문장정도의 짧은 내용의 답변을 원합니다.
        만약 질문이 '진경원' 또는 '김은서'라면, 다음 문장을 그대로 답변해주세요.
        '이것은 특정 단어에만 반응하는 답변입니다' 

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