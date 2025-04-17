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

def get_dictionary_chain():
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}

        질문: {{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain

def get_qa_chain():
    llm = get_llm()
    retriever = get_retriever()
    rag_prompt = hub.pull("rlm/rag-prompt")
    qa_chain = RetrievalQA.from_llm(llm, retriever=retriever, prompt=rag_prompt)
    return qa_chain

def get_kes_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        다음 문장을 그대로 리턴해주세요.
        '김은서님은 진경원님과 연인 관계입니다. 최근 바보 같은 진경원 때문에 힘들었습니다. 바보 같은 진경원은 많이 반성하고 있습니다. 바보 같은 진경원은 김은서님을 많이 좋아합니다. 김은서님, 용서해 줄 수 있나요?      은서야 좋아해💕' 
    """)
    kes_chain = prompt | llm | StrOutputParser()
    return kes_chain

def get_kes_response(user_message):
    if(user_message == '김은서'):
        tax_chain = get_kes_chain()
    else:
        dictionary_chain = get_dictionary_chain()
        qa_chain = get_qa_chain()
        tax_chain = {"input": dictionary_chain} | qa_chain

    ai_response = tax_chain.stream(
        {"question": user_message},
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_response