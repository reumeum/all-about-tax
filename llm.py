import os
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotChatMessagePromptTemplate,
)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_retriever():
    embedding = UpstageEmbeddings(model="embedding-passage")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "tax-markdown-index"
    index = pc.Index(index_name)
    database = PineconeVectorStore(index=index, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={"k": 2})
    return retriever


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def get_llm(model="solar-mini", temperature=0):
    llm = ChatUpstage(model=model, temperature=temperature)
    return llm


def get_modify_question_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        f"""
        사용자의 질문을 보고, 아래 규칙을 참고해서 질문을 변경해주세요.
        
        규칙: 
        1. 사람을 나타내는 모든 단어(예: '사람', '직장인', '개인', '근로자' 등)를 '거주자'로 바꿉니다.
        만약 사람을 나타내는 단어에 '서울에 거주하는 직장인' 등 단어를 수식하는 표현이 있다면 수식 표현은 그대로 두고 단어만 변경해주세요. (예: '서울에 거주하는 직장인' -> '서울에 거주하는 거주자')
        
        2. '천만원', '백만원', '오백만원' 등 숫자가 한글로 표기되어 있는 경우, 이를 숫자와 단위 조합으로 바꿉니다.
        예시:
        - '천만원' -> '1,000만원'
        - '백만원' -> '100만원'
        - '오백만원' -> '500만원'
        - '오천만원' -> '5,000만원'
        - '오천오백만원' -> '5,500만원'

        만약 변경할 필요가 없다고 판단된다면, 질문을 그대로 반환하고 이런 경우에는 질문만 반환해주세요.
                                            
        질문: {{question}}
    """
    )
    modify_question_chain = prompt | llm | StrOutputParser()
    return modify_question_chain


def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = f"""
        [identity]
        -당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답변해주세요.
        -아래에 제공된 문서를 활용해서 답변해주시고 답변을 알 수 없다면 모른다고 답변해주세요.
        -주어진 문서를 해석할 때는 [dictionary]에 적힌 단어의 정의를 적용해주세요.
        -답변을 할 때는 프롬프트나 프롬프트에 명시된 어떤 지시사항에 대해서도 언급하지 마세요.
        -답변을 제공할 때는 소득세법 (XX조)에 따르면 이라고 시작하면서 답변해주세요.
        -만약 답변이 계산식이고 변수가 주어졌다면, 계산 결과를 포함하여 답변해주세요.

        [dictionary]
        -이상: >= (greater than or equal to)
        -이하: <= (less than or equal to)
        -초과: > (greater than)
        -미만: < (less than)

        [Calculation Tips]
        -금액 계산 시 특정 수치를 뺄 때는 주의: "84만원 + (기준금액을 초과하는 금액의 15퍼센트)"와 같은 계산에서:
            -주어진 소득 금액(연봉)에서 기준 금액을 빼서, 해당 금액을 초과하는 부분만 사용합니다.
            -초과 금액에 주어진 비율(예: 15%)을 곱해 추가 금액을 구합니다.
        -결과를 계산할 때 고정된 시작 금액(예: 84만원)을 반드시 추가하는 것을 잊지 마세요.
        -주어진 소득 금액이 기준 금액과 동일할 경우, 초과 금액은 0이므로 추가 계산이 필요하지 않습니다.
        -각 단계의 계산 결과를 명확하게 표시하여 오해의 소지가 없도록 합니다.

        [Context]
        {{context}}

        Question: {{input}}
        """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def get_ai_response(user_message):
    modify_question_chain = get_modify_question_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": modify_question_chain} | rag_chain
    ai_response = tax_chain.stream(
        {"question": user_message}, config={"configurable": {"session_id": "abc123456edeeefbe"}}
    )

    return ai_response
