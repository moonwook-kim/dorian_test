import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
from PIL import Image


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
LANGSMITH_TRACING = st.secrets["LANGSMITH_TRACING"]
LANGSMITH_ENDPOINT= st.secrets["LANGSMITH_ENDPOINT"]
LANGSMITH_API_KEY= st.secrets["LANGSMITH_API_KEY"]
LANGSMITH_PROJECT= st.secrets["LANGSMITH_PROJECT"]

image = Image.open("images/dorian_01.png")
st.image(image, width=150)
# st.title("dorian")

# 대화기록 저장을 위한 세션 생성 (최초 1번만)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    clear_button = st.button("대화 초기화")
    selected_prompt = st.selectbox("프롬프트를 선택해주세요", ("기본", "SNS", "요약"), index=0)

# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).markdown(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 체인 생성
def create_chain(prompt_type):
    # 기본모드 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 도리안(dorian)이라는 이름을 가진 AI 어시스턴트입니다. 질문에 간결하게 답변을 해주세요."),
            ("user", "#Question:\n{question}"),
        ]
    )
    if prompt_type == "SNS":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "요약":
        prompt = load_prompt("prompts/요약.yaml", encoding="utf-8")
    
    
    # GPT 모델 생성
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 출력 파서 생성
    output_parser = StrOutputParser()
    # 체인 생성
    chain = prompt | llm | output_parser
    return chain

# 대화 초기화 버튼 클릭 시 실행
if clear_button:
    st.session_state["messages"] = []

print_messages()

# 사용자 입력
user_input = st.chat_input("메시지를 입력하세요")

if user_input:
    # 사용자 입력 출력
    st.chat_message("user").markdown(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt)
    # 사용자 입력 처리
    # ai_answer = chain.invoke({"question": user_input})
    # 응답 출력
    # st.chat_message("assistant").markdown(ai_answer)

    # 응답 스트리밍 처리&출력
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()
        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer + "")
    

    add_message("user", user_input)
    add_message("assistant", ai_answer)
