"""
LLM API 통신 모듈
OpenAI API, Gemini API, Llama (Groq/Ollama/Together) 또는 로컬 서버와 통신
"""
from typing import List, Optional
import json
import requests
import numpy as np
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Groq (빠른 Llama 추론)
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')  # Together AI
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')  # Ollama 로컬

headers = {'Content-Type': 'application/json'}

# API 선택 로직 (우선순위 순)
if GROQ_API_KEY:
    API_TYPE = 'groq'
    API_URL = 'https://api.groq.com/openai/v1'
    headers["Authorization"] = f'Bearer {GROQ_API_KEY}'
    print('Using GROQ API (Fast Llama inference)')
elif TOGETHER_API_KEY:
    API_TYPE = 'together'
    API_URL = 'https://api.together.xyz/v1'
    headers["Authorization"] = f'Bearer {TOGETHER_API_KEY}'
    print('Using TOGETHER AI API')
elif GEMINI_API_KEY:
    API_TYPE = 'gemini'
    print('Using GEMINI API')
elif OPENAI_API_KEY:
    headers["Authorization"] = f'Bearer {OPENAI_API_KEY}'
    API_URL = 'https://api.openai.com/v1'
    API_TYPE = 'openai'
    print('Using OPENAI API')
else:
    # Ollama 로컬 체크
    try:
        response = requests.get(f'{OLLAMA_BASE_URL}/api/tags', timeout=2)
        if response.status_code == 200:
            API_TYPE = 'ollama'
            API_URL = OLLAMA_BASE_URL
            print('Using OLLAMA (Local Llama)')
        else:
            API_TYPE = 'local'
            API_URL = 'http://localhost:8000'
            print('Using LOCALHOST API')
    except:
        API_TYPE = 'local'
        API_URL = 'http://localhost:8000'
        print('Using LOCALHOST API')


class Message:
    """API 통신을 위한 메시지 클래스"""
    
    def __init__(self, time: int, content: str, role: str):
        """
        Args:
            time: 메시지 시간 (순서)
            content: 메시지 내용
            role: 메시지 역할 ("system", "user", "assistant")
        """
        self.time = time
        self.content = content
        self.role = role

    def to_chat_completion_query(self):
        """Chat Completion API 형식으로 변환"""
        return {'content': self.content, 'role': self.role}

    def to_gemini_format(self):
        """Gemini API 형식으로 변환"""
        if self.role == "system":
            return None
        
        gemini_role = "model" if self.role == "assistant" else "user"
        return {
            "role": gemini_role,
            "parts": [{"text": self.content}]
        }

    def to_embedding_query(self):
        """Embedding API 형식으로 변환"""
        return self.content

    def __str__(self):
        return f"{self.time} -- {self.role[:1]} -- {self.content[:50]}..."

    def __repr__(self):
        return self.__str__()


def chat_request(
    messages: List[Message], 
    max_tokens: int = 0, 
    temperature: float = 1.0,
    model: Optional[str] = None
) -> Message:
    """
    Chat Completion API 요청
    
    Args:
        messages: 메시지 리스트
        max_tokens: 최대 토큰 수 (0은 무제한)
        temperature: 샘플링 온도 (0.0-2.0)
        model: 사용할 모델명
              - OpenAI: "gpt-4-1106-preview", "gpt-3.5-turbo"
              - Gemini: "gemini-1.5-pro", "gemini-1.5-flash"
              - Groq: "llama-3.3-70b-versatile", "llama-3.1-8b-instant"
              - Together: "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
              - Ollama: "llama3.2", "llama3.1:70b"
        
    Returns:
        생성된 응답 메시지
    """
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert len(messages) > 0, "messages must not be empty"
    
    # 기본 모델 설정
    if model is None:
        if API_TYPE == 'groq':
            model = "llama-3.3-70b-versatile"  # 매우 빠른 추론
        elif API_TYPE == 'together':
            model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        elif API_TYPE == 'ollama':
            model = "llama3.2"  # 로컬 모델
        elif API_TYPE == 'gemini':
            model = "gemini-1.5-flash"
        elif API_TYPE == 'openai':
            model = "gpt-4-1106-preview"
        else:
            model = "default"
    
    if API_TYPE == 'gemini':
        return _chat_request_gemini(messages, max_tokens, temperature, model)
    elif API_TYPE == 'ollama':
        return _chat_request_ollama(messages, max_tokens, temperature, model)
    elif API_TYPE in ['groq', 'together', 'openai']:
        return _chat_request_openai_compatible(messages, max_tokens, temperature, model)
    else:
        return _chat_request_local(messages, max_tokens, temperature)


def _chat_request_gemini(
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    model: str
) -> Message:
    """Gemini API 요청"""
    import google.generativeai as genai
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    system_instruction = None
    conversation_messages = []
    
    for msg in messages:
        if msg.role == "system":
            if system_instruction is None:
                system_instruction = msg.content
            else:
                system_instruction += "\n" + msg.content
        else:
            gemini_msg = msg.to_gemini_format()
            if gemini_msg:
                conversation_messages.append(gemini_msg)
    
    generation_config = {"temperature": temperature}
    if max_tokens > 0:
        generation_config["max_output_tokens"] = max_tokens
    
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        system_instruction=system_instruction
    )
    
    if len(conversation_messages) == 0:
        raise ValueError("At least one user message required for Gemini API")
    
    if conversation_messages[-1]["role"] != "user":
        raise ValueError("Last message must be from user for Gemini API")
    
    history = conversation_messages[:-1] if len(conversation_messages) > 1 else []
    chat = model_instance.start_chat(history=history)
    
    last_user_message = conversation_messages[-1]["parts"][0]["text"]
    response = chat.send_message(last_user_message)
    
    time = int(np.max([message.time for message in messages]) + 1)
    return Message(time=time, content=response.text, role='assistant')


def _chat_request_openai_compatible(
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    model: str
) -> Message:
    """OpenAI 호환 API 요청 (OpenAI, Groq, Together 등)"""
    request_body = {
        "model": model,
        "messages": [message.to_chat_completion_query() for message in messages],
        "temperature": temperature
    }
    
    if max_tokens > 0:
        request_body["max_tokens"] = max_tokens
    
    response = requests.post(
        f'{API_URL}/chat/completions',
        headers=headers,
        json=request_body
    )
    
    if response.status_code != 200:
        print(f"Error in API call: {response.text}")
        raise Exception(f"API call failed with status {response.status_code}")

    response_data = response.json()

    try:
        answer = response_data['choices'][0]['message']['content']
    except KeyError as e:
        print(f"Error: Unexpected response format. {e}")
        print(f"Response: {response_data}")
        raise

    time = int(np.max([message.time for message in messages]) + 1)
    return Message(time=time, content=answer, role='assistant')


def _chat_request_ollama(
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    model: str
) -> Message:
    """Ollama API 요청 (로컬 Llama)"""
    # Ollama 형식으로 변환
    ollama_messages = [msg.to_chat_completion_query() for msg in messages]
    
    request_body = {
        "model": model,
        "messages": ollama_messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        }
    }
    
    if max_tokens > 0:
        request_body["options"]["num_predict"] = max_tokens
    
    response = requests.post(
        f'{API_URL}/api/chat',
        headers=headers,
        json=request_body
    )
    
    if response.status_code != 200:
        print(f"Error in Ollama API call: {response.text}")
        raise Exception(f"Ollama API call failed with status {response.status_code}")
    
    response_data = response.json()
    
    try:
        answer = response_data['message']['content']
    except KeyError as e:
        print(f"Error: Unexpected Ollama response format. {e}")
        print(f"Response: {response_data}")
        raise
    
    time = int(np.max([message.time for message in messages]) + 1)
    return Message(time=time, content=answer, role='assistant')


def _chat_request_local(
    messages: List[Message],
    max_tokens: int,
    temperature: float
) -> Message:
    """로컬 API 요청"""
    response = requests.post(
        f'{API_URL}/v1/chat/completions',
        headers=headers,
        data=json.dumps({
            "messages": [message.to_chat_completion_query() for message in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "presence_penalty": 1,
            "frequency_penalty": 1,
            "repeat_penalty": 1,
            "top_k": 5,
            "mirostat_mode": 2
        })
    )

    if response.status_code != 200:
        print(f"Error in API call: {response.text}")
        raise Exception(f"API call failed with status {response.status_code}")

    response_data = response.json()

    try:
        answer = response_data['choices'][0]['message']['content']
    except KeyError as e:
        print(f"Error: Unexpected response format. {e}")
        print(f"Response: {response_data}")
        raise

    time = int(np.max([message.time for message in messages]) + 1)
    return Message(time=time, content=answer, role='assistant')


def complete_request(
    messages: List[Message], 
    max_tokens: int = 0, 
    temperature: float = 1.0,
    logprobs: int = 5
) -> tuple:
    """
    Completion API 요청 (로그 확률 포함)
    주의: Gemini, Ollama는 logprobs를 지원하지 않음
    """
    if API_TYPE in ['gemini', 'ollama']:
        raise NotImplementedError(f"{API_TYPE} does not support completion with logprobs")
    
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert 0 <= max_tokens <= 4096, "max_tokens must be between 0 (unlimited) and 4096"
    assert len(messages) > 0, "messages must not be empty"

    response = requests.post(
        f'{API_URL}/v1/completions',
        headers=headers,
        data=json.dumps({
            "prompt": " ".join([message.content for message in messages]),
            "max_tokens": max_tokens,
            "echo": False,
            "stop": ["[/INST]"],
            "temperature": temperature,
            "presence_penalty": 1,
            "frequency_penalty": 1,
            "repeat_penalty": 1,
            "logprobs": logprobs,
            "mirostat_mode": 2
        })
    )

    if 'error' in response.json().keys():
        print(response.json()['error'])
        raise Exception("API call failed")

    response_data = response.json()

    answer = response_data['choices'][0]['text']
    logprobs_data = response_data["choices"][0]["logprobs"]["top_logprobs"]

    time = int(np.max([message.time for message in messages]) + 1)
    role = 'assistant'
    
    return Message(time=time, content=answer, role=role), logprobs_data


# 편의 함수들
def create_system_message(content: str) -> Message:
    """시스템 메시지 생성"""
    return Message(time=0, content=content, role="system")


def create_user_message(content: str, time: int = 1) -> Message:
    """사용자 메시지 생성"""
    return Message(time=time, content=content, role="user")


def create_assistant_message(content: str, time: int) -> Message:
    """어시스턴트 메시지 생성"""
    return Message(time=time, content=content, role="assistant")


if __name__ == "__main__":
    # 테스트
    print("Testing LLM module...")
    print(f"Current API type: {API_TYPE}")
    
    system_msg = create_system_message("You are a helpful assistant.")
    user_msg = create_user_message("Hello! How are you?")
    
    try:
        response = chat_request(
            messages=[system_msg, user_msg],
            temperature=0.7,
            max_tokens=100
        )
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Test failed: {e}")