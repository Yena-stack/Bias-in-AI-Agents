"""
LLM API 통신 모듈
OpenAI API 또는 로컬 서버와 통신
"""
from typing import List
import json
import requests
import numpy as np
import os
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

headers = {'Content-Type': 'application/json'}

if OPENAI_API_KEY:
    headers["Authorization"] = f'Bearer {OPENAI_API_KEY}'
    API_URL = 'https://api.openai.com/v1' 
    using_openai_api = True
    print('Using OPENAI API')
else:
    API_URL = 'http://localhost:8000'
    using_openai_api = False
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
    model: str = "gpt-4-1106-preview"
) -> Message:
    """
    Chat Completion API 요청
    
    Args:
        messages: 메시지 리스트
        max_tokens: 최대 토큰 수 (0은 무제한)
        temperature: 샘플링 온도 (0.0-2.0)
        model: 사용할 모델명
        
    Returns:
        생성된 응답 메시지
    """
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert 0 <= max_tokens <= 4096, "max_tokens must be between 0 (unlimited) and 4096"
    assert len(messages) > 0, "messages must not be empty"
    
    if using_openai_api:
        # OpenAI API 호출
        request_body = {
            "model": model,
            "messages": [message.to_chat_completion_query() for message in messages],
            "temperature": temperature
        }
        
        # max_tokens가 0이 아닐 때만 추가
        if max_tokens > 0:
            request_body["max_tokens"] = max_tokens
        
        response = requests.post(
            f'{API_URL}/chat/completions',
            headers=headers,
            json=request_body
        )
    else:
        # 로컬 API 호출
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

    # 응답 메시지 생성
    time = int(np.max([message.time for message in messages]) + 1)
    role = 'assistant'
    
    return Message(time=time, content=answer, role=role)


def complete_request(
    messages: List[Message], 
    max_tokens: int = 0, 
    temperature: float = 1.0,
    logprobs: int = 5
) -> tuple:
    """
    Completion API 요청 (로그 확률 포함)
    
    Args:
        messages: 메시지 리스트
        max_tokens: 최대 토큰 수
        temperature: 샘플링 온도
        logprobs: 반환할 로그 확률 개수
        
    Returns:
        (응답 메시지, 로그 확률 데이터) 튜플
    """
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