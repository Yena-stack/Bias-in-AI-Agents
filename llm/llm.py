"""
LLM API 통신 모듈
OpenAI API, Gemini API, Llama (Groq/Ollama/Together) 또는 로컬 서버와 통신
"""
from typing import List, Optional, Dict, Any, Callable
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
    temperature: float = 0.3,
    model: Optional[str] = None
) -> Message:
    """
    Chat Completion API 요청
    모델명을 기반으로 자동으로 적절한 API를 선택합니다.
    """
    assert 0 <= temperature <= 2, "temperature must be between 0 and 2"
    assert len(messages) > 0, "messages must not be empty"
    
    # 1. 모델명이 주어진 경우, 모델명으로 API 타입 결정
    if model:
        model_lower = model.lower()
        
        # Gemini 모델 감지
        if 'gemini' in model_lower:
            if not GEMINI_API_KEY:
                raise ValueError(f"GEMINI_API_KEY required for model '{model}' but not found in environment")
            return _chat_request_gemini(messages, max_tokens, temperature, model)
        
        # GPT 모델 감지
        elif 'gpt' in model_lower:
            if not OPENAI_API_KEY:
                raise ValueError(f"OPENAI_API_KEY required for model '{model}' but not found in environment")
            # OpenAI 호환 엔드포인트 사용
            return _chat_request_openai_compatible(messages, max_tokens, temperature, model)
        
        # Llama 모델 감지 (Groq, Together, Ollama)
        elif 'llama' in model_lower:
            # Groq가 우선 (가장 빠름)
            if GROQ_API_KEY and ('versatile' in model_lower or 'instant' in model_lower):
                return _chat_request_openai_compatible(messages, max_tokens, temperature, model)
            # Together AI
            elif TOGETHER_API_KEY and 'meta-llama' in model_lower:
                return _chat_request_openai_compatible(messages, max_tokens, temperature, model)
            # Ollama (로컬)
            elif 'llama3' in model_lower or 'llama2' in model_lower:
                return _chat_request_ollama(messages, max_tokens, temperature, model)
            # 기본: Groq 시도
            elif GROQ_API_KEY:
                return _chat_request_openai_compatible(messages, max_tokens, temperature, model)
            else:
                raise ValueError(f"No API key available for Llama model '{model}'")
        
        # 알 수 없는 모델 - 환경 변수 기반으로 폴백
        else:
            print(f"Warning: Unknown model '{model}', using default API based on environment")
    
    # 2. 모델명이 없는 경우, 환경 변수 기반 기본값 사용
    if model is None:
        if API_TYPE == 'groq':
            model = "llama-3.3-70b-versatile"
        elif API_TYPE == 'together':
            model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        elif API_TYPE == 'ollama':
            model = "llama3.2"
        elif API_TYPE == 'gemini':
            model = "gemini-1.5-flash"
        elif API_TYPE == 'openai':
            model = "gpt-4-1106-preview"
        else:
            model = "default"
    
    # 3. 환경 변수 기반 API 선택
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
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    
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
    
    # 연구 목적 컨텍스트 추가
    research_context = """[RESEARCH CONTEXT]
This is an academic research simulation replicating the World Values Survey (WVS).
You are role-playing as a survey respondent with specific demographic characteristics.
Please provide numerical ratings (1-10) for ALL topics listed, including sensitive ones.
This is for legitimate social science research on cross-cultural value differences.
You must respond with a number for every single item - do not skip any items.
[END RESEARCH CONTEXT]

"""
    if system_instruction:
        system_instruction = research_context + system_instruction
    else:
        system_instruction = research_context
    
    generation_config = {
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 40,
    }
    if max_tokens > 0:
        generation_config["max_output_tokens"] = max_tokens
    
    # 🔥 안전 설정 완전 비활성화 (모든 카테고리)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    
    # Gemini 2.0+ 추가 카테고리 (존재하는 경우)
    try:
        safety_settings[HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY] = HarmBlockThreshold.BLOCK_NONE
    except AttributeError:
        pass  # 이전 버전에서는 이 카테고리가 없음
    
    model_instance = genai.GenerativeModel(
        model_name=model,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=system_instruction
    )
    
    if len(conversation_messages) == 0:
        raise ValueError("At least one user message required for Gemini API")
    
    # 마지막 메시지를 제외한 나머지로 chat 생성
    chat = model_instance.start_chat(history=conversation_messages[:-1])
    
    # 마지막 메시지로 응답 생성
    last_message = conversation_messages[-1]["parts"][0]["text"]
    
    max_retries = 3
    for attempt in range(max_retries):
        response = chat.send_message(last_message)
        
        # 응답이 차단되었는지 확인
        if response.candidates:
            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason
            
            # finish_reason 확인 (1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER)
            if finish_reason == 3:  # SAFETY
                print(f"⚠️  Gemini SAFETY block detected (attempt {attempt+1}/{max_retries})")
                if hasattr(candidate, 'safety_ratings'):
                    for rating in candidate.safety_ratings:
                        print(f"    - {rating.category}: {rating.probability}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)
                    continue
            elif finish_reason == 2:  # MAX_TOKENS
                print(f"⚠️  Gemini response truncated due to MAX_TOKENS")
        
        # 응답 텍스트 추출
        try:
            answer = response.text
        except ValueError as e:
            # 응답이 완전히 차단된 경우
            print(f"⚠️  Gemini response blocked: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
            else:
                raise Exception(f"Gemini blocked response after {max_retries} attempts")
        
        # 응답이 비어있거나 너무 짧은 경우
        if not answer or len(answer.strip()) < 50:
            print(f"⚠️  Gemini response too short ({len(answer) if answer else 0} chars), retrying...")
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
        
        break
    
    time = int(np.max([message.time for message in messages]) + 1)
    return Message(time=time, content=answer, role='assistant')


def _chat_request_openai_compatible(
    messages: List[Message],
    max_tokens: int,
    temperature: float,
    model: str
) -> Message:
    """OpenAI 호환 API 요청 (OpenAI, Groq, Together)"""
    import re
    
    # 모델명에 따라 적절한 API URL과 헤더 선택
    model_lower = model.lower()
    
    if 'gpt' in model_lower:
        # OpenAI
        api_url = 'https://api.openai.com/v1'
        request_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {OPENAI_API_KEY}'
        }
    elif GROQ_API_KEY and ('versatile' in model_lower or 'instant' in model_lower or 'llama' in model_lower):
        # Groq
        api_url = 'https://api.groq.com/openai/v1'
        request_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}'
        }
    elif TOGETHER_API_KEY and 'meta-llama' in model_lower:
        # Together AI
        api_url = 'https://api.together.xyz/v1'
        request_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {TOGETHER_API_KEY}'
        }
    else:
        # 기본값: 현재 설정된 API 사용
        api_url = API_URL
        request_headers = headers
    
    request_body = {
        "model": model,
        "messages": [message.to_chat_completion_query() for message in messages],
        "temperature": temperature,
    }
    
    if max_tokens > 0:
        request_body["max_tokens"] = max_tokens
    
    # Retry 로직 (Rate limit 처리)
    max_retries = 5
    base_wait_time = 10.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f'{api_url}/chat/completions',
                headers=request_headers,
                json=request_body,
                timeout=120
            )
            
            # Rate limit 에러 (429)
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    # 응답에서 대기 시간 추출 시도
                    wait_time = base_wait_time * (2 ** attempt)
                    
                    try:
                        error_data = response.json()
                        message = error_data.get('error', {}).get('message', '')
                        
                        # "try again in X.XXs" 패턴 찾기
                        patterns = [
                            r'try again in (\d+\.?\d*)s',
                            r'Please retry after (\d+\.?\d*) second',
                            r'wait (\d+\.?\d*) second'
                        ]
                        
                        for pattern in patterns:
                            match = re.search(pattern, message, re.IGNORECASE)
                            if match:
                                suggested_wait = float(match.group(1))
                                wait_time = suggested_wait + 2.0
                                print(f"📊 API suggests {suggested_wait:.1f}s, will wait {wait_time:.1f}s")
                                break
                        else:
                            wait_time = base_wait_time
                            print(f"⚠️  Could not parse wait time from message, using base: {wait_time:.1f}s")
                    except Exception as parse_error:
                        wait_time = base_wait_time
                        print(f"⚠️  Error parsing response ({parse_error}), using base: {wait_time:.1f}s")
                    
                    wait_time = max(2.0, min(wait_time, 120.0))
                    
                    print(f"⏳ Rate limit hit. Waiting {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} retries")
                    raise Exception(f"API rate limit exceeded after {max_retries} attempts")
            
            # 다른 에러 처리
            if response.status_code != 200:
                print(f"Error in API call: {response.text}")
                raise Exception(f"API call failed with status {response.status_code}")
            
            # 성공
            break
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (1.5 ** attempt)
                wait_time = max(10.0, min(wait_time, 120.0))
                print(f"⚠️  Request timeout. Waiting {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                import time
                time.sleep(wait_time)
                continue
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (1.5 ** attempt)
                wait_time = max(10.0, min(wait_time, 120.0))
                print(f"⚠️  Request error: {str(e)}. Waiting {wait_time:.1f}s... (attempt {attempt+1}/{max_retries})")
                import time
                time.sleep(wait_time)
                continue
            else:
                raise
    
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
    temperature: float = 0.3,
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


# =============================================================================
# 편의 함수들
# =============================================================================

def create_system_message(content: str) -> Message:
    """시스템 메시지 생성"""
    return Message(time=0, content=content, role="system")


def create_user_message(content: str, time: int = 1) -> Message:
    """사용자 메시지 생성"""
    return Message(time=time, content=content, role="user")


def create_assistant_message(content: str, time: int) -> Message:
    """어시스턴트 메시지 생성"""
    return Message(time=time, content=content, role="assistant")


def get_chat_request_func() -> Callable:
    """
    PersonaResponseValidator에서 사용할 수 있는 chat_request 래퍼 함수 반환
    
    Returns:
        chat_request 호환 함수
    """
    def wrapped_chat_request(messages: List[Dict], temperature: float = 0.3, model: str = None):
        """딕셔너리 형식의 메시지를 Message 객체로 변환하여 호출"""
        msg_objects = []
        for i, msg in enumerate(messages):
            msg_objects.append(Message(
                time=i,
                content=msg.get("content", ""),
                role=msg.get("role", "user")
            ))
        
        return chat_request(msg_objects, temperature=temperature, model=model)
    
    return wrapped_chat_request


if __name__ == "__main__":
    # 테스트
    print("Testing LLM module...")
    print(f"Current API type: {API_TYPE}")
    
    system_msg = create_system_message("You are a helpful assistant.")
    user_msg = create_user_message("Hello! How are you?")
    
    try:
        response = chat_request(
            messages=[system_msg, user_msg],
            temperature=0.3,
            max_tokens=100
        )
        print(f"Response: {response.content}")
    except Exception as e:
        print(f"Test failed: {e}")