"""
WVS 윤리 이슈 실험 패키지
LLM 페르소나 기반 설문조사 시뮬레이션
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# llm 모듈에서 핵심 클래스/함수 import
from llm import (
    Message,
    chat_request,
    complete_request,
    create_system_message,
    create_user_message,
    create_assistant_message
)

# agent 모듈에서 핵심 클래스 import
from agent import (
    WVSPersonaProfile,
    StatelessPersonaAgent,
    WVSEthicalQuestions,
    WVSPersonaGenerator
)

__all__ = [
    # llm module
    'Message',
    'chat_request',
    'complete_request',
    'create_system_message',
    'create_user_message',
    'create_assistant_message',
    
    # agent module
    'WVSPersonaProfile',
    'StatelessPersonaAgent',
    'WVSEthicalQuestions',
    'WVSPersonaGenerator',
]