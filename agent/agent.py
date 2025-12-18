import memory
import llm


class ConversationalAgent:
    """
    LLM 기반 대화형 에이전트
    과거 대화 기록을 기억하고 컨텍스트를 유지하며 응답
    """
    
    def __init__(self, agent_id: int, memory_window: int = 3, 
                 system_prompt: llm.Message = None, temp: float = 1.0):
        """
        Args:
            agent_id: 에이전트 고유 식별자
            memory_window: 회상할 과거 메시지의 시간 범위
            system_prompt: 초기 시스템 프롬프트 메시지
            temp: LLM 생성 온도 파라미터
        """
        self.agent_id = agent_id
        self.memory_window = memory_window
        self.system_prompt = system_prompt
        self.conversation_memory = memory.Memory()
        self.temp = temp

    def respond(self, incoming_message: llm.Message, **extra_params) -> llm.Message:
        """
        메시지를 받아 응답을 생성하고 대화 기록에 저장
        
        Args:
            incoming_message: 입력 메시지
            **extra_params: LLM 요청에 전달할 추가 파라미터
            
        Returns:
            생성된 응답 메시지
        """
        # 과거 대화 내역 가져오기
        past_messages = self.conversation_memory.retrieve(
            time=incoming_message.time - self.memory_window
        )
        
        # 메시지 체인 구성
        message_chain = []
        
        if self.system_prompt:
            message_chain.append(self.system_prompt)
        
        for msg in past_messages:
            message_chain.append(
                llm.Message(time=msg.time, content=msg.content, role=msg.role)
            )
        
        message_chain.append(incoming_message)
        
        # LLM 응답 생성
        generated_response = llm.chat_request(
            messages=message_chain,
            temperature=self.temp,
            **extra_params
        )
        
        # 대화 기록 저장
        self.conversation_memory.store(message=incoming_message)
        self.conversation_memory.store(message=generated_response)
        
        return generated_response


class ProbabilityAgent:
    """
    토큰 확률 분포를 출력하는 특수 목적 에이전트
    로그 확률값과 함께 응답을 생성
    """
    
    def __init__(self, agent_id: int, memory_window: int = 0,
                 system_prompt: llm.Message = None, temp: float = 1.0):
        """
        Args:
            agent_id: 에이전트 고유 식별자
            memory_window: 회상할 과거 메시지의 시간 범위
            system_prompt: 초기 시스템 프롬프트 메시지
            temp: LLM 생성 온도 파라미터
        """
        self.agent_id = agent_id
        self.memory_window = memory_window
        self.system_prompt = system_prompt
        self.conversation_memory = memory.Memory()
        self.temp = temp
    
    def generate_with_probabilities(self, incoming_message: llm.Message, 
                                    token_limit: int = 10, 
                                    top_logprobs: int = 5, 
                                    **extra_params) -> tuple[dict, any]:
        """
        확률 분포 정보와 함께 응답 생성
        
        Args:
            incoming_message: 입력 메시지
            token_limit: 생성할 최대 토큰 수
            top_logprobs: 반환할 상위 로그 확률 개수
            **extra_params: LLM 요청에 전달할 추가 파라미터
            
        Returns:
            (응답 딕셔너리, 로그 확률 데이터) 튜플
        """
        # 과거 대화 가져오기
        past_messages = self.conversation_memory.retrieve(
            time=incoming_message.time - self.memory_window
        )
        
        # 컨텍스트 구성
        full_context = []
        
        if self.system_prompt:
            full_context.append(self.system_prompt)
            
        full_context.extend(past_messages)
        full_context.append(incoming_message)
        
        # 확률 분포와 함께 생성
        response_data, probability_data = llm.complete_request(
            messages=full_context,
            max_tokens=token_limit,
            temperature=self.temp,
            logprobs=top_logprobs,
            **extra_params
        )
        
        # 메모리에 저장
        self.conversation_memory.store(message=incoming_message)
        self.conversation_memory.store(message=response_data)
        
        return response_data, probability_data