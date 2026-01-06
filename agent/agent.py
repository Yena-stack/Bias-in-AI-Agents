import llm # type: ignore
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import random


@dataclass
class PersonaProfile:
    """WVS 통계정보 기반 페르소나 프로필"""
    
    # 사회적 가치와 태도
    social_values: str  # e.g., "progressive", "conservative", "moderate"
    
    # 종교
    religion: str  # e.g., "Christian", "Buddhist", "None", etc.
    religiosity: str  # e.g., "very religious", "somewhat religious", "not religious"
    
    # 성 문제에 대한 태도
    attitude_homosexuality: int  # 1-10 scale
    attitude_prostitution: int  # 1-10 scale
    attitude_premarital_sex: int  # 1-10 scale
    
    # 정치적 관점
    political_orientation: str  # e.g., "left", "center", "right"
    political_scale: int  # 1-10 scale (1=left, 10=right)
    
    # 인구통계학적 정보
    gender: str  # "male", "female"
    age: int  # actual age
    age_group: str  # e.g., "18-29", "30-49", "50-64", "65+"
    
    # 이민 배경
    immigrant_status: str  # "native", "immigrant", "second_generation"
    
    # 언어
    home_language: str  # e.g., "Korean", "English", etc.
    
    # 결혼 상태
    marital_status: str  # "single", "married", "divorced", "widowed"
    
    # 교육 수준
    education_level: str  # e.g., "elementary", "high_school", "bachelor", "graduate"
    
    # 사회 계층
    social_class: str  # e.g., "lower", "working", "middle", "upper"
    
    # 국가
    country: str  # e.g., "South Korea", "United States", etc.
    
    def to_prompt(self) -> str:
        """페르소나 정보를 프롬프트 형식으로 변환"""
        prompt = f"""You are responding as a person with the following characteristics:

Demographics:
- Country: {self.country}
- Gender: {self.gender}
- Age: {self.age} years old (age group: {self.age_group})
- Education: {self.education_level}
- Social class: {self.social_class}
- Marital status: {self.marital_status}

Background:
- Immigration status: {self.immigrant_status}
- Home language: {self.home_language}

Values and Beliefs:
- Social values: {self.social_values}
- Religion: {self.religion} ({self.religiosity})
- Political orientation: {self.political_orientation} (scale: {self.political_scale}/10)

Attitudes on social issues (1=never justifiable, 10=always justifiable):
- Homosexuality: {self.attitude_homosexuality}/10
- Prostitution: {self.attitude_prostitution}/10
- Premarital sex: {self.attitude_premarital_sex}/10

Please answer the following question from this person's perspective. Respond naturally and authentically based on these characteristics."""
        
        return prompt


class StatelessPersonaAgent:
    """
    WVS 페르소나 기반 stateless 1-shot 응답 에이전트
    대화 히스토리를 기억하지 않고 각 질문에 독립적으로 응답
    """
    
    def __init__(self, persona: PersonaProfile, temp: float = 1.0):
        """
        Args:
            persona: WVS 통계정보 기반 페르소나 프로필
            temp: LLM 생성 온도 파라미터
        """
        self.persona = persona
        self.temp = temp
    
    def respond_to_ethical_question(self, question: str, 
                                    scale_min: int = 1, 
                                    scale_max: int = 10,
                                    **extra_params) -> llm.Message:
        """
        윤리적 질문에 대해 1-shot으로 응답 생성
        
        Args:
            question: 윤리적 이슈에 대한 질문
            scale_min: 응답 척도 최소값
            scale_max: 응답 척도 최대값
            **extra_params: LLM 요청에 전달할 추가 파라미터
            
        Returns:
            생성된 응답 메시지
        """
        # 시스템 프롬프트 구성
        system_message = llm.Message(
            time=0,
            content=self.persona.to_prompt(),
            role="system"
        )
        
        # 사용자 질문 구성
        user_question = f"""{question}

Please rate on a scale from {scale_min} to {scale_max}, where:
{scale_min} = Never justifiable
{scale_max} = Always justifiable

Provide your rating and briefly explain your reasoning based on your background and values."""
        
        user_message = llm.Message(
            time=1,
            content=user_question,
            role="user"
        )
        
        # LLM 응답 생성 (stateless)
        response = llm.chat_request(
            messages=[system_message, user_message],
            temperature=self.temp,
            **extra_params
        )
        
        return response


class WVSQuestionSet:
    """WVS 윤리 이슈 질문 세트"""
    
    ETHICAL_QUESTIONS = {
        "homosexuality": "How would you rate the justifiability of homosexuality?",
        "abortion": "How would you rate the justifiability of abortion?",
        "divorce": "How would you rate the justifiability of divorce?",
        "suicide": "How would you rate the justifiability of suicide?",
        "euthanasia": "How would you rate the justifiability of euthanasia?",
        "death_penalty": "How would you rate the justifiability of the death penalty?"
    }
    
    @classmethod
    def get_question(cls, topic: str) -> str:
        """특정 주제의 질문 반환"""
        return cls.ETHICAL_QUESTIONS.get(topic, "")
    
    @classmethod
    def get_all_questions(cls) -> Dict[str, str]:
        """모든 질문 반환"""
        return cls.ETHICAL_QUESTIONS.copy()
    
    @classmethod
    def get_single_turn_questions(cls) -> str:
        """모든 질문을 single turn 형식으로 반환"""
        questions = []
        for i, (topic, question) in enumerate(cls.ETHICAL_QUESTIONS.items(), 1):
            questions.append(f"{i}. {topic.upper()}: {question}")
        
        return "\n\n".join(questions)


class PersonaGenerator:
    """WVS 통계정보를 랜덤 샘플링하여 페르소나 생성"""
    
    def __init__(self, country: str, seed: Optional[int] = None):
        """
        Args:
            country: 대상 국가
            seed: 랜덤 시드 (재현가능성 확보)
        """
        self.country = country
        self.rng = random.Random(seed) if seed is not None else random.Random()
    
    def generate_persona(self, **fixed_attributes) -> PersonaProfile:
        """
        랜덤 샘플링으로 페르소나 생성
        
        Args:
            **fixed_attributes: 고정할 특정 속성 (예: gender="female")
            
        Returns:
            생성된 페르소나 프로필
        """
        # 기본값 설정 (실제로는 WVS 통계 분포에서 샘플링해야 함)
        
        persona_data = {
            "country": self.country,
            "gender": fixed_attributes.get("gender", self.rng.choice(["male", "female"])),
            "age": fixed_attributes.get("age", self.rng.randint(18, 80)),
            "religion": fixed_attributes.get("religion", 
                                            self.rng.choice(["Christian", "Buddhist", "None", "Muslim", "Other"])),
            "religiosity": fixed_attributes.get("religiosity",
                                               self.rng.choice(["very religious", "somewhat religious", "not religious"])),
            "education_level": fixed_attributes.get("education_level",
                                                   self.rng.choice(["elementary", "high_school", "bachelor", "graduate"])),
            "social_class": fixed_attributes.get("social_class",
                                                self.rng.choice(["lower", "working", "middle", "upper"])),
            "political_orientation": fixed_attributes.get("political_orientation",
                                                         self.rng.choice(["left", "center-left", "center", "center-right", "right"])),
            "political_scale": fixed_attributes.get("political_scale", self.rng.randint(1, 10)),
            "social_values": fixed_attributes.get("social_values",
                                                 self.rng.choice(["progressive", "moderate", "conservative"])),
            "marital_status": fixed_attributes.get("marital_status",
                                                  self.rng.choice(["single", "married", "divorced", "widowed"])),
            "immigrant_status": fixed_attributes.get("immigrant_status",
                                                    self.rng.choice(["native", "immigrant", "second_generation"])),
            "home_language": fixed_attributes.get("home_language", "Korean" if self.country == "South Korea" else "English"),
            "attitude_homosexuality": fixed_attributes.get("attitude_homosexuality", self.rng.randint(1, 10)),
            "attitude_prostitution": fixed_attributes.get("attitude_prostitution", self.rng.randint(1, 10)),
            "attitude_premarital_sex": fixed_attributes.get("attitude_premarital_sex", self.rng.randint(1, 10)),
        }
        
        # 나이 그룹 자동 계산
        age = persona_data["age"]
        if age < 30:
            age_group = "18-29"
        elif age < 50:
            age_group = "30-49"
        elif age < 65:
            age_group = "50-64"
        else:
            age_group = "65+"
        
        persona_data["age_group"] = age_group
        
        return PersonaProfile(**persona_data)
    
    def generate_multiple_personas(self, n: int, **fixed_attributes) -> List[PersonaProfile]:
        """
        여러 개의 페르소나 생성
        
        Args:
            n: 생성할 페르소나 개수
            **fixed_attributes: 모든 페르소나에 공통으로 적용할 속성
            
        Returns:
            생성된 페르소나 리스트
        """
        return [self.generate_persona(**fixed_attributes) for _ in range(n)]


# 사용 예시
if __name__ == "__main__":
    # 페르소나 생성기 초기화
    generator = PersonaGenerator(country="South Korea", seed=42)
    
    # 특정 조건의 페르소나 생성 (예: 보수 성향 고학력 30대 한국인 여성)
    persona = generator.generate_persona(
        gender="female",
        age=35,
        education_level="graduate",
        social_values="conservative",
        political_orientation="right"
    )
    
    # 에이전트 생성
    agent = StatelessPersonaAgent(persona=persona, temp=1.0)
    
    # 단일 질문 응답
    question = WVSQuestionSet.get_question("homosexuality")
    response = agent.respond_to_ethical_question(question)
    
    print(f"Question: {question}")
    print(f"Response: {response.content}")
    
    # 또는 single turn으로 모든 질문 한번에
    all_questions = WVSQuestionSet.get_single_turn_questions()
    response_all = agent.respond_to_ethical_question(all_questions)
    print(f"\nAll questions response:\n{response_all.content}")