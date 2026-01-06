import llm
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import random


@dataclass
class WVSPersonaProfile:
    """WVS Wave 7 설문지 기반 페르소나 프로필"""
    
    # === 국가 ===
    country: str  # "United States", "Germany", "Great Britain", "Japan", "South Korea", "India", "Netherlands"
    
    # === 인구통계학적 정보 (DEMOGRAPHICS) ===
    # Q260: 성별
    gender: str  # "Male", "Female"
    
    # Q262: 나이
    age: int
    
    # Q263-265: 이민 배경
    born_in_country: bool  # Q263: True if born in country
    mother_immigrant: bool  # Q264: True if mother is immigrant
    father_immigrant: bool  # Q265: True if father is immigrant
    
    # Q269: 시민권
    is_citizen: bool
    
    # Q272: 가정에서 사용하는 언어
    home_language: str
    
    # Q273: 결혼 상태
    marital_status: str  # "Married", "Living together as married", "Divorced", "Separated", "Widowed", "Single"
    
    # Q275: 교육 수준 (respondent)
    education_level: int  # 0-8 (ISCED scale)
    
    # Q276: 배우자 교육 수준
    spouse_education: Optional[int]  # 0-8 or None
    
    # Q277: 어머니 교육 수준
    mother_education: Optional[int]  # 0-8 or None
    
    # Q278: 아버지 교육 수준
    father_education: Optional[int]  # 0-8 or None
    
    # Q287: 사회 계층
    social_class: str  # "Upper class", "Upper middle class", "Lower middle class", "Working class", "Lower class"
    
    # === 가치 정보 (SOCIAL VALUES) ===
    # Q1-Q6: 삶에서 중요한 것들 (1=Very important, 4=Not at all important)
    importance_family: int  # Q1
    importance_friends: int  # Q2
    importance_leisure: int  # Q3
    importance_politics: int  # Q4
    importance_work: int  # Q5
    importance_religion: int  # Q6
    
    # === 종교 관련 (RELIGIOUS VALUES) ===
    # Q164: 신의 중요성 (1=Not at all important, 10=Very important)
    importance_god: int
    
    # Q171: 종교 예배 참석 빈도 (1=More than once a week, 7=Never)
    religious_service_attendance: int
    
    # Q172: 기도 빈도 (1=Several times a day, 8=Never)
    prayer_frequency: int
    
    # Q173: 종교인 여부
    religiosity: str  # "A religious person", "Not a religious person", "An atheist"
    
    # === 이웃으로 원하지 않는 사람들 (SOCIAL ATTITUDES) ===
    # Q22: 동성애자를 이웃으로 원하지 않음 (1=Mentioned, 2=Not mentioned)
    reject_homosexual_neighbor: bool
    
    # === 성 역할 및 가족 가치 ===
    # Q36: 동성애 커플의 부모 자격 (1=Agree strongly, 5=Disagree strongly)
    homosexual_parents_opinion: int
    
    # === 윤리적 가치 (ETHICAL VALUES) ===
    # Q182-Q195: 정당화 가능 여부 (1=Never justifiable, 10=Always justifiable)
    justifiability_homosexuality: int  # Q182
    justifiability_prostitution: int  # Q183
    justifiability_abortion: int  # Q184
    justifiability_divorce: int  # Q185
    justifiability_premarital_sex: int  # Q186 (sex before marriage)
    justifiability_suicide: int  # Q187
    justifiability_euthanasia: int  # Q188
    justifiability_casual_sex: int  # Q193
    justifiability_death_penalty: int  # Q195
    
    # === 정치 성향 (POLITICAL CULTURE) ===
    # Q240: 좌우 정치 성향 (1=Left, 10=Right)
    political_left_right: int
    
    # Q250: 민주주의 중요성 (1=Not at all important, 10=Absolutely important)
    importance_democracy: int
    
    def to_prompt(self) -> str:
        """페르소나 정보를 프롬프트 형식으로 변환"""
        
        # 교육 수준 매핑
        education_map = {
            0: "No education",
            1: "Primary education",
            2: "Lower secondary education",
            3: "Upper secondary education",
            4: "Post-secondary non-tertiary education",
            5: "Short-cycle tertiary education",
            6: "Bachelor degree",
            7: "Master degree",
            8: "Doctoral degree"
        }
        
        # 이민 배경 설명
        immigration_status = "Native born" if self.born_in_country else "Immigrant"
        parent_immigration = []
        if self.mother_immigrant:
            parent_immigration.append("mother is an immigrant")
        if self.father_immigrant:
            parent_immigration.append("father is an immigrant")
        parent_info = ", ".join(parent_immigration) if parent_immigration else "both parents are native"
        
        # 종교 참석 빈도 매핑
        religious_attendance_map = {
            1: "More than once a week",
            2: "Once a week",
            3: "Once a month",
            4: "Only on special holy days",
            5: "Once a year",
            6: "Less often",
            7: "Never, practically never"
        }
        
        # 기도 빈도 매핑
        prayer_freq_map = {
            1: "Several times a day",
            2: "Once a day",
            3: "Several times each week",
            4: "Only when attending religious services",
            5: "Only on special holy days",
            6: "Once a year",
            7: "Less often",
            8: "Never, practically never"
        }
        
        prompt = f"""You are responding as a person with the following characteristics:

BASIC DEMOGRAPHICS:
- Country: {self.country}
- Gender: {self.gender}
- Age: {self.age} years old
- Citizenship: {"Citizen" if self.is_citizen else "Non-citizen"}
- Immigration status: {immigration_status} ({parent_info})
- Home language: {self.home_language}
- Marital status: {self.marital_status}
- Education level: {education_map.get(self.education_level, "Unknown")}
- Social class: {self.social_class}

IMPORTANCE OF LIFE DOMAINS (scale 1-4, where 1=Very important, 4=Not at all important):
- Family: {self.importance_family}
- Friends: {self.importance_friends}
- Leisure time: {self.importance_leisure}
- Politics: {self.importance_politics}
- Work: {self.importance_work}
- Religion: {self.importance_religion}

RELIGIOUS VALUES:
- Importance of God in life: {self.importance_god}/10 (1=Not important, 10=Very important)
- Religious service attendance: {religious_attendance_map.get(self.religious_service_attendance, "Unknown")}
- Prayer frequency: {prayer_freq_map.get(self.prayer_frequency, "Unknown")}
- Self-identification: {self.religiosity}

SOCIAL ATTITUDES:
- Would reject homosexuals as neighbors: {"Yes" if self.reject_homosexual_neighbor else "No"}
- Opinion on homosexual couples as parents: {self.homosexual_parents_opinion}/5 (1=Agree strongly, 5=Disagree strongly)

ETHICAL VALUES - Justifiability (scale 1-10, where 1=Never justifiable, 10=Always justifiable):
- Homosexuality: {self.justifiability_homosexuality}/10
- Prostitution: {self.justifiability_prostitution}/10
- Abortion: {self.justifiability_abortion}/10
- Divorce: {self.justifiability_divorce}/10
- Premarital sex: {self.justifiability_premarital_sex}/10
- Suicide: {self.justifiability_suicide}/10
- Euthanasia: {self.justifiability_euthanasia}/10
- Casual sex: {self.justifiability_casual_sex}/10
- Death penalty: {self.justifiability_death_penalty}/10

POLITICAL ORIENTATION:
- Left-Right scale: {self.political_left_right}/10 (1=Left, 10=Right)
- Importance of democracy: {self.importance_democracy}/10 (1=Not important, 10=Absolutely important)

Please answer the following question from this person's perspective, staying true to their values, beliefs, and background."""
        
        return prompt


class StatelessPersonaAgent:
    """
    WVS 페르소나 기반 stateless 1-shot 응답 에이전트
    대화 히스토리를 기억하지 않고 각 질문에 독립적으로 응답
    """
    
    def __init__(self, persona: WVSPersonaProfile, temp: float = 1.0):
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
            question: 윤리적 이슈에 대한 질문 (WVS Q177-Q195 형식)
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
        
        # WVS 형식의 질문 구성
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


class WVSEthicalQuestions:
    """WVS Wave 7 윤리 이슈 질문 세트 (Q182-Q195)"""
    
    # 실험계획서에서 선정한 6가지 주제
    ETHICAL_QUESTIONS = {
        "homosexuality": "Please tell me whether you think homosexuality can always be justified, never be justified, or something in between.",
        "abortion": "Please tell me whether you think abortion can always be justified, never be justified, or something in between.",
        "divorce": "Please tell me whether you think divorce can always be justified, never be justified, or something in between.",
        "suicide": "Please tell me whether you think suicide can always be justified, never be justified, or something in between.",
        "euthanasia": "Please tell me whether you think euthanasia (ending the life of the incurably sick) can always be justified, never be justified, or something in between.",
        "prostitution": "Please tell me whether you think prostitution can always be justified, never be justified, or something in between.",
        "death_penalty": "Please tell me whether you think the death penalty can always be justified, never be justified, or something in between."
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


class WVSPersonaGenerator:
    """WVS 통계정보를 랜덤 샘플링하여 페르소나 생성"""
    
    COUNTRIES = ["United States", "Germany", "Great Britain", "Japan", "South Korea", "India", "Netherlands"]
    
    def __init__(self, country: str, seed: Optional[int] = None):
        """
        Args:
            country: 대상 국가 (COUNTRIES 중 하나)
            seed: 랜덤 시드 (재현가능성 확보)
        """
        if country not in self.COUNTRIES:
            raise ValueError(f"Country must be one of {self.COUNTRIES}")
        
        self.country = country
        self.rng = random.Random(seed) if seed is not None else random.Random()
    
    def generate_persona(self, **fixed_attributes) -> WVSPersonaProfile:
        """
        랜덤 샘플링으로 페르소나 생성
        
        Args:
            **fixed_attributes: 고정할 특정 속성
            
        Returns:
            생성된 페르소나 프로필
        """
        # 기본 언어 매핑
        language_map = {
            "United States": "English",
            "Germany": "German",
            "Great Britain": "English",
            "Japan": "Japanese",
            "South Korea": "Korean",
            "India": self.rng.choice(["Hindi", "English", "Bengali", "Tamil"]),
            "Netherlands": "Dutch"
        }
        
        persona_data = {
            "country": self.country,
            
            # 인구통계학적 정보
            "gender": fixed_attributes.get("gender", self.rng.choice(["Male", "Female"])),
            "age": fixed_attributes.get("age", self.rng.randint(18, 85)),
            "born_in_country": fixed_attributes.get("born_in_country", self.rng.random() > 0.1),
            "mother_immigrant": fixed_attributes.get("mother_immigrant", self.rng.random() < 0.15),
            "father_immigrant": fixed_attributes.get("father_immigrant", self.rng.random() < 0.15),
            "is_citizen": fixed_attributes.get("is_citizen", self.rng.random() > 0.05),
            "home_language": fixed_attributes.get("home_language", language_map[self.country]),
            "marital_status": fixed_attributes.get("marital_status", 
                self.rng.choice(["Married", "Living together as married", "Divorced", "Separated", "Widowed", "Single"])),
            "education_level": fixed_attributes.get("education_level", self.rng.randint(1, 7)),
            "spouse_education": fixed_attributes.get("spouse_education", 
                self.rng.randint(1, 7) if self.rng.random() > 0.3 else None),
            "mother_education": fixed_attributes.get("mother_education", self.rng.randint(0, 6)),
            "father_education": fixed_attributes.get("father_education", self.rng.randint(0, 6)),
            "social_class": fixed_attributes.get("social_class",
                self.rng.choice(["Upper class", "Upper middle class", "Lower middle class", "Working class", "Lower class"])),
            
            # 삶의 중요 영역
            "importance_family": fixed_attributes.get("importance_family", self.rng.randint(1, 4)),
            "importance_friends": fixed_attributes.get("importance_friends", self.rng.randint(1, 4)),
            "importance_leisure": fixed_attributes.get("importance_leisure", self.rng.randint(1, 4)),
            "importance_politics": fixed_attributes.get("importance_politics", self.rng.randint(1, 4)),
            "importance_work": fixed_attributes.get("importance_work", self.rng.randint(1, 4)),
            "importance_religion": fixed_attributes.get("importance_religion", self.rng.randint(1, 4)),
            
            # 종교 관련
            "importance_god": fixed_attributes.get("importance_god", self.rng.randint(1, 10)),
            "religious_service_attendance": fixed_attributes.get("religious_service_attendance", self.rng.randint(1, 7)),
            "prayer_frequency": fixed_attributes.get("prayer_frequency", self.rng.randint(1, 8)),
            "religiosity": fixed_attributes.get("religiosity",
                self.rng.choice(["A religious person", "Not a religious person", "An atheist"])),
            
            # 사회적 태도
            "reject_homosexual_neighbor": fixed_attributes.get("reject_homosexual_neighbor", self.rng.random() < 0.3),
            "homosexual_parents_opinion": fixed_attributes.get("homosexual_parents_opinion", self.rng.randint(1, 5)),
            
            # 윤리적 가치 (1-10 척도)
            "justifiability_homosexuality": fixed_attributes.get("justifiability_homosexuality", self.rng.randint(1, 10)),
            "justifiability_prostitution": fixed_attributes.get("justifiability_prostitution", self.rng.randint(1, 10)),
            "justifiability_abortion": fixed_attributes.get("justifiability_abortion", self.rng.randint(1, 10)),
            "justifiability_divorce": fixed_attributes.get("justifiability_divorce", self.rng.randint(1, 10)),
            "justifiability_premarital_sex": fixed_attributes.get("justifiability_premarital_sex", self.rng.randint(1, 10)),
            "justifiability_suicide": fixed_attributes.get("justifiability_suicide", self.rng.randint(1, 10)),
            "justifiability_euthanasia": fixed_attributes.get("justifiability_euthanasia", self.rng.randint(1, 10)),
            "justifiability_casual_sex": fixed_attributes.get("justifiability_casual_sex", self.rng.randint(1, 10)),
            "justifiability_death_penalty": fixed_attributes.get("justifiability_death_penalty", self.rng.randint(1, 10)),
            
            # 정치 성향
            "political_left_right": fixed_attributes.get("political_left_right", self.rng.randint(1, 10)),
            "importance_democracy": fixed_attributes.get("importance_democracy", self.rng.randint(1, 10)),
        }
        
        return WVSPersonaProfile(**persona_data)
    
    def generate_multiple_personas(self, n: int, **fixed_attributes) -> List[WVSPersonaProfile]:
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
    # 한국 페르소나 생성기 초기화
    generator = WVSPersonaGenerator(country="South Korea", seed=42)
    
    # 특정 조건의 페르소나 생성 (예: 보수 성향 고학력 30대 한국인 여성)
    persona = generator.generate_persona(
        gender="Female",
        age=35,
        education_level=7,  # Master degree
        social_class="Upper middle class",
        political_left_right=7,  # Conservative
        justifiability_homosexuality=3  # Low acceptance
    )
    
    # 에이전트 생성
    agent = StatelessPersonaAgent(persona=persona, temp=1.0)
    
    # 단일 질문 응답
    question = WVSEthicalQuestions.get_question("homosexuality")
    response = agent.respond_to_ethical_question(question)
    
    print(f"Question: {question}")
    print(f"Response: {response.content}")
    
    # 모든 국가에서 페르소나 생성 예시
    print("\n=== Generating personas from all countries ===")
    for country in WVSPersonaGenerator.COUNTRIES:
        gen = WVSPersonaGenerator(country=country, seed=42)
        personas = gen.generate_multiple_personas(n=3)
        print(f"\n{country}: Generated {len(personas)} personas")