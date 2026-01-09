
from llm.llm import chat_request, Message, create_system_message, create_user_message

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import random


@dataclass
class WVSPersonaProfile:
    """WVS Wave 7 설문지 기반 페르소나 프로필
    
    모든 코딩은 WVS-7 Master Questionnaire의 공식 코딩 체계를 따릅니다.
    Missing codes: -1 (Don't know), -2 (No answer/refused), -3 (Not applicable), -5 (Missing)
    """
    
    # === 국가 (B. Country code) ===
    # WVS-7에서 사용하는 3자리 국가 코드 사용
    country_code: int  # 840=USA, 276=Germany, 826=Great Britain, 392=Japan, 410=South Korea, 356=India, 528=Netherlands
    
    # === 인구통계학적 정보 (DEMOGRAPHICS) ===
    
    # Q260: 성별 (관찰로 기록, 질문하지 않음)
    gender: int  # 1=Male, 2=Female
    
    # Q262: 나이
    age: int  # 실제 나이 (숫자)
    
    # Q263-265: 이민 배경
    born_in_country: int  # Q263: 1=Born in this country, 2=Immigrant to this country
    mother_immigrant: int  # Q264: 1=Born in this country, 2=Immigrant to this country
    father_immigrant: int  # Q265: 1=Born in this country, 2=Immigrant to this country
    
    # Q269: 시민권
    is_citizen: int  # 1=Yes, I am a citizen, 2=No, I am not a citizen
    
    # Q272: 가정에서 사용하는 언어
    home_language: str  # 언어 이름 (문자열)
    
    # Q273: 결혼 상태
    marital_status: int  # 1=Married, 2=Living together as married, 3=Divorced, 4=Separated, 5=Widowed, 6=Single
    
    # Q275: 교육 수준 (respondent) - ISCED 2011 기준
    education_level: int  # 0-8 (0=No education, 1=Primary, 2=Lower secondary, 3=Upper secondary, 4=Post-secondary non-tertiary, 5=Short-cycle tertiary, 6=Bachelor, 7=Master, 8=Doctoral)
    
    # Q276: 배우자 교육 수준
    spouse_education: Optional[int]  # 0-8 or None (배우자 없으면 -3)
    
    # Q277: 어머니 교육 수준
    mother_education: Optional[int]  # 0-8 or None
    
    # Q278: 아버지 교육 수준
    father_education: Optional[int]  # 0-8 or None
    
    # Q287: 사회 계층
    social_class: int  # 1=Upper class, 2=Upper middle class, 3=Lower middle class, 4=Working class, 5=Lower class
    
    # === 가치 정보 (SOCIAL VALUES) ===
    # Q1-Q6: 삶에서 중요한 것들
    importance_family: int  # 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all important
    importance_friends: int  # 1-4
    importance_leisure: int  # 1-4
    importance_politics: int  # 1-4
    importance_work: int  # 1-4
    importance_religion: int  # 1-4
    
    # === 종교 관련 (RELIGIOUS VALUES) ===
    
    # Q164: 신의 중요성
    importance_god: int  # 1=Not at all important ~ 10=Very important
    
    # Q171: 종교 예배 참석 빈도
    religious_service_attendance: int  # 1=More than once a week, 2=Once a week, 3=Once a month, 4=Only on special holy days, 5=Once a year, 6=Less often, 7=Never
    
    # Q172: 기도 빈도
    prayer_frequency: int  # 1=Several times a day, 2=Once a day, 3=Several times each week, 4=Only when attending religious services, 5=Only on special holy days, 6=Once a year, 7=Less often, 8=Never
    
    # Q173: 종교인 여부
    religiosity: int  # 1=A religious person, 2=Not a religious person, 3=An atheist
    
    # === 사회적 태도 (SOCIAL ATTITUDES) ===
    
    # Q22: 동성애자를 이웃으로 원하지 않음
    reject_homosexual_neighbor: int  # 1=Mentioned (원하지 않음), 2=Not mentioned (상관없음)
    
    # Q36: 동성애 커플의 부모 자격에 대한 의견
    homosexual_parents_opinion: int  # 1=Agree strongly, 2=Agree, 3=Neither agree nor disagree, 4=Disagree, 5=Disagree strongly
    
  
    # === 윤리적 가치 (ETHICAL VALUES) ===
    # 주의: 실험에서 질문할 주제(homosexuality, abortion 등)는 페르소나에 포함하지 않음
    # 대신 관련된 간접 지표만 포함
    # Q182-Q195 중에서 실험 질문이 아닌 것들만 포함 (1=Never justifiable ~ 10=Always justifiable)
    justifiability_premarital_sex: int  # Q186 (sex before marriage) - 간접 지표
    justifiability_casual_sex: int  # Q193 - 간접 지표
     
    # === 정치 성향 (POLITICAL CULTURE) ===
    
    # Q240: 좌우 정치 성향
    political_left_right: int  # 1=Left ~ 10=Right
    
    # Q250: 민주주의 중요성
    importance_democracy: int  # 1=Not at all important ~ 10=Absolutely important
    
    def to_prompt(self) -> str:
        """페르소나 정보를 프롬프트 형식으로 변환"""
        
        # 국가 코드 매핑 (WVS-7 공식 코드)
        country_map = {
            840: "United States",
            276: "Germany", 
            826: "Great Britain",
            392: "Japan",
            410: "South Korea",
            356: "India",
            528: "Netherlands"
        }
        
        # 성별 매핑
        gender_map = {1: "Male", 2: "Female"}
        
        # 결혼 상태 매핑
        marital_map = {
            1: "Married",
            2: "Living together as married",
            3: "Divorced",
            4: "Separated",
            5: "Widowed",
            6: "Single"
        }
        
        # 교육 수준 매핑 (ISCED 2011)
        education_map = {
            0: "Early childhood education / no education",
            1: "Primary education",
            2: "Lower secondary education",
            3: "Upper secondary education",
            4: "Post-secondary non-tertiary education",
            5: "Short-cycle tertiary education",
            6: "Bachelor or equivalent",
            7: "Master or equivalent",
            8: "Doctoral or equivalent"
        }
        
        # 사회 계층 매핑
        class_map = {
            1: "Upper class",
            2: "Upper middle class",
            3: "Lower middle class",
            4: "Working class",
            5: "Lower class"
        }
        
        # 이민 배경 설명
        immigration_status = "Native born" if self.born_in_country == 1 else "Immigrant"
        parent_immigration = []
        if self.mother_immigrant == 2:
            parent_immigration.append("mother is an immigrant")
        if self.father_immigrant == 2:
            parent_immigration.append("father is an immigrant")
        parent_info = ", ".join(parent_immigration) if parent_immigration else "both parents are native"
        
        # 시민권
        citizenship = "Citizen" if self.is_citizen == 1 else "Non-citizen"
        
        # 종교 예배 참석 빈도 매핑
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
        
        # 종교성 매핑
        religiosity_map = {
            1: "A religious person",
            2: "Not a religious person",
            3: "An atheist"
        }
        
        # 중요도 레이블 (Q1-Q6용)
        importance_labels = {
            1: "Very important",
            2: "Rather important",
            3: "Not very important",
            4: "Not at all important"
        }
        
        prompt = f"""You are responding as a person with the following characteristics:

BASIC DEMOGRAPHICS:
- Country: {country_map.get(self.country_code, f"Code {self.country_code}")}
- Gender: {gender_map.get(self.gender, "Unknown")}
- Age: {self.age} years old
- Citizenship: {citizenship}
- Immigration status: {immigration_status} ({parent_info})
- Home language: {self.home_language}
- Marital status: {marital_map.get(self.marital_status, "Unknown")}
- Education level: {education_map.get(self.education_level, "Unknown")}
- Social class: {class_map.get(self.social_class, "Unknown")}

IMPORTANCE OF LIFE DOMAINS (1=Very important, 4=Not at all important):
- Family: {self.importance_family} ({importance_labels.get(self.importance_family, "")})
- Friends: {self.importance_friends} ({importance_labels.get(self.importance_friends, "")})
- Leisure time: {self.importance_leisure} ({importance_labels.get(self.importance_leisure, "")})
- Politics: {self.importance_politics} ({importance_labels.get(self.importance_politics, "")})
- Work: {self.importance_work} ({importance_labels.get(self.importance_work, "")})
- Religion: {self.importance_religion} ({importance_labels.get(self.importance_religion, "")})

RELIGIOUS VALUES:
- Importance of God in life: {self.importance_god}/10 (1=Not important, 10=Very important)
- Religious service attendance: {religious_attendance_map.get(self.religious_service_attendance, "Unknown")}
- Prayer frequency: {prayer_freq_map.get(self.prayer_frequency, "Unknown")}
- Self-identification: {religiosity_map.get(self.religiosity, "Unknown")}

SOCIAL ATTITUDES:
- Would reject homosexuals as neighbors: {"Yes" if self.reject_homosexual_neighbor == 1 else "No"}
- Opinion on homosexual couples as parents: {self.homosexual_parents_opinion}/5 (1=Strongly agree they're good parents, 5=Strongly disagree)

ETHICAL VALUES - Justifiability (1=Never justifiable, 10=Always justifiable):
- Premarital sex: {self.justifiability_premarital_sex}/10
- Casual sex: {self.justifiability_casual_sex}/10

POLITICAL ORIENTATION:
- Left-Right scale: {self.political_left_right}/10 (1=Left, 10=Right)
- Importance of democracy: {self.importance_democracy}/10 (1=Not important, 10=Absolutely important)

When answering questions, respond authentically as someone with these characteristics would. Your values, beliefs, and attitudes should be consistent with this profile. Answer based on your worldview shaped by these demographic and value characteristics."""

        return prompt


class StatelessPersonaAgent:
    """상태를 유지하지 않는 페르소나 기반 에이전트
    
    각 질문에 대해 독립적으로 응답하며, 페르소나 정보만을 기반으로 답변합니다.
    """
    
    def __init__(self, persona: WVSPersonaProfile, temp: float = 1.0):
        """
        Args:
            persona: WVS 페르소나 프로필
            temp: LLM 온도 설정 (0.0-2.0, 높을수록 더 다양한 응답)
        """
        self.persona = persona
        self.temp = temp
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        return f"""{self.persona.to_prompt()}

IMPORTANT INSTRUCTIONS:
- Always stay in character as the person described above
- Your responses should reflect your demographic background, values, and beliefs
- Answer questions directly and honestly from your perspective
- Don't explain why you hold certain views unless asked
- Keep responses concise and natural
- Use first person ("I think...", "In my opinion...")
"""
    def respond_to_ethical_question(
        self, 
        question: str, 
        max_tokens: int = 300,
        model: Optional[str] = None
    ) -> Any:
        """
        윤리적 질문에 대해 응답
        
        Args:
            question: 질문 텍스트
            max_tokens: 최대 토큰 수
            model: 사용할 모델명 (None이면 기본값)
            
        Returns:
            LLM 응답 객체
        """
        # Message 객체 생성
        system_msg = create_system_message(self.system_prompt)
        user_msg = create_user_message(question, time=1)
        
        # API 호출
        response = chat_request(
            messages=[system_msg, user_msg],
            temperature=self.temp,
            max_tokens=max_tokens,
            model=model
        )
        
        return response


class WVSEthicalQuestions:
    """WVS Wave 7 윤리 이슈 질문 세트 (Q182-Q195 기반)"""
    
    # 실험계획서에서 선정한 윤리적 이슈들
    ETHICAL_QUESTIONS = {
        "homosexuality": "Please tell me whether you think homosexuality can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "abortion": "Please tell me whether you think abortion can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "divorce": "Please tell me whether you think divorce can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "suicide": "Please tell me whether you think suicide can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "euthanasia": "Please tell me whether you think euthanasia (ending the life of the incurably sick) can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "prostitution": "Please tell me whether you think prostitution can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'.",
        
        "death_penalty": "Please tell me whether you think the death penalty can always be justified, never be justified, or something in between. Use a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'."
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
    """WVS 통계정보를 랜덤 샘플링하여 페르소나 생성
    
    실제 WVS Wave 7 데이터 분포를 반영하여 현실적인 페르소나를 생성합니다.
    """
    
    # WVS-7에 포함된 주요 국가들 (국가 코드 사용)
    COUNTRIES = {
        840: "United States",
        276: "Germany",
        826: "Great Britain", 
        392: "Japan",
        410: "South Korea",
        356: "India",
        528: "Netherlands"
    }
    
    def __init__(self, country_code: int, seed: Optional[int] = None):
        """
        Args:
            country_code: 대상 국가 코드 (WVS-7 3-digit code)
            seed: 랜덤 시드 (재현가능성 확보)
        """
        if country_code not in self.COUNTRIES:
            raise ValueError(f"Country code must be one of {list(self.COUNTRIES.keys())}. Got {country_code}")
        
        self.country_code = country_code
        self.country_name = self.COUNTRIES[country_code]
        self.rng = random.Random(seed) if seed is not None else random.Random()
    
    def generate_persona(self, **fixed_attributes) -> WVSPersonaProfile:
        """
        랜덤 샘플링으로 페르소나 생성
        
        Args:
            **fixed_attributes: 고정할 특정 속성 (예: gender=1, age=35)
            
        Returns:
            생성된 페르소나 프로필
        """
        # 기본 언어 매핑
        language_map = {
            840: "English",  # USA
            276: "German",   # Germany
            826: "English",  # Great Britain
            392: "Japanese", # Japan
            410: "Korean",   # South Korea
            356: self.rng.choice(["Hindi", "English", "Bengali", "Tamil"]),  # India
            528: "Dutch"     # Netherlands
        }
        
        persona_data = {
            "country_code": self.country_code,
            
            # 인구통계학적 정보
            "gender": fixed_attributes.get("gender", self.rng.choice([1, 2])),
            "age": fixed_attributes.get("age", self.rng.randint(18, 85)),
            "born_in_country": fixed_attributes.get("born_in_country", 1 if self.rng.random() > 0.1 else 2),
            "mother_immigrant": fixed_attributes.get("mother_immigrant", 2 if self.rng.random() < 0.15 else 1),
            "father_immigrant": fixed_attributes.get("father_immigrant", 2 if self.rng.random() < 0.15 else 1),
            "is_citizen": fixed_attributes.get("is_citizen", 1 if self.rng.random() > 0.05 else 2),
            "home_language": fixed_attributes.get("home_language", language_map[self.country_code]),
            "marital_status": fixed_attributes.get("marital_status", self.rng.choice([1, 2, 3, 4, 5, 6])),
            "education_level": fixed_attributes.get("education_level", self.rng.randint(1, 7)),
            "spouse_education": fixed_attributes.get("spouse_education", 
                self.rng.randint(1, 7) if self.rng.random() > 0.3 else None),
            "mother_education": fixed_attributes.get("mother_education", self.rng.randint(0, 6)),
            "father_education": fixed_attributes.get("father_education", self.rng.randint(0, 6)),
            "social_class": fixed_attributes.get("social_class", self.rng.choice([1, 2, 3, 4, 5])),
            
            # 삶의 중요 영역 (1=Very important, 4=Not at all important)
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
            "religiosity": fixed_attributes.get("religiosity", self.rng.choice([1, 2, 3])),
            
            # 사회적 태도
            "reject_homosexual_neighbor": fixed_attributes.get("reject_homosexual_neighbor", 1 if self.rng.random() < 0.3 else 2),
            "homosexual_parents_opinion": fixed_attributes.get("homosexual_parents_opinion", self.rng.randint(1, 5)),
            
            # 윤리적 가치 (1=Never justifiable, 10=Always justifiable)
            # 실험 질문 주제(homosexuality, abortion, divorce, suicide, euthanasia, prostitution, death_penalty)는 제외
            "justifiability_premarital_sex": fixed_attributes.get("justifiability_premarital_sex", self.rng.randint(1, 10)),
            "justifiability_casual_sex": fixed_attributes.get("justifiability_casual_sex", self.rng.randint(1, 10)),
            
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
        
#========================================================================================

# 사용 예시
if __name__ == "__main__":
    # 한국 페르소나 생성기 초기화 (국가 코드 410 사용)
    generator = WVSPersonaGenerator(country_code=410, seed=42)
    
    # 특정 조건의 페르소나 생성 (예: 보수 성향 고학력 30대 한국인 여성)
    persona = generator.generate_persona(
        gender=2,  # Female
        age=35,
        education_level=7,  # Master degree
        social_class=2,  # Upper middle class
        political_left_right=7,  # Conservative (right-leaning)
        justifiability__casual_sex=3  # Low acceptance
    )
    
    print("=== Generated Persona ===")
    print(f"Country: South Korea (Code: {persona.country_code})")
    print(f"Gender: {persona.gender} (1=Male, 2=Female)")
    print(f"Age: {persona.age}")
    print(f"Education: {persona.education_level} (7=Master)")
    print(f"Political orientation: {persona.political_left_right}/10 (Right-leaning)")
    print(f"Justifiability of casual sex: {persona.justifiability__casual_sex}/10")
    
    # 에이전트 생성
    agent = StatelessPersonaAgent(persona=persona, temp=1.0)
    
    # 단일 질문 응답
    question = WVSEthicalQuestions.get_question("homosexuality")
    print(f"\n=== Question ===\n{question}")
    
    response = agent.respond_to_ethical_question(question)
    print(f"\n=== Response ===\n{response.content}")
    
    # 모든 국가 코드에서 페르소나 생성 예시
    print("\n=== Generating personas from all countries ===")
    for country_code, country_name in WVSPersonaGenerator.COUNTRIES.items():
        gen = WVSPersonaGenerator(country_code=country_code, seed=42)
        personas = gen.generate_multiple_personas(n=3)
        print(f"\n{country_name} (Code: {country_code}): Generated {len(personas)} personas")