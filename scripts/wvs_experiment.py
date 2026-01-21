"""
WVS 윤리 이슈 실험 메인 스크립트
여러 LLM 모델을 사용하여 국가별 윤리 가치관 조사 시뮬레이션

변경사항:
- agent.py → persona.py로 변경
- WVS 실제 분포 기반 페르소나 생성기 사용
- 페르소나-응답 일관성 검증 기능 추가
"""
import json
import csv
import os
import re
import sys
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import asdict

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
if 'scripts' in current_file:
    scripts_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(scripts_dir)
else:
    project_root = os.path.dirname(current_file)

# 프로젝트 루트를 path에 추가
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 모듈 import (agent → persona로 변경)
from persona.persona import (
    WVSPersonaGenerator, 
    WVSPersonaProfile,
    PersonaResponseValidator
)
from llm.llm import Message, chat_request, get_chat_request_func

# 실험 설정
COUNTRIES = ["United States", "Germany", "Great Britain", "Japan", "South Korea", "India", "Netherlands"]

# 국가명 → WVS 국가 코드 매핑
COUNTRY_CODES = {
    "United States": 840,
    "Germany": 276,
    "Great Britain": 826,
    "Japan": 392,
    "South Korea": 410,
    "India": 356,
    "Netherlands": 528
}

ETHICAL_TOPICS = ["homosexuality", "abortion", "divorce", "suicide", "euthanasia", "prostitution", "death_penalty"]


# =============================================================================
# WVS 윤리 질문 클래스
# =============================================================================

class WVSEthicalQuestions:
    """WVS 윤리적 이슈 질문"""
    
    TOPICS = {
        "homosexuality": "Homosexuality",
        "abortion": "Abortion",
        "divorce": "Divorce",
        "suicide": "Suicide",
        "euthanasia": "Euthanasia (ending the life of the incurably sick)",
        "prostitution": "Prostitution",
        "death_penalty": "Death penalty"
    }
    
    @classmethod
    def get_single_turn_questions(cls, return_number_only: bool = True) -> str:
        """
        모든 윤리 질문을 한 번에 제시하는 프롬프트 생성
        
        Args:
            return_number_only: True면 숫자만 요청, False면 reasoning도 요청
        """
        topic_list = "\n".join([
            f"{i}. {cls.TOPICS[topic]}" 
            for i, topic in enumerate(ETHICAL_TOPICS, 1)
        ])
        
        if return_number_only:
            return f"""Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between.

Use a scale from 1 to 10, where:
- 1 means "Never justifiable"
- 10 means "Always justifiable"

Rate each topic based on YOUR personal values and beliefs as described in your profile:

{topic_list}

IMPORTANT: Respond with ONLY the topic name and number for each, in this exact format:
1. homosexuality: [your rating]
2. abortion: [your rating]
3. divorce: [your rating]
4. suicide: [your rating]
5. euthanasia: [your rating]
6. prostitution: [your rating]
7. death_penalty: [your rating]

Do not include any explanation or reasoning. Just the ratings."""
        else:
            return f"""Please tell me for each of the following statements whether you think it can always be justified, never be justified, or something in between.

Use a scale from 1 to 10, where:
- 1 means "Never justifiable"
- 10 means "Always justifiable"

Rate each topic based on YOUR personal values and beliefs:

{topic_list}

For each topic, briefly explain your reasoning and then give your rating."""


# =============================================================================
# Stateless 페르소나 에이전트
# =============================================================================

class StatelessPersonaAgent:
    """
    상태를 유지하지 않는 페르소나 에이전트
    각 요청마다 페르소나 프로필을 시스템 프롬프트로 전달
    """
    
    def __init__(self, persona: WVSPersonaProfile, temp: float = 0.3):
        self.persona = persona
        self.temp = temp
    
    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.persona.to_prompt()


# =============================================================================
# 파싱 함수
# =============================================================================

def parse_rating_from_response_advanced(response: str, topic: str, topic_index: int, debug: bool = False) -> int:
    """
    개선된 평점 파싱 - 더 많은 패턴 지원
    
    Args:
        response: LLM 응답 전체
        topic: 주제 이름 (예: "homosexuality")
        topic_index: 주제 번호 (1-7)
        debug: 디버그 모드
    
    Returns:
        평점 (1-10) 또는 -1 (파싱 실패)
    """
    # 응답을 줄 단위로 분리
    lines = response.strip().split('\n')
    
    # 더욱 강력한 패턴 리스트 (우선순위 순)
    patterns = [
        # 1. "1. homosexuality: 7" 형태 (가장 일반적)
        rf"^\s*{topic_index}[\.\)]\s*{topic}\s*[:=\-–]\s*(\d+)",
        
        # 2. "homosexuality: 7" (번호 없이)
        rf"^\s*{topic}\s*[:=\-–]\s*(\d+)",
        
        # 3. 번호만 있는 경우: "1. 7" 또는 "1) 7"
        rf"^\s*{topic_index}[\.\)]\s*[:=\-–]?\s*(\d+)\s*$",
        
        # 4. 중간에 텍스트가 있는 경우: "1. homosexuality - Rating: 7"
        rf"^\s*{topic_index}[\.\)]\s*{topic}.*?[:=\-–]\s*(\d+)",
        
        # 5. 줄 어디든 주제명 + 숫자 (느슨한 매칭)
        rf"\b{topic}\b.*?[:=\-–]\s*(\d+)",
    ]
    
    # 각 줄에 대해 패턴 매칭 시도
    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
            
        for pattern in patterns:
            match = re.search(pattern, line_clean, re.IGNORECASE)
            if match:
                rating = int(match.group(1))
                if 1 <= rating <= 10:
                    if debug:
                        print(f"✓ Matched '{topic}' with pattern '{pattern}' in line: '{line_clean[:80]}'")
                    return rating
    
    # 전체 응답에서 주제별로 검색 (콤마 구분 등)
    # "homosexuality: 6, abortion: 7" 같은 경우
    global_pattern = rf"\b{topic}\b\s*[:=\-–]\s*(\d+)"
    global_match = re.search(global_pattern, response, re.IGNORECASE)
    if global_match:
        rating = int(global_match.group(1))
        if 1 <= rating <= 10:
            if debug:
                print(f"✓ Matched '{topic}' globally")
            return rating
    
    # 번호 기반 검색 (주제명이 없는 경우 대비)
    # "3: 8" 또는 "3. 8" 형태 찾기
    number_pattern = rf"^\s*{topic_index}[\.\):\s]+(\d+)\s*$"
    for line in lines:
        match = re.search(number_pattern, line.strip())
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 10:
                if debug:
                    print(f"✓ Matched '{topic}' by number {topic_index} in line: '{line.strip()}'")
                return rating
    
    if debug:
        print(f"✗ Failed to parse '{topic}' (index {topic_index})")
        print(f"  Full response:")
        print(f"  {response}")
        print(f"  ---")
    
    return -1


def calculate_distribution_stats(ratings: List[int]) -> Dict:
    """평점 분포 통계 계산"""
    if not ratings:
        return {"mean": 0, "std": 0, "count": 0, "distribution": {}}
    
    import statistics
    
    distribution = defaultdict(int)
    for rating in ratings:
        if 1 <= rating <= 10:
            distribution[rating] += 1
    
    valid_ratings = [r for r in ratings if 1 <= r <= 10]
    
    return {
        "mean": statistics.mean(valid_ratings) if valid_ratings else 0,
        "std": statistics.stdev(valid_ratings) if len(valid_ratings) > 1 else 0,
        "count": len(valid_ratings),
        "distribution": dict(distribution),
        "invalid_count": len(ratings) - len(valid_ratings)
    }


# =============================================================================
# 메인 실험 함수
# =============================================================================

def run_single_turn_experiment(
    country: str,
    num_personas: int = 100,
    temp: float = 0.3,
    max_tokens: int = 500,
    model: str = None,
    debug: bool = False,
    validate_consistency: bool = True  # 일관성 검증 여부
) -> Tuple[List[Dict], Dict]:
    """
    Single turn 방식으로 모든 윤리 질문에 한번에 응답 (숫자만)
    
    Args:
        country: 국가명
        num_personas: 생성할 페르소나 수
        temp: LLM 온도
        max_tokens: 최대 토큰 수
        model: 사용할 모델
        debug: 디버그 모드
        validate_consistency: 페르소나-응답 일관성 검증 여부
    
    Returns:
        (응답 데이터 리스트, 통계 딕셔너리)
    """
    print(f"\n{'='*60}")
    print(f"Running SINGLE TURN experiment: {country}")
    print(f"Temperature: {temp}")
    print(f"Model: {model or 'default'}")
    print(f"Consistency validation: {'ON' if validate_consistency else 'OFF'}")
    print(f"{'='*60}\n")
    
    # 페르소나 생성 (국가명 → 국가 코드 변환)
    country_code = COUNTRY_CODES.get(country)
    if not country_code:
        raise ValueError(f"Unknown country: {country}")
    
    # 새로운 WVS 실제 분포 기반 생성기 사용 (seed 제거됨)
    generator = WVSPersonaGenerator(country_code=country_code)
    personas = generator.generate_multiple_personas(n=num_personas)
    
    print(f"✅ Generated {len(personas)} personas for {country} (Code: {country_code})")
    print(f"   Example persona: Age={personas[0].age}, Gender={'Male' if personas[0].gender == 1 else 'Female'}, Education={personas[0].education_level}")
    print(f"   Political L-R: {personas[0].political_left_right}/10, Religiosity: {personas[0].religiosity}\n")
    
    # 일관성 검증기 초기화
    validator = PersonaResponseValidator(use_llm=False) if validate_consistency else None
    
    # 모든 질문을 single turn 형식으로 (숫자만 요청 - reasoning 제외)
    all_questions = WVSEthicalQuestions.get_single_turn_questions(return_number_only=True)
    
    responses_data = []
    topic_ratings = {topic: [] for topic in ETHICAL_TOPICS}
    
    # 파싱 실패 추적
    parsing_failures = {topic: 0 for topic in ETHICAL_TOPICS}
    
    # 일관성 점수 추적
    consistency_scores = []
    
    for i, persona in enumerate(personas):
        agent = StatelessPersonaAgent(persona=persona, temp=temp)
        
        try:
            # 시스템 프롬프트
            system_message_content = agent.get_system_prompt()
            
            # 질문 메시지
            messages = [
                Message(time=0, content=system_message_content, role="system"),
                Message(time=1, content=all_questions, role="user")
            ]
            
            # API 호출
            response = chat_request(
                messages=messages,
                temperature=temp,
                max_tokens=1000,
                model=model
            )
            
            response_text = response.content
            
            # 디버그 모드: 첫 5개 응답 출력
            if debug and i < 5:
                print(f"\n{'='*60}")
                print(f"DEBUG: Persona {i} Response (FULL):")
                print(f"{'='*60}")
                print(response_text)
                print(f"{'='*60}")
                print(f"Response length: {len(response_text)} characters")
                print(f"{'='*60}\n")
            
            # Rate limit 방지를 위한 대기
            time.sleep(3.0)
            
            # 응답 파싱
            ratings_dict = {}
            for j, topic in enumerate(ETHICAL_TOPICS, 1):
                rating = parse_rating_from_response_advanced(response_text, topic, j, debug=debug)
                
                if 1 <= rating <= 10:
                    ratings_dict[topic] = rating
                    topic_ratings[topic].append(rating)
                else:
                    ratings_dict[topic] = -1
                    parsing_failures[topic] += 1
            
            # 일관성 검증 (선택적)
            consistency_result = None
            if validator:
                # 응답을 검증 가능한 형식으로 변환
                validation_responses = {}
                if ratings_dict.get("homosexuality", -1) > 0:
                    validation_responses["homosexual_justifiable"] = ratings_dict["homosexuality"]
                if ratings_dict.get("prostitution", -1) > 0:
                    validation_responses["casual_sex_justifiable"] = ratings_dict["prostitution"]
                
                if validation_responses:
                    consistency_result = validator.validate_response_consistency(persona, validation_responses)
                    consistency_scores.append(consistency_result["score"])
                    
                    if debug and not consistency_result["is_consistent"]:
                        print(f"⚠️  Persona {i} inconsistency detected:")
                        print(f"   {consistency_result['details']}")
            
            # 결과 저장
            persona_dict = {
                "persona_id": i,
                "country": country,
                "country_code": country_code,
                "age": persona.age,
                "gender": persona.gender,
                "education_level": persona.education_level,
                "social_class": persona.social_class,
                "marital_status": persona.marital_status,
                "born_in_country": persona.born_in_country,
                "is_citizen": persona.is_citizen,
                "political_left_right": persona.political_left_right,
                "importance_religion": persona.importance_religion,
                "importance_god": persona.importance_god,
                "religiosity": persona.religiosity,
                "religious_service_attendance": persona.religious_service_attendance,
                "reject_homosexual_neighbor": persona.reject_homosexual_neighbor,
                "homosexual_parents_opinion": persona.homosexual_parents_opinion,
                "response": response_text,
                "temperature": temp,
                "model": model or "default",
                "consistency_score": consistency_result["score"] if consistency_result else None,
                "is_consistent": consistency_result["is_consistent"] if consistency_result else None,
                **{f"rating_{topic}": ratings_dict.get(topic, -1) for topic in ETHICAL_TOPICS}
            }
            responses_data.append(persona_dict)
            
            if (i + 1) % 10 == 0:
                avg_consistency = sum(consistency_scores[-10:]) / len(consistency_scores[-10:]) if consistency_scores else 0
                print(f"Processed {i+1}/{num_personas} personas... (Avg consistency: {avg_consistency:.2f})")
        
        except Exception as e:
            error_str = str(e)
            print(f"❌ Error processing persona {i}: {error_str[:100]}")
            
            # Retry 로직
            print(f"⚠️  Will retry persona {i} after waiting...")
            time.sleep(10)
            
            try:
                messages = [
                    Message(time=0, content=system_message_content, role="system"),
                    Message(time=1, content=all_questions, role="user")
                ]
                
                response = chat_request(
                    messages=messages,
                    temperature=temp,
                    max_tokens=max_tokens,
                    model=model
                )
                
                response_text = response.content
                time.sleep(5.0)
                
                ratings_dict = {}
                for j, topic in enumerate(ETHICAL_TOPICS, 1):
                    rating = parse_rating_from_response_advanced(response_text, topic, j, debug=debug)
                    
                    if 1 <= rating <= 10:
                        ratings_dict[topic] = rating
                        topic_ratings[topic].append(rating)
                    else:
                        ratings_dict[topic] = -1
                
                persona_dict = {
                    "persona_id": i,
                    "country": country,
                    "country_code": country_code,
                    "age": persona.age,
                    "gender": persona.gender,
                    "education_level": persona.education_level,
                    "social_class": persona.social_class,
                    "political_left_right": persona.political_left_right,
                    "importance_religion": persona.importance_religion,
                    "religiosity": persona.religiosity,
                    "response": response_text,
                    "temperature": temp,
                    "model": model or "default",
                    **{f"rating_{topic}": ratings_dict.get(topic, -1) for topic in ETHICAL_TOPICS}
                }
                responses_data.append(persona_dict)
                print(f"✅ Retry succeeded for persona {i}")
                
            except Exception as retry_error:
                print(f"❌ Retry failed for persona {i}: {str(retry_error)[:100]}")
                persona_dict = {
                    "persona_id": i,
                    "country": country,
                    "age": persona.age if hasattr(persona, 'age') else -1,
                    "gender": persona.gender if hasattr(persona, 'gender') else "unknown",
                    "error": str(retry_error)[:200],
                    **{f"rating_{topic}": -1 for topic in ETHICAL_TOPICS}
                }
                responses_data.append(persona_dict)
                continue
    
    # 각 주제별 통계 계산
    all_stats = {}
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for topic in ETHICAL_TOPICS:
        stats = calculate_distribution_stats(topic_ratings[topic])
        stats["topic"] = topic
        all_stats[topic] = stats
        
        # 파싱 성공률 표시
        success_rate = (stats['count'] / num_personas) * 100
        print(f"{topic:15s}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, N={stats['count']}/{num_personas} ({success_rate:.1f}%)")
    
    # 일관성 통계
    if consistency_scores:
        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        consistent_count = sum(1 for s in consistency_scores if s >= 0.7)
        print(f"\n{'='*60}")
        print("CONSISTENCY SUMMARY")
        print(f"{'='*60}")
        print(f"Average consistency score: {avg_consistency:.3f}")
        print(f"Consistent responses: {consistent_count}/{len(consistency_scores)} ({consistent_count/len(consistency_scores)*100:.1f}%)")
        
        all_stats["_consistency"] = {
            "average_score": avg_consistency,
            "consistent_count": consistent_count,
            "total_validated": len(consistency_scores)
        }
    
    return responses_data, all_stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='WVS 윤리 이슈 실험')
    parser.add_argument('--country', type=str, 
                       help='실험 대상 국가 (생략시 모든 국가 실행)')
    parser.add_argument('--all-countries', action='store_true',
                       help='모든 국가에 대해 실험 실행')
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                       help='사용할 모델 (기본값: llama-3.3-70b-versatile)')
    parser.add_argument('--num-personas', type=int, default=100,
                       help='생성할 페르소나 수 (기본값: 100)')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='LLM 온도 (기본값: 0.3)')
    parser.add_argument('--debug', action='store_true',
                       help='디버그 모드 활성화 (상세 로그 출력)')
    parser.add_argument('--no-validation', action='store_true',
                       help='일관성 검증 비활성화')
    
    args = parser.parse_args()
    
    # 국가 리스트 결정
    if args.all_countries:
        countries_to_run = COUNTRIES
    elif args.country:
        if args.country not in COUNTRIES:
            print(f"❌ Error: '{args.country}' is not a valid country.")
            print(f"Valid countries: {', '.join(COUNTRIES)}")
            sys.exit(1)
        countries_to_run = [args.country]
    else:
        print("❌ Error: Please specify --country or --all-countries")
        print(f"Example: python scripts/wvs_experiment.py --country \"South Korea\" --model gemini-1.5-flash")
        print(f"Or: python scripts/wvs_experiment.py --all-countries --model llama-3.3-70b-versatile")
        sys.exit(1)
    
    # 모델명을 간단하게 변환
    if 'llama' in args.model.lower():
        if '70b' in args.model.lower():
            model_short = 'llama-70b-versatile'
        elif '8b' in args.model.lower():
            model_short = 'llama-8b'
        else:
            model_short = 'llama'
    elif 'gemini' in args.model.lower():
        if 'flash' in args.model.lower():
            model_short = 'gemini-flash'
        elif 'pro' in args.model.lower():
            model_short = 'gemini-pro'
        else:
            model_short = 'gemini'
    elif 'gpt' in args.model.lower():
        if '4' in args.model:
            model_short = 'gpt-4'
        elif '3.5' in args.model:
            model_short = 'gpt-3.5'
        else:
            model_short = 'gpt'
    else:
        model_short = args.model.replace('/', '-').replace('.', '_')[:20]
    
    # 온도 표기
    if args.temperature == int(args.temperature):
        temp_str = str(int(args.temperature))
    else:
        temp_str = str(args.temperature).replace('.', 'p')
    
    # 출력 디렉토리 설정 (results 폴더에 저장)
    output_dir = os.path.join(project_root, 'results', f'{model_short}_temp{temp_str}')
    
    # 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"🚀 WVS ETHICAL VALUES EXPERIMENT")
    print(f"{'='*70}")
    print(f"🌍 Countries: {', '.join(countries_to_run)}")
    print(f"🤖 Model: {args.model} (saved as: {model_short})")
    print(f"👥 Personas per country: {args.num_personas}")
    print(f"🌡️  Temperature: {args.temperature}")
    print(f"📁 Output: {output_dir}")
    print(f"✅ Consistency validation: {'OFF' if args.no_validation else 'ON'}")
    if args.debug:
        print(f"🐛 Debug mode: ENABLED")
    print(f"{'='*70}\n")
    
    # 각 국가별로 실험 실행
    for country_idx, country in enumerate(countries_to_run, 1):
        print(f"\n{'#'*70}")
        print(f"# Country {country_idx}/{len(countries_to_run)}: {country}")
        print(f"{'#'*70}")
        
        try:
            responses, all_stats = run_single_turn_experiment(
                country=country,
                num_personas=args.num_personas,
                temp=args.temperature,
                max_tokens=500,
                model=args.model,
                debug=args.debug,
                validate_consistency=not args.no_validation
            )
            
            # 결과 저장
            country_safe = country.replace(' ', '_')
            
            # CSV 저장
            responses_filename = f"responses_{country_safe}.csv"
            responses_path = os.path.join(output_dir, responses_filename)
            
            with open(responses_path, 'w', newline='', encoding='utf-8') as f:
                if responses:
                    writer = csv.DictWriter(f, fieldnames=responses[0].keys())
                    writer.writeheader()
                    writer.writerows(responses)
            
            # JSON 저장
            stats_filename = f"stats_{country_safe}.json"
            stats_path = os.path.join(output_dir, stats_filename)
            
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(all_stats, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ {country} completed!")
            print(f"   CSV: {responses_filename}")
            print(f"   JSON: {stats_filename}")
            
        except Exception as e:
            print(f"\n❌ {country} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✅ ALL EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    print(f"📊 Countries processed: {len(countries_to_run)}")
    print(f"📁 Results saved to: {output_dir}")
    print(f"{'='*70}\n")