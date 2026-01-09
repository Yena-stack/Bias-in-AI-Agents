"""
WVS 윤리 이슈 실험 메인 스크립트
llm.py (api.py를 llm.py로 rename)를 사용하여 LLM 에이전트 실험 실행
"""
import sys
import json
import csv
import os
import re
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from agent.agent import WVSPersonaGenerator, StatelessPersonaAgent, WVSEthicalQuestions, WVSPersonaProfile
from llm.llm import Message, chat_request

# 실험 설정 - WVS-7 국가 코드 사용
COUNTRIES = {
    840: "United States",
    276: "Germany",
    826: "Great Britain",
    392: "Japan",
    410: "South Korea",
    356: "India",
    528: "Netherlands"
}

ETHICAL_TOPICS = ["homosexuality", "abortion", "divorce", "suicide", "euthanasia", "prostitution", "death_penalty"]
NUM_PERSONAS_PER_COUNTRY = 200  # 국가당 생성할 페르소나 수 (계획서: 1~2천명)
RANDOM_SEEDS = [42, 123, 456, 789, 1024]  # 재현가능성을 위한 여러 시드


def get_next_file_number(directory: str, pattern: str) -> int:
    """디렉토리에서 다음 파일 번호 결정"""
    max_number = 0
    if not os.path.exists(directory):
        return 1
    for filename in os.listdir(directory):
        match = re.match(pattern, filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return max_number + 1


def parse_rating_from_response(response: str) -> int:
    """
    LLM 응답에서 1-10 척도 평점 추출
    
    Args:
        response: LLM 응답 텍스트
        
    Returns:
        1-10 사이의 평점 (추출 실패시 -1)
    """
    # "Rating: 7", "I would rate this 8", "Score: 5/10" 등의 패턴 찾기
    patterns = [
        r'(?:rating|rate|score)[\s:]+(\d+)',
        r'(\d+)\s*(?:/10|out of 10)',
        r'(?:scale|from 1 to 10).*?(\d+)',
        r'^(\d+)\b',  # 시작 부분의 숫자
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 10:
                return rating
    
    # 실패시 -1 반환
    return -1


def calculate_distribution_stats(ratings: List[int]) -> Dict:
    """
    평점 분포 통계 계산
    
    Args:
        ratings: 평점 리스트
        
    Returns:
        평균, 표준편차, 분포 등의 통계
    """
    if not ratings:
        return {"mean": 0, "std": 0, "count": 0, "distribution": {}, "invalid_count": 0}
    
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


def run_wvs_experiment(
    country_code: int,
    topic: str,
    num_personas: int = 200,
    random_seed: int = 42,
    temp: float = 1.0,
    max_tokens: int = 300
) -> Tuple[List[Dict], Dict]:
    """
    특정 국가와 주제에 대한 WVS 실험 실행
    
    Args:
        country_code: 대상 국가 코드 (WVS-7 3-digit code)
        topic: 윤리 이슈 주제
        num_personas: 생성할 페르소나 수
        random_seed: 랜덤 시드
        temp: LLM 온도
        max_tokens: 최대 토큰 수
        
    Returns:
        (개별 응답 데이터, 통계 요약) 튜플
    """
    country_name = COUNTRIES[country_code]
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {country_name} (Code: {country_code}) - {topic}")
    print(f"Random seed: {random_seed}, Temperature: {temp}")
    print(f"{'='*60}\n")
    
    # 페르소나 생성 (국가 코드 사용)
    generator = WVSPersonaGenerator(country_code=country_code, seed=random_seed)
    personas = generator.generate_multiple_personas(n=num_personas)
    
    # 질문 가져오기
    question = WVSEthicalQuestions.get_question(topic)
    
    # 결과 저장
    responses_data = []
    ratings = []
    
    for i, persona in enumerate(personas):
        # 에이전트 생성 및 응답 받기
        agent = StatelessPersonaAgent(persona=persona, temp=temp)
        
        try:
            response = agent.respond_to_ethical_question(
                question=question,
                max_tokens=max_tokens
            )
            
            response_text = response.content
            rating = parse_rating_from_response(response_text)
            
            # 개별 응답 데이터 저장
            # 주의: 실험 질문 주제의 justifiability는 페르소나에 포함되지 않음
            persona_dict = {
                "persona_id": i,
                "country_code": country_code,
                "country_name": country_name,
                "topic": topic,
                "age": persona.age,
                "gender": persona.gender,  # 1=Male, 2=Female
                "education_level": persona.education_level,  # 0-8 ISCED
                "social_class": persona.social_class,  # 1-5
                "political_left_right": persona.political_left_right,  # 1-10
                "importance_religion": persona.importance_religion,  # 1-4
                "religiosity": persona.religiosity,  # 1=Religious, 2=Not religious, 3=Atheist
                # 간접 지표만 포함 (실험 질문이 아닌 것)
                "justifiability_premarital_sex": persona.justifiability_premarital_sex,
                "justifiability_casual_sex": persona.justifiability_casual_sex,
                "response": response_text,
                "rating": rating,
                "temperature": temp,
                "random_seed": random_seed
            }
            
            responses_data.append(persona_dict)
            if rating != -1:
                ratings.append(rating)
            
            # 진행 상황 출력
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{num_personas} personas completed")
                if ratings:
                    current_stats = calculate_distribution_stats(ratings)
                    print(f"Current mean rating: {current_stats['mean']:.2f}")
        
        except Exception as e:
            print(f"Error processing persona {i}: {str(e)}")
            continue
    
    # 통계 계산
    stats = calculate_distribution_stats(ratings)
    stats["country_code"] = country_code
    stats["country_name"] = country_name
    stats["topic"] = topic
    stats["random_seed"] = random_seed
    stats["temperature"] = temp
    
    print(f"\n--- Results Summary ---")
    print(f"Valid responses: {stats['count']}/{num_personas}")
    print(f"Mean rating: {stats['mean']:.2f}")
    print(f"Std deviation: {stats['std']:.2f}")
    print(f"Distribution: {stats['distribution']}")
    
    return responses_data, stats


def save_experiment_results(
    responses_data: List[Dict],
    stats: Dict,
    output_dir: str,
    country_name: str,
    topic: str,
    seed: int
):
    """실험 결과를 파일로 저장"""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 개별 응답 데이터 저장 (CSV)
    # 파일명에 공백 제거
    country_filename = country_name.replace(" ", "_")
    responses_filename = f"responses_{country_filename}_{topic}_seed{seed}.csv"
    responses_path = os.path.join(output_dir, responses_filename)
    
    with open(responses_path, 'w', newline='', encoding='utf-8') as f:
        if responses_data:
            writer = csv.DictWriter(f, fieldnames=responses_data[0].keys())
            writer.writeheader()
            writer.writerows(responses_data)
    
    print(f"Saved responses to: {responses_path}")
    
    # 통계 요약 저장 (JSON)
    stats_filename = f"stats_{country_filename}_{topic}_seed{seed}.json"
    stats_path = os.path.join(output_dir, stats_filename)
    
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved statistics to: {stats_path}")


def run_single_turn_experiment(
    country_code: int,
    num_personas: int = 200,
    random_seed: int = 42,
    temp: float = 1.0,
    max_tokens: int = 600
) -> Tuple[List[Dict], Dict]:
    """
    Single turn 방식으로 모든 윤리 질문에 한번에 응답
    
    Args:
        country_code: 대상 국가 코드 (WVS-7 3-digit code)
        num_personas: 생성할 페르소나 수
        random_seed: 랜덤 시드
        temp: LLM 온도
        max_tokens: 최대 토큰 수
        
    Returns:
        (개별 응답 데이터, 통계 요약) 튜플
    """
    country_name = COUNTRIES[country_code]
    
    print(f"\n{'='*60}")
    print(f"Running SINGLE TURN experiment: {country_name} (Code: {country_code})")
    print(f"Random seed: {random_seed}, Temperature: {temp}")
    print(f"{'='*60}\n")
    
    # 페르소나 생성 (국가 코드 사용)
    generator = WVSPersonaGenerator(country_code=country_code, seed=random_seed)
    personas = generator.generate_multiple_personas(n=num_personas)
    
    # 모든 질문을 single turn 형식으로
    all_questions = WVSEthicalQuestions.get_single_turn_questions()
    
    responses_data = []
    topic_ratings = {topic: [] for topic in ETHICAL_TOPICS}
    
    for i, persona in enumerate(personas):
        agent = StatelessPersonaAgent(persona=persona, temp=temp)
        
        try:
            response = agent.respond_to_ethical_question(
                question=all_questions,
                max_tokens=max_tokens
            )
            
            response_text = response.content
            
            # 각 주제별 평점 추출
            ratings_dict = {}
            for topic in ETHICAL_TOPICS:
                # 주제명 찾고 그 다음 숫자 추출
                pattern = rf"{topic}.*?(\d+)(?:/10)?"
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    rating = int(match.group(1))
                    if 1 <= rating <= 10:
                        ratings_dict[topic] = rating
                        topic_ratings[topic].append(rating)
                    else:
                        ratings_dict[topic] = -1
                else:
                    ratings_dict[topic] = -1
            
            persona_dict = {
                "persona_id": i,
                "country_code": country_code,
                "country_name": country_name,
                "age": persona.age,
                "gender": persona.gender,  # 1=Male, 2=Female
                "education_level": persona.education_level,  # 0-8 ISCED
                "social_class": persona.social_class,  # 1-5
                "political_left_right": persona.political_left_right,  # 1-10
                "importance_religion": persona.importance_religion,  # 1-4
                "religiosity": persona.religiosity,  # 1-3
                # 간접 지표만 포함
                "justifiability_premarital_sex": persona.justifiability_premarital_sex,
                "justifiability_casual_sex": persona.justifiability_casual_sex,
                "response": response_text,
                "temperature": temp,
                "random_seed": random_seed,
                **{f"rating_{topic}": ratings_dict.get(topic, -1) for topic in ETHICAL_TOPICS}
            }
            
            responses_data.append(persona_dict)
            
            if (i + 1) % 50 == 0:
                print(f"Progress: {i+1}/{num_personas} personas completed")
        
        except Exception as e:
            print(f"Error processing persona {i}: {str(e)}")
            continue
    
    # 각 주제별 통계 계산
    all_stats = {}
    for topic in ETHICAL_TOPICS:
        stats = calculate_distribution_stats(topic_ratings[topic])
        stats["topic"] = topic
        stats["country_code"] = country_code
        stats["country_name"] = country_name
        all_stats[topic] = stats
        
        print(f"\n{topic}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, N={stats['count']}")
    
    return responses_data, all_stats


def compare_with_human_data(llm_stats: Dict, human_stats: Dict) -> Dict:
    """
    LLM 응답과 인간 응답 비교 (Kendall's Tau 등)
    
    Args:
        llm_stats: LLM 응답 통계
        human_stats: 인간 응답 통계 (WVS 실제 데이터)
        
    Returns:
        비교 결과 딕셔너리
    """
    # 실제 구현시 scipy.stats.kendalltau 등 사용
    # 여기서는 기본 구조만 제공
    
    comparison = {
        "llm_mean": llm_stats["mean"],
        "human_mean": human_stats.get("mean", 0),
        "mean_difference": abs(llm_stats["mean"] - human_stats.get("mean", 0)),
        "llm_distribution": llm_stats["distribution"],
        "human_distribution": human_stats.get("distribution", {}),
        # Kendall's Tau는 실제 데이터로 계산
        "kendalls_tau": None,
        "p_value": None
    }
    
    return comparison


if __name__ == '__main__':
    # 실험 모드 선택
    EXPERIMENT_MODE = "single_turn"  # "separate" 또는 "single_turn"
    
    # 출력 디렉토리 설정
    model_name = "groq_llama3.3_70b"
    model_name = "gemini-2.5-flash"
    model_name = "gpt-4-turbo"

    temperature = 1.0
    temp_str = str(temperature).replace('.', 'p')
    output_dir = f'wvs_results/{model_name}_temp{temp_str}'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if EXPERIMENT_MODE == "separate":
        # 각 질문을 개별적으로 실행 (더 정확한 분석)
        for seed in RANDOM_SEEDS[:1]:  # 첫 번째 시드만 테스트
            for country_code in list(COUNTRIES.keys())[:1]:  # 테스트: 첫 번째 국가만
                country_name = COUNTRIES[country_code]
                for topic in ETHICAL_TOPICS[:2]:  # 테스트: 처음 2개 주제만
                    responses, stats = run_wvs_experiment(
                        country_code=country_code,
                        topic=topic,
                        num_personas=10,  # 테스트: 10명만
                        random_seed=seed,
                        temp=temperature
                    )
                    
                    save_experiment_results(
                        responses_data=responses,
                        stats=stats,
                        output_dir=output_dir,
                        country_name=country_name,
                        topic=topic,
                        seed=seed
                    )
    
    elif EXPERIMENT_MODE == "single_turn":
        # Single turn 방식 (계획서에 명시된 방법)
        for seed in RANDOM_SEEDS[:1]:  # 테스트: 첫 시드만
            for country_code in list(COUNTRIES.keys())[:1]:  # 테스트: 첫 국가만
                country_name = COUNTRIES[country_code]
                responses, all_stats = run_single_turn_experiment(
                    country_code=country_code,
                    num_personas=10,  # 테스트: 10명만
                    random_seed=seed,
                    temp=temperature,
                    max_tokens=800
                )
                
                # Single turn 결과 저장
                country_filename = country_name.replace(" ", "_")
                responses_filename = f"single_turn_responses_{country_filename}_seed{seed}.csv"
                responses_path = os.path.join(output_dir, responses_filename)
                
                with open(responses_path, 'w', newline='', encoding='utf-8') as f:
                    if responses:
                        writer = csv.DictWriter(f, fieldnames=responses[0].keys())
                        writer.writeheader()
                        writer.writerows(responses)
                
                stats_filename = f"single_turn_stats_{country_filename}_seed{seed}.json"
                stats_path = os.path.join(output_dir, stats_filename)
                
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(all_stats, f, indent=2)
                
                print(f"\nCompleted {country_name} (Code: {country_code}) - saved to {output_dir}")
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)