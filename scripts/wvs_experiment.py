"""
WVS 윤리 이슈 실험 메인 스크립트
여러 LLM 모델을 사용하여 국가별 윤리 가치관 조사 시뮬레이션
"""
import json
import csv
import os
import re
import sys
import time  # 추가!
from typing import List, Dict, Tuple
from collections import defaultdict

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

# 모듈 import
from agent.agent import WVSPersonaGenerator, StatelessPersonaAgent, WVSEthicalQuestions
from llm.llm import Message, chat_request

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


def parse_rating_from_response(response: str) -> int:
    """LLM 응답에서 1-10 척도 평점 추출"""
    response = response.strip()
    if response.isdigit():
        rating = int(response)
        if 1 <= rating <= 10:
            return rating
    
    patterns = [
        r'(?:rating|rate|score)[\s:]+(\d+)',
        r'(\d+)\s*(?:/10|out of 10)',
        r'(?:scale|from 1 to 10).*?(\d+)',
        r'^(\d+)\b',
        r'\b(\d+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            rating = int(match.group(1))
            if 1 <= rating <= 10:
                return rating
    
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


def run_single_turn_experiment(
    country: str,
    num_personas: int = 100,
    random_seed: int = 42,
    temp: float = 1.0,
    max_tokens: int = 200,
    model: str = None
) -> Tuple[List[Dict], Dict]:
    """
    Single turn 방식으로 모든 윤리 질문에 한번에 응답 (숫자만)
    """
    print(f"\n{'='*60}")
    print(f"Running SINGLE TURN experiment: {country}")
    print(f"Random seed: {random_seed}, Temperature: {temp}")
    print(f"Model: {model or 'default'}")
    print(f"{'='*60}\n")
    
    # 페르소나 생성 (국가만 지정, 나머지는 랜덤 샘플링)
    country_code = COUNTRY_CODES.get(country)
    if not country_code:
        raise ValueError(f"Unknown country: {country}")
    
    generator = WVSPersonaGenerator(country_code=country_code, seed=random_seed)
    personas = generator.generate_multiple_personas(n=num_personas)
    
    print(f"✅ Generated {len(personas)} personas for {country} (Code: {country_code})")
    print(f"   Example persona: Age={personas[0].age}, Gender={personas[0].gender}, Education={personas[0].education_level}")
    print(f"   Random seed: {random_seed} (for reproducibility)\n")
    
    # 모든 질문을 single turn 형식으로 (숫자만 요청 - reasoning 제외)
    all_questions = WVSEthicalQuestions.get_single_turn_questions(return_number_only=True)
    
    responses_data = []
    topic_ratings = {topic: [] for topic in ETHICAL_TOPICS}
    
    for i, persona in enumerate(personas):
        agent = StatelessPersonaAgent(persona=persona, temp=temp)
        
        try:
            # 시스템 프롬프트
            system_message_content = agent.persona.to_prompt()
            
            # 질문 메시지
            messages = [
                Message(time=0, content=system_message_content, role="system"),
                Message(time=1, content=all_questions, role="user")
            ]
            
            # API 호출
            response = chat_request(
                messages=messages,
                temperature=temp,
                max_tokens=max_tokens,
                model=model
            )
            
            response_text = response.content
            
            # Rate limit 방지를 위한 짧은 대기 (Groq 무료 플랜: 12,000 TPM)
            time.sleep(2.0)  # 2000ms 대기
            
            # 각 주제별 평점 추출
            ratings_dict = {}
            for j, topic in enumerate(ETHICAL_TOPICS, 1):
                pattern = rf"{j}\.\s*{topic}[\s:]+(\d+)"
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
                "country": country,
                "age": persona.age,
                "gender": persona.gender,
                "education_level": persona.education_level,
                "social_class": persona.social_class,
                "political_left_right": persona.political_left_right,
                "importance_religion": persona.importance_religion,
                "religiosity": persona.religiosity,
                "response": response_text,
                "temperature": temp,
                "random_seed": random_seed,
                "model": model or "default",
                **{f"rating_{topic}": ratings_dict.get(topic, -1) for topic in ETHICAL_TOPICS}
            }
            
            responses_data.append(persona_dict)
            
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_personas} personas completed")
                valid_counts = sum(1 for topic in ETHICAL_TOPICS if ratings_dict.get(topic, -1) != -1)
                print(f"  Valid ratings: {valid_counts}/7")
        
        except Exception as e:
            error_str = str(e)
            
            # Rate limit 에러 처리
            if "rate_limit" in error_str.lower() or "429" in error_str:
                print(f"⏳ Rate limit hit at persona {i}. Waiting 5 seconds...")
                time.sleep(5)
                
                # 재시도
                try:
                    response = chat_request(
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                        model=model
                    )
                    response_text = response.content
                    
                    # 성공했으면 정상 처리 계속
                    ratings_dict = {}
                    for j, topic in enumerate(ETHICAL_TOPICS, 1):
                        pattern = rf"{j}\.\s*{topic}[\s:]+(\d+)"
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
                        "country": country,
                        "age": persona.age,
                        "gender": persona.gender,
                        "education_level": persona.education_level,
                        "social_class": persona.social_class,
                        "political_left_right": persona.political_left_right,
                        "importance_religion": persona.importance_religion,
                        "religiosity": persona.religiosity,
                        "response": response_text,
                        "temperature": temp,
                        "random_seed": random_seed,
                        "model": model or "default",
                        **{f"rating_{topic}": ratings_dict.get(topic, -1) for topic in ETHICAL_TOPICS}
                    }
                    responses_data.append(persona_dict)
                    
                except Exception as retry_error:
                    print(f"❌ Retry failed for persona {i}: {str(retry_error)}")
                    persona_dict = {
                        "persona_id": i,
                        "country": country,
                        "age": persona.age if hasattr(persona, 'age') else -1,
                        "gender": persona.gender if hasattr(persona, 'gender') else "unknown",
                        "error": str(retry_error),
                        **{f"rating_{topic}": -1 for topic in ETHICAL_TOPICS}
                    }
                    responses_data.append(persona_dict)
                    continue
            else:
                print(f"❌ Error processing persona {i}: {error_str}")
                persona_dict = {
                    "persona_id": i,
                    "country": country,
                    "age": persona.age if hasattr(persona, 'age') else -1,
                    "gender": persona.gender if hasattr(persona, 'gender') else "unknown",
                    "error": error_str,
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
        print(f"{topic:15s}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, N={stats['count']}/{num_personas}")
    
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
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='LLM 온도 (기본값: 1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='랜덤 시드 (기본값: 42)')
    
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
        print(f"Example: python scripts/wvs_experiment.py --country \"South Korea\" --model gemini-2.5-flash")
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
    print(f"🎲 Random Seed: {args.seed}")
    print(f"📁 Output: {output_dir}")
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
                random_seed=args.seed,
                temp=args.temperature,
                max_tokens=200,
                model=args.model
            )
            
            # 결과 저장
            country_safe = country.replace(' ', '_')
            
            # CSV 저장
            responses_filename = f"responses_{country_safe}_seed{args.seed}.csv"
            responses_path = os.path.join(output_dir, responses_filename)
            
            with open(responses_path, 'w', newline='', encoding='utf-8') as f:
                if responses:
                    writer = csv.DictWriter(f, fieldnames=responses[0].keys())
                    writer.writeheader()
                    writer.writerows(responses)
            
            # JSON 저장
            stats_filename = f"stats_{country_safe}_seed{args.seed}.json"
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