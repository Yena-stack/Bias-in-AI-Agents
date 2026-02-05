"""
WVS 실제 분포 vs LLM 페르소나 분포 비교 스크립트

PDF에서 추출한 실제 WVS 통계와 생성된 페르소나의 분포를 비교합니다.
"""
import sys
import os

# 프로젝트 루트 경로 설정
current_file = os.path.abspath(__file__)
if 'scripts' in current_file:
    scripts_dir = os.path.dirname(current_file)
    project_root = os.path.dirname(scripts_dir)
else:
    project_root = os.path.dirname(current_file)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from collections import Counter
from typing import Dict, List
try:
    # 프로젝트 구조에서 실행하는 경우
    from persona.persona import (
        WVSPersonaGenerator,
        GENDER_DISTRIBUTION,
        AGE_DISTRIBUTION,
        EDUCATION_DISTRIBUTION,
        SOCIAL_CLASS_DISTRIBUTION,
        RELIGIOSITY_DISTRIBUTION,
        POLITICAL_LEFT_RIGHT_DISTRIBUTION,
        IMPORTANCE_RELIGION_DISTRIBUTION,
        MARITAL_STATUS_DISTRIBUTION,
    )
except ImportError:
    # 단독 실행하는 경우
    from persona_new import (
        WVSPersonaGenerator,
        GENDER_DISTRIBUTION,
        AGE_DISTRIBUTION,
        EDUCATION_DISTRIBUTION,
        SOCIAL_CLASS_DISTRIBUTION,
        RELIGIOSITY_DISTRIBUTION,
        POLITICAL_LEFT_RIGHT_DISTRIBUTION,
        IMPORTANCE_RELIGION_DISTRIBUTION,
        MARITAL_STATUS_DISTRIBUTION,
    )

# 국가 코드 매핑
COUNTRY_CODES = {
    "United States": 840,
    "Germany": 276,
    "Great Britain": 826,
    "Japan": 392,
    "South Korea": 410,
    "India": 356,
    "Netherlands": 528
}

COUNTRIES = list(COUNTRY_CODES.keys())


def calculate_persona_distribution(personas: List, attribute: str, mapping: Dict = None) -> Dict[str, float]:
    """페르소나 리스트에서 특정 속성의 분포 계산"""
    values = [getattr(p, attribute) for p in personas]
    
    if mapping:
        values = [mapping.get(v, v) for v in values]
    
    counter = Counter(values)
    total = len(values)
    
    return {k: (v / total) * 100 for k, v in counter.items()}


def print_comparison_table(title: str, wvs_dist: Dict, persona_dist: Dict, key_order: List = None):
    """WVS 실제 분포와 페르소나 분포 비교 테이블 출력"""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    print(f"  {'Category':<25} {'WVS (%)':<12} {'Persona (%)':<12} {'Diff':<10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    
    if key_order is None:
        key_order = list(wvs_dist.keys())
    
    total_diff = 0
    for key in key_order:
        wvs_val = wvs_dist.get(key, 0)
        persona_val = persona_dist.get(key, 0)
        diff = persona_val - wvs_val
        total_diff += abs(diff)
        
        diff_str = f"{diff:+.1f}" if diff != 0 else "0.0"
        print(f"  {str(key):<25} {wvs_val:>8.1f}     {persona_val:>8.1f}     {diff_str:>8}")
    
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Total Abs Diff':<25} {'':<12} {'':<12} {total_diff:>8.1f}")


def compare_country_demographics(country: str, num_personas: int = 1000):
    """특정 국가의 WVS 실제 분포와 페르소나 분포 비교"""
    
    print(f"\n{'='*70}")
    print(f"  📊 DEMOGRAPHIC COMPARISON: {country}")
    print(f"  WVS Wave 7 (2017-2022) vs Generated Personas (n={num_personas})")
    print(f"{'='*70}")
    
    # 페르소나 생성
    country_code = COUNTRY_CODES[country]
    generator = WVSPersonaGenerator(country_code=country_code)
    personas = generator.generate_multiple_personas(n=num_personas)
    
    # 1. 성별 비교
    gender_mapping = {1: "male", 2: "female"}
    persona_gender = calculate_persona_distribution(personas, "gender", gender_mapping)
    print_comparison_table(
        "GENDER",
        GENDER_DISTRIBUTION[country],
        persona_gender,
        ["male", "female"]
    )
    
    # 2. 연령대 비교
    def age_to_group(age):
        if age <= 29:
            return "up_to_29"
        elif age <= 49:
            return "30_49"
        else:
            return "50_plus"
    
    age_groups = [age_to_group(p.age) for p in personas]
    age_counter = Counter(age_groups)
    persona_age = {k: (v / len(personas)) * 100 for k, v in age_counter.items()}
    
    wvs_age = {k: v for k, v in AGE_DISTRIBUTION[country].items() if k in ["up_to_29", "30_49", "50_plus"]}
    print_comparison_table(
        "AGE GROUP",
        wvs_age,
        persona_age,
        ["up_to_29", "30_49", "50_plus"]
    )
    
    # 3. 교육 수준 비교
    edu_mapping = {1: "lower", 2: "middle", 3: "higher"}
    persona_edu = calculate_persona_distribution(personas, "education_level", edu_mapping)
    print_comparison_table(
        "EDUCATION LEVEL",
        EDUCATION_DISTRIBUTION[country],
        persona_edu,
        ["lower", "middle", "higher"]
    )
    
    # 4. 사회 계층 비교
    class_mapping = {1: "upper", 2: "upper_middle", 3: "lower_middle", 4: "working", 5: "lower"}
    persona_class = calculate_persona_distribution(personas, "social_class", class_mapping)
    print_comparison_table(
        "SOCIAL CLASS",
        SOCIAL_CLASS_DISTRIBUTION[country],
        persona_class,
        ["upper", "upper_middle", "lower_middle", "working", "lower"]
    )
    
    # 5. 종교성 비교
    rel_mapping = {1: "religious", 2: "not_religious", 3: "atheist"}
    persona_rel = calculate_persona_distribution(personas, "religiosity", rel_mapping)
    print_comparison_table(
        "RELIGIOSITY",
        RELIGIOSITY_DISTRIBUTION[country],
        persona_rel,
        ["religious", "not_religious", "atheist"]
    )
    
    # 6. 종교 중요도 비교
    imp_rel_mapping = {1: "very_important", 2: "rather_important", 3: "not_very_important", 4: "not_at_all"}
    persona_imp_rel = calculate_persona_distribution(personas, "importance_religion", imp_rel_mapping)
    print_comparison_table(
        "IMPORTANCE OF RELIGION",
        IMPORTANCE_RELIGION_DISTRIBUTION[country],
        persona_imp_rel,
        ["very_important", "rather_important", "not_very_important", "not_at_all"]
    )
    
    # 7. 정치 성향 비교 (1-10)
    persona_political = calculate_persona_distribution(personas, "political_left_right")
    print_comparison_table(
        "POLITICAL LEFT-RIGHT (1=Left, 10=Right)",
        POLITICAL_LEFT_RIGHT_DISTRIBUTION[country],
        persona_political,
        list(range(1, 11))
    )
    
    # 8. 결혼 상태 비교
    marital_mapping = {1: "married", 2: "living_together", 3: "divorced", 4: "separated", 5: "widowed", 6: "single"}
    persona_marital = calculate_persona_distribution(personas, "marital_status", marital_mapping)
    print_comparison_table(
        "MARITAL STATUS",
        MARITAL_STATUS_DISTRIBUTION[country],
        persona_marital,
        ["married", "living_together", "divorced", "separated", "widowed", "single"]
    )


def compare_all_countries_summary(num_personas: int = 1000):
    """모든 국가의 핵심 통계 요약 비교"""
    
    print(f"\n{'='*100}")
    print(f"  📊 ALL COUNTRIES DEMOGRAPHIC SUMMARY")
    print(f"  WVS Wave 7 (2017-2022) vs Generated Personas (n={num_personas} per country)")
    print(f"{'='*100}")
    
    # 테이블 헤더
    print(f"\n{'─'*100}")
    print(f"  {'Country':<15} │ {'Metric':<20} │ {'WVS (%)':<10} │ {'Persona (%)':<12} │ {'Diff':<8}")
    print(f"{'─'*100}")
    
    for country in COUNTRIES:
        country_code = COUNTRY_CODES[country]
        generator = WVSPersonaGenerator(country_code=country_code)
        personas = generator.generate_multiple_personas(n=num_personas)
        
        # 핵심 지표들
        metrics = []
        
        # 성별 (남성 비율)
        male_pct = sum(1 for p in personas if p.gender == 1) / len(personas) * 100
        wvs_male = GENDER_DISTRIBUTION[country]["male"]
        metrics.append(("Male %", wvs_male, male_pct))
        
        # 고학력 비율
        higher_edu_pct = sum(1 for p in personas if p.education_level == 3) / len(personas) * 100
        wvs_higher = EDUCATION_DISTRIBUTION[country]["higher"]
        metrics.append(("Higher Edu %", wvs_higher, higher_edu_pct))
        
        # 종교인 비율
        religious_pct = sum(1 for p in personas if p.religiosity == 1) / len(personas) * 100
        wvs_religious = RELIGIOSITY_DISTRIBUTION[country]["religious"]
        metrics.append(("Religious %", wvs_religious, religious_pct))
        
        # 무신론자 비율
        atheist_pct = sum(1 for p in personas if p.religiosity == 3) / len(personas) * 100
        wvs_atheist = RELIGIOSITY_DISTRIBUTION[country]["atheist"]
        metrics.append(("Atheist %", wvs_atheist, atheist_pct))
        
        # 정치 성향 평균
        pol_mean = sum(p.political_left_right for p in personas) / len(personas)
        wvs_pol_mean = sum(k * v for k, v in POLITICAL_LEFT_RIGHT_DISTRIBUTION[country].items()) / 100
        metrics.append(("Political Mean", wvs_pol_mean, pol_mean))
        
        # 기혼 비율
        married_pct = sum(1 for p in personas if p.marital_status == 1) / len(personas) * 100
        wvs_married = MARITAL_STATUS_DISTRIBUTION[country]["married"]
        metrics.append(("Married %", wvs_married, married_pct))
        
        # 출력
        for i, (metric_name, wvs_val, persona_val) in enumerate(metrics):
            diff = persona_val - wvs_val
            diff_str = f"{diff:+.1f}"
            
            country_name = country if i == 0 else ""
            separator = "│" if i == 0 else "│"
            
            print(f"  {country_name:<15} {separator} {metric_name:<20} │ {wvs_val:>8.1f}   │ {persona_val:>10.1f}   │ {diff_str:>6}")
        
        print(f"  {'':<15} │ {'-'*20} │ {'-'*10} │ {'-'*12} │ {'-'*8}")
    
    print(f"{'─'*100}")


def export_comparison_csv(output_path: str, num_personas: int = 1000):
    """비교 결과를 CSV로 내보내기"""
    import csv
    
    # 출력 디렉터리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    
    for country in COUNTRIES:
        print(f"Processing {country}...")
        country_code = COUNTRY_CODES[country]
        generator = WVSPersonaGenerator(country_code=country_code)
        personas = generator.generate_multiple_personas(n=num_personas)
        
        # 성별 (no_answer = -1 제외)
        valid_personas = [p for p in personas if p.gender in [1, 2]]
        for gender_key in ["male", "female", "no_answer"]:
            wvs_val = GENDER_DISTRIBUTION[country].get(gender_key, 0)
            if gender_key == "male":
                persona_val = sum(1 for p in personas if p.gender == 1) / len(personas) * 100
            elif gender_key == "female":
                persona_val = sum(1 for p in personas if p.gender == 2) / len(personas) * 100
            else:  # no_answer
                persona_val = sum(1 for p in personas if p.gender == -1) / len(personas) * 100
            rows.append({
                "country": country,
                "category": "gender",
                "value": gender_key,
                "wvs_pct": round(wvs_val, 2),
                "persona_pct": round(persona_val, 2),
                "diff": round(persona_val - wvs_val, 2)
            })
        
        # 교육
        edu_mapping = {1: "lower", 2: "middle", 3: "higher"}
        for edu_code, edu_key in edu_mapping.items():
            wvs_val = EDUCATION_DISTRIBUTION[country].get(edu_key, 0)
            persona_val = sum(1 for p in personas if p.education_level == edu_code) / len(personas) * 100
            rows.append({
                "country": country,
                "category": "education",
                "value": edu_key,
                "wvs_pct": round(wvs_val, 2),
                "persona_pct": round(persona_val, 2),
                "diff": round(persona_val - wvs_val, 2)
            })
        
        # 종교성 (no_answer = -1 포함)
        rel_mapping = {1: "religious", 2: "not_religious", 3: "atheist", -1: "no_answer"}
        for rel_code, rel_key in rel_mapping.items():
            wvs_val = RELIGIOSITY_DISTRIBUTION[country].get(rel_key, 0)
            persona_val = sum(1 for p in personas if p.religiosity == rel_code) / len(personas) * 100
            rows.append({
                "country": country,
                "category": "religiosity",
                "value": rel_key,
                "wvs_pct": round(wvs_val, 2),
                "persona_pct": round(persona_val, 2),
                "diff": round(persona_val - wvs_val, 2)
            })
        
        # 사회 계층 (no_answer = -1 포함)
        class_mapping = {1: "upper", 2: "upper_middle", 3: "lower_middle", 4: "working", 5: "lower", -1: "no_answer"}
        for class_code, class_key in class_mapping.items():
            wvs_val = SOCIAL_CLASS_DISTRIBUTION[country].get(class_key, 0)
            persona_val = sum(1 for p in personas if p.social_class == class_code) / len(personas) * 100
            rows.append({
                "country": country,
                "category": "social_class",
                "value": class_key,
                "wvs_pct": round(wvs_val, 2),
                "persona_pct": round(persona_val, 2),
                "diff": round(persona_val - wvs_val, 2)
            })
        
        # 정치 성향 (1-10, 0 = no_answer)
        for pol_val in list(range(1, 11)) + [0]:
            pol_key = str(pol_val) if pol_val > 0 else "no_answer"
            wvs_val = POLITICAL_LEFT_RIGHT_DISTRIBUTION[country].get(pol_val, 0)
            persona_val = sum(1 for p in personas if p.political_left_right == pol_val) / len(personas) * 100
            rows.append({
                "country": country,
                "category": "political_left_right",
                "value": pol_key,
                "wvs_pct": round(wvs_val, 2),
                "persona_pct": round(persona_val, 2),
                "diff": round(persona_val - wvs_val, 2)
            })
    
    # CSV 저장
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["country", "category", "value", "wvs_pct", "persona_pct", "diff"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✅ Comparison data exported to: {output_path}")
    print(f"   Total rows: {len(rows)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WVS vs Persona 인구통계 비교')
    parser.add_argument('--country', type=str, help='특정 국가만 비교')
    parser.add_argument('--all', action='store_true', help='모든 국가 요약 비교')
    parser.add_argument('--num-personas', type=int, default=1000, help='생성할 페르소나 수 (기본값: 1000)')
    parser.add_argument('--export-csv', type=str, help='CSV 파일로 내보내기')
    parser.add_argument('--output-dir', type=str, default='results', help='결과 저장 디렉터리 (기본값: results)')
    
    args = parser.parse_args()
    
    if args.export_csv:
        export_comparison_csv(args.export_csv, args.num_personas)
    elif args.country:
        if args.country not in COUNTRIES:
            print(f"❌ Invalid country: {args.country}")
            print(f"Valid countries: {', '.join(COUNTRIES)}")
            sys.exit(1)
        compare_country_demographics(args.country, args.num_personas)
        
        # 자동 CSV 저장
        csv_path = os.path.join(args.output_dir, f"demographic_comparison_{args.country.replace(' ', '_')}.csv")
        export_comparison_csv(csv_path, args.num_personas)
    elif args.all:
        compare_all_countries_summary(args.num_personas)
        
        # 자동 CSV 저장
        csv_path = os.path.join(args.output_dir, "demographic_comparison_all.csv")
        export_comparison_csv(csv_path, args.num_personas)
    else:
        # 기본: 모든 국가 상세 비교
        print("\n" + "█" * 70)
        print("█  WVS Wave 7 vs Generated Personas - Demographic Comparison")
        print("█" * 70)
        
        for country in COUNTRIES:
            compare_country_demographics(country, args.num_personas)
        
        print("\n")
        compare_all_countries_summary(args.num_personas)
        
        # 자동 CSV 저장
        csv_path = os.path.join(args.output_dir, "demographic_comparison_all.csv")
        export_comparison_csv(csv_path, args.num_personas)