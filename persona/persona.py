"""
WVS Wave 7 (2017-2022) - 7개 주요국 실제 분포 기반 페르소나 생성기

PDF에서 추출한 실제 통계 데이터를 사용하여 페르소나를 생성합니다.
각 나라의 인구통계학적 분포에 맞게 현실적인 페르소나를 생성합니다.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import random


# =============================================================================
# 실제 WVS 분포 데이터 (PDF에서 추출)
# =============================================================================

# 성별 분포 (Q260)
GENDER_DISTRIBUTION = {
    "Germany": {"male": 48.6, "female": 51.4},
    "India": {"male": 51.1, "female": 48.9},
    "Japan": {"male": 43.6, "female": 56.4},
    "South Korea": {"male": 48.8, "female": 51.2},
    "Netherlands": {"male": 46.3, "female": 53.7},
    "Great Britain": {"male": 47.6, "female": 51.1},  # 나머지는 무응답
    "United States": {"male": 46.4, "female": 51.6},
}

# 나이 분포 (mean, std 사용)
AGE_DISTRIBUTION = {
    "Germany": {"up_to_29": 16.2, "30_49": 30.8, "50_plus": 52.9, "mean": 50.8, "std": 18.09},
    "India": {"up_to_29": 38.1, "30_49": 38.3, "50_plus": 23.6, "mean": 37.83, "std": 16.05},
    "Japan": {"up_to_29": 10.1, "30_49": 30.4, "50_plus": 59.5, "mean": 54.78, "std": 17.95},
    "South Korea": {"up_to_29": 19.9, "30_49": 37.5, "50_plus": 42.6, "mean": 45.63, "std": 15.03},
    "Netherlands": {"up_to_29": 8.9, "30_49": 31.2, "50_plus": 59.9, "mean": 53.36, "std": 16.38},
    "Great Britain": {"up_to_29": 16.3, "30_49": 32.0, "50_plus": 46.4, "mean": 49.27, "std": 18.61},
    "United States": {"up_to_29": 21.3, "30_49": 33.5, "50_plus": 45.2, "mean": 46.73, "std": 17.33},
}

# 시민권 (Q269)
CITIZENSHIP_DISTRIBUTION = {
    "Germany": {"yes": 91.6, "no": 8.4},
    "India": {"yes": 100.0, "no": 0.0},
    "Japan": {"yes": 99.0, "no": 0.1},
    "South Korea": {"yes": 99.8, "no": 0.2},
    "Netherlands": {"yes": 90.3, "no": 2.3},  # 나머지 무응답
    "Great Britain": {"yes": 92.6, "no": 6.5},
    "United States": {"yes": 95.6, "no": 3.3},
}

# 결혼 상태 (Q273)
MARITAL_STATUS_DISTRIBUTION = {
    "Germany": {
        "married": 54.5, "living_together": 10.0, "divorced": 7.4,
        "separated": 1.3, "widowed": 7.4, "single": 19.0
    },
    "India": {
        "married": 58.3, "living_together": 8.8, "divorced": 0.1,
        "separated": 0.1, "widowed": 5.5, "single": 27.0
    },
    "Japan": {
        "married": 72.4, "living_together": 1.0, "divorced": 5.0,
        "separated": 0.4, "widowed": 7.4, "single": 12.8
    },
    "South Korea": {
        "married": 72.1, "living_together": 0.1, "divorced": 0.6,
        "separated": 0.1, "widowed": 2.1, "single": 25.0
    },
    "Netherlands": {
        "married": 45.8, "living_together": 13.3, "divorced": 6.1,
        "separated": 1.2, "widowed": 4.9, "single": 21.7
    },
    "Great Britain": {
        "married": 51.9, "living_together": 10.7, "divorced": 5.4,
        "separated": 1.9, "widowed": 5.4, "single": 22.2
    },
    "United States": {
        "married": 50.6, "living_together": 7.1, "divorced": 12.1,
        "separated": 2.3, "widowed": 4.5, "single": 23.5
    },
}

# 교육 수준 (Lower/Middle/Higher)
EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 11.2, "middle": 54.1, "higher": 34.5},
    "India": {"lower": 33.9, "middle": 38.9, "higher": 27.1},
    "Japan": {"lower": 6.2, "middle": 38.1, "higher": 54.5},
    "South Korea": {"lower": 11.2, "middle": 41.9, "higher": 46.8},
    "Netherlands": {"lower": 16.3, "middle": 29.5, "higher": 45.9},
    "Great Britain": {"lower": 18.5, "middle": 40.1, "higher": 38.9},
    "United States": {"lower": 3.0, "middle": 52.9, "higher": 42.8},
}

# 어머니 교육 수준
MOTHER_EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 37.7, "middle": 49.9, "higher": 9.7},
    "India": {"lower": 75.9, "middle": 18.0, "higher": 3.5},
    "Japan": {"lower": 31.0, "middle": 40.9, "higher": 17.1},
    "South Korea": {"lower": 56.1, "middle": 35.7, "higher": 6.1},
    "Netherlands": {"lower": 55.7, "middle": 14.1, "higher": 12.6},
    "Great Britain": {"lower": 33.3, "middle": 33.3, "higher": 33.3},  # 데이터 없음 - 균등
    "United States": {"lower": 13.6, "middle": 54.6, "higher": 22.4},
}

# 아버지 교육 수준
FATHER_EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 16.3, "middle": 55.6, "higher": 23.6},
    "India": {"lower": 62.5, "middle": 26.3, "higher": 8.5},
    "Japan": {"lower": 31.9, "middle": 34.0, "higher": 21.8},
    "South Korea": {"lower": 50.3, "middle": 35.0, "higher": 13.1},
    "Netherlands": {"lower": 41.9, "middle": 17.8, "higher": 21.3},
    "Great Britain": {"lower": 33.3, "middle": 33.3, "higher": 33.3},  # 데이터 없음 - 균등
    "United States": {"lower": 16.9, "middle": 47.0, "higher": 22.7},
}

# 사회 계층 (Q287)
SOCIAL_CLASS_DISTRIBUTION = {
    "Germany": {"upper": 1.4, "upper_middle": 36.0, "lower_middle": 41.0, "working": 16.8, "lower": 1.8},
    "India": {"upper": 3.4, "upper_middle": 21.8, "lower_middle": 43.6, "working": 19.1, "lower": 9.9},
    "Japan": {"upper": 1.5, "upper_middle": 15.0, "lower_middle": 42.2, "working": 26.4, "lower": 9.6},
    "South Korea": {"upper": 0.2, "upper_middle": 15.2, "lower_middle": 74.5, "working": 1.4, "lower": 8.8},
    "Netherlands": {"upper": 4.5, "upper_middle": 36.3, "lower_middle": 18.6, "working": 16.3, "lower": 2.5},
    "Great Britain": {"upper": 2.0, "upper_middle": 25.0, "lower_middle": 35.0, "working": 30.0, "lower": 8.0},  # 추정
    "United States": {"upper": 1.3, "upper_middle": 29.4, "lower_middle": 35.8, "working": 24.8, "lower": 7.3},
}

# 출생지 (Q263)
BORN_IN_COUNTRY_DISTRIBUTION = {
    "Germany": {"born_here": 86.0, "immigrant": 13.9},
    "India": {"born_here": 100.0, "immigrant": 0.0},
    "Japan": {"born_here": 97.9, "immigrant": 1.0},
    "South Korea": {"born_here": 99.0, "immigrant": 1.0},
    "Netherlands": {"born_here": 82.3, "immigrant": 10.1},
    "Great Britain": {"born_here": 84.2, "immigrant": 14.3},
    "United States": {"born_here": 87.6, "immigrant": 9.9},
}

# 어머니 이민자 여부 (Q264)
MOTHER_IMMIGRANT_DISTRIBUTION = {
    "Germany": {"not_immigrant": 81.4, "immigrant": 16.8},
    "India": {"not_immigrant": 100.0, "immigrant": 0.0},
    "Japan": {"not_immigrant": 98.6, "immigrant": 0.1},
    "South Korea": {"not_immigrant": 99.6, "immigrant": 0.4},
    "Netherlands": {"not_immigrant": 82.7, "immigrant": 9.0},
    "Great Britain": {"not_immigrant": 77.3, "immigrant": 10.7},
    "United States": {"not_immigrant": 77.7, "immigrant": 13.5},
}

# 아버지 이민자 여부 (Q265)
FATHER_IMMIGRANT_DISTRIBUTION = {
    "Germany": {"not_immigrant": 81.2, "immigrant": 16.4},
    "India": {"not_immigrant": 100.0, "immigrant": 0.0},
    "Japan": {"not_immigrant": 98.2, "immigrant": 0.5},
    "South Korea": {"not_immigrant": 99.7, "immigrant": 0.3},
    "Netherlands": {"not_immigrant": 83.1, "immigrant": 8.2},
    "Great Britain": {"not_immigrant": 75.2, "immigrant": 11.1},
    "United States": {"not_immigrant": 77.9, "immigrant": 12.6},
}

# =============================================================================
# 가치관 분포 (Q1-Q6)
# =============================================================================

# Q1: 가족 중요도
IMPORTANCE_FAMILY_DISTRIBUTION = {
    "Germany": {"very_important": 88.4, "rather_important": 10.3, "not_very_important": 1.1, "not_at_all": 0.1},
    "India": {"very_important": 94.4, "rather_important": 4.4, "not_very_important": 0.8, "not_at_all": 0.3},
    "Japan": {"very_important": 92.0, "rather_important": 6.2, "not_very_important": 0.7, "not_at_all": 0.2},
    "South Korea": {"very_important": 88.9, "rather_important": 10.8, "not_very_important": 0.3, "not_at_all": 0.0},
    "Netherlands": {"very_important": 73.7, "rather_important": 20.8, "not_very_important": 3.3, "not_at_all": 0.7},
    "Great Britain": {"very_important": 92.3, "rather_important": 6.3, "not_very_important": 0.9, "not_at_all": 0.4},
    "United States": {"very_important": 91.0, "rather_important": 7.1, "not_very_important": 1.6, "not_at_all": 0.2},
}

# Q2: 친구 중요도
IMPORTANCE_FRIENDS_DISTRIBUTION = {
    "Germany": {"very_important": 59.8, "rather_important": 37.0, "not_very_important": 2.9, "not_at_all": 0.1},
    "India": {"very_important": 54.0, "rather_important": 28.4, "not_very_important": 11.8, "not_at_all": 5.2},
    "Japan": {"very_important": 40.4, "rather_important": 50.6, "not_very_important": 6.8, "not_at_all": 0.9},
    "South Korea": {"very_important": 40.0, "rather_important": 57.8, "not_very_important": 2.2, "not_at_all": 0.0},
    "Netherlands": {"very_important": 51.9, "rather_important": 40.2, "not_very_important": 5.9, "not_at_all": 0.7},
    "Great Britain": {"very_important": 63.0, "rather_important": 31.6, "not_very_important": 4.8, "not_at_all": 0.5},
    "United States": {"very_important": 50.7, "rather_important": 38.4, "not_very_important": 9.4, "not_at_all": 1.1},
}

# Q3: 여가 중요도
IMPORTANCE_LEISURE_DISTRIBUTION = {
    "Germany": {"very_important": 37.7, "rather_important": 54.1, "not_very_important": 7.5, "not_at_all": 0.5},
    "India": {"very_important": 39.6, "rather_important": 34.8, "not_very_important": 12.7, "not_at_all": 12.0},
    "Japan": {"very_important": 44.5, "rather_important": 45.8, "not_very_important": 7.8, "not_at_all": 0.5},
    "South Korea": {"very_important": 24.1, "rather_important": 67.5, "not_very_important": 8.4, "not_at_all": 0.0},
    "Netherlands": {"very_important": 53.6, "rather_important": 40.5, "not_very_important": 3.6, "not_at_all": 0.3},
    "Great Britain": {"very_important": 51.0, "rather_important": 41.3, "not_very_important": 6.1, "not_at_all": 0.8},
    "United States": {"very_important": 39.5, "rather_important": 49.1, "not_very_important": 10.2, "not_at_all": 0.7},
}

# Q4: 정치 중요도
IMPORTANCE_POLITICS_DISTRIBUTION = {
    "Germany": {"very_important": 15.8, "rather_important": 54.1, "not_very_important": 25.2, "not_at_all": 4.8},
    "India": {"very_important": 21.5, "rather_important": 25.1, "not_very_important": 23.3, "not_at_all": 27.3},
    "Japan": {"very_important": 15.0, "rather_important": 49.3, "not_very_important": 27.1, "not_at_all": 4.1},
    "South Korea": {"very_important": 8.0, "rather_important": 52.1, "not_very_important": 36.5, "not_at_all": 3.5},
    "Netherlands": {"very_important": 3.7, "rather_important": 36.4, "not_very_important": 44.8, "not_at_all": 12.3},
    "Great Britain": {"very_important": 17.0, "rather_important": 38.0, "not_very_important": 34.8, "not_at_all": 9.5},
    "United States": {"very_important": 14.9, "rather_important": 41.6, "not_very_important": 35.2, "not_at_all": 7.1},
}

# Q5: 일 중요도
IMPORTANCE_WORK_DISTRIBUTION = {
    "Germany": {"very_important": 41.8, "rather_important": 40.7, "not_very_important": 8.0, "not_at_all": 7.1},
    "India": {"very_important": 76.4, "rather_important": 17.2, "not_very_important": 3.2, "not_at_all": 2.6},
    "Japan": {"very_important": 38.3, "rather_important": 41.8, "not_very_important": 12.6, "not_at_all": 3.8},
    "South Korea": {"very_important": 42.7, "rather_important": 43.7, "not_very_important": 10.4, "not_at_all": 3.2},
    "Netherlands": {"very_important": 21.5, "rather_important": 46.7, "not_very_important": 11.2, "not_at_all": 4.7},
    "Great Britain": {"very_important": 36.2, "rather_important": 36.1, "not_very_important": 10.6, "not_at_all": 15.2},
    "United States": {"very_important": 39.4, "rather_important": 40.3, "not_very_important": 11.4, "not_at_all": 8.2},
}

# Q6: 종교 중요도
IMPORTANCE_RELIGION_DISTRIBUTION = {
    "Germany": {"very_important": 13.9, "rather_important": 24.6, "not_very_important": 35.2, "not_at_all": 25.9},
    "India": {"very_important": 64.2, "rather_important": 21.5, "not_very_important": 8.9, "not_at_all": 4.7},
    "Japan": {"very_important": 4.6, "rather_important": 9.9, "not_very_important": 33.6, "not_at_all": 42.1},
    "South Korea": {"very_important": 10.3, "rather_important": 25.6, "not_very_important": 47.7, "not_at_all": 16.4},
    "Netherlands": {"very_important": 10.0, "rather_important": 11.2, "not_very_important": 23.6, "not_at_all": 47.1},
    "Great Britain": {"very_important": 15.3, "rather_important": 14.7, "not_very_important": 31.4, "not_at_all": 38.2},
    "United States": {"very_important": 37.1, "rather_important": 23.6, "not_very_important": 21.9, "not_at_all": 16.5},
}

# =============================================================================
# 사회적 태도
# =============================================================================

# Q22: 동성애자를 이웃으로 원하지 않음
REJECT_HOMOSEXUAL_NEIGHBOR_DISTRIBUTION = {
    "Germany": {"mentioned": 6.4, "not_mentioned": 93.3},
    "India": {"mentioned": 62.6, "not_mentioned": 23.5},  # 나머지 무응답
    "Japan": {"mentioned": 26.4, "not_mentioned": 70.8},
    "South Korea": {"mentioned": 79.6, "not_mentioned": 20.4},
    "Netherlands": {"mentioned": 2.2, "not_mentioned": 93.7},
    "Great Britain": {"mentioned": 3.6, "not_mentioned": 95.5},
    "United States": {"mentioned": 12.7, "not_mentioned": 81.1},
}

# Q36: 동성 부부도 좋은 부모가 될 수 있다
HOMOSEXUAL_PARENTS_DISTRIBUTION = {
    "Germany": {"agree_strongly": 26.0, "agree": 36.8, "neither": 10.3, "disagree": 13.1, "disagree_strongly": 4.8},
    "India": {"agree_strongly": 13.4, "agree": 16.5, "neither": 11.8, "disagree": 14.5, "disagree_strongly": 22.6},
    "Japan": {"agree_strongly": 8.8, "agree": 35.0, "neither": 23.1, "disagree": 7.6, "disagree_strongly": 2.4},
    "South Korea": {"agree_strongly": 3.3, "agree": 19.4, "neither": 37.3, "disagree": 30.6, "disagree_strongly": 9.4},
    "Netherlands": {"agree_strongly": 31.7, "agree": 35.9, "neither": 11.9, "disagree": 6.0, "disagree_strongly": 3.6},
    "Great Britain": {"agree_strongly": 34.9, "agree": 32.6, "neither": 18.8, "disagree": 7.6, "disagree_strongly": 2.1},
    "United States": {"agree_strongly": 26.6, "agree": 26.2, "neither": 29.7, "disagree": 9.9, "disagree_strongly": 6.7},
}

# Q186: 혼전 성관계 정당화 (1-10 스케일)
PREMARITAL_SEX_DISTRIBUTION = {
    "Germany": {1: 5.7, 2: 0.7, 3: 0.9, 4: 0.8, 5: 8.0, 6: 1.8, 7: 2.9, 8: 7.2, 9: 6.1, 10: 63.9},
    "India": {1: 62.8, 2: 9.7, 3: 5.0, 4: 3.1, 5: 5.5, 6: 2.8, 7: 2.3, 8: 1.8, 9: 1.3, 10: 1.9},
    "Japan": {1: 5.2, 2: 2.6, 3: 3.6, 4: 2.4, 5: 21.7, 6: 5.2, 7: 6.0, 8: 13.8, 9: 7.2, 10: 24.9},
    "South Korea": {1: 5.1, 2: 8.4, 3: 14.3, 4: 13.2, 5: 24.3, 6: 14.5, 7: 12.9, 8: 5.4, 9: 1.1, 10: 0.8},
    "Netherlands": {1: 2.2, 2: 0.4, 3: 0.7, 4: 1.0, 5: 4.0, 6: 3.4, 7: 4.5, 8: 7.9, 9: 8.7, 10: 57.2},
    "Great Britain": {1: 5.1, 2: 0.5, 3: 1.8, 4: 1.1, 5: 10.7, 6: 2.9, 7: 6.4, 8: 9.6, 9: 7.3, 10: 52.7},
    "United States": {1: 11.2, 2: 2.3, 3: 3.2, 4: 3.3, 5: 19.8, 6: 7.5, 7: 8.3, 8: 11.7, 9: 7.3, 10: 24.0},
}

# Q193: 캐주얼 섹스 정당화 (1-10 스케일)
CASUAL_SEX_DISTRIBUTION = {
    "Germany": {1: 29.4, 2: 5.4, 3: 7.1, 4: 4.5, 5: 18.3, 6: 5.8, 7: 3.7, 8: 5.2, 9: 2.5, 10: 14.5},
    "India": {1: 67.2, 2: 8.6, 3: 5.1, 4: 3.1, 5: 2.9, 6: 2.3, 7: 1.6, 8: 0.9, 9: 1.0, 10: 1.4},
    "Japan": {1: 46.3, 2: 12.0, 3: 10.6, 4: 2.7, 5: 12.7, 6: 2.5, 7: 2.8, 8: 1.0, 9: 1.0, 10: 2.2},
    "South Korea": {1: 38.2, 2: 21.9, 3: 15.1, 4: 9.5, 5: 9.2, 6: 3.5, 7: 2.0, 8: 0.6, 9: 0.0, 10: 0.0},
    "Netherlands": {1: 7.1, 2: 2.1, 3: 2.5, 4: 3.7, 5: 10.8, 6: 6.6, 7: 7.6, 8: 8.2, 9: 5.7, 10: 33.0},
    "Great Britain": {1: 10.0, 2: 2.0, 3: 6.0, 4: 3.9, 5: 18.9, 6: 5.4, 7: 9.9, 8: 10.6, 9: 5.5, 10: 25.0},
    "United States": {1: 17.9, 2: 4.2, 3: 5.5, 4: 5.1, 5: 20.7, 6: 7.7, 7: 8.1, 8: 9.9, 9: 4.8, 10: 14.9},
}

# =============================================================================
# 정치적 성향
# =============================================================================

# Q240: 좌-우 정치 성향 (1=좌, 10=우)
POLITICAL_LEFT_RIGHT_DISTRIBUTION = {
    "Germany": {1: 3.1, 2: 5.1, 3: 14.3, 4: 14.3, 5: 30.4, 6: 13.3, 7: 7.0, 8: 3.9, 9: 0.8, 10: 1.2},
    "India": {1: 2.7, 2: 2.5, 3: 3.1, 4: 5.2, 5: 14.3, 6: 10.7, 7: 8.3, 8: 9.5, 9: 5.9, 10: 10.0},
    "Japan": {1: 1.3, 2: 1.6, 3: 6.3, 4: 7.8, 5: 20.4, 6: 10.1, 7: 9.3, 8: 8.6, 9: 3.5, 10: 3.1},
    "South Korea": {1: 0.8, 2: 4.1, 3: 13.3, 4: 15.4, 5: 22.6, 6: 17.3, 7: 14.6, 8: 10.0, 9: 1.8, 10: 0.1},
    "Netherlands": {1: 2.2, 2: 4.2, 3: 9.3, 4: 11.2, 5: 14.1, 6: 11.5, 7: 14.6, 8: 10.1, 9: 2.0, 10: 2.3},
    "Great Britain": {1: 3.5, 2: 3.7, 3: 12.4, 4: 10.8, 5: 36.2, 6: 9.9, 7: 7.3, 8: 6.7, 9: 1.6, 10: 2.2},
    "United States": {1: 8.6, 2: 4.8, 3: 10.9, 4: 7.6, 5: 26.2, 6: 9.5, 7: 8.6, 8: 8.9, 9: 4.7, 10: 7.4},
}

# Q250: 민주주의 중요도 (1=전혀 중요하지 않음, 10=매우 중요)
IMPORTANCE_DEMOCRACY_DISTRIBUTION = {
    "Germany": {1: 0.1, 2: 0.1, 3: 0.4, 4: 0.2, 5: 2.3, 6: 1.4, 7: 1.8, 8: 7.9, 9: 10.0, 10: 75.4},
    "India": {1: 2.0, 2: 2.1, 3: 1.5, 4: 1.9, 5: 3.4, 6: 5.1, 7: 7.1, 8: 10.8, 9: 12.9, 10: 48.2},
    "Japan": {1: 0.3, 2: 0.2, 3: 0.3, 4: 0.4, 5: 4.8, 6: 3.5, 7: 7.1, 8: 18.0, 9: 12.5, 10: 43.0},
    "South Korea": {1: 0.6, 2: 0.8, 3: 0.0, 4: 0.0, 5: 6.2, 6: 10.2, 7: 16.9, 8: 28.5, 9: 21.6, 10: 15.2},
    "Netherlands": {1: 0.2, 2: 0.0, 3: 0.1, 4: 0.4, 5: 2.8, 6: 3.0, 7: 7.0, 8: 17.5, 9: 16.4, 10: 40.6},
    "Great Britain": {1: 0.7, 2: 0.2, 3: 0.6, 4: 0.3, 5: 7.7, 6: 3.1, 7: 5.2, 8: 9.0, 9: 9.6, 10: 61.5},
    "United States": {1: 1.6, 2: 0.8, 3: 0.9, 4: 1.9, 5: 12.5, 6: 4.4, 7: 5.9, 8: 10.3, 9: 10.7, 10: 48.8},
}

# =============================================================================
# 종교적 가치
# =============================================================================

# Q164: 신의 중요도 (1-10)
IMPORTANCE_GOD_DISTRIBUTION = {
    "Germany": {1: 26.4, 2: 8.8, 3: 7.1, 4: 4.6, 5: 9.8, 6: 5.2, 7: 7.7, 8: 9.1, 9: 3.9, 10: 16.0},
    "India": {1: 2.5, 2: 2.1, 3: 2.9, 4: 3.2, 5: 4.8, 6: 6.8, 7: 10.9, 8: 14.8, 9: 9.8, 10: 41.6},
    "Japan": {1: 16.3, 2: 10.1, 3: 11.7, 4: 6.1, 5: 15.9, 6: 9.6, 7: 8.4, 8: 6.9, 9: 2.4, 10: 4.8},
    "South Korea": {1: 6.7, 2: 9.2, 3: 14.9, 4: 11.6, 5: 15.1, 6: 11.1, 7: 14.0, 8: 12.0, 9: 3.0, 10: 2.5},
    "Netherlands": {1: 37.0, 2: 10.3, 3: 7.2, 4: 3.7, 5: 5.7, 6: 5.1, 7: 5.0, 8: 6.1, 9: 2.2, 10: 8.2},
    "Great Britain": {1: 39.4, 2: 10.1, 3: 6.8, 4: 3.6, 5: 6.8, 6: 5.8, 7: 4.6, 8: 5.4, 9: 2.7, 10: 14.2},
    "United States": {1: 12.8, 2: 3.5, 3: 4.1, 4: 3.3, 5: 9.0, 6: 4.6, 7: 5.8, 8: 7.7, 9: 5.3, 10: 43.2},
}

# Q171: 종교 예배 참석 빈도
RELIGIOUS_SERVICE_ATTENDANCE_DISTRIBUTION = {
    "Germany": {"more_than_once_week": 1.3, "once_week": 7.1, "once_month": 10.5, "holy_days": 16.4, "once_year": 8.7, "less_often": 15.1, "never": 40.5},
    "India": {"more_than_once_week": 23.1, "once_week": 24.3, "once_month": 15.3, "holy_days": 21.2, "once_year": 2.6, "less_often": 9.0, "never": 3.9},
    "Japan": {"more_than_once_week": 1.0, "once_week": 1.8, "once_month": 9.5, "holy_days": 41.5, "once_year": 17.2, "less_often": 15.1, "never": 13.3},
    "South Korea": {"more_than_once_week": 9.2, "once_week": 9.8, "once_month": 3.9, "holy_days": 8.0, "once_year": 4.1, "less_often": 12.9, "never": 52.2},
    "Netherlands": {"more_than_once_week": 2.0, "once_week": 6.0, "once_month": 3.2, "holy_days": 5.4, "once_year": 4.3, "less_often": 8.3, "never": 61.2},
    "Great Britain": {"more_than_once_week": 3.6, "once_week": 6.9, "once_month": 5.2, "holy_days": 9.2, "once_year": 7.3, "less_often": 12.3, "never": 55.4},
    "United States": {"more_than_once_week": 9.2, "once_week": 20.0, "once_month": 9.7, "holy_days": 8.4, "once_year": 5.0, "less_often": 14.1, "never": 33.3},
}

# Q172: 기도 빈도
PRAYER_FREQUENCY_DISTRIBUTION = {
    "Germany": {"several_times_day": 6.6, "once_day": 11.4, "several_times_week": 12.2, "religious_services_only": 7.5, "holy_days_only": 4.0, "once_year": 2.3, "less_often": 15.5, "never": 39.1},
    "India": {"several_times_day": 20.0, "once_day": 34.7, "several_times_week": 13.7, "religious_services_only": 10.1, "holy_days_only": 11.3, "once_year": 1.2, "less_often": 5.4, "never": 3.4},
    "Japan": {"several_times_day": 5.3, "once_day": 14.5, "several_times_week": 5.2, "religious_services_only": 5.5, "holy_days_only": 28.6, "once_year": 6.9, "less_often": 17.1, "never": 16.6},
    "South Korea": {"several_times_day": 6.6, "once_day": 5.7, "several_times_week": 6.7, "religious_services_only": 7.2, "holy_days_only": 6.9, "once_year": 3.1, "less_often": 16.1, "never": 47.7},
    "Netherlands": {"several_times_day": 8.2, "once_day": 7.2, "several_times_week": 6.2, "religious_services_only": 2.0, "holy_days_only": 1.3, "once_year": 5.1, "less_often": 3.1, "never": 56.8},
    "Great Britain": {"several_times_day": 7.4, "once_day": 8.3, "several_times_week": 8.4, "religious_services_only": 6.0, "holy_days_only": 2.5, "once_year": 2.8, "less_often": 10.1, "never": 54.1},
    "United States": {"several_times_day": 26.1, "once_day": 15.8, "several_times_week": 20.4, "religious_services_only": 4.0, "holy_days_only": 1.9, "once_year": 2.9, "less_often": 9.0, "never": 19.0},
}

# Q173: 종교적 정체성
RELIGIOSITY_DISTRIBUTION = {
    "Germany": {"religious": 48.0, "not_religious": 37.5, "atheist": 11.6},
    "India": {"religious": 76.7, "not_religious": 16.6, "atheist": 2.4},
    "Japan": {"religious": 14.3, "not_religious": 55.7, "atheist": 19.1},
    "South Korea": {"religious": 16.1, "not_religious": 29.0, "atheist": 54.9},
    "Netherlands": {"religious": 31.5, "not_religious": 38.6, "atheist": 14.0},
    "Great Britain": {"religious": 31.8, "not_religious": 45.6, "atheist": 21.7},
    "United States": {"religious": 58.0, "not_religious": 33.3, "atheist": 7.9},
}


# =============================================================================
# 페르소나 데이터 클래스
# =============================================================================

@dataclass
class WVSPersonaProfile:
    """WVS 기반 페르소나 프로필"""
    # 기본 정보
    country_code: int
    country_name: str
    gender: int  # 1=Male, 2=Female
    age: int
    
    # 이민/시민권
    born_in_country: int  # 1=Born here, 2=Immigrant
    mother_immigrant: int  # 1=Not immigrant, 2=Immigrant
    father_immigrant: int
    is_citizen: int  # 1=Yes, 2=No
    
    # 결혼/가족
    marital_status: int  # 1=Married, 2=Living together, 3=Divorced, 4=Separated, 5=Widowed, 6=Single
    
    # 교육
    education_level: int  # 1=Lower, 2=Middle, 3=Higher
    mother_education: int
    father_education: int
    
    # 사회 계층
    social_class: int  # 1=Upper, 2=Upper middle, 3=Lower middle, 4=Working, 5=Lower
    
    # 가치관 (Q1-Q6): 1=Very important, 2=Rather important, 3=Not very important, 4=Not at all
    importance_family: int
    importance_friends: int
    importance_leisure: int
    importance_politics: int
    importance_work: int
    importance_religion: int
    
    # 종교
    importance_god: int  # 1-10
    religious_service_attendance: int  # 1=More than once a week ~ 7=Never
    prayer_frequency: int  # 1=Several times a day ~ 8=Never
    religiosity: int  # 1=Religious, 2=Not religious, 3=Atheist
    
    # 사회적 태도
    reject_homosexual_neighbor: int  # 1=Mentioned, 2=Not mentioned
    homosexual_parents_opinion: int  # 1=Agree strongly ~ 5=Disagree strongly
    justifiability_premarital_sex: int  # 1-10
    justifiability_casual_sex: int  # 1-10
    
    # 정치
    political_left_right: int  # 1=Left ~ 10=Right
    importance_democracy: int  # 1-10
    
    def to_prompt(self) -> str:
        """페르소나를 프롬프트용 텍스트로 변환"""
        gender_str = "Male" if self.gender == 1 else "Female"
        
        marital_map = {
            1: "Married", 2: "Living together as married", 3: "Divorced",
            4: "Separated", 5: "Widowed", 6: "Single/Never married"
        }
        
        education_map = {1: "Lower (primary/secondary)", 2: "Middle (high school)", 3: "Higher (university+)"}
        
        social_class_map = {
            1: "Upper class", 2: "Upper middle class", 3: "Lower middle class",
            4: "Working class", 5: "Lower class"
        }
        
        importance_map = {1: "Very important", 2: "Rather important", 3: "Not very important", 4: "Not at all important"}
        
        religiosity_map = {1: "A religious person", 2: "Not a religious person", 3: "A convinced atheist"}
        
        attendance_map = {
            1: "More than once a week", 2: "Once a week", 3: "Once a month",
            4: "Only on special holy days", 5: "Once a year", 6: "Less often", 7: "Never"
        }
        
        prayer_map = {
            1: "Several times a day", 2: "Once a day", 3: "Several times each week",
            4: "Only when attending religious services", 5: "Only on special holy days",
            6: "Once a year", 7: "Less often", 8: "Never"
        }
        
        homosexual_opinion_map = {
            1: "Agree strongly", 2: "Agree", 3: "Neither agree nor disagree",
            4: "Disagree", 5: "Disagree strongly"
        }
        
        political_str = "Left/Progressive" if self.political_left_right <= 4 else \
                       "Center" if self.political_left_right <= 6 else "Right/Conservative"
        
        return f"""=== RESPONDENT PROFILE ===
DEMOGRAPHICS:
- Country: {self.country_name}
- Gender: {gender_str}
- Age: {self.age} years old
- Marital Status: {marital_map[self.marital_status]}
- Education Level: {education_map[self.education_level]}
- Social Class: {social_class_map[self.social_class]}
- Born in country: {"Yes" if self.born_in_country == 1 else "No (immigrant)"}
- Citizen: {"Yes" if self.is_citizen == 1 else "No"}

LIFE VALUES (importance in life):
- Family: {importance_map[self.importance_family]}
- Friends: {importance_map[self.importance_friends]}
- Leisure time: {importance_map[self.importance_leisure]}
- Politics: {importance_map[self.importance_politics]}
- Work: {importance_map[self.importance_work]}
- Religion: {importance_map[self.importance_religion]}

RELIGIOUS PROFILE:
- Self-identification: {religiosity_map[self.religiosity]}
- Importance of God in life: {self.importance_god}/10
- Religious service attendance: {attendance_map[self.religious_service_attendance]}
- Prayer frequency: {prayer_map[self.prayer_frequency]}

SOCIAL ATTITUDES:
- Would NOT want homosexuals as neighbors: {"Yes" if self.reject_homosexual_neighbor == 1 else "No"}
- "Homosexual couples are as good parents as other couples": {homosexual_opinion_map[self.homosexual_parents_opinion]}
- Justifiability of sex before marriage: {self.justifiability_premarital_sex}/10 (1=never, 10=always justifiable)
- Justifiability of casual sex: {self.justifiability_casual_sex}/10 (1=never, 10=always justifiable)

POLITICAL ORIENTATION:
- Left-Right scale: {self.political_left_right}/10 (1=Left, 10=Right) → {political_str}
- Importance of living in a democracy: {self.importance_democracy}/10
"""
    
    def get_expected_response_ranges(self) -> Dict[str, Dict[str, Any]]:
        """
        페르소나 특성에 기반한 예상 응답 범위 반환
        일관성 검증에 사용
        """
        ranges = {}
        
        # 정치 성향에 따른 예상 범위
        if self.political_left_right <= 3:  # 진보
            ranges["homosexual_tolerance"] = {"min": 5, "max": 10, "weight": 0.8}
            ranges["premarital_sex"] = {"min": 4, "max": 10, "weight": 0.6}
        elif self.political_left_right >= 8:  # 보수
            ranges["homosexual_tolerance"] = {"min": 1, "max": 5, "weight": 0.8}
            ranges["premarital_sex"] = {"min": 1, "max": 6, "weight": 0.6}
        else:  # 중도
            ranges["homosexual_tolerance"] = {"min": 3, "max": 8, "weight": 0.5}
            ranges["premarital_sex"] = {"min": 2, "max": 8, "weight": 0.5}
        
        # 종교성에 따른 예상 범위
        if self.religiosity == 1 and self.importance_god >= 8:  # 매우 독실
            ranges["casual_sex"] = {"min": 1, "max": 4, "weight": 0.9}
            ranges["premarital_sex"] = {"min": 1, "max": 5, "weight": 0.8}
        elif self.religiosity == 3:  # 무신론자
            ranges["casual_sex"] = {"min": 3, "max": 10, "weight": 0.6}
        
        # 동성애 관련 일관성
        if self.reject_homosexual_neighbor == 1:  # 동성애자 이웃 거부
            ranges["homosexual_parents"] = {"min": 3, "max": 5, "weight": 0.9}  # Disagree 쪽
        elif self.homosexual_parents_opinion <= 2:  # 동성 부모 긍정
            ranges["reject_neighbor"] = {"expected": 2, "weight": 0.8}  # Not mentioned
        
        return ranges


# =============================================================================
# 페르소나-응답 일관성 검증 클래스
# =============================================================================

class PersonaResponseValidator:
    """
    페르소나와 응답 간의 일관성을 검증하는 클래스
    
    규칙 기반 검증과 LLM 기반 검증을 모두 지원
    """
    
    # 일관성 규칙 정의
    CONSISTENCY_RULES = [
        {
            "name": "progressive_homosexual_tolerance",
            "description": "진보 성향은 동성애에 관대해야 함",
            "condition": lambda p: p.political_left_right <= 3,
            "check": lambda p, r: r.get("homosexual_justifiable", 5) >= 4 or p.homosexual_parents_opinion <= 3,
            "severity": "high"
        },
        {
            "name": "conservative_traditional_values",
            "description": "보수 성향은 전통적 가치관을 가져야 함",
            "condition": lambda p: p.political_left_right >= 8,
            "check": lambda p, r: r.get("premarital_sex_justifiable", 5) <= 7,
            "severity": "medium"
        },
        {
            "name": "religious_conservative_sex",
            "description": "독실한 종교인은 캐주얼 섹스에 부정적이어야 함",
            "condition": lambda p: p.religiosity == 1 and p.importance_god >= 8,
            "check": lambda p, r: r.get("casual_sex_justifiable", 5) <= 5,
            "severity": "high"
        },
        {
            "name": "atheist_secular_values",
            "description": "무신론자는 종교의 중요성을 낮게 평가해야 함",
            "condition": lambda p: p.religiosity == 3,
            "check": lambda p, r: r.get("religion_importance", 3) >= 3,  # Not very or Not at all
            "severity": "high"
        },
        {
            "name": "homosexual_attitude_consistency",
            "description": "동성애자 이웃 거부와 동성 부모 의견이 일관되어야 함",
            "condition": lambda p: p.reject_homosexual_neighbor == 1,
            "check": lambda p, r: p.homosexual_parents_opinion >= 3,  # Neither ~ Disagree
            "severity": "high"
        },
        {
            "name": "frequent_attendee_religious",
            "description": "자주 예배 참석하는 사람은 종교적이어야 함",
            "condition": lambda p: p.religious_service_attendance <= 2,
            "check": lambda p, r: p.religiosity == 1,
            "severity": "medium"
        },
        {
            "name": "never_pray_not_religious",
            "description": "기도를 안 하는 사람은 종교적이지 않아야 함",
            "condition": lambda p: p.prayer_frequency == 8,
            "check": lambda p, r: p.religiosity >= 2,
            "severity": "medium"
        },
    ]
    
    def __init__(self, use_llm: bool = False, chat_request_func=None):
        """
        Args:
            use_llm: LLM 기반 검증 사용 여부
            chat_request_func: LLM API 호출 함수 (use_llm=True일 때 필요)
        """
        self.use_llm = use_llm
        self.chat_request = chat_request_func
    
    def validate_internal_consistency(self, persona: WVSPersonaProfile) -> Dict[str, Any]:
        """
        페르소나 내부 일관성 검증 (생성 시점에서 체크)
        
        Returns:
            {
                "is_consistent": bool,
                "score": float (0-1),
                "violations": List[str],
                "warnings": List[str]
            }
        """
        violations = []
        warnings = []
        
        for rule in self.CONSISTENCY_RULES:
            if rule["condition"](persona):
                if not rule["check"](persona, {}):
                    msg = f"[{rule['name']}] {rule['description']}"
                    if rule["severity"] == "high":
                        violations.append(msg)
                    else:
                        warnings.append(msg)
        
        total_rules = len([r for r in self.CONSISTENCY_RULES if r["condition"](persona)])
        violated_count = len(violations) + len(warnings) * 0.5
        
        score = 1.0 - (violated_count / max(total_rules, 1))
        
        return {
            "is_consistent": len(violations) == 0,
            "score": max(0, score),
            "violations": violations,
            "warnings": warnings
        }
    
    def validate_response_consistency(
        self, 
        persona: WVSPersonaProfile, 
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        페르소나와 응답 간의 일관성 검증
        
        Args:
            persona: 페르소나 프로필
            responses: 응답 딕셔너리 (예: {"homosexual_justifiable": 8, "casual_sex_justifiable": 3})
        
        Returns:
            {
                "is_consistent": bool,
                "score": float (0-1),
                "violations": List[Dict],
                "details": str
            }
        """
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        # 1. 정치 성향 - 동성애 관련 일관성
        if "homosexual_justifiable" in responses:
            checks_performed += 1
            value = responses["homosexual_justifiable"]
            
            if persona.political_left_right <= 3 and value <= 2:
                violations.append({
                    "type": "political_homosexual_mismatch",
                    "message": f"진보 성향(정치={persona.political_left_right})인데 동성애 정당화={value}로 너무 보수적",
                    "severity": "high"
                })
            elif persona.political_left_right >= 8 and value >= 9:
                violations.append({
                    "type": "political_homosexual_mismatch",
                    "message": f"보수 성향(정치={persona.political_left_right})인데 동성애 정당화={value}로 너무 진보적",
                    "severity": "high"
                })
            else:
                checks_passed += 1
        
        # 2. 종교성 - 성적 보수성 일관성
        if "casual_sex_justifiable" in responses:
            checks_performed += 1
            value = responses["casual_sex_justifiable"]
            
            if persona.religiosity == 1 and persona.importance_god >= 8 and value >= 8:
                violations.append({
                    "type": "religious_sex_mismatch",
                    "message": f"독실한 종교인(신 중요도={persona.importance_god})인데 캐주얼 섹스 정당화={value}",
                    "severity": "high"
                })
            elif persona.religiosity == 3 and persona.importance_god <= 2 and value <= 2:
                violations.append({
                    "type": "atheist_sex_mismatch",
                    "message": f"무신론자인데 캐주얼 섹스 정당화={value}로 너무 보수적",
                    "severity": "medium"
                })
            else:
                checks_passed += 1
        
        # 3. 혼전 성관계 일관성
        if "premarital_sex_justifiable" in responses:
            checks_performed += 1
            value = responses["premarital_sex_justifiable"]
            
            # 종교적 보수와 비교
            if persona.religiosity == 1 and persona.religious_service_attendance <= 2 and value >= 9:
                violations.append({
                    "type": "religious_premarital_mismatch",
                    "message": f"주 1회 이상 예배 참석하는 종교인인데 혼전 성관계 정당화={value}",
                    "severity": "medium"
                })
            else:
                checks_passed += 1
        
        # 4. 동성애자 이웃 거부 일관성
        if "reject_homosexual_neighbor_response" in responses:
            checks_performed += 1
            value = responses["reject_homosexual_neighbor_response"]
            
            # 페르소나의 동성 부모 의견과 비교
            if persona.homosexual_parents_opinion <= 2 and value == 1:  # 동성 부모 긍정인데 이웃 거부
                violations.append({
                    "type": "homosexual_attitude_inconsistency",
                    "message": "동성 부모에 긍정적인데 동성애자 이웃은 거부",
                    "severity": "high"
                })
            elif persona.homosexual_parents_opinion >= 4 and value == 2:  # 동성 부모 부정인데 이웃 수용
                # 이건 가능할 수 있음 (관용적이지만 양육에는 반대)
                checks_passed += 1
            else:
                checks_passed += 1
        
        # 5. 민주주의 중요도 일관성 (정치 관심도와)
        if "democracy_importance_response" in responses:
            checks_performed += 1
            value = responses["democracy_importance_response"]
            
            if persona.importance_politics <= 2 and value <= 3:
                violations.append({
                    "type": "democracy_politics_mismatch",
                    "message": f"정치가 중요하다고 했는데 민주주의 중요도={value}로 낮음",
                    "severity": "low"
                })
            else:
                checks_passed += 1
        
        # 점수 계산
        if checks_performed > 0:
            base_score = checks_passed / checks_performed
        else:
            base_score = 1.0
        
        # 위반 심각도에 따른 감점
        high_violations = len([v for v in violations if v["severity"] == "high"])
        medium_violations = len([v for v in violations if v["severity"] == "medium"])
        
        penalty = high_violations * 0.2 + medium_violations * 0.1
        final_score = max(0, base_score - penalty)
        
        return {
            "is_consistent": len([v for v in violations if v["severity"] == "high"]) == 0,
            "score": final_score,
            "violations": violations,
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
            "details": self._generate_details(persona, responses, violations)
        }
    
    def _generate_details(
        self, 
        persona: WVSPersonaProfile, 
        responses: Dict, 
        violations: List
    ) -> str:
        """검증 결과 상세 설명 생성"""
        details = []
        details.append(f"페르소나 핵심 특성:")
        details.append(f"  - 정치 성향: {persona.political_left_right}/10 ({'진보' if persona.political_left_right <= 4 else '보수' if persona.political_left_right >= 7 else '중도'})")
        details.append(f"  - 종교성: {['', '종교인', '비종교인', '무신론자'][persona.religiosity]}")
        details.append(f"  - 신 중요도: {persona.importance_god}/10")
        details.append(f"  - 동성 부모 의견: {persona.homosexual_parents_opinion}/5")
        
        if violations:
            details.append(f"\n발견된 불일치 ({len(violations)}건):")
            for v in violations:
                details.append(f"  [{v['severity'].upper()}] {v['message']}")
        else:
            details.append("\n✓ 모든 응답이 페르소나와 일관됩니다.")
        
        return "\n".join(details)
    
    def validate_with_llm(
        self, 
        persona: WVSPersonaProfile, 
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM을 사용한 일관성 검증
        
        Args:
            persona: 페르소나 프로필
            responses: 응답 딕셔너리
        
        Returns:
            {"score": float, "explanation": str, "is_consistent": bool}
        """
        if not self.use_llm or not self.chat_request:
            raise ValueError("LLM validation requires use_llm=True and chat_request_func")
        
        import json
        
        validation_prompt = f"""You are evaluating whether survey responses are consistent with the respondent's profile.

RESPONDENT PROFILE:
{persona.to_prompt()}

RESPONDENT'S SURVEY RESPONSES:
{json.dumps(responses, indent=2)}

TASK: Evaluate the consistency between the profile and responses.

Consider these key consistency checks:
1. Political orientation (left=progressive, right=conservative) should align with social attitudes
2. Religious people should have more conservative views on sex/marriage
3. Atheists should rate religion/God as unimportant
4. Attitudes toward homosexuality should be internally consistent

Return your evaluation in this exact JSON format:
{{
    "score": <float between 0 and 1>,
    "is_consistent": <true if score >= 0.7>,
    "explanation": "<brief explanation of consistency issues found>",
    "violations": ["<list of specific inconsistencies found>"]
}}

Return ONLY the JSON, no other text."""

        try:
            # chat_request 함수 호출 (사용자 구현에 따라 다를 수 있음)
            response = self.chat_request(
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.1
            )
            
            # 응답 파싱
            import json
            result = json.loads(response.content.strip())
            
            return {
                "score": max(0, min(1, float(result.get("score", 0.5)))),
                "is_consistent": result.get("is_consistent", False),
                "explanation": result.get("explanation", ""),
                "violations": result.get("violations", []),
                "method": "llm"
            }
        except Exception as e:
            return {
                "score": -1.0,
                "is_consistent": False,
                "explanation": f"LLM validation failed: {str(e)}",
                "violations": [],
                "method": "llm",
                "error": str(e)
            }
    
    def full_validation(
        self, 
        persona: WVSPersonaProfile, 
        responses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        규칙 기반 + LLM 검증을 모두 수행
        
        Returns:
            종합 검증 결과
        """
        # 규칙 기반 검증
        rule_based = self.validate_response_consistency(persona, responses)
        
        result = {
            "rule_based": rule_based,
            "combined_score": rule_based["score"],
            "is_consistent": rule_based["is_consistent"]
        }
        
        # LLM 검증 (활성화된 경우)
        if self.use_llm and self.chat_request:
            llm_based = self.validate_with_llm(persona, responses)
            result["llm_based"] = llm_based
            
            # 점수 결합 (규칙 기반 70%, LLM 30%)
            if llm_based["score"] >= 0:
                result["combined_score"] = rule_based["score"] * 0.7 + llm_based["score"] * 0.3
                result["is_consistent"] = result["combined_score"] >= 0.7
        
        return result


# =============================================================================
# 샘플링 유틸리티 함수
# =============================================================================

def sample_from_distribution(distribution: Dict[str, float]) -> str:
    """확률 분포에서 샘플링"""
    items = list(distribution.keys())
    weights = list(distribution.values())
    
    # 가중치 합이 100이 아닐 수 있으므로 정규화
    total = sum(weights)
    if total > 0:
        weights = [w / total * 100 for w in weights]
    
    return random.choices(items, weights=weights)[0]


def sample_from_scale_distribution(distribution: Dict[int, float]) -> int:
    """1-10 스케일 분포에서 샘플링"""
    items = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(items, weights=weights)[0]


# =============================================================================
# 메인 페르소나 생성기 클래스
# =============================================================================

class WVSPersonaGenerator:
    """WVS Wave 7 실제 분포 기반 페르소나 생성기"""
    
    COUNTRIES = {
        840: "United States",
        276: "Germany",
        826: "Great Britain",
        392: "Japan",
        410: "South Korea",
        356: "India",
        528: "Netherlands"
    }
    
    def __init__(self, country_code: int):
        if country_code not in self.COUNTRIES:
            raise ValueError(f"Country code must be one of {list(self.COUNTRIES.keys())}")
        
        self.country_code = country_code
        self.country_name = self.COUNTRIES[country_code]
    
    def _sample_gender(self) -> int:
        """성별 샘플링"""
        dist = GENDER_DISTRIBUTION[self.country_name]
        total = dist["male"] + dist["female"]
        return 1 if random.random() * total < dist["male"] else 2
    
    def _sample_age(self) -> int:
        """나이 샘플링 (정규분포 + 범위 제한)"""
        dist = AGE_DISTRIBUTION[self.country_name]
        
        # 정규분포에서 샘플링
        age = int(random.gauss(dist["mean"], dist["std"]))
        
        # 18-90 범위로 제한
        return max(18, min(90, age))
    
    def _sample_citizenship(self) -> int:
        """시민권 샘플링"""
        dist = CITIZENSHIP_DISTRIBUTION[self.country_name]
        total = dist["yes"] + dist["no"]
        return 1 if random.random() * total < dist["yes"] else 2
    
    def _sample_marital_status(self) -> int:
        """결혼 상태 샘플링"""
        dist = MARITAL_STATUS_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "married": 1, "living_together": 2, "divorced": 3,
            "separated": 4, "widowed": 5, "single": 6
        }
        return mapping[result]
    
    def _sample_education(self) -> int:
        """교육 수준 샘플링"""
        dist = EDUCATION_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {"lower": 1, "middle": 2, "higher": 3}
        return mapping[result]
    
    def _sample_parent_education(self, parent: str) -> int:
        """부모 교육 수준 샘플링"""
        if parent == "mother":
            dist = MOTHER_EDUCATION_DISTRIBUTION[self.country_name]
        else:
            dist = FATHER_EDUCATION_DISTRIBUTION[self.country_name]
        
        result = sample_from_distribution(dist)
        mapping = {"lower": 1, "middle": 2, "higher": 3}
        return mapping[result]
    
    def _sample_social_class(self) -> int:
        """사회 계층 샘플링"""
        dist = SOCIAL_CLASS_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "upper": 1, "upper_middle": 2, "lower_middle": 3,
            "working": 4, "lower": 5
        }
        return mapping[result]
    
    def _sample_born_in_country(self) -> int:
        """출생지 샘플링"""
        dist = BORN_IN_COUNTRY_DISTRIBUTION[self.country_name]
        total = dist["born_here"] + dist["immigrant"]
        return 1 if random.random() * total < dist["born_here"] else 2
    
    def _sample_parent_immigrant(self, parent: str) -> int:
        """부모 이민자 여부 샘플링"""
        if parent == "mother":
            dist = MOTHER_IMMIGRANT_DISTRIBUTION[self.country_name]
        else:
            dist = FATHER_IMMIGRANT_DISTRIBUTION[self.country_name]
        
        total = dist["not_immigrant"] + dist["immigrant"]
        return 1 if random.random() * total < dist["not_immigrant"] else 2
    
    def _sample_importance(self, question: str) -> int:
        """중요도 샘플링 (Q1-Q6)"""
        distributions = {
            "family": IMPORTANCE_FAMILY_DISTRIBUTION,
            "friends": IMPORTANCE_FRIENDS_DISTRIBUTION,
            "leisure": IMPORTANCE_LEISURE_DISTRIBUTION,
            "politics": IMPORTANCE_POLITICS_DISTRIBUTION,
            "work": IMPORTANCE_WORK_DISTRIBUTION,
            "religion": IMPORTANCE_RELIGION_DISTRIBUTION,
        }
        
        dist = distributions[question][self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "very_important": 1, "rather_important": 2,
            "not_very_important": 3, "not_at_all": 4
        }
        return mapping[result]
    
    def _sample_importance_god(self) -> int:
        """신 중요도 샘플링 (1-10)"""
        dist = IMPORTANCE_GOD_DISTRIBUTION[self.country_name]
        return sample_from_scale_distribution(dist)
    
    def _sample_religious_service_attendance(self) -> int:
        """종교 예배 참석 빈도 샘플링"""
        dist = RELIGIOUS_SERVICE_ATTENDANCE_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "more_than_once_week": 1, "once_week": 2, "once_month": 3,
            "holy_days": 4, "once_year": 5, "less_often": 6, "never": 7
        }
        return mapping[result]
    
    def _sample_prayer_frequency(self) -> int:
        """기도 빈도 샘플링"""
        dist = PRAYER_FREQUENCY_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "several_times_day": 1, "once_day": 2, "several_times_week": 3,
            "religious_services_only": 4, "holy_days_only": 5, "once_year": 6,
            "less_often": 7, "never": 8
        }
        return mapping[result]
    
    def _sample_religiosity(self) -> int:
        """종교적 정체성 샘플링"""
        dist = RELIGIOSITY_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {"religious": 1, "not_religious": 2, "atheist": 3}
        return mapping[result]
    
    def _sample_reject_homosexual_neighbor(self) -> int:
        """동성애자 이웃 거부 샘플링"""
        dist = REJECT_HOMOSEXUAL_NEIGHBOR_DISTRIBUTION[self.country_name]
        total = dist["mentioned"] + dist["not_mentioned"]
        return 1 if random.random() * total < dist["mentioned"] else 2
    
    def _sample_homosexual_parents_opinion(self) -> int:
        """동성 부부 양육 의견 샘플링"""
        dist = HOMOSEXUAL_PARENTS_DISTRIBUTION[self.country_name]
        result = sample_from_distribution(dist)
        
        mapping = {
            "agree_strongly": 1, "agree": 2, "neither": 3,
            "disagree": 4, "disagree_strongly": 5
        }
        return mapping[result]
    
    def _sample_premarital_sex(self) -> int:
        """혼전 성관계 정당화 샘플링"""
        dist = PREMARITAL_SEX_DISTRIBUTION[self.country_name]
        return sample_from_scale_distribution(dist)
    
    def _sample_casual_sex(self) -> int:
        """캐주얼 섹스 정당화 샘플링"""
        dist = CASUAL_SEX_DISTRIBUTION[self.country_name]
        return sample_from_scale_distribution(dist)
    
    def _sample_political_left_right(self) -> int:
        """정치 성향 샘플링"""
        dist = POLITICAL_LEFT_RIGHT_DISTRIBUTION[self.country_name]
        return sample_from_scale_distribution(dist)
    
    def _sample_importance_democracy(self) -> int:
        """민주주의 중요도 샘플링"""
        dist = IMPORTANCE_DEMOCRACY_DISTRIBUTION[self.country_name]
        return sample_from_scale_distribution(dist)
    
    def generate_persona(self, **fixed_attributes) -> WVSPersonaProfile:
        """
        WVS 실제 분포에 따라 페르소나 생성
        
        Args:
            **fixed_attributes: 고정할 속성들 (예: age=30, gender=1)
        
        Returns:
            WVSPersonaProfile 객체
        """
        return WVSPersonaProfile(
            country_code=self.country_code,
            country_name=self.country_name,
            gender=fixed_attributes.get("gender", self._sample_gender()),
            age=fixed_attributes.get("age", self._sample_age()),
            born_in_country=fixed_attributes.get("born_in_country", self._sample_born_in_country()),
            mother_immigrant=fixed_attributes.get("mother_immigrant", self._sample_parent_immigrant("mother")),
            father_immigrant=fixed_attributes.get("father_immigrant", self._sample_parent_immigrant("father")),
            is_citizen=fixed_attributes.get("is_citizen", self._sample_citizenship()),
            marital_status=fixed_attributes.get("marital_status", self._sample_marital_status()),
            education_level=fixed_attributes.get("education_level", self._sample_education()),
            mother_education=fixed_attributes.get("mother_education", self._sample_parent_education("mother")),
            father_education=fixed_attributes.get("father_education", self._sample_parent_education("father")),
            social_class=fixed_attributes.get("social_class", self._sample_social_class()),
            importance_family=fixed_attributes.get("importance_family", self._sample_importance("family")),
            importance_friends=fixed_attributes.get("importance_friends", self._sample_importance("friends")),
            importance_leisure=fixed_attributes.get("importance_leisure", self._sample_importance("leisure")),
            importance_politics=fixed_attributes.get("importance_politics", self._sample_importance("politics")),
            importance_work=fixed_attributes.get("importance_work", self._sample_importance("work")),
            importance_religion=fixed_attributes.get("importance_religion", self._sample_importance("religion")),
            importance_god=fixed_attributes.get("importance_god", self._sample_importance_god()),
            religious_service_attendance=fixed_attributes.get("religious_service_attendance", self._sample_religious_service_attendance()),
            prayer_frequency=fixed_attributes.get("prayer_frequency", self._sample_prayer_frequency()),
            religiosity=fixed_attributes.get("religiosity", self._sample_religiosity()),
            reject_homosexual_neighbor=fixed_attributes.get("reject_homosexual_neighbor", self._sample_reject_homosexual_neighbor()),
            homosexual_parents_opinion=fixed_attributes.get("homosexual_parents_opinion", self._sample_homosexual_parents_opinion()),
            justifiability_premarital_sex=fixed_attributes.get("justifiability_premarital_sex", self._sample_premarital_sex()),
            justifiability_casual_sex=fixed_attributes.get("justifiability_casual_sex", self._sample_casual_sex()),
            political_left_right=fixed_attributes.get("political_left_right", self._sample_political_left_right()),
            importance_democracy=fixed_attributes.get("importance_democracy", self._sample_importance_democracy()),
        )
    
    def generate_multiple_personas(self, n: int, **fixed_attributes) -> List[WVSPersonaProfile]:
        """여러 페르소나 생성"""
        return [self.generate_persona(**fixed_attributes) for _ in range(n)]


# =============================================================================
# 테스트 및 검증
# =============================================================================

def validate_distribution(generated: List[Any], expected_dist: Dict[str, float], name: str):
    """생성된 분포가 예상 분포와 유사한지 검증"""
    from collections import Counter
    counts = Counter(generated)
    total = len(generated)
    
    print(f"\n{name} Distribution Validation:")
    print("-" * 50)
    
    for key, expected_pct in expected_dist.items():
        actual_count = counts.get(key, 0)
        actual_pct = (actual_count / total) * 100
        diff = abs(actual_pct - expected_pct)
        status = "✓" if diff < 5 else "△" if diff < 10 else "✗"
        print(f"  {key:20s}: Expected {expected_pct:5.1f}%, Got {actual_pct:5.1f}% {status}")


if __name__ == "__main__":
    print("=" * 70)
    print("WVS Wave 7 - Real Distribution Based Persona Generator")
    print("=" * 70)
    
    # 각 나라별 테스트
    for code, name in WVSPersonaGenerator.COUNTRIES.items():
        print(f"\n{'='*70}")
        print(f"Testing: {name} (Code: {code})")
        print("=" * 70)
        
        generator = WVSPersonaGenerator(country_code=code)
        personas = generator.generate_multiple_personas(1000)
        
        # 성별 분포 검증
        genders = ["male" if p.gender == 1 else "female" for p in personas]
        validate_distribution(genders, GENDER_DISTRIBUTION[name], "Gender")
        
        # 결혼 상태 분포 검증
        marital_map = {1: "married", 2: "living_together", 3: "divorced", 
                       4: "separated", 5: "widowed", 6: "single"}
        marital = [marital_map[p.marital_status] for p in personas]
        validate_distribution(marital, MARITAL_STATUS_DISTRIBUTION[name], "Marital Status")
        
        # 사회 계층 분포 검증
        class_map = {1: "upper", 2: "upper_middle", 3: "lower_middle", 
                     4: "working", 5: "lower"}
        social = [class_map[p.social_class] for p in personas]
        validate_distribution(social, SOCIAL_CLASS_DISTRIBUTION[name], "Social Class")
        
        # 종교성 분포 검증
        religiosity_map = {1: "religious", 2: "not_religious", 3: "atheist"}
        religiosity = [religiosity_map[p.religiosity] for p in personas]
        validate_distribution(religiosity, RELIGIOSITY_DISTRIBUTION[name], "Religiosity")
        
        # 샘플 페르소나 출력
        print(f"\nSample Personas from {name}:")
        print("-" * 50)
        for i in range(3):
            p = personas[i]
            print(f"  Persona {i+1}:")
            print(f"    Gender: {'Male' if p.gender == 1 else 'Female'}, Age: {p.age}")
            print(f"    Education: {['', 'Lower', 'Middle', 'Higher'][p.education_level]}")
            print(f"    Social Class: {class_map[p.social_class]}")
            print(f"    Religiosity: {religiosity_map[p.religiosity]}")
            print(f"    Political L-R: {p.political_left_right}/10")
    
    print("\n" + "=" * 70)
    print("All distribution tests completed!")
    print("=" * 70)
    
    # ==========================================================================
    # 일관성 검증 테스트
    # ==========================================================================
    print("\n" + "=" * 70)
    print("CONSISTENCY VALIDATION TEST")
    print("=" * 70)
    
    validator = PersonaResponseValidator(use_llm=False)
    
    # 테스트 케이스 1: 일관된 진보적 페르소나
    print("\n[Test 1] Consistent Progressive Persona")
    print("-" * 50)
    generator = WVSPersonaGenerator(country_code=840)
    
    # 진보적 페르소나 생성 (고정 속성)
    progressive_persona = generator.generate_persona(
        political_left_right=2,  # 진보
        religiosity=3,  # 무신론자
        importance_god=2,
        homosexual_parents_opinion=1,  # 강하게 동의
        reject_homosexual_neighbor=2,  # 거부 안 함
    )
    
    # 일관된 응답
    consistent_responses = {
        "homosexual_justifiable": 9,
        "casual_sex_justifiable": 7,
        "premarital_sex_justifiable": 9,
    }
    
    result = validator.validate_response_consistency(progressive_persona, consistent_responses)
    print(f"Score: {result['score']:.2f}")
    print(f"Is Consistent: {result['is_consistent']}")
    print(result['details'])
    
    # 테스트 케이스 2: 불일치 페르소나 (진보인데 보수적 응답)
    print("\n[Test 2] Inconsistent Progressive Persona (conservative responses)")
    print("-" * 50)
    
    inconsistent_responses = {
        "homosexual_justifiable": 1,  # 진보인데 동성애 절대 불가?
        "casual_sex_justifiable": 1,
        "premarital_sex_justifiable": 1,
    }
    
    result = validator.validate_response_consistency(progressive_persona, inconsistent_responses)
    print(f"Score: {result['score']:.2f}")
    print(f"Is Consistent: {result['is_consistent']}")
    print(result['details'])
    
    # 테스트 케이스 3: 독실한 종교인 페르소나
    print("\n[Test 3] Devout Religious Persona")
    print("-" * 50)
    
    religious_persona = generator.generate_persona(
        political_left_right=8,  # 보수
        religiosity=1,  # 종교인
        importance_god=10,
        religious_service_attendance=1,  # 주 1회 이상
        homosexual_parents_opinion=5,  # 강하게 반대
        reject_homosexual_neighbor=1,  # 거부
    )
    
    # 종교인에게 맞는 응답
    religious_responses = {
        "homosexual_justifiable": 2,
        "casual_sex_justifiable": 1,
        "premarital_sex_justifiable": 2,
    }
    
    result = validator.validate_response_consistency(religious_persona, religious_responses)
    print(f"Score: {result['score']:.2f}")
    print(f"Is Consistent: {result['is_consistent']}")
    print(result['details'])
    
    # 테스트 케이스 4: 독실한 종교인인데 진보적 응답 (불일치)
    print("\n[Test 4] Devout Religious with Progressive Responses (inconsistent)")
    print("-" * 50)
    
    liberal_responses = {
        "homosexual_justifiable": 10,
        "casual_sex_justifiable": 10,  # 독실한 종교인이 캐주얼 섹스 완전 찬성?
        "premarital_sex_justifiable": 10,
    }
    
    result = validator.validate_response_consistency(religious_persona, liberal_responses)
    print(f"Score: {result['score']:.2f}")
    print(f"Is Consistent: {result['is_consistent']}")
    print(result['details'])
    
    # 테스트 케이스 5: 내부 일관성 검증
    print("\n[Test 5] Internal Consistency Check")
    print("-" * 50)
    
    # 내부적으로 불일치한 페르소나 (무신론자인데 주 1회 예배 참석)
    inconsistent_persona = generator.generate_persona(
        religiosity=3,  # 무신론자
        religious_service_attendance=1,  # 주 1회 이상 참석?
        importance_god=1,
        prayer_frequency=8,  # 기도 안 함
    )
    
    internal_result = validator.validate_internal_consistency(inconsistent_persona)
    print(f"Internal Consistency Score: {internal_result['score']:.2f}")
    print(f"Is Internally Consistent: {internal_result['is_consistent']}")
    if internal_result['violations']:
        print("Violations:")
        for v in internal_result['violations']:
            print(f"  - {v}")
    if internal_result['warnings']:
        print("Warnings:")
        for w in internal_result['warnings']:
            print(f"  - {w}")
    
    print("\n" + "=" * 70)
    print("All validation tests completed!")
    print("=" * 70)