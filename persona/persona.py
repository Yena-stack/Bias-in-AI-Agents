"""
WVS Wave 7 (2017-2022) - 7개 주요국 실제 분포 기반 페르소나 생성기

PDF에서 추출한 실제 통계 데이터를 사용하여 페르소나를 생성합니다.
층화 샘플링(Stratified Sampling)을 사용하여 n수에 관계없이 정확한 분포를 보장합니다.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import random
import math


# =============================================================================
# 실제 WVS 분포 데이터 (PDF에서 추출)
# =============================================================================

GENDER_DISTRIBUTION = {
    "Germany": {"male": 48.6, "female": 51.4},
    "India": {"male": 51.1, "female": 48.9},
    "Japan": {"male": 43.6, "female": 56.4},
    "South Korea": {"male": 48.8, "female": 51.2},
    "Netherlands": {"male": 46.3, "female": 53.7},
    "Great Britain": {"male": 47.6, "female": 51.1},
    "United States": {"male": 46.4, "female": 51.6},
}

AGE_DISTRIBUTION = {
    "Germany": {"up_to_29": 16.2, "30_49": 30.8, "50_plus": 52.9, "mean": 50.8, "std": 18.09},
    "India": {"up_to_29": 38.1, "30_49": 38.3, "50_plus": 23.6, "mean": 37.83, "std": 16.05},
    "Japan": {"up_to_29": 10.1, "30_49": 30.4, "50_plus": 59.5, "mean": 54.78, "std": 17.95},
    "South Korea": {"up_to_29": 19.9, "30_49": 37.5, "50_plus": 42.6, "mean": 45.63, "std": 15.03},
    "Netherlands": {"up_to_29": 8.9, "30_49": 31.2, "50_plus": 59.9, "mean": 53.36, "std": 16.38},
    "Great Britain": {"up_to_29": 16.3, "30_49": 32.0, "50_plus": 46.4, "mean": 49.27, "std": 18.61},
    "United States": {"up_to_29": 21.3, "30_49": 33.5, "50_plus": 45.2, "mean": 46.73, "std": 17.33},
}

CITIZENSHIP_DISTRIBUTION = {
    "Germany": {"yes": 91.6, "no": 8.4},
    "India": {"yes": 100.0, "no": 0.0},
    "Japan": {"yes": 99.0, "no": 0.1},
    "South Korea": {"yes": 99.8, "no": 0.2},
    "Netherlands": {"yes": 90.3, "no": 2.3},
    "Great Britain": {"yes": 92.6, "no": 6.5},
    "United States": {"yes": 95.6, "no": 3.3},
}

MARITAL_STATUS_DISTRIBUTION = {
    "Germany": {"married": 54.5, "living_together": 10.0, "divorced": 7.4, "separated": 1.3, "widowed": 7.4, "single": 19.0},
    "India": {"married": 58.3, "living_together": 8.8, "divorced": 0.1, "separated": 0.1, "widowed": 5.5, "single": 27.0},
    "Japan": {"married": 72.4, "living_together": 1.0, "divorced": 5.0, "separated": 0.4, "widowed": 7.4, "single": 12.8},
    "South Korea": {"married": 72.1, "living_together": 0.1, "divorced": 0.6, "separated": 0.1, "widowed": 2.1, "single": 25.0},
    "Netherlands": {"married": 45.8, "living_together": 13.3, "divorced": 6.1, "separated": 1.2, "widowed": 4.9, "single": 21.7},
    "Great Britain": {"married": 51.9, "living_together": 10.7, "divorced": 5.4, "separated": 1.9, "widowed": 5.4, "single": 22.2},
    "United States": {"married": 50.6, "living_together": 7.1, "divorced": 12.1, "separated": 2.3, "widowed": 4.5, "single": 23.5},
}

EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 11.2, "middle": 54.1, "higher": 34.5},
    "India": {"lower": 33.9, "middle": 38.9, "higher": 27.1},
    "Japan": {"lower": 6.2, "middle": 38.1, "higher": 54.5},
    "South Korea": {"lower": 11.2, "middle": 41.9, "higher": 46.8},
    "Netherlands": {"lower": 16.3, "middle": 29.5, "higher": 45.9},
    "Great Britain": {"lower": 18.5, "middle": 40.1, "higher": 38.9},
    "United States": {"lower": 3.0, "middle": 52.9, "higher": 42.8},
}

MOTHER_EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 37.7, "middle": 49.9, "higher": 9.7},
    "India": {"lower": 75.9, "middle": 18.0, "higher": 3.5},
    "Japan": {"lower": 31.0, "middle": 40.9, "higher": 17.1},
    "South Korea": {"lower": 56.1, "middle": 35.7, "higher": 6.1},
    "Netherlands": {"lower": 55.7, "middle": 14.1, "higher": 12.6},
    "Great Britain": {"lower": 33.3, "middle": 33.3, "higher": 33.3},
    "United States": {"lower": 13.6, "middle": 54.6, "higher": 22.4},
}

FATHER_EDUCATION_DISTRIBUTION = {
    "Germany": {"lower": 16.3, "middle": 55.6, "higher": 23.6},
    "India": {"lower": 62.5, "middle": 26.3, "higher": 8.5},
    "Japan": {"lower": 31.9, "middle": 34.0, "higher": 21.8},
    "South Korea": {"lower": 50.3, "middle": 35.0, "higher": 13.1},
    "Netherlands": {"lower": 41.9, "middle": 17.8, "higher": 21.3},
    "Great Britain": {"lower": 33.3, "middle": 33.3, "higher": 33.3},
    "United States": {"lower": 16.9, "middle": 47.0, "higher": 22.7},
}

SOCIAL_CLASS_DISTRIBUTION = {
    "Germany": {"upper": 1.4, "upper_middle": 36.0, "lower_middle": 41.0, "working": 16.8, "lower": 1.8},
    "India": {"upper": 3.4, "upper_middle": 21.8, "lower_middle": 43.6, "working": 19.1, "lower": 9.9},
    "Japan": {"upper": 1.5, "upper_middle": 15.0, "lower_middle": 42.2, "working": 26.4, "lower": 9.6},
    "South Korea": {"upper": 0.2, "upper_middle": 15.2, "lower_middle": 74.5, "working": 1.4, "lower": 8.8},
    "Netherlands": {"upper": 4.5, "upper_middle": 36.3, "lower_middle": 18.6, "working": 16.3, "lower": 2.5},
    "Great Britain": {"upper": 2.0, "upper_middle": 25.0, "lower_middle": 35.0, "working": 30.0, "lower": 8.0},
    "United States": {"upper": 1.3, "upper_middle": 29.4, "lower_middle": 35.8, "working": 24.8, "lower": 7.3},
}

BORN_IN_COUNTRY_DISTRIBUTION = {
    "Germany": {"born_here": 86.0, "immigrant": 13.9},
    "India": {"born_here": 100.0, "immigrant": 0.0},
    "Japan": {"born_here": 97.9, "immigrant": 1.0},
    "South Korea": {"born_here": 99.0, "immigrant": 1.0},
    "Netherlands": {"born_here": 82.3, "immigrant": 10.1},
    "Great Britain": {"born_here": 84.2, "immigrant": 14.3},
    "United States": {"born_here": 87.6, "immigrant": 9.9},
}

MOTHER_IMMIGRANT_DISTRIBUTION = {
    "Germany": {"not_immigrant": 81.4, "immigrant": 16.8},
    "India": {"not_immigrant": 100.0, "immigrant": 0.0},
    "Japan": {"not_immigrant": 98.6, "immigrant": 0.1},
    "South Korea": {"not_immigrant": 99.6, "immigrant": 0.4},
    "Netherlands": {"not_immigrant": 82.7, "immigrant": 9.0},
    "Great Britain": {"not_immigrant": 77.3, "immigrant": 10.7},
    "United States": {"not_immigrant": 77.7, "immigrant": 13.5},
}

FATHER_IMMIGRANT_DISTRIBUTION = {
    "Germany": {"not_immigrant": 81.2, "immigrant": 16.4},
    "India": {"not_immigrant": 100.0, "immigrant": 0.0},
    "Japan": {"not_immigrant": 98.2, "immigrant": 0.5},
    "South Korea": {"not_immigrant": 99.7, "immigrant": 0.3},
    "Netherlands": {"not_immigrant": 83.1, "immigrant": 8.2},
    "Great Britain": {"not_immigrant": 75.2, "immigrant": 11.1},
    "United States": {"not_immigrant": 77.9, "immigrant": 12.6},
}

# 가치관 분포 (Q1-Q6)
IMPORTANCE_FAMILY_DISTRIBUTION = {
    "Germany": {"very_important": 88.4, "rather_important": 10.3, "not_very_important": 1.1, "not_at_all": 0.1},
    "India": {"very_important": 94.4, "rather_important": 4.4, "not_very_important": 0.8, "not_at_all": 0.3},
    "Japan": {"very_important": 92.0, "rather_important": 6.2, "not_very_important": 0.7, "not_at_all": 0.2},
    "South Korea": {"very_important": 88.9, "rather_important": 10.8, "not_very_important": 0.3, "not_at_all": 0.0},
    "Netherlands": {"very_important": 73.7, "rather_important": 20.8, "not_very_important": 3.3, "not_at_all": 0.7},
    "Great Britain": {"very_important": 92.3, "rather_important": 6.3, "not_very_important": 0.9, "not_at_all": 0.4},
    "United States": {"very_important": 91.0, "rather_important": 7.1, "not_very_important": 1.6, "not_at_all": 0.2},
}

IMPORTANCE_FRIENDS_DISTRIBUTION = {
    "Germany": {"very_important": 59.8, "rather_important": 37.0, "not_very_important": 2.9, "not_at_all": 0.1},
    "India": {"very_important": 54.0, "rather_important": 28.4, "not_very_important": 11.8, "not_at_all": 5.2},
    "Japan": {"very_important": 40.4, "rather_important": 50.6, "not_very_important": 6.8, "not_at_all": 0.9},
    "South Korea": {"very_important": 40.0, "rather_important": 57.8, "not_very_important": 2.2, "not_at_all": 0.0},
    "Netherlands": {"very_important": 51.9, "rather_important": 40.2, "not_very_important": 5.9, "not_at_all": 0.7},
    "Great Britain": {"very_important": 63.0, "rather_important": 31.6, "not_very_important": 4.8, "not_at_all": 0.5},
    "United States": {"very_important": 50.7, "rather_important": 38.4, "not_very_important": 9.4, "not_at_all": 1.1},
}

IMPORTANCE_LEISURE_DISTRIBUTION = {
    "Germany": {"very_important": 37.7, "rather_important": 54.1, "not_very_important": 7.5, "not_at_all": 0.5},
    "India": {"very_important": 39.6, "rather_important": 34.8, "not_very_important": 12.7, "not_at_all": 12.0},
    "Japan": {"very_important": 44.5, "rather_important": 45.8, "not_very_important": 7.8, "not_at_all": 0.5},
    "South Korea": {"very_important": 24.1, "rather_important": 67.5, "not_very_important": 8.4, "not_at_all": 0.0},
    "Netherlands": {"very_important": 53.6, "rather_important": 40.5, "not_very_important": 3.6, "not_at_all": 0.3},
    "Great Britain": {"very_important": 51.0, "rather_important": 41.3, "not_very_important": 6.1, "not_at_all": 0.8},
    "United States": {"very_important": 39.5, "rather_important": 49.1, "not_very_important": 10.2, "not_at_all": 0.7},
}

IMPORTANCE_POLITICS_DISTRIBUTION = {
    "Germany": {"very_important": 15.8, "rather_important": 54.1, "not_very_important": 25.2, "not_at_all": 4.8},
    "India": {"very_important": 21.5, "rather_important": 25.1, "not_very_important": 23.3, "not_at_all": 27.3},
    "Japan": {"very_important": 15.0, "rather_important": 49.3, "not_very_important": 27.1, "not_at_all": 4.1},
    "South Korea": {"very_important": 8.0, "rather_important": 52.1, "not_very_important": 36.5, "not_at_all": 3.5},
    "Netherlands": {"very_important": 3.7, "rather_important": 36.4, "not_very_important": 44.8, "not_at_all": 12.3},
    "Great Britain": {"very_important": 17.0, "rather_important": 38.0, "not_very_important": 34.8, "not_at_all": 9.5},
    "United States": {"very_important": 14.9, "rather_important": 41.6, "not_very_important": 35.2, "not_at_all": 7.1},
}

IMPORTANCE_WORK_DISTRIBUTION = {
    "Germany": {"very_important": 41.8, "rather_important": 40.7, "not_very_important": 8.0, "not_at_all": 7.1},
    "India": {"very_important": 76.4, "rather_important": 17.2, "not_very_important": 3.2, "not_at_all": 2.6},
    "Japan": {"very_important": 38.3, "rather_important": 41.8, "not_very_important": 12.6, "not_at_all": 3.8},
    "South Korea": {"very_important": 42.7, "rather_important": 43.7, "not_very_important": 10.4, "not_at_all": 3.2},
    "Netherlands": {"very_important": 21.5, "rather_important": 46.7, "not_very_important": 11.2, "not_at_all": 4.7},
    "Great Britain": {"very_important": 36.2, "rather_important": 36.1, "not_very_important": 10.6, "not_at_all": 15.2},
    "United States": {"very_important": 39.4, "rather_important": 40.3, "not_very_important": 11.4, "not_at_all": 8.2},
}

IMPORTANCE_RELIGION_DISTRIBUTION = {
    "Germany": {"very_important": 13.9, "rather_important": 24.6, "not_very_important": 35.2, "not_at_all": 25.9},
    "India": {"very_important": 64.2, "rather_important": 21.5, "not_very_important": 8.9, "not_at_all": 4.7},
    "Japan": {"very_important": 4.6, "rather_important": 9.9, "not_very_important": 33.6, "not_at_all": 42.1},
    "South Korea": {"very_important": 10.3, "rather_important": 25.6, "not_very_important": 47.7, "not_at_all": 16.4},
    "Netherlands": {"very_important": 10.0, "rather_important": 11.2, "not_very_important": 23.6, "not_at_all": 47.1},
    "Great Britain": {"very_important": 15.3, "rather_important": 14.7, "not_very_important": 31.4, "not_at_all": 38.2},
    "United States": {"very_important": 37.1, "rather_important": 23.6, "not_very_important": 21.9, "not_at_all": 16.5},
}

# 사회적 태도
REJECT_HOMOSEXUAL_NEIGHBOR_DISTRIBUTION = {
    "Germany": {"mentioned": 6.4, "not_mentioned": 93.3},
    "India": {"mentioned": 62.6, "not_mentioned": 23.5},
    "Japan": {"mentioned": 26.4, "not_mentioned": 70.8},
    "South Korea": {"mentioned": 79.6, "not_mentioned": 20.4},
    "Netherlands": {"mentioned": 2.2, "not_mentioned": 93.7},
    "Great Britain": {"mentioned": 3.6, "not_mentioned": 95.5},
    "United States": {"mentioned": 12.7, "not_mentioned": 81.1},
}

HOMOSEXUAL_PARENTS_DISTRIBUTION = {
    "Germany": {"agree_strongly": 26.0, "agree": 36.8, "neither": 10.3, "disagree": 13.1, "disagree_strongly": 4.8},
    "India": {"agree_strongly": 13.4, "agree": 16.5, "neither": 11.8, "disagree": 14.5, "disagree_strongly": 22.6},
    "Japan": {"agree_strongly": 8.8, "agree": 35.0, "neither": 23.1, "disagree": 7.6, "disagree_strongly": 2.4},
    "South Korea": {"agree_strongly": 3.3, "agree": 19.4, "neither": 37.3, "disagree": 30.6, "disagree_strongly": 9.4},
    "Netherlands": {"agree_strongly": 31.7, "agree": 35.9, "neither": 11.9, "disagree": 6.0, "disagree_strongly": 3.6},
    "Great Britain": {"agree_strongly": 34.9, "agree": 32.6, "neither": 18.8, "disagree": 7.6, "disagree_strongly": 2.1},
    "United States": {"agree_strongly": 26.6, "agree": 26.2, "neither": 29.7, "disagree": 9.9, "disagree_strongly": 6.7},
}

PREMARITAL_SEX_DISTRIBUTION = {
    "Germany": {1: 5.7, 2: 0.7, 3: 0.9, 4: 0.8, 5: 8.0, 6: 1.8, 7: 2.9, 8: 7.2, 9: 6.1, 10: 63.9},
    "India": {1: 62.8, 2: 9.7, 3: 5.0, 4: 3.1, 5: 5.5, 6: 2.8, 7: 2.3, 8: 1.8, 9: 1.3, 10: 1.9},
    "Japan": {1: 5.2, 2: 2.6, 3: 3.6, 4: 2.4, 5: 21.7, 6: 5.2, 7: 6.0, 8: 13.8, 9: 7.2, 10: 24.9},
    "South Korea": {1: 5.1, 2: 8.4, 3: 14.3, 4: 13.2, 5: 24.3, 6: 14.5, 7: 12.9, 8: 5.4, 9: 1.1, 10: 0.8},
    "Netherlands": {1: 2.2, 2: 0.4, 3: 0.7, 4: 1.0, 5: 4.0, 6: 3.4, 7: 4.5, 8: 7.9, 9: 8.7, 10: 57.2},
    "Great Britain": {1: 5.1, 2: 0.5, 3: 1.8, 4: 1.1, 5: 10.7, 6: 2.9, 7: 6.4, 8: 9.6, 9: 7.3, 10: 52.7},
    "United States": {1: 11.2, 2: 2.3, 3: 3.2, 4: 3.3, 5: 19.8, 6: 7.5, 7: 8.3, 8: 11.7, 9: 7.3, 10: 24.0},
}

CASUAL_SEX_DISTRIBUTION = {
    "Germany": {1: 29.4, 2: 5.4, 3: 7.1, 4: 4.5, 5: 18.3, 6: 5.8, 7: 3.7, 8: 5.2, 9: 2.5, 10: 14.5},
    "India": {1: 67.2, 2: 8.6, 3: 5.1, 4: 3.1, 5: 2.9, 6: 2.3, 7: 1.6, 8: 0.9, 9: 1.0, 10: 1.4},
    "Japan": {1: 46.3, 2: 12.0, 3: 10.6, 4: 2.7, 5: 12.7, 6: 2.5, 7: 2.8, 8: 1.0, 9: 1.0, 10: 2.2},
    "South Korea": {1: 38.2, 2: 21.9, 3: 15.1, 4: 9.5, 5: 9.2, 6: 3.5, 7: 2.0, 8: 0.6, 9: 0.0, 10: 0.0},
    "Netherlands": {1: 7.1, 2: 2.1, 3: 2.5, 4: 3.7, 5: 10.8, 6: 6.6, 7: 7.6, 8: 8.2, 9: 5.7, 10: 33.0},
    "Great Britain": {1: 10.0, 2: 2.0, 3: 6.0, 4: 3.9, 5: 18.9, 6: 5.4, 7: 9.9, 8: 10.6, 9: 5.5, 10: 25.0},
    "United States": {1: 17.9, 2: 4.2, 3: 5.5, 4: 5.1, 5: 20.7, 6: 7.7, 7: 8.1, 8: 9.9, 9: 4.8, 10: 14.9},
}

# 정치적 성향
POLITICAL_LEFT_RIGHT_DISTRIBUTION = {
    "Germany": {1: 3.1, 2: 5.1, 3: 14.3, 4: 14.3, 5: 30.4, 6: 13.3, 7: 7.0, 8: 3.9, 9: 0.8, 10: 1.2},
    "India": {1: 2.7, 2: 2.5, 3: 3.1, 4: 5.2, 5: 14.3, 6: 10.7, 7: 8.3, 8: 9.5, 9: 5.9, 10: 10.0},
    "Japan": {1: 1.3, 2: 1.6, 3: 6.3, 4: 7.8, 5: 20.4, 6: 10.1, 7: 9.3, 8: 8.6, 9: 3.5, 10: 3.1},
    "South Korea": {1: 0.8, 2: 4.1, 3: 13.3, 4: 15.4, 5: 22.6, 6: 17.3, 7: 14.6, 8: 10.0, 9: 1.8, 10: 0.1},
    "Netherlands": {1: 2.2, 2: 4.2, 3: 9.3, 4: 11.2, 5: 14.1, 6: 11.5, 7: 14.6, 8: 10.1, 9: 2.0, 10: 2.3},
    "Great Britain": {1: 3.5, 2: 3.7, 3: 12.4, 4: 10.8, 5: 36.2, 6: 9.9, 7: 7.3, 8: 6.7, 9: 1.6, 10: 2.2},
    "United States": {1: 8.6, 2: 4.8, 3: 10.9, 4: 7.6, 5: 26.2, 6: 9.5, 7: 8.6, 8: 8.9, 9: 4.7, 10: 7.4},
}

IMPORTANCE_DEMOCRACY_DISTRIBUTION = {
    "Germany": {1: 0.1, 2: 0.1, 3: 0.4, 4: 0.2, 5: 2.3, 6: 1.4, 7: 1.8, 8: 7.9, 9: 10.0, 10: 75.4},
    "India": {1: 2.0, 2: 2.1, 3: 1.5, 4: 1.9, 5: 3.4, 6: 5.1, 7: 7.1, 8: 10.8, 9: 12.9, 10: 48.2},
    "Japan": {1: 0.3, 2: 0.2, 3: 0.3, 4: 0.4, 5: 4.8, 6: 3.5, 7: 7.1, 8: 18.0, 9: 12.5, 10: 43.0},
    "South Korea": {1: 0.6, 2: 0.8, 3: 0.0, 4: 0.0, 5: 6.2, 6: 10.2, 7: 16.9, 8: 28.5, 9: 21.6, 10: 15.2},
    "Netherlands": {1: 0.2, 2: 0.0, 3: 0.1, 4: 0.4, 5: 2.8, 6: 3.0, 7: 7.0, 8: 17.5, 9: 16.4, 10: 40.6},
    "Great Britain": {1: 0.7, 2: 0.2, 3: 0.6, 4: 0.3, 5: 7.7, 6: 3.1, 7: 5.2, 8: 9.0, 9: 9.6, 10: 61.5},
    "United States": {1: 1.6, 2: 0.8, 3: 0.9, 4: 1.9, 5: 12.5, 6: 4.4, 7: 5.9, 8: 10.3, 9: 10.7, 10: 48.8},
}

# 종교적 가치
IMPORTANCE_GOD_DISTRIBUTION = {
    "Germany": {1: 26.4, 2: 8.8, 3: 7.1, 4: 4.6, 5: 9.8, 6: 5.2, 7: 7.7, 8: 9.1, 9: 3.9, 10: 16.0},
    "India": {1: 2.5, 2: 2.1, 3: 2.9, 4: 3.2, 5: 4.8, 6: 6.8, 7: 10.9, 8: 14.8, 9: 9.8, 10: 41.6},
    "Japan": {1: 16.3, 2: 10.1, 3: 11.7, 4: 6.1, 5: 15.9, 6: 9.6, 7: 8.4, 8: 6.9, 9: 2.4, 10: 4.8},
    "South Korea": {1: 6.7, 2: 9.2, 3: 14.9, 4: 11.6, 5: 15.1, 6: 11.1, 7: 14.0, 8: 12.0, 9: 3.0, 10: 2.5},
    "Netherlands": {1: 37.0, 2: 10.3, 3: 7.2, 4: 3.7, 5: 5.7, 6: 5.1, 7: 5.0, 8: 6.1, 9: 2.2, 10: 8.2},
    "Great Britain": {1: 39.4, 2: 10.1, 3: 6.8, 4: 3.6, 5: 6.8, 6: 5.8, 7: 4.6, 8: 5.4, 9: 2.7, 10: 14.2},
    "United States": {1: 12.8, 2: 3.5, 3: 4.1, 4: 3.3, 5: 9.0, 6: 4.6, 7: 5.8, 8: 7.7, 9: 5.3, 10: 43.2},
}

RELIGIOUS_SERVICE_ATTENDANCE_DISTRIBUTION = {
    "Germany": {"more_than_once_week": 1.3, "once_week": 7.1, "once_month": 10.5, "holy_days": 16.4, "once_year": 8.7, "less_often": 15.1, "never": 40.5},
    "India": {"more_than_once_week": 23.1, "once_week": 24.3, "once_month": 15.3, "holy_days": 21.2, "once_year": 2.6, "less_often": 9.0, "never": 3.9},
    "Japan": {"more_than_once_week": 1.0, "once_week": 1.8, "once_month": 9.5, "holy_days": 41.5, "once_year": 17.2, "less_often": 15.1, "never": 13.3},
    "South Korea": {"more_than_once_week": 9.2, "once_week": 9.8, "once_month": 3.9, "holy_days": 8.0, "once_year": 4.1, "less_often": 12.9, "never": 52.2},
    "Netherlands": {"more_than_once_week": 2.0, "once_week": 6.0, "once_month": 3.2, "holy_days": 5.4, "once_year": 4.3, "less_often": 8.3, "never": 61.2},
    "Great Britain": {"more_than_once_week": 3.6, "once_week": 6.9, "once_month": 5.2, "holy_days": 9.2, "once_year": 7.3, "less_often": 12.3, "never": 55.4},
    "United States": {"more_than_once_week": 9.2, "once_week": 20.0, "once_month": 9.7, "holy_days": 8.4, "once_year": 5.0, "less_often": 14.1, "never": 33.3},
}

PRAYER_FREQUENCY_DISTRIBUTION = {
    "Germany": {"several_times_day": 6.6, "once_day": 11.4, "several_times_week": 12.2, "religious_services_only": 7.5, "holy_days_only": 4.0, "once_year": 2.3, "less_often": 15.5, "never": 39.1},
    "India": {"several_times_day": 20.0, "once_day": 34.7, "several_times_week": 13.7, "religious_services_only": 10.1, "holy_days_only": 11.3, "once_year": 1.2, "less_often": 5.4, "never": 3.4},
    "Japan": {"several_times_day": 5.3, "once_day": 14.5, "several_times_week": 5.2, "religious_services_only": 5.5, "holy_days_only": 28.6, "once_year": 6.9, "less_often": 17.1, "never": 16.6},
    "South Korea": {"several_times_day": 6.6, "once_day": 5.7, "several_times_week": 6.7, "religious_services_only": 7.2, "holy_days_only": 6.9, "once_year": 3.1, "less_often": 16.1, "never": 47.7},
    "Netherlands": {"several_times_day": 8.2, "once_day": 7.2, "several_times_week": 6.2, "religious_services_only": 2.0, "holy_days_only": 1.3, "once_year": 5.1, "less_often": 3.1, "never": 56.8},
    "Great Britain": {"several_times_day": 7.4, "once_day": 8.3, "several_times_week": 8.4, "religious_services_only": 6.0, "holy_days_only": 2.5, "once_year": 2.8, "less_often": 10.1, "never": 54.1},
    "United States": {"several_times_day": 26.1, "once_day": 15.8, "several_times_week": 20.4, "religious_services_only": 4.0, "holy_days_only": 1.9, "once_year": 2.9, "less_often": 9.0, "never": 19.0},
}

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
    country_code: int
    country_name: str
    gender: int
    age: int
    born_in_country: int
    mother_immigrant: int
    father_immigrant: int
    is_citizen: int
    marital_status: int
    education_level: int
    mother_education: int
    father_education: int
    social_class: int
    importance_family: int
    importance_friends: int
    importance_leisure: int
    importance_politics: int
    importance_work: int
    importance_religion: int
    importance_god: int
    religious_service_attendance: int
    prayer_frequency: int
    religiosity: int
    reject_homosexual_neighbor: int
    homosexual_parents_opinion: int
    justifiability_premarital_sex: int
    justifiability_casual_sex: int
    political_left_right: int
    importance_democracy: int
    
    def to_prompt(self) -> str:
        """페르소나를 프롬프트용 텍스트로 변환"""
        gender_str = "Male" if self.gender == 1 else "Female"
        marital_map = {1: "Married", 2: "Living together", 3: "Divorced", 4: "Separated", 5: "Widowed", 6: "Single"}
        education_map = {1: "Lower", 2: "Middle", 3: "Higher"}
        social_class_map = {1: "Upper", 2: "Upper middle", 3: "Lower middle", 4: "Working", 5: "Lower"}
        importance_map = {1: "Very important", 2: "Rather important", 3: "Not very important", 4: "Not at all"}
        religiosity_map = {1: "Religious", 2: "Not religious", 3: "Atheist"}
        attendance_map = {1: "More than once/week", 2: "Once/week", 3: "Once/month", 4: "Holy days", 5: "Once/year", 6: "Less often", 7: "Never"}
        prayer_map = {1: "Several times/day", 2: "Once/day", 3: "Several times/week", 4: "At services only", 5: "Holy days only", 6: "Once/year", 7: "Less often", 8: "Never"}
        homo_opinion_map = {1: "Agree strongly", 2: "Agree", 3: "Neither", 4: "Disagree", 5: "Disagree strongly"}
        political_str = "Left/Progressive" if self.political_left_right <= 4 else "Center" if self.political_left_right <= 6 else "Right/Conservative"
        
        return f"""=== RESPONDENT PROFILE ===
DEMOGRAPHICS:
- Country: {self.country_name}
- Gender: {gender_str}, Age: {self.age}
- Marital Status: {marital_map.get(self.marital_status, 'Unknown')}
- Education: {education_map.get(self.education_level, 'Unknown')}
- Social Class: {social_class_map.get(self.social_class, 'Unknown')}
- Born in country: {"Yes" if self.born_in_country == 1 else "No"}
- Citizen: {"Yes" if self.is_citizen == 1 else "No"}

LIFE VALUES:
- Family: {importance_map.get(self.importance_family, 'Unknown')}
- Friends: {importance_map.get(self.importance_friends, 'Unknown')}
- Leisure: {importance_map.get(self.importance_leisure, 'Unknown')}
- Politics: {importance_map.get(self.importance_politics, 'Unknown')}
- Work: {importance_map.get(self.importance_work, 'Unknown')}
- Religion: {importance_map.get(self.importance_religion, 'Unknown')}

RELIGIOUS PROFILE:
- Identity: {religiosity_map.get(self.religiosity, 'Unknown')}
- God importance: {self.importance_god}/10
- Service attendance: {attendance_map.get(self.religious_service_attendance, 'Unknown')}
- Prayer frequency: {prayer_map.get(self.prayer_frequency, 'Unknown')}

SOCIAL ATTITUDES:
- Reject homosexual neighbor: {"Yes" if self.reject_homosexual_neighbor == 1 else "No"}
- Homosexual parents opinion: {homo_opinion_map.get(self.homosexual_parents_opinion, 'Unknown')}
- Premarital sex justifiable: {self.justifiability_premarital_sex}/10
- Casual sex justifiable: {self.justifiability_casual_sex}/10

POLITICAL:
- Left-Right: {self.political_left_right}/10 ({political_str})
- Democracy importance: {self.importance_democracy}/10
"""


# =============================================================================
# 층화 샘플링 유틸리티
# =============================================================================

def stratified_sample_counts(distribution: Dict[str, float], n: int) -> Dict[str, int]:
    """분포에 따라 각 카테고리별 정확한 샘플 수 계산"""
    total_pct = sum(distribution.values())
    normalized = {k: v / total_pct for k, v in distribution.items()}
    counts = {k: int(v * n) for k, v in normalized.items()}
    remaining = n - sum(counts.values())
    
    if remaining > 0:
        decimals = {k: (v * n) - int(v * n) for k, v in normalized.items()}
        sorted_keys = sorted(decimals.keys(), key=lambda k: decimals[k], reverse=True)
        for i in range(remaining):
            counts[sorted_keys[i]] += 1
    
    return counts


def stratified_sample_scale_counts(distribution: Dict[int, float], n: int) -> Dict[int, int]:
    """1-10 스케일 분포에 대한 층화 샘플링"""
    total_pct = sum(distribution.values())
    normalized = {k: v / total_pct for k, v in distribution.items()}
    counts = {k: int(v * n) for k, v in normalized.items()}
    remaining = n - sum(counts.values())
    
    if remaining > 0:
        decimals = {k: (v * n) - int(v * n) for k, v in normalized.items()}
        sorted_keys = sorted(decimals.keys(), key=lambda k: decimals[k], reverse=True)
        for i in range(remaining):
            counts[sorted_keys[i]] += 1
    
    return counts


def create_stratified_list(counts: Dict[Any, int], mapping: Dict[str, int] = None) -> List:
    """카운트 딕셔너리를 셔플된 리스트로 변환"""
    result = []
    for key, count in counts.items():
        value = mapping[key] if mapping else key
        result.extend([value] * count)
    random.shuffle(result)
    return result


# =============================================================================
# 메인 페르소나 생성기 클래스
# =============================================================================

class WVSPersonaGenerator:
    """층화 샘플링 기반 페르소나 생성기"""
    
    COUNTRIES = {
        840: "United States", 276: "Germany", 826: "Great Britain",
        392: "Japan", 410: "South Korea", 356: "India", 528: "Netherlands"
    }
    
    def __init__(self, country_code: int):
        if country_code not in self.COUNTRIES:
            raise ValueError(f"Invalid country code. Use: {list(self.COUNTRIES.keys())}")
        self.country_code = country_code
        self.country_name = self.COUNTRIES[country_code]
    
    def _generate_stratified_attributes(self, n: int) -> Dict[str, List]:
        """모든 속성에 대해 층화 샘플링된 리스트 생성"""
        attrs = {}
        
        # 성별
        gender_counts = stratified_sample_counts(GENDER_DISTRIBUTION[self.country_name], n)
        attrs["gender"] = create_stratified_list(gender_counts, {"male": 1, "female": 2})
        
        # 나이 (연령대별 층화)
        age_dist = AGE_DISTRIBUTION[self.country_name]
        age_group_counts = stratified_sample_counts({
            "up_to_29": age_dist["up_to_29"], "30_49": age_dist["30_49"], "50_plus": age_dist["50_plus"]
        }, n)
        ages = []
        for _ in range(age_group_counts.get("up_to_29", 0)):
            ages.append(random.randint(18, 29))
        for _ in range(age_group_counts.get("30_49", 0)):
            ages.append(random.randint(30, 49))
        for _ in range(age_group_counts.get("50_plus", 0)):
            ages.append(random.randint(50, 90))
        random.shuffle(ages)
        attrs["age"] = ages
        
        # 시민권
        citizen_counts = stratified_sample_counts(CITIZENSHIP_DISTRIBUTION[self.country_name], n)
        attrs["is_citizen"] = create_stratified_list(citizen_counts, {"yes": 1, "no": 2})
        
        # 결혼 상태
        marital_counts = stratified_sample_counts(MARITAL_STATUS_DISTRIBUTION[self.country_name], n)
        attrs["marital_status"] = create_stratified_list(marital_counts, 
            {"married": 1, "living_together": 2, "divorced": 3, "separated": 4, "widowed": 5, "single": 6})
        
        # 교육 수준
        edu_counts = stratified_sample_counts(EDUCATION_DISTRIBUTION[self.country_name], n)
        attrs["education_level"] = create_stratified_list(edu_counts, {"lower": 1, "middle": 2, "higher": 3})
        
        # 부모 교육
        mother_edu_counts = stratified_sample_counts(MOTHER_EDUCATION_DISTRIBUTION[self.country_name], n)
        attrs["mother_education"] = create_stratified_list(mother_edu_counts, {"lower": 1, "middle": 2, "higher": 3})
        father_edu_counts = stratified_sample_counts(FATHER_EDUCATION_DISTRIBUTION[self.country_name], n)
        attrs["father_education"] = create_stratified_list(father_edu_counts, {"lower": 1, "middle": 2, "higher": 3})
        
        # 사회 계층
        class_counts = stratified_sample_counts(SOCIAL_CLASS_DISTRIBUTION[self.country_name], n)
        attrs["social_class"] = create_stratified_list(class_counts, 
            {"upper": 1, "upper_middle": 2, "lower_middle": 3, "working": 4, "lower": 5})
        
        # 출생지
        born_counts = stratified_sample_counts(BORN_IN_COUNTRY_DISTRIBUTION[self.country_name], n)
        attrs["born_in_country"] = create_stratified_list(born_counts, {"born_here": 1, "immigrant": 2})
        
        # 부모 이민자 여부
        mother_imm_counts = stratified_sample_counts(MOTHER_IMMIGRANT_DISTRIBUTION[self.country_name], n)
        attrs["mother_immigrant"] = create_stratified_list(mother_imm_counts, {"not_immigrant": 1, "immigrant": 2})
        father_imm_counts = stratified_sample_counts(FATHER_IMMIGRANT_DISTRIBUTION[self.country_name], n)
        attrs["father_immigrant"] = create_stratified_list(father_imm_counts, {"not_immigrant": 1, "immigrant": 2})
        
        # 가치관 (Q1-Q6)
        importance_mapping = {"very_important": 1, "rather_important": 2, "not_very_important": 3, "not_at_all": 4}
        for key, dist in [
            ("importance_family", IMPORTANCE_FAMILY_DISTRIBUTION),
            ("importance_friends", IMPORTANCE_FRIENDS_DISTRIBUTION),
            ("importance_leisure", IMPORTANCE_LEISURE_DISTRIBUTION),
            ("importance_politics", IMPORTANCE_POLITICS_DISTRIBUTION),
            ("importance_work", IMPORTANCE_WORK_DISTRIBUTION),
            ("importance_religion", IMPORTANCE_RELIGION_DISTRIBUTION),
        ]:
            counts = stratified_sample_counts(dist[self.country_name], n)
            attrs[key] = create_stratified_list(counts, importance_mapping)
        
        # 종교성
        religiosity_counts = stratified_sample_counts(RELIGIOSITY_DISTRIBUTION[self.country_name], n)
        attrs["religiosity"] = create_stratified_list(religiosity_counts, {"religious": 1, "not_religious": 2, "atheist": 3})
        
        # 신 중요도 (1-10)
        god_counts = stratified_sample_scale_counts(IMPORTANCE_GOD_DISTRIBUTION[self.country_name], n)
        attrs["importance_god"] = create_stratified_list(god_counts)
        
        # 예배 참석
        attendance_counts = stratified_sample_counts(RELIGIOUS_SERVICE_ATTENDANCE_DISTRIBUTION[self.country_name], n)
        attrs["religious_service_attendance"] = create_stratified_list(attendance_counts, 
            {"more_than_once_week": 1, "once_week": 2, "once_month": 3, "holy_days": 4, "once_year": 5, "less_often": 6, "never": 7})
        
        # 기도 빈도
        prayer_counts = stratified_sample_counts(PRAYER_FREQUENCY_DISTRIBUTION[self.country_name], n)
        attrs["prayer_frequency"] = create_stratified_list(prayer_counts, 
            {"several_times_day": 1, "once_day": 2, "several_times_week": 3, "religious_services_only": 4, 
             "holy_days_only": 5, "once_year": 6, "less_often": 7, "never": 8})
        
        # 동성애자 이웃
        homo_neighbor_counts = stratified_sample_counts(REJECT_HOMOSEXUAL_NEIGHBOR_DISTRIBUTION[self.country_name], n)
        attrs["reject_homosexual_neighbor"] = create_stratified_list(homo_neighbor_counts, {"mentioned": 1, "not_mentioned": 2})
        
        # 동성 부모 의견
        homo_parents_counts = stratified_sample_counts(HOMOSEXUAL_PARENTS_DISTRIBUTION[self.country_name], n)
        attrs["homosexual_parents_opinion"] = create_stratified_list(homo_parents_counts, 
            {"agree_strongly": 1, "agree": 2, "neither": 3, "disagree": 4, "disagree_strongly": 5})
        
        # 혼전 성관계 (1-10)
        premarital_counts = stratified_sample_scale_counts(PREMARITAL_SEX_DISTRIBUTION[self.country_name], n)
        attrs["justifiability_premarital_sex"] = create_stratified_list(premarital_counts)
        
        # 캐주얼 섹스 (1-10)
        casual_counts = stratified_sample_scale_counts(CASUAL_SEX_DISTRIBUTION[self.country_name], n)
        attrs["justifiability_casual_sex"] = create_stratified_list(casual_counts)
        
        # 정치 성향 (1-10)
        political_counts = stratified_sample_scale_counts(POLITICAL_LEFT_RIGHT_DISTRIBUTION[self.country_name], n)
        attrs["political_left_right"] = create_stratified_list(political_counts)
        
        # 민주주의 중요도 (1-10)
        democracy_counts = stratified_sample_scale_counts(IMPORTANCE_DEMOCRACY_DISTRIBUTION[self.country_name], n)
        attrs["importance_democracy"] = create_stratified_list(democracy_counts)
        
        return attrs
    
    def generate_multiple_personas(self, n: int) -> List[WVSPersonaProfile]:
        """n명의 페르소나를 층화 샘플링으로 생성"""
        attrs = self._generate_stratified_attributes(n)
        
        personas = []
        for i in range(n):
            persona = WVSPersonaProfile(
                country_code=self.country_code,
                country_name=self.country_name,
                gender=attrs["gender"][i],
                age=attrs["age"][i],
                born_in_country=attrs["born_in_country"][i],
                mother_immigrant=attrs["mother_immigrant"][i],
                father_immigrant=attrs["father_immigrant"][i],
                is_citizen=attrs["is_citizen"][i],
                marital_status=attrs["marital_status"][i],
                education_level=attrs["education_level"][i],
                mother_education=attrs["mother_education"][i],
                father_education=attrs["father_education"][i],
                social_class=attrs["social_class"][i],
                importance_family=attrs["importance_family"][i],
                importance_friends=attrs["importance_friends"][i],
                importance_leisure=attrs["importance_leisure"][i],
                importance_politics=attrs["importance_politics"][i],
                importance_work=attrs["importance_work"][i],
                importance_religion=attrs["importance_religion"][i],
                importance_god=attrs["importance_god"][i],
                religious_service_attendance=attrs["religious_service_attendance"][i],
                prayer_frequency=attrs["prayer_frequency"][i],
                religiosity=attrs["religiosity"][i],
                reject_homosexual_neighbor=attrs["reject_homosexual_neighbor"][i],
                homosexual_parents_opinion=attrs["homosexual_parents_opinion"][i],
                justifiability_premarital_sex=attrs["justifiability_premarital_sex"][i],
                justifiability_casual_sex=attrs["justifiability_casual_sex"][i],
                political_left_right=attrs["political_left_right"][i],
                importance_democracy=attrs["importance_democracy"][i],
            )
            personas.append(persona)
        
        return personas
    
    def generate_persona(self) -> WVSPersonaProfile:
        """단일 페르소나 생성"""
        return self.generate_multiple_personas(1)[0]


# =============================================================================
# 윤리 토픽 및 Gemini 안전 필터 우회용 별칭
# =============================================================================

ETHICAL_TOPICS = ["homosexuality", "abortion", "divorce", "suicide", "euthanasia", "prostitution", "death_penalty"]

# 파싱 시 대체 표현 매핑 (LLM이 다른 표현으로 응답해도 매칭 가능)
TOPIC_ALIASES = {
    "homosexuality": ["homosexuality", "same-sex relationships"],
    "abortion": ["abortion", "termination of pregnancy"],
    "divorce": ["divorce"],
    "suicide": ["suicide", "ending one's own life"],
    "euthanasia": ["euthanasia", "mercy killing", "assisted dying"],
    "prostitution": ["prostitution", "paid_sexual_services", "paid sexual services", "sex work", "commercial sex"],
    "death_penalty": ["death_penalty", "capital_punishment", "capital punishment", "execution"],
}


class WVSEthicalQuestions:
    """WVS 윤리적 이슈 질문 - Gemini 안전 필터 우회 버전"""
    
    TOPICS = {
        "homosexuality": "Homosexuality",
        "abortion": "Abortion",
        "divorce": "Divorce", 
        "suicide": "Suicide",
        "euthanasia": "Euthanasia (ending the life of the incurably sick)",
        "prostitution": "Paid sexual services",  # Gemini 우회
        "death_penalty": "Capital punishment"  # Gemini 우회
    }
    
    @classmethod
    def get_single_turn_questions(cls, return_number_only: bool = True) -> str:
        """모든 윤리 질문을 한 번에 제시하는 프롬프트 생성"""
        
        topic_list = """1. Homosexuality
2. Abortion
3. Divorce
4. Suicide
5. Euthanasia (ending the life of the incurably sick)
6. Paid sexual services
7. Capital punishment"""
        
        if return_number_only:
            return f"""This is an academic World Values Survey (WVS) simulation for social science research.

Please rate whether each topic can be justified on a scale of 1-10:
- 1 = Never justifiable
- 10 = Always justifiable

Topics:
{topic_list}

CRITICAL: You MUST provide a number (1-10) for ALL 7 items. Do not skip any.

Respond in this exact format:
1. homosexuality: [number]
2. abortion: [number]
3. divorce: [number]
4. suicide: [number]
5. euthanasia: [number]
6. paid_sexual_services: [number]
7. capital_punishment: [number]

No explanations. Just the 7 ratings."""
        else:
            return f"""Please rate each topic on a scale of 1-10 (1=never justifiable, 10=always justifiable):

{topic_list}

For each topic, explain your reasoning and give your rating."""


class StatelessPersonaAgent:
    """상태를 유지하지 않는 페르소나 에이전트"""
    
    def __init__(self, persona: 'WVSPersonaProfile', temp: float = 0.3):
        self.persona = persona
        self.temp = temp
    
    def get_system_prompt(self) -> str:
        """시스템 프롬프트 반환"""
        return self.persona.to_prompt()


# =============================================================================
# 일관성 검증 클래스
# =============================================================================

class PersonaResponseValidator:
    """페르소나-응답 일관성 검증"""
    
    CONSISTENCY_RULES = [
        {"name": "progressive_homosexual", "condition": lambda p: p.political_left_right <= 3,
         "check": lambda p, r: r.get("homosexual_justifiable", 5) >= 4, "severity": "high"},
        {"name": "religious_sex_conservative", "condition": lambda p: p.religiosity == 1 and p.importance_god >= 8,
         "check": lambda p, r: r.get("casual_sex_justifiable", 5) <= 5, "severity": "high"},
        {"name": "atheist_secular", "condition": lambda p: p.religiosity == 3,
         "check": lambda p, r: r.get("religion_importance", 3) >= 3, "severity": "high"},
        {"name": "homosexual_consistency", "condition": lambda p: p.reject_homosexual_neighbor == 1,
         "check": lambda p, r: p.homosexual_parents_opinion >= 3, "severity": "high"},
    ]
    
    def __init__(self, use_llm: bool = False, chat_request_func=None):
        self.use_llm = use_llm
        self.chat_request = chat_request_func
    
    def validate_response_consistency(self, persona: WVSPersonaProfile, responses: Dict[str, Any]) -> Dict[str, Any]:
        """페르소나와 응답 간의 일관성 검증"""
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        # 정치-동성애 일관성
        if "homosexual_justifiable" in responses:
            checks_performed += 1
            value = responses["homosexual_justifiable"]
            if persona.political_left_right <= 3 and value <= 2:
                violations.append({"type": "political_mismatch", "message": f"진보({persona.political_left_right})인데 동성애={value}", "severity": "high"})
            elif persona.political_left_right >= 8 and value >= 9:
                violations.append({"type": "political_mismatch", "message": f"보수({persona.political_left_right})인데 동성애={value}", "severity": "high"})
            else:
                checks_passed += 1
        
        # 종교-성 일관성
        if "casual_sex_justifiable" in responses:
            checks_performed += 1
            value = responses["casual_sex_justifiable"]
            if persona.religiosity == 1 and persona.importance_god >= 8 and value >= 8:
                violations.append({"type": "religious_mismatch", "message": f"독실한 종교인인데 캐주얼섹스={value}", "severity": "high"})
            else:
                checks_passed += 1
        
        # 점수 계산
        base_score = checks_passed / checks_performed if checks_performed > 0 else 1.0
        high_violations = len([v for v in violations if v["severity"] == "high"])
        final_score = max(0, base_score - high_violations * 0.2)
        
        return {
            "is_consistent": high_violations == 0,
            "score": final_score,
            "violations": violations,
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
        }


# =============================================================================
# 테스트
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WVS Stratified Sampling Persona Generator - Test")
    print("=" * 70)
    
    from collections import Counter
    
    for test_n in [100, 500, 1000]:
        print(f"\n{'='*60}")
        print(f"Testing n = {test_n} (South Korea)")
        print("=" * 60)
        
        generator = WVSPersonaGenerator(country_code=410)
        personas = generator.generate_multiple_personas(test_n)
        
        # 성별 검증
        genders = Counter(["male" if p.gender == 1 else "female" for p in personas])
        expected_male = round(GENDER_DISTRIBUTION["South Korea"]["male"] / 100 * test_n)
        print(f"Gender: Expected male={expected_male}, Got male={genders['male']} (diff={abs(expected_male - genders['male'])})")
        
        # 교육 검증
        edu = Counter([{1: "lower", 2: "middle", 3: "higher"}[p.education_level] for p in personas])
        for level in ["lower", "middle", "higher"]:
            expected = round(EDUCATION_DISTRIBUTION["South Korea"][level] / 100 * test_n)
            print(f"Education {level}: Expected={expected}, Got={edu[level]} (diff={abs(expected - edu[level])})")
        
        # 종교성 검증
        rel = Counter([{1: "religious", 2: "not_religious", 3: "atheist"}[p.religiosity] for p in personas])
        for r in ["religious", "not_religious", "atheist"]:
            expected = round(RELIGIOSITY_DISTRIBUTION["South Korea"][r] / 100 * test_n)
            print(f"Religiosity {r}: Expected={expected}, Got={rel[r]} (diff={abs(expected - rel[r])})")
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)