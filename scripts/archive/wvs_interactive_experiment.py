import os
import re
import json
import pandas as pd
import random
from typing import List, Dict

# 기존 agent 및 llm 모듈 임포트 (사용자 환경에 맞춰 유지)
# sys.path 설정을 통해 경로를 맞춰주세요.
try:
    import agent
    from llm import Message
except ImportError:
    print("Warning: 'agent' or 'llm' modules not found. Please ensure they are in your path.")

# 가치관 조사 항목 (World Values Survey 기준)
TOPICS = [
    "Homosexuality", "Abortion", "Divorce", "Suicide", 
    "Euthanasia", "Prostitution", "Death penalty"
]

class ValueSurveySimulator:
    def __init__(self, persona_csv_path: str, model_name: str = "wvs_agent"):
        self.persona_csv_path = persona_csv_path
        self.model_name = model_name
        self.output_dir = f"survey_results/{model_name}"
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def sample_personas(self, n: int) -> List[str]:
        """수천 개의 데이터 중 n개를 랜덤 샘플링하여 페르소나 리스트 생성"""
        df = pd.read_csv(self.persona_csv_path)
        # 'Description' 컬럼 안에 페르소나 문장 포함
        sampled_df = df.sample(n=min(n, len(df)))
        return sampled_df['Description'].tolist()

    def create_persona_context(self, persona_desc: str) -> str:
        """에이전트에게 부여할 초기 가치관 및 배경 설정"""
        return (
            f"You are a participant in the World Values Survey. {persona_desc} "
            "Your task is to provide your honest opinion on various social issues. "
            "For each issue, you will justify your stance on a scale from 1 (Never justifiable) "
            "to 10 (Always justifiable)."
        )

    def parse_scores(self, text: str) -> Dict[str, int]:
        """LLM 응답에서 각 항목별 점수를 추출 (정규표현식 활용)"""
        results = {}
        for topic in TOPICS:
            # 'Topic: 숫자' 또는 'Topic - 숫자' 형태를 찾음
            match = re.search(rf"{topic}.*?(\d+)", text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                # 1~10 범위를 벗어나지 않게 조정
                results[topic] = max(1, min(10, score))
            else:
                results[topic] = None # 응답 누락 시
        return results

    def run_simulation(self, n_agents: int, temp: float = 0.7):
        """가치관 조사 시뮬레이션 실행 (CoT 방식 적용)"""
        personas = self.sample_personas(n_agents)
        collected_data = []

        print(f"--- Starting Survey with {n_agents} Agents ---")

        for i, persona in enumerate(personas):
            # 에이전트 초기화
            sim_agent = agent.Agent(
                aid=i, 
                recall=2, 
                initial_context=Message(time=0, content=self.create_persona_context(persona), role="system"),
                temperature=temp
            )

            # 1단계: 사고 과정 유도 (Thought Process)
            thought_prompt = Message(
                time=1,
                content="Reflect on your background, religious beliefs, and cultural values. "
                        "How do these factors shape your view on sensitive moral issues?",
                role="user"
            )
            thought_process = sim_agent.perceive(message=thought_prompt, max_tokens=500).content

            # 2단계: 실제 점수 매기기
            survey_prompt = Message(
                time=2,
                content=(
                    "Based on your reflections, please rate how justifiable the following topics are "
                    "on a scale of 1 to 10. List each topic and your score clearly.\n"
                    f"Topics: {', '.join(TOPICS)}"
                ),
                role="user"
            )
            response = sim_agent.perceive(message=survey_prompt, max_tokens=500).content
            
            # 데이터 정리
            scores = self.parse_scores(response)
            entry = {
                "agent_id": i,
                "persona": persona,
                "reflection": thought_process,
                "full_response": response,
                "scores": scores
            }
            collected_data.append(entry)
            
            print(f"Agent {i} completed. Scores: {scores}")

        self.save_results(collected_data)

    def save_results(self, data: List[Dict]):
        """결과를 JSON 및 CSV로 저장"""
        # JSON 저장
        json_path = os.path.join(self.output_dir, "survey_raw.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # CSV 요약 저장 (분석용)
        summary_list = []
        for d in data:
            row = {"agent_id": d["agent_id"]}
            row.update(d["scores"])
            summary_list.append(row)
        
        df_summary = pd.DataFrame(summary_list)
        df_summary.to_csv(os.path.join(self.output_dir, "survey_summary.csv"), index=False)
        print(f"Results saved to {self.output_dir}")

if __name__ == "__main__":
    # 설정값
    CSV_PATH = 'data/world_values_persona_large.csv' # 수천 줄의 페르소나 데이터 경로
    N_SAMPLE = 50 # 샘플링할 에이전트 수
    
    simulator = ValueSurveySimulator(persona_csv_path=CSV_PATH)
    simulator.run_simulation(n_agents=N_SAMPLE, temp=0.8)