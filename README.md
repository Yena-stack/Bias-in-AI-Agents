# WVS Ethical Values in AI Agents

This repository hosts the source code and data for research investigating whether Large Language Models (LLMs) exhibit cultural biases in ethical judgments similar to humans, based on the World Values Survey (WVS) Wave 7 framework.

## Abstract

This research examines whether LLM-based agents reproduce human cultural patterns in ethical judgments across different countries. Using the World Values Survey (WVS) Wave 7 data as a baseline, we generate synthetic personas representing diverse demographics from seven countries (United States, Germany, Great Britain, Japan, South Korea, India, Netherlands) and evaluate their responses to ethical questions about homosexuality, abortion, divorce, suicide, euthanasia, prostitution, and death penalty. Our methodology compares LLM agent responses with actual WVS survey data to identify potential biases and measure alignment with human cultural values. We test multiple LLM providers (OpenAI GPT-4, Google Gemini, Meta Llama via Groq) to assess consistency and variation in ethical reasoning across different model architectures.

## Key Features

- **WVS-7 Compliant Personas**: Generate synthetic personas using official WVS-7 coding schemes (3-digit country codes, ISCED education levels)
- **Multi-Model Support**: Compatible with Groq (fast Llama), Gemini, OpenAI, Together AI, and Ollama (local)
- **Ethical Question Framework**: Seven controversial topics from WVS-7 using justifiability scales (1-10)
- **Single-Turn Design**: All questions in one prompt for efficiency and consistency
- **Statistical Analysis**: Distribution analysis, Kendall's Tau correlation with human data
- **Rate Limit Handling**: Automatic retry and wait mechanisms for free API tiers
- **Reproducibility**: Random seed control for consistent persona generation

## Repository Structure

```
.
├── agent/
│   ├── __init__.py
│   └── agent.py              # Core persona and agent classes
├── llm/
│   ├── __init__.py
│   └── llm.py                # Multi-provider LLM API wrapper
├── scripts/
│   ├── __init__.py
│   ├── wvs_experiment.py     # Main experiment runner
│   └── archive/              # Deprecated code
├── data/
│   └── filtered_wvs_data.csv # Human baseline data
├── results/                  # Experiment outputs (gitignored)
│   ├── llama-70b-versatile_temp1/
│   ├── gemini-flash_temp1/
│   └── gpt-4_temp1/
├── .env.example              # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/Bias-in-AI-Agents.git
cd Bias-in-AI-Agents

# Create Python virtual environment (Python 3.9+)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
```txt
python-dotenv>=1.0.0
requests>=2.31.0
numpy>=1.24.0
google-generativeai>=0.3.0  # For Gemini
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Choose one of the supported LLM providers:

**Option A: Groq (Recommended - Free & Very Fast)**
```bash
# Get API key: https://console.groq.com/keys
echo "GROQ_API_KEY=gsk_your_key_here" >> .env
```
- **Limit**: 12,000 tokens/minute (free tier)
- **Best for**: Fast experimentation, large batches with rate limit handling

**Option B: Google Gemini (Free Tier, Large Quota)**
```bash
# Get API key: https://aistudio.google.com/app/apikey
echo "GEMINI_API_KEY=AIza_your_key_here" >> .env
```
- **Limit**: 15 requests/minute, 1,500/day (free tier)
- **Best for**: Stable experiments without rate limits

**Option C: OpenAI (Paid)**
```bash
echo "OPENAI_API_KEY=sk-your_key_here" >> .env
```
- **Best for**: Highest quality responses (but expensive)

**Option D: Ollama (Local, Free)**
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
# No API key needed - auto-detected at http://localhost:11434
```
- **Best for**: Completely offline experiments

### 3. Run Experiments

#### Basic Commands

**Single Country Experiment:**
```bash
python scripts/wvs_experiment.py --country "South Korea" --model llama-3.3-70b-versatile
```

**All Countries at Once:**
```bash
python scripts/wvs_experiment.py --all-countries --model llama-3.3-70b-versatile
```

**Quick Test (10 personas):**
```bash
python scripts/wvs_experiment.py --country "United States" --model gemini-2.5-flash --num-personas 10
```

#### Advanced Commands

**Different Models:**
```bash
# Llama 70B via Groq (free, fast)
python scripts/wvs_experiment.py --all-countries --model llama-3.3-70b-versatile

# Gemini Flash (free, stable)
python scripts/wvs_experiment.py --all-countries --model gemini-2.5-flash

# Gemini Pro (free, higher quality)
python scripts/wvs_experiment.py --country "Japan" --model gemini-2.5-pro

# GPT-4 (paid, best quality)
python scripts/wvs_experiment.py --country "Germany" --model gpt-4-turbo

# Ollama Local (free, offline)
python scripts/wvs_experiment.py --country "India" --model llama3.2
```

**Different Sample Sizes:**
```bash
# Small test (10 personas)
python scripts/wvs_experiment.py --country "South Korea" --model llama-3.3-70b-versatile --num-personas 10

# Standard (100 personas, default)
python scripts/wvs_experiment.py --country "South Korea" --model llama-3.3-70b-versatile --num-personas 100

# Large sample (200 personas)
python scripts/wvs_experiment.py --country "South Korea" --model llama-3.3-70b-versatile --num-personas 200
```

**Temperature Variations:**
```bash
# Low temperature (more consistent)
python scripts/wvs_experiment.py --country "United States" --model gemini-2.5-flash --temperature 0.7

# Default temperature (balanced)
python scripts/wvs_experiment.py --country "United States" --model gemini-2.5-flash --temperature 1.0

# High temperature (more diverse)
python scripts/wvs_experiment.py --country "United States" --model gemini-2.5-flash --temperature 1.5
```

**Different Random Seeds:**
```bash
# Seed 42 (default)
python scripts/wvs_experiment.py --country "Germany" --model llama-3.3-70b-versatile --seed 42

# Seed 123 (for reproducibility study)
python scripts/wvs_experiment.py --country "Germany" --model llama-3.3-70b-versatile --seed 123
```

**Complete Example:**
```bash
python scripts/wvs_experiment.py \
  --country "Japan" \
  --model gemini-2.5-flash \
  --num-personas 100 \
  --temperature 1.0 \
  --seed 42
```

#### Batch Processing Script (All Combinations)

Create `run_all_experiments.sh`:
```bash
#!/bin/bash

# Models to test
MODELS=("llama-3.3-70b-versatile" "gemini-2.5-flash")

# Countries to test
COUNTRIES=("United States" "Germany" "Great Britain" "Japan" "South Korea" "India" "Netherlands")

# Run experiments
for model in "${MODELS[@]}"; do
  for country in "${COUNTRIES[@]}"; do
    echo "Running: $country with $model"
    python scripts/wvs_experiment.py \
      --country "$country" \
      --model "$model" \
      --num-personas 100 \
      --seed 42
    sleep 2  # Brief pause between experiments
  done
done

echo "All experiments completed!"
```

Run:
```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh
```

### 4. Analyze Results

Results are saved in `results/{model}_temp{temp}/`:

```
results/
├── llama-70b-versatile_temp1/
│   ├── responses_South_Korea_seed42.csv
│   ├── stats_South_Korea_seed42.json
│   ├── responses_United_States_seed42.csv
│   └── stats_United_States_seed42.json
├── gemini-flash_temp1/
│   └── ...
└── gpt-4_temp1/
    └── ...
```

**CSV Columns:**
- `persona_id`, `country`, `age`, `gender`, `education_level`, `social_class`
- `political_left_right`, `importance_religion`, `religiosity`
- `rating_homosexuality`, `rating_abortion`, `rating_divorce`, `rating_suicide`
- `rating_euthanasia`, `rating_prostitution`, `rating_death_penalty`
- `response` (full LLM response text), `temperature`, `random_seed`, `model`

**Analysis with Python:**
```python
import pandas as pd
import json

# Load responses
df = pd.read_csv('results/llama-70b-versatile_temp1/responses_South_Korea_seed42.csv')

# Basic statistics
print(f"Total personas: {len(df)}")
print(f"Mean rating for homosexuality: {df['rating_homosexuality'].mean():.2f}")
print(f"Gender distribution:\n{df['gender'].value_counts()}")

# Load statistics
with open('results/llama-70b-versatile_temp1/stats_South_Korea_seed42.json') as f:
    stats = json.load(f)

for topic, data in stats.items():
    print(f"\n{topic}:")
    print(f"  Mean: {data['mean']:.2f}")
    print(f"  Std: {data['std']:.2f}")
    print(f"  Valid responses: {data['count']}")
    print(f"  Distribution: {data['distribution']}")
```

## Project Components

### Agent Module (`agent/agent.py`)

**Core Classes:**

1. **`WVSPersonaProfile`** (dataclass)
   - All fields use official WVS-7 integer coding
   - **Country codes**: 840 (USA), 276 (Germany), 826 (Great Britain), 392 (Japan), 410 (South Korea), 356 (India), 528 (Netherlands)
   - **Gender**: 1 (Male), 2 (Female)
   - **Education**: 0-8 (ISCED 2011: 0=No education, 6=Bachelor, 7=Master, 8=Doctoral)
   - **Political orientation**: 1 (Left) to 10 (Right)
   - **Important**: Excludes direct ethical values for experimental topics to prevent priming

2. **`WVSPersonaGenerator`**
   - Generates random personas with country-specific distributions
   - Uses `country_code` (int) instead of country name (string)
   - Supports `seed` parameter for reproducibility
   - Example:
   ```python
   generator = WVSPersonaGenerator(country_code=410, seed=42)  # South Korea
   personas = generator.generate_multiple_personas(n=100)
   ```

3. **`StatelessPersonaAgent`**
   - Stateless LLM agent (no conversation history)
   - Responds to questions in character based on persona
   - Each question is independent (1-shot)

4. **`WVSEthicalQuestions`**
   - Standardized questions from WVS-7 Q182-Q195
   - Topics: homosexuality, abortion, divorce, suicide, euthanasia, prostitution, death_penalty
   - Justifiability scale: 1 (never) to 10 (always)
   - `get_single_turn_questions(return_number_only=True)` for number-only responses

### LLM Module (`llm/llm.py`)

**Multi-Provider API Wrapper:**
- Unified `chat_request()` interface
- Automatic provider detection based on API keys
- Priority: Groq → Together → Gemini → OpenAI → Ollama → localhost

**Supported Models:**
- **Groq**: `llama-3.3-70b-versatile` (recommended), `llama-3.1-8b-instant`
- **Gemini**: `gemini-2.5-flash` (recommended), `gemini-2.5-pro`, `gemini-2.0-flash-exp`
- **OpenAI**: `gpt-4-turbo`, `gpt-4-1106-preview`, `gpt-3.5-turbo`
- **Together**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- **Ollama**: `llama3.2`, `llama3.1:8b`, `llama3.1:70b`

**Rate Limit Handling:**
- Automatic 0.5s delay between requests
- Detects 429 errors (rate limit exceeded)
- Automatic 5s wait and retry on rate limits
- Continues experiment even if individual requests fail

### Experiment Script (`scripts/wvs_experiment.py`)

**Single-Turn Design:**
- All 7 ethical questions in one prompt
- Requests number-only responses (no reasoning)
- More efficient and cost-effective
- Evaluates consistency across topics

**Key Features:**
- **Country code mapping**: Converts country names to WVS-7 codes
- **Automatic rate limiting**: 0.5s delay + retry on 429 errors
- **Error recovery**: Logs errors but continues experiment
- **Progress tracking**: Prints progress every 10 personas
- **Result validation**: Checks for valid ratings (1-10)

**Configuration Variables:**
```python
COUNTRIES = ["United States", "Germany", "Great Britain", "Japan", 
             "South Korea", "India", "Netherlands"]
COUNTRY_CODES = {
    "United States": 840,
    "Germany": 276,
    "Great Britain": 826,
    "Japan": 392,
    "South Korea": 410,
    "India": 356,
    "Netherlands": 528
}
ETHICAL_TOPICS = ["homosexuality", "abortion", "divorce", "suicide", 
                  "euthanasia", "prostitution", "death_penalty"]
```

## Experimental Design

### Persona Generation
- **Sample size**: 100 personas per country (default, adjustable 10-200+)
- **Random sampling**: All demographic and value attributes randomized
- **Reproducibility**: Uses random seed (default 42)
- **Country-specific**: Language automatically assigned based on country
- **Exclusion principle**: Direct ethical values for test topics excluded to prevent circular reasoning

### Response Collection
- **Prompt**: WVS-7 exact question wording
- **Format**: Single-turn with all 7 questions
- **Instructions**: "Respond ONLY with numbers in the following format..."
- **Parsing**: Regex extraction `r"1\. homosexuality: (\d+)"`
- **Temperature**: Default 1.0 (adjustable for diversity study)
- **Rate limiting**: 0.5s delay + auto-retry on 429

### Data Analysis
- **Distribution comparison**: LLM vs. human histograms
- **Mean/Std deviation**: Central tendency and variance
- **Valid response rate**: Percentage of successfully parsed ratings
- **Kendall's Tau**: Rank correlation with human data (TODO)
- **Cross-country comparison**: Cultural pattern alignment

## Cost & Time Estimation

**For 100 personas × 7 countries = 700 requests:**

| Provider | Model | Total Cost | Time | Rate Limit |
|----------|-------|------------|------|------------|
| **Groq** | Llama 3.3 70B | **FREE** | ~6-10 min | 12K TPM |
| **Gemini** | 2.5 Flash | **FREE** | ~10 min | 15 RPM |
| **OpenAI** | GPT-4 Turbo | ~$35 | ~15 min | Depends on tier |
| **Ollama** | Llama 3.2 | **FREE** | ~1-2 hours | No limit |

*Groq recommended for speed, Gemini for stability*

## Troubleshooting

### Rate Limit Errors (429)

**Problem:**
```
❌ Error: rate_limit_exceeded
Limit 12000, Used 11772, Requested 1022
```

**Solution:**
1. **Already implemented**: Auto-retry with 5s wait
2. **If persistent**: Increase delay in `wvs_experiment.py`:
   ```python
   time.sleep(0.5)  # Change to 1.0
   ```
3. **Switch models**:
   ```bash
   # Switch from Groq to Gemini
   python scripts/wvs_experiment.py --all-countries --model gemini-2.5-flash
   ```

### Module Import Errors

**Problem:**
```
ModuleNotFoundError: No module named 'agent'
```

**Solution:**
```bash
# Make sure you're in project root
cd /path/to/Bias-in-AI-Agents

# Run from root, not from scripts/
python scripts/wvs_experiment.py --country "South Korea" --model llama-3.3-70b-versatile
```

### API Key Errors

**Problem:**
```
Error: GROQ_API_KEY not found
```

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Check content
cat .env

# Should contain:
GROQ_API_KEY=gsk_your_key_here
# or
GEMINI_API_KEY=AIza_your_key_here
```

### Gemini Setup

**Problem:**
```
ModuleNotFoundError: No module named 'google.generativeai'
```

**Solution:**
```bash
pip install google-generativeai
```

## Advanced Usage

### Custom Persona Generation
```python
from agent.agent import WVSPersonaGenerator, StatelessPersonaAgent, WVSEthicalQuestions

# Generate specific persona
generator = WVSPersonaGenerator(country_code=410, seed=42)  # South Korea
persona = generator.generate_persona(
    gender=2,                    # Female
    age=35,
    education_level=7,           # Master's degree
    political_left_right=8,      # Conservative
    importance_religion=1        # Very important
)

# Create agent
agent = StatelessPersonaAgent(persona=persona, temp=1.0)

# Ask question
question = WVSEthicalQuestions.get_question("abortion")
response = agent.respond_to_ethical_question(question, model="gemini-2.5-flash")
print(response.content)
```

### Compare Models
```python
# Compare Llama vs Gemini for same persona
models = ["llama-3.3-70b-versatile", "gemini-2.5-flash"]

for model in models:
    response = agent.respond_to_ethical_question(question, model=model)
    print(f"{model}: {response.content}")
```

## File Structure Details

### Important Files
- **`.env`**: API keys (gitignored, create from `.env.example`)
- **`requirements.txt`**: Python dependencies
- **`.gitignore`**: Excludes results/, .env, __pycache__/, etc.

### Code Organization
- **`agent/agent.py`**: 650 lines, core persona and agent logic
- **`llm/llm.py`**: 400 lines, multi-provider API wrapper
- **`scripts/wvs_experiment.py`**: 350 lines, experiment runner

## Citation

If you use this repository for your research, please cite:

```bibtex
@article{Choi2026WVSBias,
  title={Cultural Bias in AI Ethical Reasoning: A World Values Survey Analysis},
  author={Yena Choi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- World Values Survey Association for providing WVS Wave 7 data and questionnaire
- Groq for providing free fast Llama API access
- Google for Gemini API free tier
- Anthropic Claude for assistance in code development

## Contact

For questions or collaboration inquiries:
- Open an issue on GitHub
- Email: yenachoi@hanyang.ac.kr

## Additional Resources

- **WVS-7 Official Website**: https://www.worldvaluessurvey.org/
- **Groq Console**: https://console.groq.com/
- **Google AI Studio**: https://aistudio.google.com/
- **Ollama**: https://ollama.com/