# Bias-in-AI-Agents
# WVS Ethical Values in AI Agents

This repository hosts the source code and data for research investigating whether Large Language Models (LLMs) exhibit cultural biases in ethical judgments similar to humans, based on the World Values Survey (WVS) Wave 7 framework.

## Abstract

This research examines whether LLM-based agents reproduce human cultural patterns in ethical judgments across different countries. Using the World Values Survey (WVS) Wave 7 data as a baseline, we generate synthetic personas representing diverse demographics from seven countries (United States, Germany, Great Britain, Japan, South Korea, India, Netherlands) and evaluate their responses to ethical questions about homosexuality, abortion, divorce, suicide, euthanasia, prostitution, and death penalty. Our methodology compares LLM agent responses with actual WVS survey data to identify potential biases and measure alignment with human cultural values. We test multiple LLM providers (OpenAI GPT-4, Google Gemini, Meta Llama) to assess consistency and variation in ethical reasoning across different model architectures.

## Key Features

- **WVS-7 Compliant Personas**: Generate synthetic personas using official WVS-7 coding schemes (3-digit country codes, ISCED education levels, etc.)
- **Multi-Model Support**: Compatible with OpenAI, Gemini, Groq (fast Llama), Together AI, and Ollama (local)
- **Ethical Question Framework**: Seven controversial topics from WVS-7 using justifiability scales (1-10)
- **Experimental Modes**: 
  - Single-turn: All questions in one prompt (faster, more efficient)
  - Separate: Each question individually (more accurate, detailed)
- **Statistical Analysis**: Distribution analysis, Kendall's Tau correlation with human data

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
│   └── wvs_experiment.py     # Main experiment runner
├── data/
│   ├── wvs_results/          # Experiment outputs (CSV/JSON)
│   └── filtered_wvs_data_csv.xlsx  # Human baseline data
├── docs/
│   ├── API_SETUP_GUIDE.md    # API configuration guide
│   └── LLAMA_GUIDE.md        # Llama-specific setup
├── .env.example              # Environment variables template
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/wvs-ai-bias.git
cd wvs-ai-bias

# Create Python virtual environment (Python 3.9+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Choose one of the supported LLM providers:

**Option A: Groq (Recommended - Free & Fast)**
```bash
# Get API key: https://console.groq.com/keys
cp .env.example .env
echo "GROQ_API_KEY=gsk_your_key_here" >> .env
```

**Option B: Google Gemini (Free Tier)**
```bash
# Get API key: https://aistudio.google.com/app/apikey
echo "GEMINI_API_KEY=AIza_your_key_here" >> .env
```

**Option C: OpenAI (Paid)**
```bash
echo "OPENAI_API_KEY=sk-your_key_here" >> .env
```

**Option D: Ollama (Local, Free)**
```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3.2
# No API key needed - auto-detected
```

For detailed setup instructions, see `docs/API_SETUP_GUIDE.md` and `docs/LLAMA_GUIDE.md`.

### 3. Run Experiments

**Quick Test (10 personas, 2 topics)**
```bash
cd scripts
python wvs_experiment.py
```

**Full Experiment (Modify `wvs_experiment.py`)**
```python
# In wvs_experiment.py, change:

# Test mode (default)
for seed in RANDOM_SEEDS[:1]:           # First seed only
    for country_code in list(COUNTRIES.keys())[:1]:  # First country
        for topic in ETHICAL_TOPICS[:2]:   # First 2 topics
            num_personas=10                # 10 personas

# Full experiment
for seed in RANDOM_SEEDS:               # All 5 seeds
    for country_code in list(COUNTRIES.keys()):  # All 7 countries
        for topic in ETHICAL_TOPICS:       # All 7 topics
            num_personas=200               # 200 personas per country
```

### 4. Analyze Results

Results are saved in `data/wvs_results/{model}_temp{temp}/`:

```
wvs_results/
└── gpt4_temp1p0/
    ├── responses_United_States_homosexuality_seed42.csv  # Individual responses
    ├── stats_United_States_homosexuality_seed42.json     # Summary statistics
    └── ...
```

**CSV Columns:**
- `persona_id`, `country_code`, `country_name`, `topic`
- Demographics: `age`, `gender`, `education_level`, `social_class`, `political_left_right`
- Religious: `importance_religion`, `religiosity`
- Indirect indicators: `justifiability_premarital_sex`, `justifiability_casual_sex`
- Response: `response` (full text), `rating` (1-10 extracted score)

**Analysis with Python:**
```python
import pandas as pd
import json

# Load responses
df = pd.read_csv('wvs_results/gpt4_temp1p0/responses_United_States_homosexuality_seed42.csv')

# Load statistics
with open('wvs_results/gpt4_temp1p0/stats_United_States_homosexuality_seed42.json') as f:
    stats = json.load(f)

print(f"Mean rating: {stats['mean']:.2f}")
print(f"Distribution: {stats['distribution']}")
```

## Project Components

### Agent Module (`agent/agent.py`)

**Core Classes:**

1. **`WVSPersonaProfile`**
   - Dataclass representing WVS-7 compliant persona
   - All fields use official WVS-7 integer coding
   - Example fields:
     - `country_code`: 840 (USA), 410 (South Korea), etc.
     - `gender`: 1 (Male), 2 (Female)
     - `education_level`: 0-8 (ISCED 2011)
     - `political_left_right`: 1 (Left) to 10 (Right)

2. **`WVSPersonaGenerator`**
   - Generates random personas with realistic distributions
   - Supports fixed attributes for controlled experiments
   - Country-specific language assignment

3. **`StatelessPersonaAgent`**
   - Stateless LLM agent with persona context
   - Responds to ethical questions in character
   - Temperature-controlled response variability

4. **`WVSEthicalQuestions`**
   - Standardized ethical questions from WVS-7
   - Topics: homosexuality, abortion, divorce, suicide, euthanasia, prostitution, death_penalty
   - Justifiability scale: 1 (never justifiable) to 10 (always justifiable)

### LLM Module (`llm/llm.py`)

**Multi-Provider API Wrapper:**
- Unified interface for multiple LLM providers
- Automatic provider selection based on available API keys
- Priority: Groq → Together → Gemini → OpenAI → Ollama → localhost

**Supported Models:**
- **Groq**: `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`
- **Gemini**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-2.0-flash-exp`
- **OpenAI**: `gpt-4-1106-preview`, `gpt-3.5-turbo`
- **Together**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
- **Ollama**: `llama3.2`, `llama3.1:8b`, `llama3.1:70b`

### Experiment Script (`scripts/wvs_experiment.py`)

**Two Experiment Modes:**

1. **Single-turn** (Recommended)
   - All 7 questions in one prompt
   - Faster, more cost-efficient
   - Evaluates consistency across topics

2. **Separate** (More Detailed)
   - Each question in separate conversation
   - More accurate individual ratings
   - Better for detailed analysis

**Configuration Variables:**
```python
COUNTRIES = {840: "United States", 410: "South Korea", ...}  # 7 countries
ETHICAL_TOPICS = ["homosexuality", "abortion", ...]          # 7 topics
NUM_PERSONAS_PER_COUNTRY = 200                               # Personas per country
RANDOM_SEEDS = [42, 123, 456, 789, 1024]                     # For reproducibility
```

## Experimental Design

### Persona Generation
- **Sample size**: 200 personas per country (adjustable)
- **Demographics**: Age (18-85), gender, education, social class, etc.
- **Values**: Religion importance, political orientation, indirect ethical indicators
- **Exclusion**: Direct ethical values for experimental topics (prevents priming)

### Response Collection
- **Prompt**: WVS-7 exact question wording
- **Format**: "Please tell me... on a scale from 1 to 10, where 1 means 'never justifiable' and 10 means 'always justifiable'"
- **Parsing**: Regex extraction of numerical rating (1-10)
- **Temperature**: Default 1.0 (adjustable for diversity study)

### Data Analysis
- **Distribution comparison**: LLM vs. human histograms
- **Mean/Std deviation**: Central tendency and variance
- **Kendall's Tau**: Rank correlation with human data
- **Cross-country comparison**: Cultural pattern alignment

## Cost Estimation

**For 200 personas × 7 topics × 7 countries = 9,800 requests:**

| Provider | Model | Total Cost | Time |
|----------|-------|------------|------|
| **Groq** | Llama 3.3 70B | **FREE** | ~20 min |
| **Gemini** | 1.5 Flash | **FREE** | ~30 min |
| **OpenAI** | GPT-4 Turbo | ~$200 | ~1 hour |
| **Ollama** | Llama 3.2 | **FREE** | ~2-4 hours |

*Groq recommended for fastest free option*

## Advanced Usage

### Custom Persona Generation
```python
from agent import WVSPersonaGenerator, StatelessPersonaAgent

# Generate specific persona
generator = WVSPersonaGenerator(country_code=410, seed=42)  # South Korea
persona = generator.generate_persona(
    gender=2,                    # Female
    age=35,
    education_level=7,           # Master's degree
    political_left_right=7,      # Conservative
    religiosity=1                # Religious
)

# Create agent
agent = StatelessPersonaAgent(persona=persona, temp=1.0)

# Ask question
from agent import WVSEthicalQuestions
question = WVSEthicalQuestions.get_question("abortion")
response = agent.respond_to_ethical_question(question)
print(response.content)
```

### Model Selection
```python
# Specify model explicitly
response = agent.respond_to_ethical_question(
    question,
    model="gemini-1.5-pro",      # Higher quality
    max_tokens=500,              # Longer response
    temperature=0.7              # Less random
)
```

### Batch Processing
```python
# Generate multiple personas
personas = generator.generate_multiple_personas(
    n=100,
    education_level=6  # All with Bachelor's degree
)
```

## Citation

If you use this repository for your research, please cite:

```bibtex
@article{yourname2024wvsbias,
  title={Cultural Bias in AI Ethical Reasoning: A World Values Survey Analysis},
  author={Yena Choi},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- World Values Survey Association for providing the WVS Wave 7 data and questionnaire
- Anthropic, OpenAI, Google, Meta for providing API access to language models
- Inspired by "LLM Voting: Human Choices and AI Collective Decision Making" (Yang et al., 2024)

## Contact

For questions or collaboration inquiries, please open an issue or contact [your email].

## Additional Resources

- **WVS-7 Official Website**: https://www.worldvaluessurvey.org/
- **WVS-7 Codebook**: See `data/F00011055-WVS7_Codebook_Variables_report_V6_0.pdf`
- **API Setup Guide**: See `docs/API_SETUP_GUIDE.md`
- **Llama Guide**: See `docs/LLAMA_GUIDE.md`