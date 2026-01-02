# E*TRADE AI Portfolio Analysis

AI-powered portfolio analysis and recommendations using E*TRADE data and advanced language models.

## 📁 Project Structure

```
etrade-ai-portfolio-analysis/
├── etrade/                      # E*TRADE data fetching
│   ├── get_all_data.py         # Fetch portfolio data from E*TRADE
│   ├── generate_ai_prompt.py   # Generate prompts for AI analysis
│   ├── requirements.txt        # E*TRADE dependencies
│   └── etrade_reports/         # Downloaded portfolio data (gitignored)
│
├── ai/                          # AI analysis tools
│   ├── portfolio_advisor.py    # Main AI portfolio advisor
│   ├── daily_analysis.py       # Automated daily analysis
│   ├── view_analyses.py        # View past analyses
│   ├── requirements.txt        # AI dependencies
│   └── README.md              # AI module documentation
│
├── .env                         # Your API keys (create from .env.example)
├── .env.example                # Template for API keys
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd etrade-ai-portfolio-analysis
```

### 2. Set Up API Keys

**Create a `.env` file in the root directory and add your API keys:**

#### E*TRADE API (Required)
1. Go to [E*TRADE Developer Portal](https://developer.etrade.com/home)
2. Create an application to get your Consumer Key and Secret
3. Add them to `.env`:
   ```
   ETRADE_CONSUMER_KEY=your_actual_key
   ETRADE_CONSUMER_SECRET=your_actual_secret
   ```

#### OpenAI API (ChatGPT)
1. Get API key from [OpenAI Platform](https://platform.openai.com/api-keys)
2. Add to `.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```

### 3. Install Dependencies

```bash
pip install -r etrade/requirements.txt
pip install openai python-dotenv
```

### 4. Run Complete Analysis (Easiest!)

**One command to do everything:**
```bash
python run_analysis.py
```

This will:
- Fetch your portfolio data from E*TRADE
- Run AI analysis
- Display results and save to file

**Or run components separately:**

**Fetch Portfolio Data:**
```bash
python etrade/get_all_data.py
```

**Run AI Analysis:**
```bash
python ai/portfolio_advisor.py
```

**Daily Analysis:**
```bash
python ai/daily_analysis.py
```

## 📊 What You Can Do

### Complete Analysis (Recommended)
```bash
python run_analysis.py
```
One script that does everything - fetches data and runs AI analysis!

### Daily Analysis
```bash
python ai/daily_analysis.py
```
Automatically fetches fresh data and runs AI analysis.

### View Past Analyses
```bash
python ai/view_analyses.py
```
Browse and compare historical AI recommendations.

### Manual Steps
If you prefer to run each step manually:
1. Fetch data: `python etrade/get_all_data.py`
2. Run analysis: `python ai/portfolio_advisor.py`

## 🛠️ Troubleshooting

### "No module named 'rauth'"
```bash
cd etrade
pip install -r requirements.txt
```

### E*TRADE Authentication Issues
- Ensure your Consumer Key and Secret are correct
- Check that your E*TRADE app is approved for production access
- Try generating new credentials if needed

## 📝 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here]
