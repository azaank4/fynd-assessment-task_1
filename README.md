# Yelp Review Rating Prediction via Prompting Design

A comprehensive evaluation of different prompting strategies for predicting Yelp review star ratings (1-5) using Google Gemini AI.

## 📋 Overview

This project compares three distinct prompt engineering approaches to classify Yelp reviews into star ratings:

1. **Baseline**: Simple, direct sentiment analysis
2. **Chain-of-Thought (CoT)**: Step-by-step reasoning process
3. **Criteria-Based**: Explicit rating rubric with multiple evaluation dimensions

The system evaluates each approach on accuracy, JSON validity, consistency, and prediction error metrics.

## 🎯 Project Objectives

- Design and test multiple prompting strategies for rating prediction
- Compare effectiveness of different prompt engineering techniques
- Evaluate trade-offs between speed, accuracy, and reliability
- Provide actionable insights for production deployment

## 🚀 Features

- **Multiple Prompting Strategies**: Three distinct approaches with documented rationale
- **Comprehensive Metrics**: Accuracy, JSON validity rate, consistency score, and MAE
- **Robust Error Handling**: Multiple JSON extraction strategies and API error recovery
- **Rate Limiting**: Built-in delays to prevent API throttling
- **Detailed Analysis**: Automated comparison table and insights generation
- **Exportable Results**: JSON output with timestamp for reproducibility

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Google Cloud account with Gemini API access
- Yelp Reviews dataset from Kaggle

### Step 1: Clone or Download

```bash
# If using git
git clone <your-repo-url>
cd yelp-rating-prediction

# Or download and extract the ZIP file
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install google-generativeai python-dotenv
```

### Step 3: Download Dataset

1. Visit [Yelp Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset)
2. Download the CSV file
3. Rename it to `yelp.csv` and place it in the project root directory

### Step 4: Configure API Key

Create a `.env` file in the project root:

```bash
# .env file
GOOGLE_API_KEY=your_gemini_api_key_here
```

**To get your Gemini API key:**
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Get API Key"
3. Create a new API key or use an existing one
4. Copy the key to your `.env` file

## 📁 Project Structure

```
yelp-rating-prediction/
│
├── main.py                          # Main evaluation script
├── yelp.csv                         # Yelp reviews dataset (download separately)
├── .env                             # API key configuration (create this)
├── README.md                        # This file
│
└── Output Files (generated):
    ├── rating_prediction_results_<timestamp>.json
    └── Console output with comparison tables
```

## 🎮 Usage

### Basic Usage

Run the evaluation on 50 test reviews:

```bash
python main.py
```

### Customize Parameters

Edit the configuration section in `main.py`:

```python
SAMPLE_SIZE = 200      # Total reviews to sample from dataset
TEST_SIZE = 50         # Number of reviews to test (use first N)
RANDOM_SEED = 42       # For reproducibility
RATE_LIMIT_DELAY = 1.0 # Seconds between API calls
```

### Expected Runtime

- **50 reviews × 3 approaches = 150 API calls**
- **With 1-second delay = ~2.5-3 minutes**
- Actual time may vary based on API response time

## 📊 Output

### Console Output

The script prints real-time progress and generates:

1. **Live Evaluation Progress**
   ```
   [1/50] Processing review... ✓ Actual: 5, Predicted: 5
   [2/50] Processing review... ✗ Actual: 4, Predicted: 3
   ```

2. **Comparison Table**
   ```
   ==========================================================================================
   COMPARISON TABLE: PROMPTING APPROACHES
   ==========================================================================================
   
   Metric                         | Prompt v1 (Baseline) | Prompt v2 (CoT)      | Prompt v3 (Criteria)
   ------------------------------------------------------------------------------------------
   Accuracy                       | 64.0%                | 68.0%                | 72.0%
   JSON Validity Rate             | 98.0%                | 100.0%               | 100.0%
   Consistency Score              | 85.2/100             | 88.7/100             | 91.3/100
   Mean Absolute Error            | 0.52                 | 0.44                 | 0.38
   Correct Predictions            | 32/50                | 34/50                | 36/50
   ```

3. **Detailed Analysis**
   - Best performer identification
   - Approach-specific breakdowns
   - Trade-offs and recommendations

### JSON Output File

Results are saved to `rating_prediction_results_<timestamp>.json`:

```json
{
  "timestamp": "2025-01-07T10:30:00",
  "model": "gemini-2.5-flash",
  "sample_size": 200,
  "test_size": 50,
  "results": [...],
  "comparison_metrics": {...}
}
```

## 🔍 Prompting Approaches Explained

### Approach 1: Baseline
**Design Philosophy**: Simple and direct

- Asks for overall sentiment classification
- Minimal instructions
- Fastest response time
- **Best for**: High-volume, speed-critical applications

**Prompt Structure**:
```
You are an expert review analyst. Analyze the review and predict the star rating.
Consider overall sentiment, tone, and specific statements.
```

### Approach 2: Chain-of-Thought (CoT)
**Design Philosophy**: Structured reasoning

- Breaks down analysis into explicit steps
- Identifies positive and negative indicators separately
- Evaluates sentiment balance
- **Best for**: Accuracy-critical applications requiring transparency

**Prompt Structure**:
```
Follow these steps:
1. List positive indicators
2. List negative indicators
3. Assess the balance
4. Consider language intensity
5. Predict rating
```

### Approach 3: Criteria-Based
**Design Philosophy**: Standardized evaluation

- Provides explicit 1-5 star rating definitions
- Evaluates multiple dimensions (quality, service, value, atmosphere)
- Most structured and consistent
- **Best for**: Production systems requiring reliability

**Prompt Structure**:
```
Rating Scale:
★ 1 (Poor): Major problems, negative experience...
★ 2 (Fair): Mixed experience with notable issues...
★ 3 (Good): Generally positive, meets expectations...
★ 4 (Very Good): Clearly positive, exceeds expectations...
★ 5 (Excellent): Outstanding experience...

Evaluate dimensions: Quality, Service, Value, Experience
```

## 📈 Evaluation Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Accuracy** | Percentage of exact matches | Higher = better predictions |
| **JSON Validity Rate** | % of properly formatted responses | Higher = more reliable API interaction |
| **Consistency Score** | Inverse of prediction variance | Higher = more stable predictions |
| **Mean Absolute Error (MAE)** | Average star difference | Lower = closer predictions |

## ⚙️ Configuration Options

### Model Selection

Change the Gemini model:
```python
MODEL = "gemini-2.5-flash"  # Fast, efficient
# MODEL = "gemini-pro"      # More capable, slower
```

### Temperature Adjustment

Control randomness in predictions:
```python
temperature = 0.7  # Default (balanced)
temperature = 0.3  # More deterministic
temperature = 1.0  # More creative/varied
```

### Sample Size

For quick testing:
```python
TEST_SIZE = 10  # Quick test on 10 reviews
```

For comprehensive evaluation:
```python
TEST_SIZE = 200  # Full evaluation (takes ~10 minutes)
```

## 🐛 Troubleshooting

### Common Issues

**1. "GOOGLE_API_KEY not found"**
```bash
# Solution: Create .env file with your API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

**2. "Dataset not found at yelp.csv"**
```bash
# Solution: Download from Kaggle and rename to yelp.csv
# Place in project root directory
```

**3. API Rate Limiting Errors**
```python
# Solution: Increase delay between calls
RATE_LIMIT_DELAY = 2.0  # Increase from 1.0 to 2.0 seconds
```

**4. JSON Parsing Failures**
```
# The code has multiple fallback strategies
# Check raw_response in output JSON for debugging
# May need to adjust temperature or prompt wording
```

**5. Low Accuracy Results**
```
# This is expected! Sentiment analysis is challenging
# Typical accuracy ranges: 50-75%
# Focus on comparing approaches, not absolute accuracy
```

## 📚 Dependencies

```
google-generativeai==0.3.0+
python-dotenv==1.0.0+
Python 3.8+
```

Full requirements:
```bash
pip install google-generativeai python-dotenv
```

## 🔬 Research Notes

### Why These Metrics?

- **Accuracy**: Direct measure of correctness
- **JSON Validity**: Tests prompt instruction-following
- **Consistency**: Important for production reliability
- **MAE**: More nuanced than accuracy (off-by-1 vs off-by-4)

### Expected Results

Based on similar studies:
- Baseline: 55-65% accuracy
- Chain-of-Thought: 60-70% accuracy
- Criteria-Based: 65-75% accuracy

Your mileage may vary based on:
- Dataset characteristics
- Model version
- Random sampling
- Temperature settings

## 📝 License

This project is for educational purposes. Please respect:
- Kaggle dataset usage terms
- Google Gemini API usage policies
- Academic integrity guidelines

## 🤝 Contributing

Suggestions for improvement:
1. Add visualization functions (matplotlib/seaborn)
2. Implement cross-validation
3. Add more prompting strategies
4. Test on other review datasets
5. Add confusion matrix analysis

## 📧 Support

For issues:
1. Check troubleshooting section above
2. Review Google Gemini API documentation
3. Verify dataset format matches expected structure

## 🎓 Academic Use

If using for coursework, please:
- Cite the Yelp dataset properly
- Document your methodology clearly
- Include comparison tables and analysis
- Discuss limitations and future work

## 📖 References

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Yelp Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/omkarsabnis/yelp-reviews-dataset)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
