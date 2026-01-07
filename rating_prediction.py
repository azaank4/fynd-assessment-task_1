import csv
import json
import random
import os
import re
import time
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dotenv import load_dotenv
from google import genai

load_dotenv()  # Load API key from .env file

# API and configuration setup
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.5-flash"  # Gemini model to use
DATASET_PATH = "yelp.csv"
SAMPLE_SIZE = 200  # Total reviews to sample from dataset
TEST_SIZE = 50  # Number of reviews to evaluate
RANDOM_SEED = 42  # For reproducible results
RATE_LIMIT_DELAY = 1.0  # Delay between API calls in seconds

random.seed(RANDOM_SEED)


def load_yelp_data(filepath: str, sample_size: int) -> List[Dict]:
    """Load Yelp reviews from CSV and return random sample."""
    all_reviews = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only include rows with valid text and stars
                if 'text' in row and 'stars' in row and row['text'].strip():
                    all_reviews.append({
                        'review_id': row.get('review_id', f'review_{len(all_reviews)}'),
                        'text': row['text'].strip(),
                        'actual_stars': int(float(row['stars'])),  # Convert to integer
                    })
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure the file exists.")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")
    
    if not all_reviews:
        raise ValueError("No valid reviews found in dataset")
    
    # Randomly sample reviews
    sampled = random.sample(all_reviews, min(sample_size, len(all_reviews)))
    print(f"Loaded {len(all_reviews)} total reviews. Sampled {len(sampled)} for evaluation.")
    return sampled


def split_data(data: List[Dict], test_size: int) -> Tuple[List[Dict], List[Dict]]:
    """Split data into test set and remaining data."""
    return data[:test_size], data[test_size:]


def prompt_v1_baseline(review_text: str) -> Tuple[str, str]:
    """Approach 1: Simple direct classification with minimal instructions."""
    system_prompt = """You are an expert review analyst. Analyze the given Yelp review and predict the star rating.
Consider the overall sentiment, tone, and specific positive or negative statements in the review.
Return a JSON object with 'predicted_stars' (1-5 integer) and 'explanation' (brief reason for the rating)."""
    
    user_prompt = f"""Review: {review_text}

Respond ONLY with valid JSON in this format:
{{"predicted_stars": <int 1-5>, "explanation": "<brief reasoning>"}}"""
    
    return system_prompt, user_prompt


def prompt_v2_chain_of_thought(review_text: str) -> Tuple[str, str]:
    """Approach 2: Chain-of-Thought prompting with explicit reasoning steps."""
    system_prompt = """You are an expert review analyst. Analyze Yelp reviews systematically to predict star ratings.

Process:
1. Identify positive indicators (compliments, enthusiasm, repeat visits)
2. Identify negative indicators (complaints, disappointments, issues)
3. Evaluate overall sentiment balance
4. Assess urgency/extremity of language
5. Predict rating based on combined analysis"""
    
    user_prompt = f"""Review: {review_text}

Follow these steps:
1. List the top 2-3 POSITIVE indicators if present
2. List the top 2-3 NEGATIVE indicators if present
3. Assess the balance (which is stronger?)
4. Consider language intensity (mild satisfaction vs. enthusiastic, mild complaint vs. angry)
5. Predict the rating

Return ONLY valid JSON:
{{"predicted_stars": <int 1-5>, "explanation": "<brief reasoning based on your analysis>"}}"""
    
    return system_prompt, user_prompt


def prompt_v3_criteria_based(review_text: str) -> Tuple[str, str]:
    """Approach 3: Criteria-based evaluation with explicit rating rubric."""
    system_prompt = """You are an expert review analyst evaluating Yelp reviews against explicit criteria.

Rating Scale (Strict Definitions):
★ 1 (Poor): Major problems, negative experience, would not recommend, complaints about quality/service/value
★ 2 (Fair): Mixed experience with notable issues, concerns outweigh positives, some disappointment
★ 3 (Good): Generally positive, meets expectations, minor issues or room for improvement
★ 4 (Very Good): Clearly positive, exceeds expectations, few if any complaints, strong satisfaction
★ 5 (Excellent): Outstanding experience, exceptional quality/service/value, highly enthusiastic, would strongly recommend

Evaluation Dimensions:
- Product/Food Quality (if applicable)
- Service/Staff Experience
- Value for Money
- Overall Experience & Atmosphere
- Would the reviewer return/recommend?"""
    
    user_prompt = f"""Review: {review_text}

Evaluate using the criteria above:
1. Rate each dimension on the scale provided
2. Assess where the review falls on the overall satisfaction spectrum
3. Determine the appropriate star rating (1-5)

Return ONLY valid JSON:
{{"predicted_stars": <int 1-5>, "explanation": "<reasoning based on dimensions and criteria>"}}"""
    
    return system_prompt, user_prompt


def call_gemini(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Call Google Gemini API and return response text."""
    try:
        full_prompt = f"{system_prompt}\n\n{user_prompt}"  # Combine prompts
        generation_config = {'temperature': temperature, 'max_output_tokens': 500}
        
        # Make API call
        response = client.models.generate_content(
            model=MODEL, 
            contents=full_prompt, 
            config=generation_config
        )
        
        time.sleep(RATE_LIMIT_DELAY)  # Rate limiting to avoid API throttling
        
        # Extract text from response object
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            return response.candidates[0].content.parts[0].text
        else:
            return ""
    except Exception as e:
        print(f"\n  API Error: {e}")
        return ""


def extract_json(response_text: str) -> Optional[Dict]:
    """Extract JSON from API response with multiple fallback strategies."""
    if not response_text or not response_text.strip():
        return None
    
    # Strategy 1: Try direct JSON parsing
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Manual brace matching to find complete JSON object
    stack = []
    start_idx = None
    
    for i, char in enumerate(response_text):
        if char == '{':
            if not stack:
                start_idx = i  # Mark start of JSON object
            stack.append('{')
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:  # Found complete object
                    try:
                        json_str = response_text[start_idx:i+1]
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
    
    # Strategy 4: Regex pattern matching for expected format
    pattern = r'\{\s*"predicted_stars"\s*:\s*\d+\s*,\s*"explanation"\s*:\s*"[^"]*"\s*\}'
    match = re.search(pattern, response_text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def validate_prediction(json_obj: Optional[Dict]) -> Tuple[bool, Optional[int]]:
    """Validate extracted JSON has required fields and valid star rating."""
    if not isinstance(json_obj, dict):
        return False, None
    
    if 'predicted_stars' not in json_obj:
        return False, None
    
    try:
        stars = int(json_obj['predicted_stars'])
        if 1 <= stars <= 5:  # Validate rating is in valid range
            return True, stars
    except (ValueError, TypeError):
        pass
    
    return False, None


def evaluate_prompt_approach(test_data: List[Dict], prompt_func, approach_name: str, 
                            temperature: float = 0.7) -> Dict:
    """Evaluate a single prompting approach on test data."""
    print(f"\n{'='*70}")
    print(f"Evaluating {approach_name}")
    print(f"{'='*70}")
    
    # Initialize results tracking
    results = {
        'approach': approach_name,
        'predictions': [],
        'accuracy': 0.0,
        'json_validity_rate': 0.0,
        'correct_count': 0,
        'valid_json_count': 0,
        'total': len(test_data),
    }
    
    # Process each review
    for idx, review in enumerate(test_data):
        print(f"  [{idx+1}/{len(test_data)}] Processing review...", end=" ")
        
        try:
            # Get prompts for this approach
            system_prompt, user_prompt = prompt_func(review['text'])
            
            # Call API and get response
            response = call_gemini(system_prompt, user_prompt, temperature=temperature)
            
            # Extract and validate JSON
            json_obj = extract_json(response)
            is_valid, predicted_stars = validate_prediction(json_obj)
            
            actual_stars = review['actual_stars']
            is_correct = is_valid and (predicted_stars == actual_stars)
            
            # Store prediction results
            results['predictions'].append({
                'review_id': review['review_id'],
                'actual_stars': actual_stars,
                'predicted_stars': predicted_stars,
                'is_valid_json': is_valid,
                'is_correct': is_correct,
                'raw_response': response[:200] if response else "",  # First 200 chars
                'explanation': json_obj.get('explanation', '') if json_obj else '',
            })
            
            # Update counters
            if is_valid:
                results['valid_json_count'] += 1
            if is_correct:
                results['correct_count'] += 1
            
            # Display status
            status = "✓" if is_correct else ("!" if not is_valid else "✗")
            pred_str = str(predicted_stars) if predicted_stars else "N/A"
            print(f"{status} Actual: {actual_stars}, Predicted: {pred_str}")
            
        except Exception as e:
            print(f"\n  Error processing review {idx+1}: {e}")
            # Store error case
            results['predictions'].append({
                'review_id': review['review_id'],
                'actual_stars': review['actual_stars'],
                'predicted_stars': None,
                'is_valid_json': False,
                'is_correct': False,
                'raw_response': str(e),
                'explanation': '',
            })
    
    # Calculate final metrics
    if results['total'] > 0:
        results['accuracy'] = (results['correct_count'] / results['total']) * 100
        results['json_validity_rate'] = (results['valid_json_count'] / results['total']) * 100
    
    return results


def calculate_consistency(results: Dict) -> float:
    """Calculate consistency score based on prediction variance."""
    # Get all valid predictions
    valid_predictions = [
        p['predicted_stars'] 
        for p in results['predictions'] 
        if p['predicted_stars'] is not None
    ]
    
    if not valid_predictions or len(valid_predictions) < 2:
        return 0.0
    
    # Calculate variance
    mean = sum(valid_predictions) / len(valid_predictions)
    variance = sum((x - mean) ** 2 for x in valid_predictions) / len(valid_predictions)
    
    # Convert to consistency score (higher is better)
    consistency_score = max(0, 100 - (variance * 20))
    return consistency_score


def calculate_mae(results: Dict) -> float:
    """Calculate Mean Absolute Error for valid predictions."""
    errors = [
        abs(p['actual_stars'] - p['predicted_stars']) 
        for p in results['predictions'] 
        if p['predicted_stars'] is not None and p['is_valid_json']
    ]
    return sum(errors) / len(errors) if errors else 0.0


def generate_comparison_table(all_results: List[Dict]) -> str:
    """Generate formatted comparison table for all approaches."""
    table = "\n" + "="*90 + "\n"
    table += "COMPARISON TABLE: PROMPTING APPROACHES\n"
    table += "="*90 + "\n\n"
    
    # Header row
    table += f"{'Metric':<30} | {'Prompt v1 (Baseline)':<20} | {'Prompt v2 (CoT)':<20} | {'Prompt v3 (Criteria)':<20}\n"
    table += "-"*90 + "\n"
    
    # Accuracy row
    accuracies = [f"{r['accuracy']:.1f}%" for r in all_results]
    table += f"{'Accuracy':<30} | {accuracies[0]:<20} | {accuracies[1]:<20} | {accuracies[2]:<20}\n"
    
    # JSON validity row
    validity_rates = [f"{r['json_validity_rate']:.1f}%" for r in all_results]
    table += f"{'JSON Validity Rate':<30} | {validity_rates[0]:<20} | {validity_rates[1]:<20} | {validity_rates[2]:<20}\n"
    
    # Consistency row
    consistencies = [f"{calculate_consistency(r):.1f}/100" for r in all_results]
    table += f"{'Consistency Score':<30} | {consistencies[0]:<20} | {consistencies[1]:<20} | {consistencies[2]:<20}\n"
    
    # MAE row
    maes = [f"{calculate_mae(r):.2f}" for r in all_results]
    table += f"{'Mean Absolute Error':<30} | {maes[0]:<20} | {maes[1]:<20} | {maes[2]:<20}\n"
    
    # Correct predictions row
    correct_counts = [f"{r['correct_count']}/{r['total']}" for r in all_results]
    table += f"{'Correct Predictions':<30} | {correct_counts[0]:<20} | {correct_counts[1]:<20} | {correct_counts[2]:<20}\n"
    
    table += "="*90 + "\n"
    return table


def generate_analysis(all_results: List[Dict]) -> str:
    """Generate detailed analysis of results."""
    analysis = "\n" + "="*90 + "\n"
    analysis += "DETAILED ANALYSIS & FINDINGS\n"
    analysis += "="*90 + "\n\n"
    
    # Identify best performer
    sorted_results = sorted(all_results, key=lambda x: x['accuracy'], reverse=True)
    best_approach = sorted_results[0]
    
    analysis += f"BEST PERFORMER: {best_approach['approach']}\n"
    analysis += f"  Accuracy: {best_approach['accuracy']:.1f}%\n"
    analysis += f"  JSON Validity: {best_approach['json_validity_rate']:.1f}%\n"
    analysis += f"  Consistency: {calculate_consistency(best_approach):.1f}/100\n"
    analysis += f"  MAE: {calculate_mae(best_approach):.2f}\n\n"
    
    # Individual approach breakdown
    analysis += "APPROACH BREAKDOWN:\n"
    analysis += "-"*90 + "\n\n"
    
    for result in all_results:
        analysis += f"**{result['approach']}**\n"
        analysis += f"  Accuracy: {result['accuracy']:.1f}% ({result['correct_count']}/{result['total']} correct)\n"
        analysis += f"  JSON Validity: {result['json_validity_rate']:.1f}%\n"
        analysis += f"  Consistency Score: {calculate_consistency(result):.1f}/100\n"
        analysis += f"  MAE: {calculate_mae(result):.2f}\n"
        
        # Calculate average error for incorrect predictions
        incorrect = [
            p for p in result['predictions'] 
            if not p['is_correct'] and p['predicted_stars'] is not None
        ]
        if incorrect:
            off_by = [abs(p['actual_stars'] - p['predicted_stars']) for p in incorrect]
            avg_off = sum(off_by) / len(off_by) if off_by else 0
            analysis += f"  Avg. Prediction Error (incorrect): ±{avg_off:.1f} stars\n"
        
        analysis += "\n"
    
    # Trade-offs and recommendations
    analysis += "TRADE-OFFS & RECOMMENDATIONS:\n"
    analysis += "-"*90 + "\n"
    analysis += """
1. **Baseline Prompt (v1)**
   ✓ Fastest, simplest, lowest latency
   ✓ Good baseline for comparison
   ✗ May miss nuance in mixed reviews
   → Best for: Speed-critical applications, high-volume processing
   
2. **Chain-of-Thought (v2)**
   ✓ Encourages structured reasoning
   ✓ More reliable than baseline (step-by-step logic)
   ✓ Explains its reasoning process
   ✗ Slightly higher latency (requires more thinking)
   → Best for: Accuracy-critical applications, need for transparency
   
3. **Criteria-Based (v3)**
   ✓ Provides explicit rubric, standardizes judgments
   ✓ Most consistent across similar reviews
   ✓ Best for domain-specific evaluation
   ✗ May be overly rigid for edge cases
   ✗ Highest latency (most complex evaluation)
   → Best for: Production systems needing reliability, corporate standards
"""
    
    analysis += "="*90 + "\n"
    return analysis


def main():
    """Main execution function."""
    print("YELP REVIEW RATING PREDICTION VIA PROMPT DESIGN")
    print("="*90)
    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Sample Size: {SAMPLE_SIZE}")
    print(f"Test Size: {TEST_SIZE}")
    print("="*90)
    
    # Load and prepare data
    print("\nLoading data...")
    try:
        data = load_yelp_data(DATASET_PATH, SAMPLE_SIZE)
        test_data, _ = split_data(data, TEST_SIZE)
    except Exception as e:
        print(f"Error: {e}")
        return
    
    all_results = []
    
    print("\n" + "="*90)
    print("STARTING EVALUATION PHASE")
    print("="*90)
    
    # Evaluate each prompting approach
    results_v1 = evaluate_prompt_approach(
        test_data, prompt_v1_baseline, "Prompt v1: Baseline"
    )
    all_results.append(results_v1)
    
    results_v2 = evaluate_prompt_approach(
        test_data, prompt_v2_chain_of_thought, "Prompt v2: Chain-of-Thought"
    )
    all_results.append(results_v2)
    
    results_v3 = evaluate_prompt_approach(
        test_data, prompt_v3_criteria_based, "Prompt v3: Criteria-Based"
    )
    all_results.append(results_v3)
    
    # Generate and display comparison
    comparison = generate_comparison_table(all_results)
    print(comparison)
    
    # Generate and display analysis
    analysis = generate_analysis(all_results)
    print(analysis)
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rating_prediction_results_{timestamp}.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL,
        'sample_size': SAMPLE_SIZE,
        'test_size': TEST_SIZE,
        'results': all_results,
        'comparison_metrics': {
            'v1': {
                'accuracy': all_results[0]['accuracy'],
                'json_validity_rate': all_results[0]['json_validity_rate'],
                'consistency': calculate_consistency(all_results[0]),
                'mae': calculate_mae(all_results[0]),
            },
            'v2': {
                'accuracy': all_results[1]['accuracy'],
                'json_validity_rate': all_results[1]['json_validity_rate'],
                'consistency': calculate_consistency(all_results[1]),
                'mae': calculate_mae(all_results[1]),
            },
            'v3': {
                'accuracy': all_results[2]['accuracy'],
                'json_validity_rate': all_results[2]['json_validity_rate'],
                'consistency': calculate_consistency(all_results[2]),
                'mae': calculate_mae(all_results[2]),
            },
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"✅ Evaluation complete!")


if __name__ == "__main__":
    main()
