"""
AI Portfolio Advisor
Connects your portfolio data with AI APIs for investment analysis and advice
Supports: OpenAI, Anthropic Claude, Google Gemini, and local models
"""
import os
import sys
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

# Add parent directory to path to import generate_ai_prompt
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from etrade.generate_ai_prompt import load_portfolio_data, generate_prompt

class AIPortfolioAdvisor:
    """Manages AI API connections for portfolio analysis"""
    
    def __init__(self):
        self.provider = None
        self.api_key = None
        
    def setup_openai(self, api_key):
        """Setup OpenAI API (GPT-4, GPT-3.5-turbo, etc.)"""
        try:
            import openai
            self.provider = 'openai'
            self.api_key = api_key
            openai.api_key = api_key
            print("✓ OpenAI API configured")
            return True
        except ImportError:
            print("✗ OpenAI library not installed. Run: pip install openai")
            return False
    
    def setup_anthropic(self, api_key):
        """Setup Anthropic Claude API"""
        try:
            import anthropic
            self.provider = 'anthropic'
            self.api_key = api_key
            self.client = anthropic.Anthropic(api_key=api_key)
            print("✓ Anthropic Claude API configured")
            return True
        except ImportError:
            print("✗ Anthropic library not installed. Run: pip install anthropic")
            return False
    
    def setup_google(self, api_key):
        """Setup Google Gemini API"""
        try:
            import google.generativeai as genai
            self.provider = 'google'
            self.api_key = api_key
            genai.configure(api_key=api_key)
            print("✓ Google Gemini API configured")
            return True
        except ImportError:
            print("✗ Google Generative AI library not installed. Run: pip install google-generativeai")
            return False
    
    def get_ai_analysis(self, prompt_text, model=None):
        """Get AI analysis based on the provider"""
        
        if self.provider == 'openai':
            return self._query_openai(prompt_text, model or 'gpt-4o')
        
        elif self.provider == 'anthropic':
            return self._query_anthropic(prompt_text, model or 'claude-3-5-sonnet-20241022')
        
        elif self.provider == 'google':
            return self._query_google(prompt_text, model or 'gemini-1.5-pro')
        
        else:
            return "Error: No AI provider configured"
    
    def _query_openai(self, prompt, model):
        """Query OpenAI API"""
        import openai
        
        print(f"\n🤖 Querying OpenAI ({model})...")
        print("⏳ This may take 30-60 seconds for detailed analysis...\n")
        
        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor and portfolio analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error querying OpenAI: {str(e)}"
    
    def _query_anthropic(self, prompt, model):
        """Query Anthropic Claude API"""
        print(f"\n🤖 Querying Anthropic Claude ({model})...")
        print("⏳ This may take 30-60 seconds for detailed analysis...\n")
        
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error querying Anthropic: {str(e)}"
    
    def _query_google(self, prompt, model):
        """Query Google Gemini API"""
        import google.generativeai as genai
        
        print(f"\n🤖 Querying Google Gemini ({model})...")
        print("⏳ This may take 30-60 seconds for detailed analysis...\n")
        
        try:
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            return f"Error querying Google: {str(e)}"


def print_formatted_response(response_text):
    """Print the AI response with nice formatting"""
    print("\n" + "="*80)
    print(" AI PORTFOLIO ANALYSIS & RECOMMENDATIONS")
    print("="*80 + "\n")
    print(response_text)
    print("\n" + "="*80 + "\n")


def save_analysis(response_text):
    """Save the analysis to a file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ai_analysis_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(" AI PORTFOLIO ANALYSIS & RECOMMENDATIONS\n")
        f.write(f" Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        f.write(response_text)
    
    print(f"💾 Analysis saved to: {filename}")
    return filename


def main():
    """Main execution flow"""
    print("\n" + "="*80)
    print(" AI PORTFOLIO ADVISOR")
    print("="*80 + "\n")
    
    # Load portfolio data
    print("📊 Loading portfolio data...")
    try:
        portfolio_data = load_portfolio_data()
        print(f"✓ Portfolio loaded: ${portfolio_data['summary']['total_portfolio_value']:,.2f}")
    except Exception as e:
        print(f"✗ Error loading portfolio data: {e}")
        return
    
    # Generate prompt
    print("📝 Generating AI prompt...")
    try:
        prompt_text = generate_prompt(portfolio_data)
        print(f"✓ Prompt generated ({len(prompt_text)} characters)")
    except Exception as e:
        print(f"✗ Error generating prompt: {e}")
        return
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️  OpenAI API key not found in .env file")
        api_key = input("Enter your OpenAI API key: ").strip()
        
        if not api_key:
            print("❌ No API key provided. Exiting.\n")
            return
    
    openai.api_key = api_key
    print("✓ ChatGPT API configured\n")
    
    # Get AI analysis
    response = get_ai_analysis(prompt_text)
    
    # Display and save
    print_formatted_response(response)
    save_analysis(response)
    
    print("✓ Analysis complete!")


if __name__ == "__main__":
    main()
