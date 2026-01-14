"""
AI Portfolio Prompt Generator
Automatically populates the AI prompt template with live portfolio data
"""
import json
import os
from datetime import datetime
from pathlib import Path

def load_portfolio_data():
    """Load the most recent portfolio JSON data from canonical path with legacy fallback"""
    candidate_dirs = [Path('etrade/etrade_reports'), Path('etrade_reports')]
    json_candidates = []

    for reports_dir in candidate_dirs:
        if reports_dir.is_dir():
            json_candidates.extend(reports_dir.glob('etrade_data_*.json'))

    if not json_candidates:
        raise FileNotFoundError("No portfolio data found in etrade/etrade_reports or legacy etrade_reports")

    filepath = max(json_candidates, key=os.path.getmtime)

    with open(filepath, 'r') as f:
        return json.load(f)

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:.2f}%"

def generate_positions_summary(data):
    """Generate formatted position details"""
    positions_text = "#### Account Holdings Breakdown\n\n"
    
    for account in data['accounts']:
        positions_text += f"**Account: {account['description']} ({account['account_id']})**\n\n"
        
        if not account.get('positions'):
            positions_text += "No positions in this account.\n\n"
            continue
        
        for position in account['positions']:
            symbol = position['symbol']
            qty = position['quantity']
            cost_basis = format_currency(position['total_cost'])
            current_value = format_currency(position['market_value'])
            gain_loss = format_currency(position['total_gain'])
            gain_loss_pct = format_percentage(position['total_gain_pct'])
            current_price = format_currency(position['current_price'])
            
            positions_text += f"**{symbol}** - {qty} shares\n"
            positions_text += f"  - Cost Basis: {cost_basis} ({format_currency(position['cost_per_share'])}/share)\n"
            positions_text += f"  - Current Value: {current_value} (@{current_price})\n"
            positions_text += f"  - Gain/Loss: {gain_loss} ({gain_loss_pct})\n"
            positions_text += f"  - Today's Change: {format_currency(position['todays_gain'])} ({format_percentage(position['todays_gain_pct'])})\n\n"
    
    return positions_text

def generate_prompt(data):
    """Generate the populated prompt"""
    summary = data['summary']
    timestamp = data['extraction_timestamp']
    
    # Calculate totals
    total_value = format_currency(summary['total_portfolio_value'])
    total_cash = format_currency(summary['total_cash'])
    buying_power = format_currency(summary['total_buying_power'])
    symbols = ", ".join(summary['unique_symbols'])
    
    # Calculate overall gain/loss
    total_cost = 0
    total_current = 0
    for account in data['accounts']:
        for position in account.get('positions', []):
            total_cost += position['total_cost']
            total_current += position['market_value']
    
    overall_gain_loss = total_current - total_cost
    overall_gain_loss_pct = (overall_gain_loss / total_cost * 100) if total_cost > 0 else 0
    
    # Generate positions summary
    positions_text = generate_positions_summary(data)
    
    # Calculate account values
    roth_value = 0
    brokerage_value = 0
    recent_trades = "No recent trades found."
    
    for account in data['accounts']:
        account_value = account['cash_allocation']['total_account_value']
        
        if 'Roth' in account['description']:
            roth_value = account_value
        else:
            brokerage_value = account_value
        
        # Extract recent trades if available
        if account.get('order_history'):
            trades_list = "- " + "\n- ".join([
                f"{order['action']} {order['quantity']} {order['symbol']} @ {format_currency(order['avg_price'])}"
                for order in account['order_history'][:3]
            ])
            recent_trades = trades_list
    
    # Build the prompt directly
    prompt = f"""# AI Portfolio Analysis & Optimization Prompt

## System Context
You are an expert financial advisor and portfolio analyst specializing in personalized investment strategy, risk management, and portfolio optimization. Your role is to analyze the user's current portfolio data and provide actionable advice to maximize returns while managing risk appropriately.

## User Portfolio Data (As of {timestamp})

### Portfolio Summary
- **Total Portfolio Value:** {total_value}
- **Total Cash Available:** {total_cash}
- **Total Buying Power:** {buying_power}
- **Total Positions:** {summary['total_positions']}
- **Unique Symbols:** {symbols}
- **Overall Portfolio Gain/Loss:** {format_currency(overall_gain_loss)} ({format_percentage(overall_gain_loss_pct)})

### Current Holdings
{positions_text}

### Account Breakdown
- **Roth IRA Account:** {format_currency(roth_value)}
- **Individual Brokerage Account:** {format_currency(brokerage_value)}

### Recent Trading Activity
{recent_trades}

## Analysis Request

Please provide a comprehensive portfolio analysis addressing the following:

### 1. Portfolio Health Assessment
- Evaluate current diversification across sectors, asset classes, and individual positions
- Identify concentration risks (positions that are too large relative to total portfolio)
- Assess risk profile based on current holdings
- Comment on the cash allocation strategy

### 2. Performance Analysis
- Analyze overall portfolio performance and individual position performance
- Identify top performers and underperformers
- Evaluate whether gains/losses align with market trends or are position-specific

### 3. Optimization Recommendations
- Suggest rebalancing strategies if needed
- Recommend specific actions: HOLD, BUY more, SELL/reduce, or TRIM positions
- Identify opportunities to improve risk-adjusted returns
- Suggest potential new positions that could enhance diversification

### 4. Tax Optimization (Roth IRA vs Brokerage)
- Recommend which types of investments belong in Roth IRA vs taxable brokerage
- Identify tax-loss harvesting opportunities in the taxable account
- Suggest strategies to maximize tax efficiency

### 5. Action Plan
Provide a prioritized list of specific actions with:
- What to do (buy/sell/hold specific symbols and quantities)
- Why (clear reasoning for each recommendation)
- Expected impact on portfolio risk and returns
- Timeline for execution

### 6. Risk Warnings
- Highlight any red flags or concerns about specific positions
- Note upcoming economic events or market conditions that could impact the portfolio
- Identify positions that may need monitoring

## Guidelines for Your Response
- Be specific with symbol names and approximate percentages/amounts
- Prioritize actionable recommendations
- Consider both short-term tactical moves and long-term strategic positioning
- Balance growth potential with risk management
- Assume the investor has moderate risk tolerance unless holdings suggest otherwise
- Current date for context: {datetime.now().strftime('%B %d, %Y')}

Please provide your analysis now.
"""
    
    return prompt

def save_populated_prompt():
    """Generate and save the populated prompt"""
    try:
        data = load_portfolio_data()
        populated_prompt = generate_prompt(data)
        
        # Save to file
        output_file = 'ai_portfolio_prompt_populated.md'
        with open(output_file, 'w') as f:
            f.write(populated_prompt)
        
        print(f"✓ Populated prompt saved to: {output_file}")
        print(f"✓ Portfolio data loaded from: {data['extraction_timestamp']}")
        print(f"✓ Portfolio value: ${data['summary']['total_portfolio_value']:,.2f}")
        
        return populated_prompt
        
    except Exception as e:
        print(f"✗ Error generating prompt: {e}")
        raise

if __name__ == "__main__":
    save_populated_prompt()
