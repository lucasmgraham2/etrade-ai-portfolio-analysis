"""
Streamlined ETrade API Data Extraction for Investment Analysis
Focuses on positions, performance, cash allocation, and trading patterns
"""
import webbrowser
import json
from rauth import OAuth1Service
from datetime import datetime
import traceback
import os
import glob
from dotenv import load_dotenv

# Load environment variables from parent directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

def oauth_authenticate():
    """Authenticate with ETrade OAuth and return session"""
    etrade = OAuth1Service(
        name="etrade",
        consumer_key=os.getenv("ETRADE_CONSUMER_KEY"),
        consumer_secret=os.getenv("ETRADE_CONSUMER_SECRET"),
        request_token_url="https://api.etrade.com/oauth/request_token",
        access_token_url="https://api.etrade.com/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url="https://api.etrade.com")

    base_url = "https://api.etrade.com"
    
    print("\n" + "="*80)
    print("ETRADE PRODUCTION AUTHENTICATION")
    print("="*80)
    
    print("\nStep 1: Getting authorization token...")
    request_token, request_token_secret = etrade.get_request_token(
        params={"oauth_callback": "oob", "format": "json"},
        timeout=10
    )
    print("[OK] Token received")
    
    authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)
    
    print("\nStep 2: Opening browser for authorization...")
    try:
        webbrowser.open(authorize_url)
        print("[OK] Browser opened")
    except:
        print("[!] Could not open browser automatically")
    
    print("\nAuthorization URL:")
    print(f"  {authorize_url}")
    print("\nInstructions:")
    print("  1. Log in with your ETrade PRODUCTION credentials")
    print("  2. Accept the authorization request")
    print("  3. Copy the verification code")
    print("  4. Paste it below\n")
    
    text_code = input("Enter verification code: ").strip()
    
    print("\nStep 3: Exchanging code for access token...")
    session = etrade.get_auth_session(
        request_token,
        request_token_secret,
        params={"oauth_verifier": text_code},
        timeout=10
    )
    
    print("[OK] Successfully authenticated!")
    print("="*80 + "\n")
    return session, base_url


def make_api_call(session, url, params=None):
    """Make API call and return response data"""
    try:
        kwargs = {'header_auth': True}
        if params:
            kwargs['params'] = params
        
        response = session.get(url, **kwargs)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 204:
            return None
        else:
            return None
    except Exception as e:
        return None


def extract_position_data(position):
    """Extract essential position information for analysis"""
    # Validate date_acquired (flag if before 1990)
    date_acquired = position.get("dateAcquired", 0)
    if date_acquired < 631152000:  # Unix timestamp for Jan 1, 1990
        date_acquired = None  # Mark as invalid
    
    # Convert portfolio percentage from decimal to percentage (API returns 0.0018 = 0.18%)
    pct_of_portfolio = position.get("pctOfPortfolio", 0) * 100
    
    return {
        "symbol": position.get("symbolDescription", "Unknown"),
        "security_type": position.get("Product", {}).get("securityType", "Unknown"),
        "quantity": position.get("quantity", 0),
        "position_type": position.get("positionType", "Unknown"),  # LONG or SHORT
        "date_acquired": date_acquired,
        "price_paid": position.get("pricePaid", 0),
        "cost_per_share": position.get("costPerShare", 0),
        "total_cost": position.get("totalCost", 0),
        "current_price": position.get("Quick", {}).get("lastTrade", 0),
        "market_value": position.get("marketValue", 0),
        "total_gain": position.get("totalGain", 0),
        "total_gain_pct": position.get("totalGainPct", 0),
        "todays_gain": position.get("daysGain", 0),
        "todays_gain_pct": position.get("daysGainPct", 0),
        "percent_of_portfolio": pct_of_portfolio,
        "data_quality_flags": get_position_quality_flags(position)
    }


def get_position_quality_flags(position):
    """Identify data quality issues with a position"""
    flags = []
    
    total_cost = position.get("totalCost", 0)
    total_gain_pct = position.get("totalGainPct", 0)
    date_acquired = position.get("dateAcquired", 0)
    
    # Flag zero cost basis
    if total_cost == 0:
        flags.append("ZERO_COST_BASIS")
    
    # Flag invalid dates (before 1990)
    if date_acquired < 631152000 and date_acquired != 0:
        flags.append("INVALID_DATE")
    
    # Flag suspicious percentage gains with zero cost
    if total_cost == 0 and total_gain_pct != 0:
        flags.append("INVALID_CALCULATION")
    
    return flags


def extract_order_data(order):
    """Extract essential order information for trading pattern analysis"""
    orders_list = []
    
    if "OrderDetail" in order:
        for detail in order["OrderDetail"]:
            if "Instrument" in detail:
                for instrument in detail["Instrument"]:
                    order_info = {
                        "order_id": order.get("orderId", "Unknown"),
                        "symbol": instrument.get("Product", {}).get("symbol", "Unknown"),
                        "security_type": instrument.get("Product", {}).get("securityType", "Unknown"),
                        "action": instrument.get("orderAction", "Unknown"),  # BUY, SELL, BUY_TO_COVER, SELL_SHORT
                        "quantity_ordered": instrument.get("orderedQuantity", 0),
                        "quantity_filled": instrument.get("filledQuantity", 0),
                        "price_type": detail.get("priceType", "Unknown"),  # MARKET, LIMIT, etc.
                        "limit_price": detail.get("limitPrice", 0),
                        "order_term": detail.get("orderTerm", "Unknown"),  # GOOD_FOR_DAY, etc.
                        "status": detail.get("status", "Unknown"),
                        "avg_execution_price": instrument.get("averageExecutionPrice", 0)
                    }
                    orders_list.append(order_info)
    
    return orders_list


def deduplicate_orders(orders_list):
    """Remove duplicate orders based on order_id, symbol, action, and quantity"""
    seen = set()
    deduplicated = []
    
    for order in orders_list:
        # Create a unique key for each order
        order_key = (
            order.get("order_id"),
            order.get("symbol"),
            order.get("action"),
            order.get("quantity_ordered")
        )
        
        if order_key not in seen:
            seen.add(order_key)
            deduplicated.append(order)
    
    return deduplicated


def main(existing_session=None, existing_base_url=None):
    """Main function to fetch investment analysis data"""
    
    print("="*80)
    print(" ETRADE INVESTMENT ANALYSIS DATA EXTRACTION")
    print("="*80)
    
    # Authenticate
    if existing_session and existing_base_url:
        session = existing_session
        base_url = existing_base_url
        print("Using existing authenticated session...")
    else:
        session, base_url = oauth_authenticate()
    
    # Master data structure for analysis
    analysis_data = {
        "extraction_timestamp": datetime.now().isoformat(),
        "summary": {
            "total_accounts": 0,
            "total_portfolio_value": 0,
            "total_cash": 0,
            "total_buying_power": 0,
            "total_positions": 0,
            "unique_symbols": set()
        },
        "accounts": []
    }
    
    print("\n" + "="*80)
    print("FETCHING ACCOUNT DATA")
    print("="*80)
    
    # Get account list
    url = base_url + "/v1/accounts/list.json"
    account_list_data = make_api_call(session, url)
    
    accounts = []
    if account_list_data and "AccountListResponse" in account_list_data:
        if "Accounts" in account_list_data["AccountListResponse"]:
            if "Account" in account_list_data["AccountListResponse"]["Accounts"]:
                accounts = account_list_data["AccountListResponse"]["Accounts"]["Account"]
                accounts = [acc for acc in accounts if acc.get('accountStatus') != 'CLOSED']
    
    analysis_data["summary"]["total_accounts"] = len(accounts)
    
    # Process each account
    for idx, account in enumerate(accounts, 1):
        account_id = account.get("accountId", f"account_{idx}")
        print(f"\n--- Processing Account: {account_id} ---")
        
        account_analysis = {
            "account_id": account_id,
            "account_type": account.get("institutionType", "Unknown"),
            "account_mode": account.get("accountMode", "Unknown"),
            "description": account.get("accountDesc", ""),
            "cash_allocation": {},
            "positions": [],
            "trading_activity": {
                "open_orders": [],
                "recent_executed": [],
                "rejected_orders": []
            },
            "performance_metrics": {
                "total_value": 0,
                "total_gain_loss": 0,
                "total_gain_loss_pct": 0,
                "winning_positions": 0,
                "losing_positions": 0
            }
        }
        
        # Get balance for cash allocation
        print(f"  Fetching balance...")
        balance_url = base_url + "/v1/accounts/" + account["accountIdKey"] + "/balance.json"
        balance_params = {"instType": account["institutionType"], "realTimeNAV": "true"}
        balance_data = make_api_call(session, balance_url, balance_params)
        
        if balance_data and "BalanceResponse" in balance_data:
            computed = balance_data["BalanceResponse"].get("Computed", {})
            
            account_analysis["cash_allocation"] = {
                "total_account_value": computed.get("RealTimeValues", {}).get("totalAccountValue", 0),
                "cash_available_for_investment": computed.get("cashAvailableForInvestment", 0),
                "cash_available_for_withdrawal": computed.get("cashAvailableForWithdrawal", 0),
                "net_cash": computed.get("netCash", 0),
                "settled_cash": computed.get("settledCashForInvestment", 0),
                "unsettled_cash": computed.get("unSettledCashForInvestment", 0),
                "margin_buying_power": computed.get("marginBuyingPower", 0),
                "cash_buying_power": computed.get("cashBuyingPower", 0)
            }
            
            total_value = computed.get("RealTimeValues", {}).get("totalAccountValue", 0)
            account_analysis["performance_metrics"]["total_value"] = total_value
            analysis_data["summary"]["total_portfolio_value"] += total_value
            analysis_data["summary"]["total_cash"] += computed.get("netCash", 0)
            analysis_data["summary"]["total_buying_power"] += computed.get("cashBuyingPower", 0)
        
        # Get portfolio positions (only for brokerage accounts)
        if account.get("institutionType") == "BROKERAGE":
            print(f"  Fetching portfolio...")
            portfolio_url = base_url + "/v1/accounts/" + account["accountIdKey"] + "/portfolio.json"
            portfolio_data = make_api_call(session, portfolio_url)
            
            if portfolio_data and "PortfolioResponse" in portfolio_data:
                for acct_portfolio in portfolio_data["PortfolioResponse"].get("AccountPortfolio", []):
                    for position in acct_portfolio.get("Position", []):
                        pos_data = extract_position_data(position)
                        account_analysis["positions"].append(pos_data)
                        
                        # Update performance metrics
                        analysis_data["summary"]["total_positions"] += 1
                        analysis_data["summary"]["unique_symbols"].add(pos_data["symbol"])
                        
                        total_gain = pos_data.get("total_gain", 0)
                        account_analysis["performance_metrics"]["total_gain_loss"] += total_gain
                        
                        if total_gain > 0:
                            account_analysis["performance_metrics"]["winning_positions"] += 1
                        elif total_gain < 0:
                            account_analysis["performance_metrics"]["losing_positions"] += 1
            
            # Calculate overall gain/loss percentage
            if account_analysis["performance_metrics"]["total_value"] > 0:
                total_cost = sum(pos.get("total_cost", 0) for pos in account_analysis["positions"])
                if total_cost > 0:
                    account_analysis["performance_metrics"]["total_gain_loss_pct"] = (
                        account_analysis["performance_metrics"]["total_gain_loss"] / total_cost * 100
                    )
            
            # Get trading activity
            print(f"  Fetching orders...")
            orders_url = base_url + "/v1/accounts/" + account["accountIdKey"] + "/orders.json"
            
            # Open orders - shows current strategy
            open_data = make_api_call(session, orders_url, {"status": "OPEN"})
            if open_data and "OrdersResponse" in open_data:
                for order in open_data["OrdersResponse"].get("Order", []):
                    account_analysis["trading_activity"]["open_orders"].extend(extract_order_data(order))
            
            # Deduplicate open orders
            account_analysis["trading_activity"]["open_orders"] = deduplicate_orders(
                account_analysis["trading_activity"]["open_orders"]
            )
            
            # Recent executed orders - shows trading patterns
            executed_data = make_api_call(session, orders_url, {"status": "EXECUTED"})
            if executed_data and "OrdersResponse" in executed_data:
                for order in executed_data["OrdersResponse"].get("Order", []):
                    account_analysis["trading_activity"]["recent_executed"].extend(extract_order_data(order))
            
            # Deduplicate executed orders
            account_analysis["trading_activity"]["recent_executed"] = deduplicate_orders(
                account_analysis["trading_activity"]["recent_executed"]
            )
            
            # Rejected orders - shows potential issues
            rejected_data = make_api_call(session, orders_url, {"status": "REJECTED"})
            if rejected_data and "OrdersResponse" in rejected_data:
                for order in rejected_data["OrdersResponse"].get("Order", []):
                    account_analysis["trading_activity"]["rejected_orders"].extend(extract_order_data(order))
        
        analysis_data["accounts"].append(account_analysis)
    
    # Convert set to list for JSON serialization
    analysis_data["summary"]["unique_symbols"] = list(analysis_data["summary"]["unique_symbols"])
    
    print("\n" + "="*80)
    print("SAVING ANALYSIS DATA")
    print("="*80)
    
    # Create output directory
    output_dir = "etrade_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    # Keep only the last 3 per pattern to preserve history without bloat
    print("\nPruning old reports (keep last 3 per type)...")
    for pattern in ["etrade_data_*.json", "etrade_data_*.txt", "etrade_summary_*.txt"]:
        files = sorted(glob.glob(os.path.join(output_dir, pattern)), key=os.path.getmtime, reverse=True)
        for old_file in files[3:]:
            try:
                os.remove(old_file)
                print(f"  [OK] Deleted: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"  [!] Could not delete {os.path.basename(old_file)}: {e}")
    
    # Save JSON data
    json_filename = os.path.join(output_dir, f"etrade_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(json_filename, 'w') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    print(f"\n[OK] Analysis data saved to: {json_filename}")
    
    # Save readable summary
    txt_filename = os.path.join(output_dir, f"etrade_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(txt_filename, 'w') as f:
        f.write("="*80 + "\n")
        f.write(" ETRADE INVESTMENT ANALYSIS SUMMARY\n")
        f.write(f" Generated: {analysis_data['extraction_timestamp']}\n")
        f.write("="*80 + "\n\n")
        
        # Overall summary
        f.write("PORTFOLIO OVERVIEW\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Accounts: {analysis_data['summary']['total_accounts']}\n")
        f.write(f"Total Portfolio Value: ${analysis_data['summary']['total_portfolio_value']:,.2f}\n")
        f.write(f"Total Cash Position: ${analysis_data['summary']['total_cash']:,.2f}\n")
        f.write(f"Total Buying Power: ${analysis_data['summary']['total_buying_power']:,.2f}\n")
        f.write(f"Total Positions: {analysis_data['summary']['total_positions']}\n")
        f.write(f"Unique Symbols: {len(analysis_data['summary']['unique_symbols'])}\n")
        f.write(f"Symbols: {', '.join(sorted(analysis_data['summary']['unique_symbols']))}\n\n")
        
        # Account details
        for account in analysis_data["accounts"]:
            f.write("="*80 + "\n")
            f.write(f"ACCOUNT: {account['account_id']}\n")
            f.write("="*80 + "\n")
            f.write(f"Type: {account['account_type']} ({account['account_mode']})\n")
            f.write(f"Description: {account['description']}\n\n")
            
            # Cash allocation
            f.write("CASH ALLOCATION:\n")
            f.write("-"*80 + "\n")
            cash = account["cash_allocation"]
            f.write(f"Total Account Value: ${cash.get('total_account_value', 0):,.2f}\n")
            f.write(f"Available Cash: ${cash.get('cash_available_for_investment', 0):,.2f}\n")
            f.write(f"Settled Cash: ${cash.get('settled_cash', 0):,.2f}\n")
            f.write(f"Unsettled Cash: ${cash.get('unsettled_cash', 0):,.2f}\n")
            f.write(f"Buying Power: ${cash.get('cash_buying_power', 0):,.2f}\n\n")
            
            # Performance
            f.write("PERFORMANCE:\n")
            f.write("-"*80 + "\n")
            perf = account["performance_metrics"]
            f.write(f"Total Gain/Loss: ${perf['total_gain_loss']:,.2f} ({perf['total_gain_loss_pct']:.2f}%)\n")
            f.write(f"Winning Positions: {perf['winning_positions']}\n")
            f.write(f"Losing Positions: {perf['losing_positions']}\n\n")
            
            # Positions
            f.write("POSITIONS:\n")
            f.write("-"*80 + "\n")
            for pos in account["positions"]:
                # Check for data quality flags
                flags = pos.get("data_quality_flags", [])
                flags_str = f" [FLAGS: {', '.join(flags)}]" if flags else ""
                
                f.write(f"\n{pos['symbol']} ({pos['security_type']}) - {pos['position_type']}{flags_str}\n")
                f.write(f"  Quantity: {pos['quantity']:,} shares\n")
                f.write(f"  Cost Basis: ${pos['total_cost']:,.2f} (${pos['cost_per_share']:.2f}/share)\n")
                f.write(f"  Current Value: ${pos['market_value']:,.2f} (@${pos['current_price']:.2f})\n")
                
                # Calculate percentage safely (handle zero cost basis)
                if pos['total_cost'] > 0:
                    pct = (pos['total_gain'] / pos['total_cost']) * 100
                    f.write(f"  Gain/Loss: ${pos['total_gain']:,.2f} ({pct:.2f}%)\n")
                else:
                    f.write(f"  Gain/Loss: ${pos['total_gain']:,.2f} (N/A - zero cost basis)\n")
                
                f.write(f"  Today's Change: ${pos['todays_gain']:,.2f} ({pos['todays_gain_pct']:.2f}%)\n")
                f.write(f"  % of Portfolio: {pos['percent_of_portfolio']:.2f}%\n")
            
            # Trading activity
            f.write("\n\nTRADING ACTIVITY:\n")
            f.write("-"*80 + "\n")
            
            trading = account["trading_activity"]
            if trading["open_orders"]:
                f.write(f"\nOpen Orders ({len(trading['open_orders'])}):\n")
                for order in trading["open_orders"]:
                    f.write(f"  #{order['order_id']}: {order['action']} {order['quantity_ordered']} {order['symbol']}\n")
                    f.write(f"    Type: {order['price_type']}, Limit: ${order['limit_price']:.2f}\n")
            
            if trading["recent_executed"]:
                f.write(f"\nRecent Executed Orders ({len(trading['recent_executed'])}):\n")
                for order in trading["recent_executed"]:
                    f.write(f"  #{order['order_id']}: {order['action']} {order['quantity_filled']} {order['symbol']}\n")
                    f.write(f"    Avg Price: ${order['avg_execution_price']:.2f}\n")
            
            if trading["rejected_orders"]:
                f.write(f"\nRejected Orders ({len(trading['rejected_orders'])}):\n")
                for order in trading["rejected_orders"]:
                    f.write(f"  #{order['order_id']}: {order['action']} {order['quantity_ordered']} {order['symbol']}\n")
            
            f.write("\n")
    
    print(f"[OK] Summary report saved to: {txt_filename}")
    
    print("\n" + "="*80)
    print("DATA EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. {json_filename} - Structured analysis data")
    print(f"  2. {txt_filename} - Human-readable summary")
    print("\nData is optimized for AI investment analysis!")


if __name__ == "__main__":
    main()
