import os
import sys
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import requests
from google import genai
# ==========================================
# CONFIGURATION: ADD YOUR STOCKS HERE (MAX 50)
# ==========================================
# For Indian Stocks (NSE), append '.NS' (e.g., "RELIANCE.NS")
WATCHLIST = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "AAPL", "MSFT", "NVDA", "TSLA", "AMZN"
]

if len(WATCHLIST) > 50:
    print("Error: Maximum watchlist constraint of 50 stocks exceeded.")
    sys.exit(1)

def calculate_technicals(df):
    if len(df) < 30:
        return {}
    
    # 1. RSI Calculation (14 Days)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD Calculation (12, 26, 9)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Breakout indicators (20-day High/Low Channel)
    df['Upper_20'] = df['High'].shift(1).rolling(window=20).max()
    df['Lower_20'] = df['Low'].shift(1).rolling(window=20).min()
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Check MACD Crossover
    macd_cross = "None"
    if prev['MACD'] <= prev['Signal'] and latest['MACD'] > latest['Signal']:
        macd_cross = "BULLISH_CROSSOVER"
    elif prev['MACD'] >= prev['Signal'] and latest['MACD'] < latest['Signal']:
        macd_cross = "BEARISH_CROSSOVER"
        
    return {
        "Close": round(latest['Close'], 2),
        "RSI_14": round(latest['RSI'], 2),
        "MACD_Status": macd_cross,
        "Daily_Volume": int(latest['Volume']),
        "Avg_Volume_20D": int(df['Volume'].tail(20).mean()),
        "Near_20D_High_Breakout": bool(latest['Close'] >= latest['Upper_20'] * 0.98),
        "Volume_Spike": bool(latest['Volume'] > (df['Volume'].tail(20).mean() * 1.5))
    }

def fetch_stock_context(ticker_symbol):
    print(f"Processing data for: {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="60d")
    technicals = calculate_technicals(hist)
    
    info = ticker.info
    fundamentals = {
        "P_E_Ratio": info.get("trailingPE", "N/A"),
        "Forward_PE": info.get("forwardPE", "N/A"),
        "Market_Cap_Billions": round(info.get("marketCap", 0) / 1e9, 2) if info.get("marketCap") else "N/A",
        "Debt_To_Equity": info.get("debtToEquity", "N/A"),
    }
    
    news_items = ticker.news[:3]
    processed_news = [{"title": item.get("title"), "publisher": item.get("publisher")} for item in news_items]
        
    return {
        "ticker": ticker_symbol,
        "technicals": technicals,
        "fundamentals": fundamentals,
        "news": processed_news
    }

def send_push_notification(title, body):
    token = os.environ.get("PUSHBULLET_ACCESS_TOKEN")
    if not token:
        print("Pushbullet token missing. Skipping notification.")
        return
    
    url = "https://pushbullet.com"
    headers = {"Access-Token": token, "Content-Type": "application/json"}
    payload = {"type": "note", "title": title, "body": body}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("Notification dispatched successfully to system apps.")
        else:
            print(f"Pushbullet Error: {response.text}")
    except Exception as e:
        print(f"Failed to transmit notification: {e}")

def main():

    
    master_report = []
    breakout_alerts = []
    client=genai.Client()
    
    for ticker in WATCHLIST:
        try:
            raw_ctx = fetch_stock_context(ticker)
            if not raw_ctx['technicals']:
                continue
                
            prompt = f"""
            Analyze the following market dataset for {raw_ctx['ticker']}:
            Technical metrics: {json.dumps(raw_ctx['technicals'])}
            Fundamental metrics: {json.dumps(raw_ctx['fundamentals'])}
            Recent news: {json.dumps(raw_ctx['news'])}
            
            Generate a concise JSON summary:
            {{
               "ticker": "{raw_ctx['ticker']}",
               "technical_summary": "1 sentence on trend, RSI, MACD",
               "breakout_prediction": "IMMINENT / ACCELERATING / CONSOLIDATING / WEAK",
               "action_flag": "URGENT_BUY / MONITOR / AVOID"
            }}
            """
                    response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config={
                'response_mime_type': 'application/json'
            }
        )
        res = json.loads(response.text)

            master_report.append(res)
            
            if res.get("breakout_prediction") in ["IMMINENT", "ACCELERATING"] or res.get("action_flag") == "URGENT_BUY":
                breakout_alerts.append(f"⚠️ {res['ticker']}: {res['breakout_prediction']} breakout! Action: {res['action_flag']}")
                
        except Exception as e:
            print(f"Error {ticker}: {e}")
            
    with open("daily_stock_insights.json", "w") as f:
        json.dump(master_report, f, indent=4)
        
    if breakout_alerts:
        notification_body = "\n".join(breakout_alerts)
    else:
        notification_body = "All stocks checked. No imminent breakouts or critical alerts flagged today."
        
    send_push_notification(f"📊 Stock Agent Report ({datetime.now().strftime('%d-%b')})", notification_body)

if __name__ == "__main__":
    main()
