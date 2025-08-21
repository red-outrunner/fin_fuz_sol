import yfinance as yf
import json

tickers = ["GLN.JO", "PRX.JO", "AGL.JO", "SOL.JO", "NED.JO", "MTN.JO", "PPE.JO", "CTA.JO",
           "EXX.JO", "PMR.JO", "RNI.JO", "INP.JO", "ABG.JO", "RDF.JO", "APN.JO", "BVT.JO", "REM.JO"]
prices = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1] if not stock.history(period="1d").empty else 0
    prices[ticker.replace(".JO", "")] = price
with open("prices.json", "w") as f:
    json.dump(prices, f, indent=4)
