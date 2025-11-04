# -*- coding: utf-8 -*-

"""
Bitkub Hybrid AI Grid Trading Bot with Telegram Interface (Gemini Upgraded Version)

This script implements a sophisticated, hybrid trading bot.
It fuses Technical Analysis (TA) with Generative AI (Gemini) for market sentiment analysis.

Upgraded Features:
- Layer 1 AI (Tactical): TA-based score (MACD, RSI, BBands, etc.) runs every 1 minute.
- Layer 2 AI (Strategic): Gemini Flash API + Google Search runs every 15 minutes
  to get a real-time sentiment_score (-1.0 to 1.0).
- Veto System: A negative sentiment score can block "buy" signals from the TA layer.
- FIXED (Bug 1): Unrealized P/L calculation to use bot's internal state.
- FIXED (Bug 2): Test Mode P/L logic to correctly track coin amount.
- FIXED (Bug 3): Robust selling logic in _enter_safe_mode.
- FIXED (Bug 4): Deletion logic (confirm_delete) to liquidate assets based on
                 *actual exchange balance*, ignoring corrupted internal state.
- NEW (Feature Request): Re-enabled Compounding Capital.
                 Bot will now use 'original_capital' + 'all_time_realized_pnl'
                 when starting a new mode (GRID or TRAILING_UP) from SAFE.
- NEW (Feature Request): Hourly report now shows P/L (Realized) instead of P/L (Total).
- NEW (Feature Request): Bot now runs full AI check (TA + Sentiment) *on start*
                 to determine the correct initial mode, preventing rapid
                 mode switching just after starting.
"""

import os
import logging
import hmac
import hashlib
import time
import json
import httpx # Changed from requests
import asyncio
import sys
import signal
import html
import re
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

# Third-party libraries
import pandas as pd
import ta

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton
from telegram import Update as TgUpdate
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)
from telegram.constants import ParseMode
from telegram.error import BadRequest, TimedOut

# --- Configuration Loading ---
try:
    from config import TELEGRAM_TOKEN, BITKUB_API_KEY, BITKUB_API_SECRET, GEMINI_API_KEY
except ImportError:
    print("Error: config.py not found or variables are missing.")
    print("Please create a config.py file with your TELEGRAM_TOKEN, BITKUB_API_KEY, and BITKUB_API_SECRET.")
    sys.exit(1)

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if script is reloaded
if not logger.handlers:
    file_handler = logging.FileHandler('bot.log', mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)


# --- Safe Callback Answer Helper ---
async def answer_cb(query):
    """Safely answers a callback query, ignoring 'Query is too old' errors."""
    if not query: return
    try:
        await query.answer()
    except BadRequest as e:
        if "Query is too old" in str(e) or "query id is invalid" in str(e):
            logger.info(f"Ignoring late/invalid callback: {e}")
        else:
            logger.warning(f"Callback answer BadRequest: {e}")
    except Exception as e:
        logger.warning(f"Callback answer error: {e}")

# --- Global State Management ---
BOT_STATE_FILE = 'bot_state.json'
user_bots = {}

# --- Persistence Functions (JSON Implementation) ---
def save_bots_state():
    """Serializes all running bot instances into a JSON file."""
    state_to_save = {}
    for bot_id, bot_instance in user_bots.items():
        bot_state = bot_instance.__getstate__()
        state_to_save[bot_id] = bot_state
    try:
        with open(BOT_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state_to_save, f, ensure_ascii=False, indent=4)
        logger.info(f"Successfully saved state for {len(user_bots)} bots to {BOT_STATE_FILE}")
    except Exception as e:
        logger.error(f"Failed to save bot state to JSON: {e}")

def load_bots_state(application: Application):
    """Loads bot states from a JSON file and recreates the bot instances."""
    global user_bots
    if not os.path.exists(BOT_STATE_FILE):
        logger.info("No state file found. Starting fresh.")
        return

    try:
        with open(BOT_STATE_FILE, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)

        for bot_id, bot_data in loaded_state.items():
            # Basic check for new logic compatibility
            if 'profit_percent' not in bot_data.get('settings', {}):
                logger.warning(f"Skipping incompatible old bot state for {bot_id}. Please recreate it.")
                continue
            
            client_cfg = bot_data.get('client_settings', {})
            if 'gemini_api_key' not in client_cfg:
                 client_cfg['gemini_api_key'] = GEMINI_API_KEY
                 client_cfg['api_key'] = BITKUB_API_KEY 
                 client_cfg['api_secret'] = BITKUB_API_SECRET
            
            bot_data['client_settings'] = client_cfg

            bot_instance = HybridGridBot(bot_data['chat_id'], client_cfg, bot_data['settings'])
            bot_instance.__setstate__(bot_data)
            bot_instance.set_context(application)
            user_bots[bot_id] = bot_instance
            if bot_instance.is_running:
                logger.info(f"Resuming bot: {bot_id}")
        logger.info(f"Successfully loaded state for {len(user_bots)} bots from {BOT_STATE_FILE}.")
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to load or parse bot state from JSON: {e}. Starting fresh.")
        user_bots = {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading state: {e}. Starting fresh.")
        user_bots = {}


# --- Bitkub API Client Class (Now using httpx for async requests) ---
class BitkubClient:
    BASE_URL = "https://api.bitkub.com"
    def __init__(self, api_key, api_secret):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def _get_server_time(self):
        try:
            response = await self.http_client.get(f"{self.BASE_URL}/api/v3/servertime")
            response.raise_for_status()
            return int(response.text)
        except httpx.RequestError as e:
            logger.error(f"Could not get server time from Bitkub API: {e}. Falling back to local time.")
            return int(time.time() * 1000)

    def _sign(self, payload_string: str) -> str:
        secret = self.api_secret.encode('utf-8')
        return hmac.new(secret, payload_string.encode('utf-8'), hashlib.sha256).hexdigest()

    async def _make_request(self, method, path, params=None, data=None):
        MAX_RETRIES = 3
        RETRY_DELAY = 1

        for attempt in range(MAX_RETRIES):
            try:
                ts = await self._get_server_time()
                headers = {"Accept": "application/json", "Content-Type": "application/json", "X-BTK-APIKEY": self.api_key}
                
                query_string = f"?{urlencode(sorted(params.items()))}" if params else ""
                body_string = json.dumps(data if data is not None else {}) if method.upper() == 'POST' else ""
                
                payload_to_sign = f"{ts}{method.upper()}{path}{query_string}{body_string}"
                sig = self._sign(payload_to_sign)

                headers["X-BTK-TIMESTAMP"] = str(ts)
                headers["X-BTK-SIGN"] = sig
                
                url = f"{self.BASE_URL}{path}{query_string}"
                
                response = await self.http_client.request(method.upper(), url, headers=headers, content=body_string)
                
                response.raise_for_status()
                json_response = response.json()

                if json_response.get('error') == 8 and attempt < MAX_RETRIES - 1:
                    logger.warning(f"Timestamp error (Error 8) on attempt {attempt + 1}/{MAX_RETRIES}. Retrying...")
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                
                return json_response

            except httpx.RequestError as e:
                logger.error(f"API Request Failed on attempt {attempt + 1}/{MAX_RETRIES}. Error: {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    error_msg = str(e)
                    if isinstance(e, httpx.HTTPStatusError):
                        try: error_msg = e.response.json()
                        except (json.JSONDecodeError): pass
                    return {"error": 100, "message": f"API request failed after {MAX_RETRIES} attempts: {error_msg}"}
        
        return {"error": 101, "message": f"Request failed after {MAX_RETRIES} retries."}

    async def get_balances(self): return await self._make_request("POST", "/api/v3/market/balances")
    async def place_bid(self, sym, amt, rat, typ='limit'): return await self._make_request("POST", "/api/v3/market/place-bid", data={"sym": sym, "amt": amt, "rat": rat, "typ": typ})
    async def place_ask(self, sym, amt, rat, typ='limit'): return await self._make_request("POST", "/api/v3/market/place-ask", data={"sym": sym, "amt": amt, "rat": rat, "typ": typ})
    async def my_open_orders(self, sym): return await self._make_request("GET", "/api/v3/market/my-open-orders", params={"sym": sym})
    async def my_order_history(self, sym, p=1, l=10): return await self._make_request("GET", "/api/v3/market/my-order-history", params={"sym": sym, "p": p, "l": l})
    async def cancel_order(self, sym, order_id, side): return await self._make_request("POST", "/api/v3/market/cancel-order", data={"sym": sym, "id": order_id, "sd": side})

async def get_market_data(symbol: str, timeframe: str = '60', limit: int = 500):
    resolution_map = {'15': '15', '60': '60', '240': '240', '1D': 'D'}
    resolution = resolution_map.get(timeframe, '60')
    to_timestamp = int(time.time())
    minutes_per_candle = {'15': 15, '60': 60, '240': 240, '1D': 1440}.get(timeframe, 60)
    from_timestamp = to_timestamp - (limit * minutes_per_candle * 60)
    url = f"https://api.bitkub.com/tradingview/history?symbol={symbol}_THB&resolution={resolution}&from={from_timestamp}&to={to_timestamp}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data['s'] == 'ok' and len(data['t']) > 0:
                df = pd.DataFrame({'timestamp': data['t'], 'open': data['o'], 'high': data['h'], 'low': data['l'], 'close': data['c'], 'volume': data['v']})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
                return df
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
    return pd.DataFrame()

async def get_latest_price(symbol: str):
    ticker_symbol = f"THB_{symbol.upper()}"
    url = f"https://api.bitkub.com/api/market/ticker?sym={ticker_symbol}"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return float(data.get(ticker_symbol, {}).get('last', 0))
        except Exception as e:
            logger.error(f"Could not fetch latest price for {symbol}: {e}")
            return None

def format_display_price(price: float) -> str:
    if price is None:
        return "N/A"
    return f"{price:,.8f}".rstrip('0').rstrip('.')

# --- Trading Bot Logic Class ---
class HybridGridBot:
    def __init__(self, chat_id, api_client_settings, bot_settings):
        self.context = None
        self.chat_id = chat_id
        self.client_settings = api_client_settings
        self.settings = bot_settings
        self.is_running = False
        self.current_mode = "SAFE"
        self.ai_score = 0
        self.adx = 0
        self.grid_levels = []
        self.grid_capital_per_level = 0
        self.open_buy_orders = {} 
        self.open_sell_orders = {}
        self.last_checked_price = 0
        
        self.initial_capital = self.settings.get('capital', 0) 
        self.original_capital = self.settings.get('capital', 0)
        
        self.realized_pnl = 0.0 # PNL for *this cycle*
        self.total_investment = 0.0 
        self.total_coins_held = 0.0
        self.all_time_realized_pnl = 0.0 # PNL *total* (Used for compounding/reporting)

        self.mode_change_candidate = None
        self.mode_change_counter = 0
        self.MODE_CHANGE_CONFIRMATIONS = 3
        
        self.is_shifting_grid = False
        
        self.sentiment_score = 0.0
        self.avg_sentiment_score = 0.0
        self.sentiment_score_history = []
        self.sentiment_justification = "N/A"
        self.SMOOTHING_PERIOD = 3

        self.gemini_api_key = self.client_settings.get('gemini_api_key', "") 
        self.gemini_http_client = httpx.AsyncClient(timeout=30.0)

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ['context', 'gemini_http_client']:
            if key in state:
                del state[key]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'open_sell_orders' in self.__dict__ and self.__dict__['open_sell_orders']:
            self.open_sell_orders = {float(k): v for k, v in self.open_sell_orders.items()}
        if 'open_buy_orders' in self.__dict__ and self.__dict__['open_buy_orders']:
            self.open_buy_orders = {float(k): v for k, v in self.open_buy_orders.items()}
            
        if 'original_capital' not in self.__dict__ or self.__dict__['original_capital'] == 0:
            self.original_capital = self.initial_capital
            
        self.realized_pnl = self.__dict__.get('realized_pnl', 0.0)
        self.total_investment = self.__dict__.get('total_investment', 0.0)
        self.total_coins_held = self.__dict__.get('total_coins_held', 0.0)
        self.all_time_realized_pnl = self.__dict__.get('all_time_realized_pnl', 0.0)
        
        self.sentiment_score = state.get('sentiment_score', 0.0)
        self.sentiment_score_history = state.get('sentiment_score_history', [])
        self.SMOOTHING_PERIOD = state.get('SMOOTHING_PERIOD', 3)
        
        if self.sentiment_score_history:
            self.avg_sentiment_score = sum(self.sentiment_score_history) / len(self.sentiment_score_history)
        else:
            self.avg_sentiment_score = self.sentiment_score
            
        self.sentiment_justification = state.get('sentiment_justification', "N/A")
        
        self.context = None
        self.gemini_http_client = httpx.AsyncClient(timeout=30.0)

    def get_client(self): return BitkubClient(self.client_settings['api_key'], self.client_settings['api_secret'])
    def set_context(self, context): self.context = context
    
    async def start(self):
        self.is_running = True
        logger.info(f"[{self.settings['symbol']}] Bot instance STARTED.")
        await self.send_telegram_message("✅ Bot started successfully!\n⏳ Initializing... Fetching market data and AI sentiment...")

        symbol = self.settings['symbol']
        
        try:
            # 1. Fetch Sentiment
            await self.update_sentiment_analysis()
            
            # 2. Fetch Market Data for TA
            df = await get_market_data(symbol, self.settings['timeframe'])
            if df.empty:
                await self.send_telegram_message(f"⚠️ Could not fetch market data for {symbol} on start. Bot stopping.")
                self.is_running = False
                return

            # 3. Calculate TA
            self.ai_score, self.adx = self._calculate_ai_score(df)
            
            # 4. Determine initial strategy
            # We are currently in "SAFE" mode (default), so _determine_strategy will pick the correct entry mode
            initial_mode = await self._determine_strategy() 
            
            await self.send_telegram_message(
                f"✅ Initialization complete.\n"
                f"AI (TA) score: <code>{self.ai_score:+.2f}</code>\n"
                f"AI (Sentiment) avg: <code>{self.avg_sentiment_score:+.2f}</code>\n"
                f"Entering <code>{initial_mode}</code> mode...",
                use_html=True
            )
            
            # 5. Handle the transition from SAFE to the chosen mode
            # We pass 'df' so it doesn't need to be fetched again
            if initial_mode != "SAFE":
                await self._handle_mode_transition(initial_mode, df)
            else:
                self.current_mode = "SAFE"
                await self.send_telegram_message("ℹ️ Market conditions require staying in 🛡️ SAFE mode.")

        except Exception as e:
            logger.error(f"[{symbol}] Error during initial start sequence: {e}", exc_info=True)
            await self.send_telegram_message(f"‼️ Error during initialization: {e}. Bot stopping.")
            self.is_running = False
            
        # The regular job_queue will now take over for subsequent cycles
    
    def stop(self): self.is_running = False; logger.info(f"[{self.settings['symbol']}] Bot instance STOPPED.")

    async def run_logic_cycle(self):
        if not self.is_running or self.is_shifting_grid: return
        symbol = self.settings['symbol']
        logger.info(f"[{symbol}] Running logic cycle...")
        
        df = await get_market_data(symbol, self.settings['timeframe'])
        if df.empty:
            await self.send_telegram_message(f"⚠️ Could not fetch market data for {symbol}. Skipping cycle.")
            return

        self.ai_score, self.adx = self._calculate_ai_score(df)
        new_mode = await self._determine_strategy()
        
        if new_mode == self.current_mode:
            if self.mode_change_counter > 0:
                await self.send_telegram_message(f"ℹ️ Mode change to {self.mode_change_candidate} cancelled. Market condition reverted.")
                self.mode_change_candidate = None
                self.mode_change_counter = 0
            
            if self.current_mode == "GRID":
                await self._manage_grid(df)
        
        else: # Potential mode change
            if new_mode == self.mode_change_candidate:
                self.mode_change_counter += 1
                await self.send_telegram_message(f"⏳ Confirmation check ({self.mode_change_counter}/{self.MODE_CHANGE_CONFIRMATIONS}) to switch to <code>{new_mode}</code>...", use_html=True)
            else:
                self.mode_change_candidate = new_mode
                self.mode_change_counter = 1
                await self.send_telegram_message(f"⏳ Potential mode change detected: <code>{self.current_mode}</code> ➡️ <code>{new_mode}</code>. Waiting for confirmation (1/{self.MODE_CHANGE_CONFIRMATIONS})...", use_html=True)

            if self.mode_change_counter >= self.MODE_CHANGE_CONFIRMATIONS:
                await self._handle_mode_transition(new_mode, df)
                self.mode_change_candidate = None
                self.mode_change_counter = 0

    async def update_sentiment_analysis(self):
        symbol = self.settings['symbol']
        if not self.gemini_api_key:
            logger.info(f"[{symbol}] Skipping sentiment analysis, GEMINI_API_KEY is not set.")
            self.sentiment_score = 0.0
            self.avg_sentiment_score = 0.0
            self.sentiment_justification = "Gemini API Key is not configured."
            return
            
        logger.info(f"[{symbol}] Running strategic sentiment analysis...")
        
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.gemini_api_key
        }
        
        summarized_news = ""

        # --- Call 1: Search for news ---
        try:
            search_prompt = f"ค้นหาข่าวสารและบทวิเคราะห์ล่าสุด (24 ชั่วโมงที่ผ่านมา) เกี่ยวกับเหรียญ {symbol} และสรุปประเด็นสำคัญทั้งหมดใน 3-4 บรรทัด"
            payload_search = {
                "contents": [{"parts": [{"text": search_prompt}]}],
                "tools": [{"google_search": {}}]
            }
            
            response_search = await self.gemini_http_client.post(api_url, json=payload_search, headers=headers)
            response_search.raise_for_status()
            result_search = response_search.json()
            summarized_news = result_search.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            
            if not summarized_news:
                raise Exception("No news summary returned from search.")
            
            logger.info(f"[{symbol}] News summary retrieved: {summarized_news[:50]}...")

        except Exception as e:
            logger.error(f"[{symbol}] Failed during sentiment analysis (Step 1 - Search): {e}", exc_info=True)
            await self.send_telegram_message(f"⚠️ [{symbol}] ไม่สามารถดึงข้อมูลข่าวสาร (AI Layer 2) ได้ในรอบนี้")
            return

        # --- Call 2: Analyze sentiment from the news ---
        try:
            system_prompt_analyze = (
                "คุณคือนักวิเคราะห์การเงินอาวุโสที่เชี่ยวชาญตลาดคริปโตเคอร์เรนซี "
                "หน้าที่ของคุณคือวิเคราะห์สภาวะอารมณ์ตลาด (Sentiment) จาก 'บทสรุปข่าว' ที่ได้รับ "
                "ประเมินอารมณ์ตลาดเป็นคะแนน -1.0 (แย่มาก/FUD) ถึง +1.0 (ดีมาก/FOMO) และสรุปเหตุผล "
                "ตอบกลับเป็น JSON object ที่มีโครงสร้างนี้เท่านั้น: "
                "{\"sentiment_score\": <ตัวเลข>, \"justification\": \"<เหตุผลสรุปสั้นๆ>\"}"
            )
            
            analyze_prompt = f"จากข้อมูลสรุปข่าวต่อไปนี้: \"{summarized_news}\" โปรดวิเคราะห์และให้คะแนน sentiment"

            payload_analyze = {
                "contents": [{"parts": [{"text": analyze_prompt}]}],
                "systemInstruction": {"parts": [{"text": system_prompt_analyze}]},
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": {
                        "type": "OBJECT",
                        "properties": {
                            "sentiment_score": {"type": "NUMBER"},
                            "justification": {"type": "STRING"}
                        },
                        "required": ["sentiment_score", "justification"]
                    }
                }
            }

            response_analyze = await self.gemini_http_client.post(api_url, json=payload_analyze, headers=headers)
            response_analyze.raise_for_status()
            result_analyze = response_analyze.json()

            text_content = result_analyze.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
            parsed_json = json.loads(text_content)

            latest_score = float(parsed_json.get('sentiment_score', 0.0))
            self.sentiment_justification = parsed_json.get('justification', 'No justification provided.')
            self.sentiment_score = latest_score

            self.sentiment_score_history.append(latest_score)
            if len(self.sentiment_score_history) > self.SMOOTHING_PERIOD:
                self.sentiment_score_history.pop(0) 
            
            self.avg_sentiment_score = sum(self.sentiment_score_history) / len(self.sentiment_score_history)
            
            logger.info(f"[{symbol}] Sentiment updated. Latest: {self.sentiment_score:+.1f}, Smoothed Avg (n={len(self.sentiment_score_history)}): {self.avg_sentiment_score:+.2f}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Failed during sentiment analysis (Step 2 - Analyze): {e}", exc_info=True)
            self.sentiment_justification = "Failed to analyze sentiment."
            await self.send_telegram_message(f"⚠️ [{symbol}] ไม่สามารถวิเคราะห์ข้อมูล Sentiment (AI Layer 2) ได้ในรอบนี้")


    def _calculate_ai_score(self, df: pd.DataFrame):
        score = 0
        weights = {'ema': 1.5, 'macd': 1.5, 'rsi': 1.0, 'bbands': 1.0, 'obv': 0.5}

        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        if df['ema_20'].iloc[-2] < df['ema_50'].iloc[-2] and df['ema_20'].iloc[-1] > df['ema_50'].iloc[-1]:
            score += weights['ema']
        if df['ema_20'].iloc[-2] > df['ema_50'].iloc[-2] and df['ema_20'].iloc[-1] < df['ema_50'].iloc[-1]:
            score -= weights['ema']

        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macdsignal'] = macd.macd_signal()
        if df['macd'].iloc[-2] < df['macdsignal'].iloc[-2] and df['macd'].iloc[-1] > df['macdsignal'].iloc[-1]:
            score += weights['macd']
        if df['macd'].iloc[-2] > df['macdsignal'].iloc[-2] and df['macd'].iloc[-1] < df['macdsignal'].iloc[-1]:
            score -= weights['macd']

        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        if df['rsi'].iloc[-1] < 30: score += weights['rsi']
        if df['rsi'].iloc[-1] > 70: score -= weights['rsi']
        
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        if df['close'].iloc[-1] < df['bb_low'].iloc[-1]:
            score += weights['bbands']
        if df['close'].iloc[-1] > df['bb_high'].iloc[-1]:
            score -= weights['bbands']

        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        if df['obv'].iloc[-1] > df['obv'].iloc[-2] and df['close'].iloc[-1] > df['close'].iloc[-2]:
             score += weights['obv']
        if df['obv'].iloc[-1] < df['obv'].iloc[-2] and df['close'].iloc[-1] < df['close'].iloc[-2]:
             score -= weights['obv']

        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        adx_value = adx_indicator.adx().iloc[-1]
        
        return score, adx_value

    def get_ta_interpretation(self):
        score = self.ai_score
        if score >= 2.5: return "📈", "Bullish แรง"
        elif score >= 1.0: return "↗️", "Bullish อ่อน"
        elif score > -1.0: return "↔️", "Sideways"
        elif score > -2.5: return "↘️", "Bearish อ่อน"
        else: return "📉", "Bearish แรง"

    def get_sentiment_interpretation(self, score_value: float):
        if score_value >= 0.5: return "🤩", "ดีมาก"
        elif score_value >= 0.1: return "🙂", "ค่อนข้างดี"
        elif score_value > -0.1: return "😐", "เป็นกลาง"
        elif score_value > -0.5: return "😟", "ค่อนข้างแย่"
        else: return "😨", "แย่มาก (FUD)"


    async def _determine_strategy(self):
        if self.avg_sentiment_score < -0.5:
            if self.current_mode != "SAFE":
                await self.send_telegram_message(f"‼️ [{self.settings['symbol']}] Smoothed Sentiment Veto! Average score ({self.avg_sentiment_score:+.2f}) is critical. Forcing SAFE mode.")
            return "SAFE"

        if self.current_mode == "TRAILING_UP":
            if self.adx > 30 and self.ai_score <= -2.5: return "SAFE"
            if self.ai_score < 1.0: return "GRID"
            return "TRAILING_UP"

        if self.current_mode == "SAFE":
            if self.ai_score > -1.0 and self.avg_sentiment_score >= -0.5: 
                return "GRID"
            return "SAFE"

        if self.adx > 30 and self.ai_score >= 2.5:
            if self.avg_sentiment_score < -0.2:
                await self.send_telegram_message(f"⚠️ [{self.settings['symbol']}] TA signal is Bullish, but Smoothed Sentiment is not positive ({self.avg_sentiment_score:+.2f}). Blocking switch to TRAILING_UP.")
                return "GRID"
            return "TRAILING_UP"
            
        if self.adx > 30 and self.ai_score <= -2.5: return "SAFE"
        
        return "GRID"

    async def _handle_mode_transition(self, new_mode, df):
        client = self.get_client()
        await self._cancel_all_orders(client)
        
        if self.current_mode != "SAFE":
            await self._enter_safe_mode(client)
            await asyncio.sleep(3) # Wait for sale to settle

        if new_mode == "TRAILING_UP": await self._enter_trailing_up_mode(client)
        elif new_mode == "GRID": await self._enter_grid_mode(client, df)
        
        self.current_mode = new_mode

    async def _enter_safe_mode(self, client: BitkubClient):
        """
        Robustly sells assets held by the bot and resets the state.
        This prevents orphaned coins and ensures correct P/L calculation.
        """
        await self.send_telegram_message("⏳ Entering SAFE mode. Selling assets...")
        
        # --- ROBUST LOGIC (FIX 3) ---
        
        # 1. Determine the amount to sell based on the bot's *internal* state.
        #    This is CRITICAL for P/L calculation.
        amount_to_sell = self.total_coins_held
        
        if amount_to_sell <= 0.0001:
            # If the bot thinks it has no coins, just reset the state.
            await self.send_telegram_message("ℹ️ No assets held by the bot. Resetting state.")
            self.total_investment, self.total_coins_held = 0.0, 0.0
            return

        # 2. If we are in Live Mode, check if we *actually* have the coins.
        if self.settings.get('trade_mode') == 'Live':
            balances = await client.get_balances()
            if balances.get('error') != 0:
                await self.send_telegram_message("❌ Could not get balances to sell assets. State not reset.")
                return # Abort!
            
            symbol_balance = float(balances.get('result', {}).get(self.settings['symbol'], {}).get('available', 0))
            
            if symbol_balance < amount_to_sell:
                logger.warning(f"[{self.settings['symbol']}] State desync! Bot thought it had {amount_to_sell:.8f}, but only {symbol_balance:.8f} is available. Selling available amount.")
                await self.send_telegram_message(f"⚠️ [{self.settings['symbol']}] State desync! Bot thought it had {amount_to_sell:.8f}, but only {symbol_balance:.8f} is available. Selling available amount.")
                amount_to_sell = symbol_balance # Sell what we can

        # 3. Proceed with the sale (Live or Test)
        if amount_to_sell > 0.0001:
            proceeds = 0.0
            profit = 0.0
            
            if self.settings.get('trade_mode') == 'Live':
                api_symbol = f"{self.settings['symbol']}_THB"
                res = await client.place_ask(api_symbol, amt=amount_to_sell, rat=0, typ='market')
                if res.get('error') == 0:
                    fill = res['result']
                    proceeds = float(fill['amt'])
                    
                    # Calculate P/L based *only* on the amount we *actually* sold
                    cost_of_sold_coins = (self.total_investment / self.total_coins_held) * amount_to_sell if self.total_coins_held > 0 else 0
                    profit = proceeds - cost_of_sold_coins
                    
                    await self.send_telegram_message(f"✅ ปิด Cycle (ขาย {amount_to_sell:.8f} {self.settings['symbol']}). P/L ของรอบนี้: {profit:+.2f} THB (จะถูกนำไปทบต้น)")
                else:
                    await self.send_telegram_message(f"❌ Error selling: {res}. State not reset.")
                    return # Abort! Sale failed, do not reset state.
            
            else: # Test Mode (This logic is now correct from the last file)
                latest_price = await get_latest_price(self.settings['symbol']) or self.last_checked_price
                proceeds = self.total_coins_held * latest_price
                profit = proceeds - self.total_investment
                await self.send_telegram_message(f"🧪 TEST MODE: ปิด Cycle. P/L ของรอบนี้: {profit:+.2f} THB (จะถูกนำไปทบต้น)")

            # 4. Success! Update P/L and reset state.
            self.realized_pnl += profit # This is the cycle PNL
            self.all_time_realized_pnl += profit # This is the cumulative PNL
            
            # Reset state *only* after a successful sale
            self.total_investment, self.total_coins_held = 0.0, 0.0
        
        else:
            # This case might be hit if the desync logic above sets amount_to_sell to 0
            await self.send_telegram_message("ℹ️ No assets available to sell after sync. Resetting state.")
            self.total_investment, self.total_coins_held = 0.0, 0.0
        
        # --- END ROBUST LOGIC ---

    async def _enter_trailing_up_mode(self, client: BitkubClient):
        await self.send_telegram_message("⏳ Entering TRAILING_UP mode. Re-investing capital...")

        # --- MODIFICATION: Use Compounding Capital (User Request) ---
        total_equity = self.original_capital + self.all_time_realized_pnl # Use original capital + all realized P/L
        # total_equity = self.original_capital (Old Fixed Capital Logic)
        # --- END MODIFICATION ---
        
        if total_equity < 100:
            await self.send_telegram_message(f"❌ Not enough capital to enter TRAILING_UP. Total equity is only {total_equity:,.2f} THB.")
            self.stop()
            return

        # We keep this message, but it's now informational.
        if self.realized_pnl != 0:
             await self.send_telegram_message(f"ℹ️ P/L from last cycle: {self.realized_pnl:+.2f} THB. Using Compounding Capital {total_equity:,.2f} THB for this cycle.")
        
        self.realized_pnl = 0.0 # Reset cycle PNL
        capital_to_use = total_equity * 0.995

        balances = await client.get_balances()
        if balances.get('error') == 0:
            thb_balance = float(balances.get('result', {}).get('THB', {}).get('available', 0))
            if thb_balance < capital_to_use:
                logger.warning(f"[{self.settings['symbol']}] Insufficient balance ({thb_balance:,.2f}) for compounding capital ({capital_to_use:,.2f}). Using available balance.")
                capital_to_use = thb_balance * 0.995

            if capital_to_use > 10 and self.settings.get('trade_mode') == 'Live':
                api_symbol = f"{self.settings['symbol']}_THB"
                res = await client.place_bid(api_symbol, amt=capital_to_use, rat=0, typ='market')
                if res.get('error') == 0:
                    fill = res['result']
                    self.total_investment = float(fill['amt']) 
                    self.total_coins_held = float(fill['cre']) 
                    await self.send_telegram_message(f"✅ Re-invested {capital_to_use:,.2f} THB for Trailing Up.")
                else:
                    await self.send_telegram_message(f"❌ Error buying for Trailing Up: {res}")
            
            elif capital_to_use > 10 and self.settings.get('trade_mode') == 'Test':
                latest_price = await get_latest_price(self.settings['symbol']) or self.last_checked_price
                if latest_price > 0:
                    self.total_investment = capital_to_use
                    self.total_coins_held = capital_to_use / latest_price
                    await self.send_telegram_message(f"🧪 TEST MODE: Re-invested {capital_to_use:,.2f} THB for Trailing Up.")
                else:
                    await self.send_telegram_message(f"❌ TEST MODE: Could not get price to simulate buy.")
            
        else:
            await self.send_telegram_message(f"❌ Could not check balance to enter Trailing Up mode: {balances}")

    async def _enter_grid_mode(self, client: BitkubClient, df):
        symbol = self.settings['symbol']
        api_symbol = f"{symbol}_THB"

        await self.send_telegram_message("⏳ Entering GRID mode. Setting up grid...")
        await self._calculate_grid_levels(df)
        if not self.grid_levels:
            await self.send_telegram_message("❌ Grid setup failed: Could not calculate levels."); return await self._enter_safe_mode(client)
        
        latest_price = await get_latest_price(symbol)
        if not latest_price:
            await self.send_telegram_message("❌ Grid setup failed: could not get latest price."); return await self._enter_safe_mode(client)
        
        # --- MODIFICATION: Use Compounding Capital (User Request) ---
        total_equity = self.original_capital + self.all_time_realized_pnl # Use original capital + all realized P/L
        # total_equity = self.original_capital (Old Fixed Capital Logic)
        # --- END MODIFICATION ---

        if total_equity < 100:
            await self.send_telegram_message(f"❌ Not enough capital to enter GRID. Total equity is only {total_equity:,.2f} THB.")
            self.stop()
            return
        
        if self.realized_pnl != 0:
             await self.send_telegram_message(f"ℹ️ P/L from last cycle: {self.realized_pnl:+.2f} THB. Using Compounding Capital {total_equity:,.2f} THB for this cycle.")

        self.realized_pnl = 0.0 # Reset cycle PNL
        self.grid_capital_per_level = total_equity / len(self.grid_levels) if self.grid_levels else 0
        
        if self.grid_capital_per_level < 10:
             await self.send_telegram_message(f"❌ Capital per grid ({self.grid_capital_per_level:.2f} THB) is below the 10 THB minimum. Please restart with more capital or a different grid strategy.")
             self.stop()
             return

        sell_levels_prices = [p for p in self.grid_levels if p > latest_price]
        buy_levels = [p for p in self.grid_levels if p <= latest_price]
        
        capital_for_initial_sells = self.grid_capital_per_level * len(sell_levels_prices)
        
        coin_available = 0.0

        if self.settings.get('trade_mode') == 'Live':
            balances = await client.get_balances()
            if balances.get('error') != 0:
                await self.send_telegram_message(f"❌ Could not check balance to enter Grid mode: {balances}"); return await self._enter_safe_mode(client)
            
            available_thb = float(balances.get('result', {}).get('THB', {}).get('available', 0))
            capital_to_buy = min(capital_for_initial_sells, available_thb * 0.995)

            if capital_to_buy > 10:
                await self.send_telegram_message(f"Buying initial coins with {capital_to_buy:,.2f} THB...")
                res = await client.place_bid(api_symbol, amt=capital_to_buy, rat=0, typ='market')
                
                if res.get('error') == 0:
                    fill = res['result']
                    self.total_investment += float(fill['amt'])
                    self.total_coins_held += float(fill['cre'])
                    await self.send_telegram_message("✅ Initial coins purchased.")
                    await asyncio.sleep(2)
                else:
                    await self.send_telegram_message(f"❌ Failed to buy initial coins: {res}"); return await self._enter_safe_mode(client)
            
            balances = await client.get_balances() # Re-check balances
            coin_available = float(balances.get('result', {}).get(symbol, {}).get('available', 0))
        
        else: # Test Mode
            capital_to_buy = capital_for_initial_sells
            if capital_to_buy > 10:
                # --- FIX: Test mode initial buy (Bug 2 fix) ---
                self.total_investment += capital_to_buy
                self.total_coins_held += capital_to_buy / latest_price # Correctly add coins
                await self.send_telegram_message(f"🧪 TEST MODE: Simulating initial buy with {capital_to_buy:,.2f} THB.")
            coin_available = self.total_coins_held # Use internal record
        
        # --- FIX: Use correct coin_available for both modes ---
        coin_per_sell_grid = coin_available / len(sell_levels_prices) if sell_levels_prices else 0
        
        for buy_price_level in sell_levels_prices:
            profit_pct = self.settings.get('profit_percent', 0)
            sell_price = self._format_price(buy_price_level * (1 + profit_pct / 100))
            
            if self.settings.get('trade_mode') == 'Live':
                if coin_per_sell_grid * sell_price > 10:
                    res = await client.place_ask(api_symbol, amt=coin_per_sell_grid, rat=sell_price, typ='limit')
                    if res.get('error') == 0:
                        actual_rate = float(res['result']['rat'])
                        self.open_sell_orders[actual_rate] = res['result']['id']
            else: # Test Mode
                 # We assume we have the coins to sell
                 self.open_sell_orders[sell_price] = f"test_sell_{int(time.time())}"

        for price in buy_levels:
            if self.grid_capital_per_level > 10: # Check applies to both modes
                if self.settings.get('trade_mode') == 'Live':
                    res = await client.place_bid(api_symbol, amt=self.grid_capital_per_level, rat=price, typ='limit')
                    if res.get('error') == 0:
                        actual_rate = float(res['result']['rat'])
                        self.open_buy_orders[actual_rate] = res['result']['id']
                else: # Test Mode
                     self.open_buy_orders[price] = f"test_buy_{int(time.time())}"
                 
        await self.send_telegram_message("✅ Grid Initialized with Limit Buy/Sell orders.")

    async def _calculate_grid_levels(self, df):
        symbol = self.settings['symbol']
        latest_price = await get_latest_price(symbol)
        if not latest_price: return
        gs = self.settings['grid_settings']; num_levels = int(gs.get('levels', 0))
        
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['std_dev_20'] = df['close'].rolling(window=20).std()
        
        p_high_sd = df['ema_20'].iloc[-1] + (2 * df['std_dev_20'].iloc[-1])
        p_low_sd = df['ema_20'].iloc[-1] - (2 * df['std_dev_20'].iloc[-1])
        
        if gs['type'] == 'Comprehensive':
            p_high = p_high_sd if gs.get('subtype') == 'Auto' else float(gs['p_max'])
            p_low = p_low_sd if gs.get('subtype') == 'Auto' else float(gs['p_min'])
            
            last_atr = df['atr'].iloc[-1]
            if last_atr > 0:
                num_levels = max(2, min(100, int((p_high - p_low) / last_atr)))
                step = (p_high - p_low) / num_levels if num_levels > 0 else 0
                self.grid_levels = [p_low + i * step for i in range(num_levels + 1)]
        elif gs['type'] == 'Fixed':
            p_high = p_high_sd
            p_low = p_low_sd
            step = (p_high - p_low) / (num_levels - 1) if num_levels > 1 else 0
            self.grid_levels = [p_low + i * step for i in range(num_levels)]
        elif gs['type'] == 'Manual':
            spread_pct = (float(gs['spread']) / 100); prices = [latest_price]
            for i in range(1, num_levels // 2 + 1): prices.append(latest_price * (1 + i * spread_pct))
            for i in range(1, (num_levels - 1) // 2 + 1): prices.append(latest_price * (1 - i * spread_pct))
            self.grid_levels = sorted(prices)
        if self.grid_levels: self.grid_levels = sorted(list(set(self._format_price(p) for p in self.grid_levels)))

    async def _handle_grid_shift(self, df):
        client = self.get_client()
        symbol = self.settings['symbol']
        api_symbol = f"{symbol}_THB"

        self.is_shifting_grid = True
        await self.send_telegram_message(f"📈 [{symbol}] Price has broken out of the grid! Shifting grid up...")

        await self._cancel_all_orders(client)

        # --- Grid Shift *still* compounds (This is by design) ---
        # A grid shift is part of the *same* cycle, so we re-invest P/L
        total_equity = self.original_capital + self.all_time_realized_pnl
        
        if total_equity < 100:
            await self.send_telegram_message(f"❌ Not enough capital to shift grid. Total equity is only {total_equity:,.2f} THB.")
            self.stop()
            self.is_shifting_grid = False
            return

        if self.realized_pnl != 0:
            # This "realized_pnl" is from the *current* cycle (which we are still in)
            await self.send_telegram_message(f"📈 Compounding! Total equity for new grid is now {total_equity:,.2f} THB.")
        
        self.realized_pnl = 0.0 # Reset cycle PNL
        # --- END PNL REFACTOR ---

        await self._calculate_grid_levels(df)
        if not self.grid_levels:
            await self.send_telegram_message("❌ Grid shift failed: Could not recalculate levels."); 
            self.is_shifting_grid = False
            return await self._handle_mode_transition("SAFE", df)

        self.grid_capital_per_level = total_equity / len(self.grid_levels) if self.grid_levels else 0

        if self.grid_capital_per_level < 10:
             await self.send_telegram_message(f"❌ Capital per grid ({self.grid_capital_per_level:.2f} THB) is too low after compounding. Stopping bot.")
             self.stop()
             self.is_shifting_grid = False
             return

        await self.send_telegram_message(f"✅ Placing {len(self.grid_levels)} new Limit Buy orders in the shifted grid (Wait for Pullback strategy).")

        for price in self.grid_levels:
            if self.grid_capital_per_level > 10:
                if self.settings.get('trade_mode') == 'Live':
                    res = await client.place_bid(api_symbol, amt=self.grid_capital_per_level, rat=price, typ='limit')
                    if res.get('error') == 0:
                        actual_rate = float(res['result']['rat'])
                        self.open_buy_orders[actual_rate] = res['result']['id']
                else: # Test Mode
                     self.open_buy_orders[price] = f"test_buy_{int(time.time())}"
        
        self.open_sell_orders = {}
        self.total_investment = 0.0
        self.total_coins_held = 0.0
        self.is_shifting_grid = False

    async def _manage_grid(self, df):
        if self.is_shifting_grid: return

        symbol = self.settings['symbol']; client = self.get_client()
        api_symbol = f"{symbol}_THB"

        current_price = await get_latest_price(symbol)
        if not current_price: return
        self.last_checked_price = current_price
        
        atr_value = df['atr'].iloc[-1]
        sl_multiplier = self.settings.get('stop_loss_atr', 0)
        open_buy_prices = list(self.open_buy_orders.keys())
        if sl_multiplier > 0 and open_buy_prices:
            sl_price = min(open_buy_prices) - (atr_value * sl_multiplier)
            if current_price <= sl_price:
                await self.send_telegram_message(f"🚨 [{symbol}] ATR STOP-LOSS TRIGGERED at {format_display_price(sl_price)}!"); 
                return await self._handle_mode_transition("SAFE", df)

        filled_buys = {}
        filled_sells = {}
        processed_prices_this_cycle = set()
        
        if self.settings.get('trade_mode') == 'Live':
            open_orders_res = await client.my_open_orders(api_symbol)
            if open_orders_res.get('error') != 0:
                logger.warning(f"Could not fetch open orders for {symbol}. Skipping management cycle.")
                return
            current_open_ids = {o['id'] for o in open_orders_res.get('result', [])}
            
            filled_sells = {p: oid for p, oid in self.open_sell_orders.items() if oid not in current_open_ids}
            filled_buys = {p: oid for p, oid in self.open_buy_orders.items() if oid not in current_open_ids}
        
        else: # Test Mode Simulation
            for sell_price in list(self.open_sell_orders.keys()):
                if current_price >= sell_price:
                    filled_sells[sell_price] = self.open_sell_orders[sell_price]
            
            for buy_price in list(self.open_buy_orders.keys()):
                if current_price <= buy_price:
                    filled_buys[buy_price] = self.open_buy_orders[buy_price]

        for sell_price, order_id in filled_sells.items():
            profit_pct = self.settings.get('profit_percent', 0)
            original_buy_price = self._format_price(sell_price / (1 + profit_pct / 100))

            if original_buy_price in processed_prices_this_cycle: continue
            processed_prices_this_cycle.add(original_buy_price)

            await self.send_telegram_message(f"📈 [{symbol}] 🔴 SELL Filled! Price: <code>{format_display_price(sell_price)}</code>", use_html=True)
            del self.open_sell_orders[sell_price]
            
            coin_amount = self.grid_capital_per_level / original_buy_price
            profit = (sell_price - original_buy_price) * coin_amount
            self.all_time_realized_pnl += profit # Use all_time PNL
            self.total_investment -= self.grid_capital_per_level 
            self.total_coins_held -= coin_amount
            await self.send_telegram_message(f"   -> Realized Profit: <code>{profit:+.2f} THB</code>", use_html=True)
            
            if self.settings.get('trade_mode') == 'Live':
                res = await client.place_bid(api_symbol, amt=self.grid_capital_per_level, rat=original_buy_price, typ='limit')
                if res.get('error') == 0:
                    actual_rate = float(res['result']['rat'])
                    self.open_buy_orders[actual_rate] = res['result']['id']
                    await self.send_telegram_message(f"   -> 🆕 New BUY order placed at <code>{format_display_price(actual_rate)}</code>", use_html=True)
            else:
                self.open_buy_orders[original_buy_price] = f"test_buy_{int(time.time())}"
                await self.send_telegram_message(f"   -> 🆕 [Test] New BUY order placed at <code>{format_display_price(original_buy_price)}</code>", use_html=True)


        for buy_price, order_id in filled_buys.items():
            profit_pct = self.settings.get('profit_percent', 0)
            sell_price = self._format_price(buy_price * (1 + profit_pct / 100))

            if sell_price in processed_prices_this_cycle: continue
            processed_prices_this_cycle.add(sell_price)
            
            await self.send_telegram_message(f"📈 [{symbol}] 🟢 BUY Filled! Price: <code>{format_display_price(buy_price)}</code>", use_html=True)
            del self.open_buy_orders[buy_price]
            
            # --- FIX: Test mode P/L tracking (Bug 2 fix) ---
            self.total_investment += self.grid_capital_per_level
            coin_amount_bought = self.grid_capital_per_level / buy_price
            self.total_coins_held += coin_amount_bought
            # --- END FIX ---
            
            if self.settings.get('trade_mode') == 'Live':
                res = await client.place_ask(api_symbol, amt=coin_amount_bought, rat=sell_price, typ='limit')
                if res.get('error') == 0:
                    actual_rate = float(res['result']['rat'])
                    self.open_sell_orders[actual_rate] = res['result']['id']
                    await self.send_telegram_message(f"   -> 🆕 New SELL order placed at <code>{format_display_price(actual_rate)}</code>", use_html=True)
            else:
                self.open_sell_orders[sell_price] = f"test_sell_{int(time.time())}"
                await self.send_telegram_message(f"   -> 🆕 [Test] New SELL order placed at <code>{format_display_price(sell_price)}</code>", use_html=True)

        if len(self.open_sell_orders) == 0 and not self.is_shifting_grid and self.current_mode == "GRID" and self.is_running:
            self.is_shifting_grid = True # Set lock
            await self._handle_grid_shift(df)
            self.is_shifting_grid = False # Release lock

    async def _cancel_all_orders(self, client: BitkubClient):
        symbol = self.settings['symbol']
        api_symbol = f"{symbol}_THB"
        if self.settings.get('trade_mode') == 'Live':
            res = await client.my_open_orders(api_symbol)
            if res.get('error') == 0:
                for o in res.get('result', []): await client.cancel_order(api_symbol, o['id'], o['side'])
        self.open_sell_orders = {}
        self.open_buy_orders = {}

    def _calculate_pnl_components(self, latest_price, coin_balance):
        average_cost = self.total_investment / self.total_coins_held if self.total_coins_held > 0 else 0
        
        # --- FIX (Bug 1): Use self.total_coins_held (Bot's record) for Unrealized P/L ---
        unrealized_pnl = (latest_price - average_cost) * self.total_coins_held if average_cost > 0 else 0
        
        # --- PNL REFACTOR: Use all_time_realized_pnl ---
        # This (current_equity) is the "true value" of the bot right now
        current_equity = self.original_capital + self.all_time_realized_pnl + unrealized_pnl
        
        # Total P/L is how much the "true value" has changed from the start
        total_pnl = current_equity - self.original_capital
        # --- END PNL REFACTOR ---
        
        pnl_percent = (total_pnl / self.original_capital) * 100 if self.original_capital > 0 else 0
        
        return {
            "unrealized_pnl": unrealized_pnl, 
            "total_pnl": total_pnl, 
            "pnl_percent": pnl_percent,
            "all_time_realized_pnl": self.all_time_realized_pnl # Pass this through
        }

    async def _generate_report_text(self, report_type: str) -> str:
        symbol = self.settings['symbol']; client = self.get_client()
        api_symbol = f"{symbol}_THB"
        status_text = '🟢 Active' if self.is_running else '🔴 Inactive'
        latest_price = await get_latest_price(symbol) or self.last_checked_price
        
        if self.is_running and (self.sentiment_justification == "N/A" or not self.sentiment_justification):
            logger.info(f"[{symbol}] Triggering on-demand sentiment analysis for report.")
            await self.send_telegram_message(f"⏳ [{symbol}] กำลังอัปเดต Sentiment (ครั้งแรก)...")
            await self.update_sentiment_analysis() 

        if self.ai_score == 0 and self.is_running:
            df = await get_market_data(symbol, self.settings['timeframe'])
            if not df.empty: self.ai_score, self.adx = self._calculate_ai_score(df)

        ai_icon, ai_text = self.get_ta_interpretation()
        senti_icon, senti_text = self.get_sentiment_interpretation(self.sentiment_score)
        avg_senti_icon, avg_senti_text = self.get_sentiment_interpretation(self.avg_sentiment_score)

        mode_icon = {'SAFE': '🛡️', 'GRID': '↔️', 'TRAILING_UP': '📈'}.get(self.current_mode, '❓')
        
        coin_balance = 0
        live_open_buys = []
        live_open_sells = []

        if self.settings.get('trade_mode') == 'Live':
            balances = await client.get_balances()
            # We get coin_balance from exchange, but P/L calculation uses bot's internal state
            coin_balance = float(balances.get('result',{}).get(symbol,{}).get('available',0)) if balances.get('error') == 0 else self.total_coins_held
            
            open_orders_res = await client.my_open_orders(api_symbol)
            if open_orders_res.get('error') == 0:
                for order in open_orders_res.get('result', []):
                    if order['side'].lower() == 'buy':
                        live_open_buys.append(float(order['rate']))
                    elif order['side'].lower() == 'sell':
                        live_open_sells.append(float(order['rate']))
        else: # Test Mode
            coin_balance = self.total_coins_held
            live_open_buys = list(self.open_buy_orders.keys())
            live_open_sells = list(self.open_sell_orders.keys())

        # --- FIX (Bug 1): P/L calculation is now correct because _calculate_pnl_components uses
        # self.total_coins_held (internal) instead of coin_balance (exchange)
        pnl = self._calculate_pnl_components(latest_price, coin_balance)

        report_lines = [
            f"Status: {status_text}",
            f"Last Price: 🪙 <code>{format_display_price(latest_price)} THB</code>",
            f"AI Mode: {mode_icon} <code>{self.current_mode}</code>",
            f"AI (TA): {ai_icon} {html.escape(ai_text)} (<code>{self.ai_score:+.2f}</code>, ADX: <code>{self.adx:.1f}</code>)",
            f"AI (Sentiment): {senti_icon} {html.escape(senti_text)} (Latest: <code>{self.sentiment_score:+.1f}</code>)",
            f"AI (Sentiment Avg): {avg_senti_icon} {html.escape(avg_senti_text)} (Avg: <code>{self.avg_sentiment_score:+.2f}</code> <b>[Logic]</b>)",
            f"Grid Orders: 🟢 BUY <code>{len(live_open_buys)}</code> | 🔴 SELL <code>{len(live_open_sells)}</code>",
            # --- PNL REFACTOR: Report Original Capital ---
            f"Original Capital: 💰 <code>{self.original_capital:,.2f} THB</code>",
            # --- END PNL REFACTOR ---
        ]
        
        # --- MODIFICATION: Show Realized P/L for Hourly, Total P/L for Data ---
        if report_type == 'hourly':
            # Hourly report shows Realized P/L as requested
            realized_pnl_color = "🟢" if pnl['all_time_realized_pnl'] >= 0 else "🔴"
            realized_pnl_percent = (pnl['all_time_realized_pnl'] / self.original_capital) * 100 if self.original_capital > 0 else 0
            report_lines.append(f"P/L (Realized): {realized_pnl_color} <code>{pnl['all_time_realized_pnl']:+,.2f} THB ({realized_pnl_percent:+.2f}%)</code>")
        else:
            # Data report (management card) shows Total P/L
            total_pnl_color = "🟢" if pnl['total_pnl'] >= 0 else "🔴"
            report_lines.append(f"P/L (Total): {total_pnl_color} <code>{pnl['total_pnl']:+,.2f} THB ({pnl['pnl_percent']:+.2f}%)</code>")
        # --- END MODIFICATION ---

        report_lines.append(f"Sentiment Justification: <i>{html.escape(self.sentiment_justification)}</i>")

        if report_type == 'data':
            # --- PNL REFACTOR: Report All Time Realized PNL ---
            realized_pnl_color = "🟢" if pnl['all_time_realized_pnl'] >= 0 else "🔴"
            unrealized_pnl_color = "🟢" if pnl['unrealized_pnl'] >= 0 else "🔴"
            report_lines.append(f"  - P/L (Realized): {realized_pnl_color} <code>{pnl['all_time_realized_pnl']:+,.2f} THB</code>")
            report_lines.append(f"  - Unrealized: {unrealized_pnl_color} <code>{pnl['unrealized_pnl']:+,.2f} THB</code>")
            # --- END PNL REFACTOR ---
            
            report_lines.append(f"\nGrid Orders ({'Live' if self.settings.get('trade_mode') == 'Live' else 'Test Mode'}):")
            
            for price in sorted(live_open_sells, reverse=True): 
                report_lines.append(f"🔴 Sell at <code>{format_display_price(price)}</code>")
            for price in sorted(live_open_buys, reverse=True): 
                report_lines.append(f"🟢 Buy at <code>{format_display_price(price)}</code>")
                
        return "\n".join(report_lines)

    async def generate_report(self, is_running=None):
        symbol = self.settings['symbol'].upper()
        report_body = await self._generate_report_text(report_type='hourly')
        header = f"📊 <b>{html.escape(symbol)} Hourly Report</b>\n{'-'*40}\n"
        await self.send_telegram_message(header + report_body, use_html=True)

    async def get_management_card(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        symbol = self.settings['symbol'].upper()
        await update.message.reply_text("🔄 กำลังอัปเดตข้อมูล (อาจใช้เวลาสักครู่หากต้องดึง Sentiment ใหม่)...", reply_markup=ReplyKeyboardRemove())
        report_body = await self._generate_report_text(report_type='data')
        header = f"⚙️ <b>{html.escape(symbol)} Management</b>\n---\n"
        
        toggle_text = "🛑 Stop Bot" if self.is_running else "▶️ Start Bot"
        
        keyboard = [
            [KeyboardButton(toggle_text), KeyboardButton("🔄 Refresh Info")],
            [KeyboardButton("🗑️ Delete Bot"), KeyboardButton("⬅️ Back to Bot List")],
            [KeyboardButton("⬅️ กลับไปเมนูหลัก")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)

        await update.message.reply_html(
            text=header + report_body,
            reply_markup=reply_markup
        )

    async def send_telegram_message(self, text: str, use_html: bool = False, reply_markup=None):
        if self.context:
            final_text = text if use_html else f"<code>{html.escape(text)}</code>"
            try:
                await self.context.bot.send_message(self.chat_id, text=final_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
            except BadRequest:
                logger.warning(f"HTML parsing failed. Sending as plain text.")
                clean_text = re.sub('<[^<]+?>', '', text)
                await self.context.bot.send_message(self.chat_id, text=clean_text, reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Failed to send message to {self.chat_id}: {e}")

    def _format_price(self, price): return float(Decimal(price).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))

# --- Main Menu and General Commands ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    user = update.effective_user
    keyboard = [
        [KeyboardButton("➕ สร้างบอทใหม่"), KeyboardButton("🤖 จัดการบอท")],
        [KeyboardButton("❓ ช่วยเหลือ")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    start_text = (f"สวัสดีครับ {user.mention_html()}! 👋\n\n"
                  "ผมคือ <b>Bitkub Hybrid AI Bot</b> พร้อมช่วยคุณเทรดอัตโนมัติ.\n\n"
                  "เลือกเมนูด้านล่างเพื่อเริ่มต้น:")

    await update.message.reply_html(start_text, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [[KeyboardButton("⬅️ กลับไปเมนูหลัก")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    try:
        with open('help_guide.md', 'r', encoding='utf-8') as f:
            help_text = f.read()
        await update.message.reply_text(help_text, reply_markup=reply_markup)
        
    except FileNotFoundError:
        logger.error("help_guide.md not found. Sending fallback help message.")
        fallback_text = (
            "<b><u>คำสั่งหลัก</u></b>\n"
            "<b>/start</b> - กลับไปที่เมนูหลัก\n"
            "<b>➕ สร้างบอทใหม่</b> - เริ่มขั้นตอนการสร้างบอทใหม่\n"
            "<b>🤖 จัดการบอท</b> - ดูสถานะ, เริ่ม/หยุด, หรือลบบอทที่มีอยู่\n\n"
            "<i>(Error: ไม่สามารถโหลดคู่มือฉบับเต็มจาก help_guide.md)</i>"
        )
        await update.message.reply_html(fallback_text, reply_markup=reply_markup)
    except Exception as e:
        logger.error(f"Error reading help_guide.md: {e}")
        await update.message.reply_text("เกิดข้อผิดพลาดในการโหลดคู่มือ", reply_markup=reply_markup)


# --- Create Bot Conversation States & Handlers ---
(SYMBOL, TIMEFRAME, CAPITAL, GRID_TYPE, GRID_SETTINGS, 
 PROFIT_PERCENT, STOP_LOSS, TRADE_MODE, CONFIRMATION) = range(9)

async def create_bot_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data['new_bot'] = {}
    await update.message.reply_text(
        "<b>➕ เริ่มขั้นตอนสร้างบอท</b>\n\nขั้นตอนที่ 1: กรุณาพิมพ์ชื่อย่อของเหรียญที่คุณต้องการเทรด (เช่น BTC, KUB, DOGE)",
        reply_markup=ReplyKeyboardRemove(),
        parse_mode=ParseMode.HTML
    )
    return SYMBOL

async def get_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data['new_bot']['symbol'] = update.message.text.upper()
    keyboard = [
        [KeyboardButton("15 นาที (Scalping)"), KeyboardButton("1 ชั่วโมง (Day Trading)")],
        [KeyboardButton("4 ชั่วโมง (Swing)"), KeyboardButton("1 วัน (Position)")],
    ]
    await update.message.reply_text(
        "ขั้นตอนที่ 2: เลือก Timeframe",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    )
    return TIMEFRAME

async def get_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if "15" in text: timeframe = "15"
    elif "1 ชั่วโมง" in text: timeframe = "60"
    elif "4 ชั่วโมง" in text: timeframe = "240"
    elif "1 วัน" in text: timeframe = "1D"
    else:
        await update.message.reply_text("Timeframe ไม่ถูกต้อง กรุณาเลือกใหม่")
        return TIMEFRAME
        
    context.user_data['new_bot']['timeframe'] = timeframe
    await update.message.reply_text(
        "ขั้นตอนที่ 3: ระบุเงินทุน (THB) ที่ต้องการใช้สำหรับบอทนี้ (ขั้นต่ำ 500 บาท)",
        reply_markup=ReplyKeyboardRemove()
    )
    return CAPITAL

async def get_capital(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    try:
        capital = float(update.message.text)
        if capital < 500:
            await update.message.reply_text("ทุนต้องมีอย่างน้อย 500 THB กรุณาลองใหม่อีกครั้ง")
            return CAPITAL
        context.user_data['new_bot']['capital'] = capital
        keyboard = [
            [KeyboardButton("Fixed (มือใหม่)"), KeyboardButton("Comprehensive (ขั้นสูง)")],
            [KeyboardButton("Manual (กำหนดเอง)")],
        ]
        await update.message.reply_text(
            "ขั้นตอนที่ 4: เลือกโหมด Grid",
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        )
        return GRID_TYPE
    except ValueError:
        await update.message.reply_text("ตัวเลขไม่ถูกต้อง กรุณาใส่เงินทุนเป็นตัวเลข (THB)")
        return CAPITAL

async def get_grid_type(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if "Fixed" in text: grid_type = 'Fixed'
    elif "Comprehensive" in text: grid_type = 'Comprehensive'
    elif "Manual" in text: grid_type = 'Manual'
    else: 
        await update.message.reply_text("โหมด Grid ไม่ถูกต้อง กรุณาเลือกใหม่")
        return GRID_TYPE

    context.user_data['new_bot']['grid_settings'] = {'type': grid_type}
    
    if grid_type == 'Comprehensive':
        keyboard = [
            [KeyboardButton("Auto (อัตโนมัติ)"), KeyboardButton("Manual (ระบุเอง)")],
        ]
        await update.message.reply_text("Grid: Comprehensive\n\nต้องการกำหนดกรอบราคาอย่างไร?",
            reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
        return GRID_SETTINGS
    elif grid_type == 'Fixed':
        await update.message.reply_text("Grid: Fixed\n\nกรุณากำหนด 'จำนวนชั้น' ของกริด (เช่น 20)", reply_markup=ReplyKeyboardRemove())
        return GRID_SETTINGS
    elif grid_type == 'Manual':
        await update.message.reply_text("Grid: Manual\n\nกรุณากำหนด '% ระยะห่าง' ของแต่ละชั้น Grid (เช่น 1.5)", reply_markup=ReplyKeyboardRemove())
        return GRID_SETTINGS
    return ConversationHandler.END


async def get_grid_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_message = update.message
    settings = context.user_data['new_bot']['grid_settings']
    user_input = user_message.text

    try:
        if 'subtype' not in settings and settings['type'] == 'Comprehensive': # Coming from Comprehensive choice
             if 'Auto' in user_input:
                 settings['subtype'] = 'Auto'
             elif 'Manual' in user_input:
                 settings['subtype'] = 'Manual'
                 await user_message.reply_text("กรุณากำหนดกรอบราคาต่ำสุดและสูงสุด คั่นด้วยเว้นวรรค (เช่น 3.5 5.0)", reply_markup=ReplyKeyboardRemove())
                 return GRID_SETTINGS
             else:
                 await user_message.reply_text("การเลือกไม่ถูกต้อง กรุณาเลือก Auto หรือ Manual")
                 return GRID_SETTINGS
        
        elif settings['type'] == 'Comprehensive' and settings.get('subtype') == 'Manual':
            p_min, p_max = map(float, user_input.split())
            if p_min >= p_max: 
                await user_message.reply_text("❌ ราคาสูงสุดต้องมากกว่าราคาต่ำสุด")
                return GRID_SETTINGS
            settings['p_min'], settings['p_max'] = p_min, p_max
        elif settings['type'] == 'Fixed':
            settings['levels'] = int(user_input)
        elif settings['type'] == 'Manual':
            if 'spread' not in settings:
                settings['spread'] = float(user_input)
                await user_message.reply_text("ต่อไป, กรุณากำหนด 'จำนวนชั้น' ของ Grid (เช่น 20)", reply_markup=ReplyKeyboardRemove())
                return GRID_SETTINGS
            else: 
                settings['levels'] = int(user_input)
    except (ValueError, IndexError):
        await user_message.reply_text("❌ รูปแบบข้อมูลไม่ถูกต้อง กรุณาลองอีกครั้ง")
        return GRID_SETTINGS

    keyboard = [
        [KeyboardButton("1.0%"), KeyboardButton("1.5%"), KeyboardButton("2.0%")],
        [KeyboardButton("2.5%"), KeyboardButton("3.0%")],
    ]
    await user_message.reply_text("ขั้นตอนที่ 5: เลือก % กำไรต่อไม้ที่ต้องการ",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
    return PROFIT_PERCENT

async def get_profit_percent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    try:
        profit_val = float(text.replace('%', ''))
        if not (1.0 <= profit_val <= 3.0):
            raise ValueError("Profit out of range")
        context.user_data['new_bot']['profit_percent'] = profit_val
    except (ValueError, IndexError):
        await update.message.reply_text("การเลือก % กำไรไม่ถูกต้อง กรุณาเลือกจากเมนู")
        return PROFIT_PERCENT

    keyboard = [
        [KeyboardButton("ต่ำ (1.5x ATR)"), KeyboardButton("กลาง (2.0x ATR)")],
        [KeyboardButton("สูง (2.5x ATR)"), KeyboardButton("ไม่ใช้ (No SL)")],
    ]
    await update.message.reply_text("ขั้นตอนที่ 6: ตั้งค่า Stop-Loss",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
    return STOP_LOSS

async def get_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if "1.5" in text: sl_value = 1.5
    elif "2.0" in text: sl_value = 2.0
    elif "2.5" in text: sl_value = 2.5
    elif "ไม่ใช้" in text: sl_value = 0
    else:
        await update.message.reply_text("การเลือก Stop-Loss ไม่ถูกต้อง")
        return STOP_LOSS
        
    context.user_data['new_bot']['stop_loss_atr'] = sl_value
    keyboard = [
        [KeyboardButton("🚀 Live Trading (เงินจริง)"), KeyboardButton("🧪 Test Mode (จำลอง)")],
    ]
    await update.message.reply_text("ขั้นตอนที่ 7: เลือกโหมดเทรด",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True))
    return TRADE_MODE

async def show_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    if "Live" in text: mode = 'Live'
    elif "Test" in text: mode = 'Test'
    else:
        await update.message.reply_text("โหมดเทรดไม่ถูกต้อง")
        return TRADE_MODE

    context.user_data['new_bot']['trade_mode'] = mode
    await update.message.reply_text("⏳ กำลังสร้างข้อมูลสรุป กรุณารอสักครู่...", reply_markup=ReplyKeyboardRemove())

    try:
        settings = context.user_data['new_bot']
        df = await get_market_data(settings['symbol'], settings['timeframe'])
        if df.empty:
            await update.message.reply_text(f"ไม่สามารถดึงข้อมูลสำหรับ {settings['symbol']} ได้")
            return ConversationHandler.END

        client_cfg = {'api_key': BITKUB_API_KEY, 'api_secret': BITKUB_API_SECRET, 'gemini_api_key': GEMINI_API_KEY}
        temp_bot = HybridGridBot(update.effective_chat.id, client_cfg, settings)
        
        await temp_bot._calculate_grid_levels(df) 
        ai_score, adx = temp_bot._calculate_ai_score(df)
        
        predicted_mode = await temp_bot._determine_strategy()
        
        price = await get_latest_price(settings['symbol'])
        profit_pct = settings.get('profit_percent', 0)

        summary_lines = [
            f"📋 <b>สรุปข้อมูลบอท</b>", '-'*40,
            f"<b>เหรียญ:</b> <code>{html.escape(settings['symbol'])}</code>",
            f"<b>Timeframe:</b> <code>{settings['timeframe']}</code>",
            f"<b>เงินทุน:</b> <code>{settings['capital']:,.2f} THB</code>",
            f"<b>% กำไรต่อไม้:</b> <code>{profit_pct:.1f}%</code>",
            f"<b>โหมดเทรด:</b> {'🚀 Live' if settings['trade_mode'] == 'Live' else '🧪 Test'}",
            f"<b>โหมดเริ่มต้นที่คาดการณ์:</b> <code>{html.escape(predicted_mode)}</code>",
            f"<b>ราคาล่าสุด:</b> <code>{format_display_price(price or 0)}</code>",
            f"<b>แนวโน้ม AI:</b> Score <code>{ai_score:+.2f}</code>, ADX <code>{adx:.2f}</code>",
            "\n<b>Grid Levels (ตัวอย่าง):</b>"
        ]
        
        if temp_bot.grid_levels and price:
            buy_levels_below_price = sorted([l for l in temp_bot.grid_levels if l <= price], reverse=True)[:5]
            for buy_level in buy_levels_below_price:
                sell_level = buy_level * (1 + profit_pct / 100)
                summary_lines.append(f"🟢 Buy at <code>{format_display_price(buy_level)}</code> ➡️ Sell at <code>{format_display_price(sell_level)}</code>")
        
        summary = "\n".join(summary_lines)
        keyboard = [
            [KeyboardButton("✅ ยืนยันและเริ่มทำงาน"), KeyboardButton("❌ ยกเลิก")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_html(summary, reply_markup=reply_markup)
        return CONFIRMATION

    except Exception as e:
        logger.error(f"Error during summary creation: {e}", exc_info=True)
        await update.message.reply_text("เกิดข้อผิดพลาด กรุณาลองใหม่ด้วย /start")
        return ConversationHandler.END

async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if "✅ ยืนยัน" in update.message.text:
        try:
            chat_id = update.effective_chat.id
            settings = context.user_data['new_bot']
            bot_id = f"{chat_id}_{settings['symbol']}"

            if bot_id in user_bots and user_bots[bot_id].is_running:
                await update.message.reply_html(f"❌ มีบอทสำหรับ <b>{html.escape(settings['symbol'])}</b> ทำงานอยู่แล้ว")
            else:
                client_cfg = {'api_key': BITKUB_API_KEY, 'api_secret': BITKUB_API_SECRET, 'gemini_api_key': GEMINI_API_KEY}
                bot = HybridGridBot(chat_id, client_cfg, settings)
                bot.set_context(context.application)
                user_bots[bot_id] = bot
                await update.message.reply_html(f"✔️ สร้างบอทสำหรับ <b>{html.escape(settings['symbol'])}</b> สำเร็จแล้ว!")
                await bot.start() # This will now run the new, smarter start logic
                save_bots_state()
        except Exception as e:
            logger.error(f"Error during bot startup: {e}", exc_info=True)
            await update.message.reply_text("เกิดข้อผิดพลาดระหว่างการเริ่มการทำงานของบอท")
    else: 
        await update.message.reply_text("ยกเลิกการสร้างบอทแล้ว")

    context.user_data.clear()
    
    kb = [[KeyboardButton("🤖 จัดการบอท")], [KeyboardButton("⬅️ กลับไปเมนูหลัก")]]
    
    await update.message.reply_html(
        "กด จัดการบอท เพื่อดู/สั่งงานบอทของคุณ",
        reply_markup=ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=True)
    )
    return ConversationHandler.END

async def cancel_creation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("ยกเลิกการสร้างบอทแล้ว")
    await start(update, context)
    return ConversationHandler.END


# --- Manage Bots Conversation States & Handlers ---
MANAGE_SELECT_BOT, MANAGE_SELECT_ACTION, MANAGE_CONFIRM_DELETE = range(3)

async def manage_bots_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    chat_id = update.effective_chat.id
    bots = [b for b_id, b in user_bots.items() if b_id.startswith(str(chat_id))]
    
    if not bots:
        await update.message.reply_text("คุณยังไม่มีบอทที่ทำงานอยู่")
        await start(update, context)
        return ConversationHandler.END

    keyboard = []
    for bot in bots:
        status_icon = '🟢' if bot.is_running else '🔴'
        symbol = bot.settings['symbol']
        keyboard.append([KeyboardButton(f"{status_icon} {symbol}")])
    
    keyboard.append([KeyboardButton("⬅️ กลับไปเมนูหลัก")])
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    
    await update.message.reply_html("🤖 <b>จัดการบอท</b>\n\nกรุณาเลือกบอทที่ต้องการดูข้อมูล:", reply_markup=reply_markup)
    return MANAGE_SELECT_BOT

async def select_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    selection = update.message.text
    chat_id = update.effective_chat.id
    
    try:
        symbol = selection.split(' ')[1]
        bot_id = f"{chat_id}_{symbol}"
        bot = user_bots.get(bot_id)

        if not bot:
            await update.message.reply_text("ไม่พบบอทดังกล่าว กรุณาเลือกใหม่")
            return MANAGE_SELECT_BOT
        
        context.user_data['selected_bot_id'] = bot_id
        await bot.get_management_card(update, context)
        return MANAGE_SELECT_ACTION
        
    except (IndexError, KeyError):
        await update.message.reply_text("การเลือกไม่ถูกต้อง กรุณาเลือกจากรายการ")
        return await manage_bots_start(update, context)


async def select_bot_action(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    action_text = update.message.text
    bot_id = context.user_data.get('selected_bot_id')
    if not bot_id:
        await update.message.reply_text("เกิดข้อผิดพลาด: ไม่พบบอทที่เลือกไว้")
        return await manage_bots_start(update, context)

    bot = user_bots.get(bot_id)
    if not bot:
        await update.message.reply_text("ไม่พบบอทดังกล่าว อาจถูกลบไปแล้ว")
        return await manage_bots_start(update, context)

    if "Start" in action_text:
        await bot.start() # This will now run the new, smarter start logic
        await update.message.reply_html(f"🟢 บอท <b>{html.escape(bot.settings['symbol'])}</b> เริ่มทำงานแล้ว")
        save_bots_state()
        return await manage_bots_start(update, context)
        
    elif "Stop" in action_text:
        bot.stop()
        await update.message.reply_html(f"🔴 บอท <b>{html.escape(bot.settings['symbol'])}</b> หยุดทำงานแล้ว")
        save_bots_state()
        return await manage_bots_start(update, context)

    elif "Refresh" in action_text:
        await bot.get_management_card(update, context) 
        return MANAGE_SELECT_ACTION

    elif "Delete" in action_text:
        keyboard = [
            [KeyboardButton("ใช่, ขายเหรียญแล้วลบ"), KeyboardButton("ใช่, ลบแต่เก็บเหรียญ")],
            [KeyboardButton("ไม่, ยกเลิก")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_html(f"🗑️ แน่ใจหรือไม่ว่าจะลบบอท <b>{html.escape(bot.settings['symbol'])}</b>?",
                                       reply_markup=reply_markup)
        return MANAGE_CONFIRM_DELETE
        
    elif "Back to Bot List" in action_text:
        return await manage_bots_start(update, context)
        
    else:
        await update.message.reply_text("คำสั่งไม่ถูกต้อง กรุณาเลือกจากเมนู")
        return MANAGE_SELECT_ACTION

async def confirm_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = update.message.text
    bot_id = context.user_data.get('selected_bot_id')
    bot_to_delete = user_bots.get(bot_id)

    if not bot_to_delete:
        await update.message.reply_text("ไม่พบบอทดังกล่าว อาจถูกลบไปแล้ว")
    elif "ไม่, ยกเลิก" in choice:
        await update.message.reply_text("ยกเลิกการลบ")
        await bot_to_delete.get_management_card(update, context)
        return MANAGE_SELECT_ACTION
    else:
        symbol = bot_to_delete.settings['symbol']
        api_symbol = f"{symbol}_THB"
        bot_to_delete.stop()
        client = bot_to_delete.get_client()

        if 'ขายเหรียญ' in choice:
            # --- NEW ROBUST DELETE-SELL LOGIC (BUG 4 FIX) ---
            # We can't trust _enter_safe_mode if the state is corrupted (e.g., total_coins_held=0)
            # We must liquidate based on *actual* exchange balance.
            if bot_to_delete.settings.get('trade_mode') == 'Live':
                await update.message.reply_text(f"Attempting to liquidate all {symbol} assets on Bitkub...")
                balances = await client.get_balances()
                if balances.get('error') == 0:
                    symbol_balance = float(balances.get('result', {}).get(symbol, {}).get('available', 0))
                    if symbol_balance > 0.0001:
                        res = await client.place_ask(api_symbol, amt=symbol_balance, rat=0, typ='market')
                        if res.get('error') == 0:
                            await update.message.reply_text(f"✅ Successfully sold {symbol_balance:.8f} {symbol}.")
                        else:
                            await update.message.reply_text(f"❌ Error liquidating {symbol}: {res.get('message', 'Unknown Error')}")
                    else:
                        await update.message.reply_text(f"ℹ️ No {symbol} assets found on Bitkub to sell.")
                else:
                    await update.message.reply_text(f"❌ Could not check balance to liquidate: {balances.get('message', 'Unknown Error')}")
            else:
                await update.message.reply_text(f"ℹ️ Bot is in Test Mode. No real assets to sell.")
            # --- END NEW LOGIC ---
        
        await bot_to_delete._cancel_all_orders(client) # This is still needed

        del user_bots[bot_id]
        save_bots_state()
        await update.message.reply_html(f"🗑️ บอทสำหรับ <b>{html.escape(symbol)}</b> ถูกลบแล้ว")

    context.user_data.clear()
    await start(update, context)
    return ConversationHandler.END

async def cancel_manage(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("ยกเลิกการจัดการบอท")
    await start(update, context)
    return ConversationHandler.END


# --- Bot Jobs ---
async def run_all_bots_periodically(context: ContextTypes.DEFAULT_TYPE):
    active_bots = [b for b in user_bots.values() if b.is_running]
    if not active_bots: return
    logger.info(f"Periodic check running for {len(active_bots)} active bots...")
    tasks = [bot.run_logic_cycle() for bot in active_bots]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for bot, result in zip(active_bots, results):
        if isinstance(result, Exception):
            logger.error(f"Error in logic cycle for bot {bot.settings['symbol']}: {result}", exc_info=True)
            await bot.send_telegram_message(f"‼️ [{bot.settings['symbol']}] An error occurred. Bot stopped to prevent issues. Check logs.")
            bot.stop()
            
    save_bots_state()
    
async def run_sentiment_analysis_periodically(context: ContextTypes.DEFAULT_TYPE):
    active_bots = [b for b in user_bots.values() if b.is_running]
    if not active_bots: return
    logger.info(f"Running sentiment analysis job for {len(active_bots)} bots...")
    tasks = [bot.update_sentiment_analysis() for bot in active_bots]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for bot, result in zip(active_bots, results):
        if isinstance(result, Exception):
            logger.error(f"Error in sentiment analysis for bot {bot.settings['symbol']}: {result}", exc_info=True)

async def send_hourly_report(context: ContextTypes.DEFAULT_TYPE):
    active_bots = [b for b in user_bots.values() if b.is_running]
    if not active_bots: return
    logger.info(f"Running hourly report job for {len(active_bots)} bots...")
    for bot in active_bots:
        try:
            await bot.generate_report(is_running=True)
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Failed to send hourly report for {bot.settings['symbol']}: {e}")

# --- Error Handler ---
async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    err = context.error
    if isinstance(err, BadRequest) and ("Query is too old" in str(err) or "query id is invalid" in str(err)):
        logger.info(f"Ignored Telegram BadRequest: {err}")
        return
    if isinstance(err, TimedOut):
        logger.warning(f"Telegram API call timed out: {err}")
        return
        
    if isinstance(err, BadRequest) and "Can't parse entities" in str(err):
        logger.error(f"HTML PARSING ERROR: {err}. This usually means mismatched tags in a sent message.")
        if update and hasattr(update, 'effective_chat_id'):
            try:
                await context.bot.send_message(
                    chat_id=update.effective_chat_id, 
                    text="⚠️ เกิดข้อผิดพลาดในการแสดงผล (HTML Error). กรุณาตรวจสอบ Log."
                )
            except Exception as e:
                logger.error(f"Failed to send error fallback message: {e}")
        return
        
    logger.error("Unhandled error", exc_info=True)

# --- Main Application Setup ---
def main() -> None:
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    load_bots_state(application)

    create_conv = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex('^➕ สร้างบอทใหม่$'), create_bot_start)],
        states={
            SYMBOL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_symbol)],
            TIMEFRAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_timeframe)],
            CAPITAL: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_capital)],
            GRID_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_grid_type)],
            GRID_SETTINGS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_grid_settings)],
            PROFIT_PERCENT: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_profit_percent)],
            STOP_LOSS: [MessageHandler(filters.TEXT & ~filters.COMMAND, get_stop_loss)],
            TRADE_MODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, show_confirmation)],
            CONFIRMATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, handle_confirmation)]
        },
        fallbacks=[MessageHandler(filters.Regex('^❌ ยกเลิก$'), cancel_creation)],
    )

    manage_conv = ConversationHandler(
        entry_points=[MessageHandler(filters.Regex('^🤖 จัดการบอท$'), manage_bots_start)],
        states={
            MANAGE_SELECT_BOT: [
                MessageHandler(filters.Regex('^🤖 จัดการบอท$'), manage_bots_start),
                MessageHandler(filters.Regex('^[🟢🔴]'), select_bot),
            ],
            MANAGE_SELECT_ACTION: [
                MessageHandler(filters.Regex('^🤖 จัดการบอท$'), manage_bots_start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, select_bot_action),
            ],
            MANAGE_CONFIRM_DELETE: [
                MessageHandler(filters.Regex('^🤖 จัดการบอท$'), manage_bots_start),
                MessageHandler(filters.TEXT & ~filters.COMMAND, confirm_delete),
            ],
        },
        fallbacks=[
            MessageHandler(filters.Regex('^⬅️ กลับไปเมนูหลัก$'), cancel_manage),
            CommandHandler("start", cancel_manage),
        ],
        allow_reentry=True,
        per_user=True,
        per_chat=True,
    )

    # Handler Registration
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.Regex('^⬅️ กลับไปเมนูหลัก$'), start))
    application.add_handler(MessageHandler(filters.Regex('^❓ ช่วยเหลือ$'), help_command))
    application.add_handler(create_conv)
    application.add_handler(manage_conv)
    application.add_error_handler(on_error)

    # Job Queue Setup
    job_queue = application.job_queue
    # We run the logic cycle *after* the sentiment job to ensure sentiment is fresh
    job_queue.run_repeating(run_sentiment_analysis_periodically, interval=900, first=60) # 15 mins
    job_queue.run_repeating(run_all_bots_periodically, interval=60, first=70) # 1 min, starts after first sentiment
    job_queue.run_repeating(send_hourly_report, interval=3600, first=3600) # 1 hour

    # Signal Handling for Graceful Shutdown
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received. Saving bot states...")
        save_bots_state()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Bot is running...")
    application.run_polling(allowed_updates=TgUpdate.ALL_TYPES)

if __name__ == "__main__":
    main()

