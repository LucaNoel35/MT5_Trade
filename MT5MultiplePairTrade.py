"""
Optimized MT5 multi-pair trading bot
- Centralized market/tick cache (1 call per symbol per loop)
- Centralized positions cache (1 call total per loop)
- Centralized rates cache with smart throttling (only refresh when a new bar appears)
- Single scheduler loop (no per-trader busy threads)
- Traders read-only views over caches; only order_send makes API calls
- Correlation checks throttled

NOTE: Credentials are left as variables. Consider securing them via env vars.
"""

import math
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sys
import ta
import random
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

# =========================
# ==== USER CONFIG =========
# =========================

# ⚠️ Move these to environment variables in production
nombre =  62219880              
pwd = 'Sephiroth35*'
server_name = 'OANDATMS-MT5'
path_name = r'C:\Program Files\OANDA TMS MT5 Terminal\terminal64.exe'

number_of_instrument = 8

total_gain = 2
total_loss = 1
hedge_factor = 1.0

time_frame = mt5.TIMEFRAME_M1
base_currency = 'EUR'

add_spread = 1
apply_quota = 0
apply_spread_avg = 0

global_leverage = 16
value_spread_multiplier = 10
minimal_pip_multiplier = 20
minimal_avg_pip_multiplier = 25

correlation_number = 120
correlation_multiplier = 4
correlation_divider = 2

#correlation inversed (-1) means high risk high reward, and vice versa
correlation_inverse=1
high_correlation_value = 0.75
low_correlation_value = high_correlation_value/3

corr_by_name=1

selection_condition_buy_sell=-1

selection_gain_loss=2

gain_plus=2
loss_plus=1
gain_minus=2
loss_minus=1

if selection_gain_loss==1:
  gain_plus=2
  loss_plus=1
  gain_minus=1.5
  loss_minus=1.5
elif selection_gain_loss==2:
  gain_plus=1.5
  loss_plus=1
  gain_minus=1
  loss_minus=2

# Time to wait to check double instrument in s
time_check_double = 5.0

# Time to wait to check neutral position
time_check_position = 1.0

# US and Japanese market
Watch_List = ['AUDJPY.pro', 'EURJPY.pro','GBPJPY.pro', 'CHFJPY.pro',
              'USDJPY.pro','CADJPY.pro','NZDJPY.pro','AUDUSD.pro', 
              'EURUSD.pro','GBPUSD.pro', 'USDCHF.pro',
              'USDCAD.pro','NZDUSD.pro']

trader1_instrument='EURJPY.pro'
trader2_instrument='USDJPY.pro'
trader3_instrument='CADJPY.pro'
trader4_instrument='AUDJPY.pro'

# Add more if needed (watch list 2)
trader5_instrument='EURUSD.pro'
trader6_instrument='GBPUSD.pro'
trader7_instrument='AUDUSD.pro'
trader8_instrument='NZDUSD.pro'

# =========================
# ==== MARKET MANAGER =====
# =========================

class MarketManager:
    """Centralizes *all* MT5 calls. Traders consume cached data only."""

    def __init__(self, symbols: List[str], timeframe=time_frame, rates_depth=correlation_number*correlation_multiplier):
        self.symbols = sorted(set(symbols))
        self.timeframe = timeframe
        self.rates_depth = rates_depth
        self.ticks: Dict[str, object] = {}
        self.positions: Dict[str, object] = {}
        self.rates: Dict[str, pd.DataFrame] = {}
        self.last_bar_time: Dict[str, int] = {}  # unix seconds of last bar per symbol
        self._last_positions_pull = 0.0
        self._last_ticks_pull = 0.0
        self._last_rates_pull = 0.0

        # throttling
        self.tick_interval_s = 1.0           # pull ticks at most once per second
        self.positions_interval_s = 1.0      # pull positions once per second
        self.rates_interval_s = 1.0          # scan bars once every 2s (per symbol new-bar check)

    # --------- pulls ---------
    def pull_ticks(self):
        now = time.time()
        if now - self._last_ticks_pull < self.tick_interval_s:
            return
        new_ticks = {}
        for s in self.symbols:
            t = mt5.symbol_info_tick(s)
            if t:
                new_ticks[s] = t
        self.ticks = new_ticks
        self._last_ticks_pull = now

    def pull_positions(self):
        now = time.time()
        if now - self._last_positions_pull < self.positions_interval_s:
            return
        pos_list = mt5.positions_get() or []
        self.positions = {p.symbol: p for p in pos_list}
        self._last_positions_pull = now

    def _refresh_rates_for_symbol(self, s: str):
        # fetch N bars and keep as df with columns c/o/h/l + time index
        rates = mt5.copy_rates_from_pos(s, self.timeframe, 0, self.rates_depth)
        if rates is None or len(rates) == 0:
            return
        df = pd.DataFrame(rates)
        df.rename(columns={"close":"c","open":"o","low":"l","high":"h"}, inplace=True)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        self.rates[s] = df
        # keep last bar time as integer seconds for quick compare
        self.last_bar_time[s] = int(df.index[-1].timestamp())

    def pull_rates(self):
        now = time.time()
        if now - self._last_rates_pull < self.rates_interval_s:
            return
        # only refresh symbols where we detect a new bar OR we don't have data yet
        for s in self.symbols:
            last = self.last_bar_time.get(s)
            # peek the latest bar time via 1-bar pull to avoid full N fetch
            rr = mt5.copy_rates_from_pos(s, self.timeframe, 0, 1)
            if rr is None or len(rr) == 0:
                continue
            last_time = int(rr[0]['time'])
            if last is None or last_time != last:
                self._refresh_rates_for_symbol(s)
        self._last_rates_pull = now

    # --------- accessors ---------
    def get_tick(self, symbol: str):
        return self.ticks.get(symbol)

    def get_mid(self, symbol: str, decimals: int) -> Optional[float]:
        t = self.get_tick(symbol)
        if not t:
            return None
        return round((t.bid + t.ask)/2, decimals)

    def get_positions(self, symbol: str):
        p = self.positions.get(symbol)
        return [p] if p else []

    def get_rates(self, symbol: str) -> Optional[pd.DataFrame]:
        return self.rates.get(symbol)


# =========================
# ===== TRADER CLASS ======
# =========================

class ConTrader:
    def __init__(self, mm: MarketManager, instrument, pip, decimal, strat, strat_close, gain, loss, space,
                 instrument_b, pourcentage, hedge, initialize, first_run, safe, inverse):
        self.mm = mm
        self.instrument = instrument
        self.instrument_b = instrument_b
        self.gain = gain
        self.loss = loss
        self.safe = safe
        self.gain_b = gain
        self.loss_b = loss
        self.inverse = inverse

        self.gain_original = gain
        self.loss_original = loss
        self.pourcentage = pourcentage
        self.pip = pip
        self.pip_b = pip
        self.decimal = decimal
        self.decimal_b = decimal
        self.strat = strat
        self.strat_close = strat_close
        self.strat_org = strat
        self.strat_close_org = strat_close
        self.strat_b = strat
        self.strat_close_b = strat_close
        self.position = 0
        self.previous_position = 0
        self.latest_seen_position = 0
        self.position_b = 0
        self.hedge = hedge
        self.hedge_b = hedge
        self.initialize = initialize
        self.initialize_origin = initialize
        self.beginning = 1
        self.beginning_origin = self.beginning

        self.units = 0.1
        self.initial_units = 0.1
        self.raw_data: Optional[pd.DataFrame] = None
        self.raw_data_b: Optional[pd.DataFrame] = None
        self.last_bar = None
        self.config = 0
        self.config_b = 0
        self.PL = 0
        self.PL_b = 0
        self.PL_tot = 0

        self.pip_loss = minimal_pip_multiplier * self.pip
        self.spread = minimal_pip_multiplier * self.pip
        self.spread_total = minimal_pip_multiplier * self.pip
        self.spread_count = 1
        self.spread_average = minimal_pip_multiplier * self.pip
        self.score = 0
        self.score_b = 0
        self.bid = 0
        self.ask = 0
        self.count = 0
        self.close = None
        self.close_b = None

        self.avg = None
        self.std = None

        self.space = space
        self.leverage = global_leverage / number_of_instrument
        self.tolerance = 0.001

        self.atr = 0
        self.atr_avg = 0
        self.stop_loss = None
        self.take_profit = None
        self.val = value_spread_multiplier * minimal_pip_multiplier * self.pip
        self.val_instant = value_spread_multiplier * minimal_pip_multiplier * self.pip
        self.price = None
        self.price_b = None

        self.avg_space = 0
        self.quota = False
        self.original_balance = None
        self.global_equity = None

        self.correlation = 1
        self.replacement = self.instrument
        self.replacement_b = self.instrument_b
        self.replaced = 0
        self.price_bought = None
        self.price_sold = None

        self.instrument_b_obj_reached_sell = False
        self.instrument_b_obj_reached_buy = False

        # throttle heavy calcs
        self._last_corr_check = 0.0
        self.corr_interval_s = 10.0  # check correlation at most every 10s


        self.emergency=1
        self.double_instrument=0
        self.first_run=first_run
        self.first_run_origin = first_run



    # ---------- utils ----------

    def setUnits(self, watchlist, assigned_symbols=None):
        if assigned_symbols is None:
            assigned_symbols = []

        account_info = mt5.account_info()
        balance = account_info.balance if account_info else 0
        self.original_balance = balance

        # Liste des symboles possibles pour ce trader
        possible_symbols = [s for s in watchlist if s not in assigned_symbols]

        # Cherche position existante sur symboles disponibles
        for s in possible_symbols:
            positions = self.mm.get_positions(s)
            if positions:
                p = positions[0]
                self.instrument = s
                self.position = 1 if p.type == mt5.ORDER_TYPE_BUY else -1
                self.latest_seen_position = self.position
                self.price = p.price_open
                self.units = p.volume
                self.initial_units = self.units
                self._set_pip_decimal(self.instrument)
                print(f"{self.instrument} assigned from existing position, lots={self.units}")
                self.beginning = -1
                self.first_run=0
                assigned_symbols.append(self.instrument)
                return

        # Aucun position existante → assigner premier symbole dispo
        if possible_symbols:
            self.instrument = possible_symbols[0]
        # fallback si plus rien dispo
        else:
            self.instrument = self.instrument

        hedge_fac = hedge_factor if self.hedge == -1 else 1
        lots = max(round((math.floor(((balance / 100000) * self.leverage) * 100)) / 100, 2), 0.01)
        self.units = round(hedge_fac * lots, 2)
        self.initial_units = self.units
        self.position = 0
        self.latest_seen_position = 0
        self.price = None
        self._set_pip_decimal(self.instrument)
        print(f"{self.instrument} no existing position, lots={self.units}")
        self.beginning = 1
        assigned_symbols.append(self.instrument)

    # ---------- Helper pour assigner decimal/pip ----------
    def _set_pip_decimal(self, instrument):
        if instrument in ['USDJPY.pro','EURJPY.pro','AUDJPY.pro']:
            self.decimal = 3; self.pip = 0.001
        elif instrument in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
            self.decimal = 3; self.pip = 0.002
        elif instrument in ['CHFJPY.pro']:
            self.decimal = 3; self.pip = 0.0025
        elif instrument in ['EURCAD.pro']:
            self.decimal = 5; self.pip = 0.000025
        elif instrument in ['EURGBP.pro','EURCHF.pro']:
            self.decimal = 5; self.pip = 0.000015
        else:
            self.decimal = 5; self.pip = 0.00001



    def _update_spread(self, tick):
        self.ask = tick.ask
        self.bid = tick.bid
        self.close = round((self.bid + self.ask)/2, self.decimal)
        self.spread = abs(self.ask - self.bid)
        self.spread_total += self.spread
        self.spread_count += 1
        self.spread_average = self.spread_total / self.spread_count
        # quality score
        if self.atr_avg:
            self.score = self.atr_avg / max(self.spread_average, 1e-12)

    def get_most_recent(self):
        df = self.mm.get_rates(self.instrument)
        if df is None or df.empty:
            return
        self.raw_data = df
        #self.last_bar = self.raw_data.index[-1]

    # ---------- indicators ----------
    @staticmethod
    def wwma(values, n):
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr_fct(self, df, n=14):
        data = df.copy()
        high = data["h"]; low = data["l"]; close = data["c"]
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0','tr1','tr2']].max(axis=1)
        df["ATR"] = self.wwma(tr, n)

    def correlate(self):
        now = time.time()
        if now - self._last_corr_check < self.corr_interval_s:
            return
        if self.raw_data is None or self.raw_data_b is None:
            self.correlation = 0
            self._last_corr_check = now
            return
        data = {
            self.instrument_b: self.raw_data_b['c'],
            self.instrument: self.raw_data['c']
        }
        df = pd.DataFrame(data)
        corr_df = df.corr()
        if corr_df.where(corr_df < 1).stack().empty:
            self.correlation = 0
        else:
            max_corr_index = corr_df.where(corr_df < 1).stack().idxmax()
            corr = corr_df.loc[max_corr_index]
            self.correlation = 1 if corr*correlation_inverse > low_correlation_value*correlation_inverse and (self.instrument != self.instrument_b) else 0
        self._last_corr_check = now

    def highly_correlate(self):
        if self.raw_data is None or self.raw_data_b is None:
            self.correlation = 0
            return
        data = {
            self.instrument_b: self.raw_data_b['c'],
            self.instrument: self.raw_data['c']
        }
        df = pd.DataFrame(data)
        corr_df = df.corr()
        if corr_df.where(corr_df < 1).stack().empty:
            self.correlation = 0
        else:
            max_corr_index = corr_df.where(corr_df < 1).stack().idxmax()
            corr = corr_df.loc[max_corr_index]
            self.correlation = 1 if corr*correlation_inverse > high_correlation_value*correlation_inverse and (self.instrument != self.instrument_b) else 0

    def getEMA(self, df: pd.DataFrame):
        df['EMA_5'] = df['c'].ewm(span=5, min_periods=5).mean()
        df['EMA_10'] = df['c'].ewm(span=10, min_periods=10).mean()
        df['EMA_spread'] = (df['EMA_5'] - df['EMA_10']).abs()
        df['EMA_spread_avg'] = df['EMA_spread'].ewm(span=5, min_periods=5).mean()
        df['EMA_spread_bin'] = np.where((df['EMA_spread'] > df['EMA_spread_avg']), 1, 0)
        df['config'] = np.where((df['EMA_5'] > df['EMA_10']), 1, 0)
        df['config'] = np.where((df['EMA_5'] < df['EMA_10']), -1, df['config'])
        df['ATR'] = ta.volatility.AverageTrueRange(df['h'], df['l'], df['c'], window=14, fillna=False).average_true_range()

        self.avg = df['c'].mean(); self.std = df['c'].std()
        self.atr = df['ATR'].iloc[-1]
        self.atr_avg = df['ATR'].mean()
        self.config = int(df['config'].iloc[-1])
        self.avg_space = int(df['EMA_spread_bin'].iloc[-1])

    # ---------- objectives ----------
    def objectif_reached_buy(self, price):
        if self.close is None or price is None:
            return False
        return ((abs(self.close-price) > self.gain*max((self.val+(self.spread*add_spread)), self.atr) and self.close > price)
                or (abs(self.close-price) > self.loss*max(self.val, self.atr) and self.close < price))

    def objectif_reached_sell(self, price):
        if self.close is None or price is None:
            return False
        return ((abs(self.close-price) > self.gain*max((self.val+(self.spread*add_spread)), self.atr) and self.close < price)
                or (abs(self.close-price) > self.loss*max(self.val, self.atr) and self.close > price))

    # ---------- trading core ----------
    def execute_trades(self):
        # ticks
        tick = self.mm.get_tick(self.instrument)
        if not tick:
            return
        self._update_spread(tick)

        # rates
        self.get_most_recent()
        if self.raw_data is None:
            return
        self.getEMA(self.raw_data)
        self.val_instant = max(value_spread_multiplier * self.spread, self.atr)
        val = self.val_instant

        if self.replaced == 1:
            self.price = self.close
            self.replaced =0

        # counterpart info from paired trader (set by place_info)
        # correlation check throttled
        self.correlate()

        # positions from cache
        positions = self.mm.get_positions(self.instrument)

        if apply_quota == 1:
            account_info = mt5.account_info()
            self.global_equity = getattr(account_info, 'equity', None)
            if self.original_balance is None and account_info:
                self.original_balance = account_info.balance
            if self.original_balance and self.global_equity:
                ratio = self.global_equity / self.original_balance
                if ratio > (1 + (total_gain*self.pourcentage)) or ratio < (1 - (total_loss*self.pourcentage)):
                    self.quota = True
                elif (1 - (total_loss*self.pourcentage) + self.tolerance) < ratio < (1 + (total_gain*self.pourcentage) - self.tolerance):
                    self.quota = False
            if self.position == 0 and self.position_b == 0 and self.quota and len(positions) == 0:
                self.original_balance = account_info.balance if account_info else self.original_balance
                self.price = self.close

        now = datetime.now(timezone.utc)
        # reset spread averaging on each new bar (approx via last_bar change)
        if self.raw_data is not None and (self.last_bar is not None) and (self.raw_data.index[-1] == self.last_bar):
            pass
        else:
            self.spread = minimal_pip_multiplier*self.pip
            self.spread_total = minimal_pip_multiplier*self.pip
            self.spread_count = 1
            self.spread_average = ((minimal_pip_multiplier+minimal_avg_pip_multiplier)/2)*self.pip
            self.count = 0
            self.last_bar = self.raw_data.index[-1]
            phrasing="\n {} {} {} correlation {} gain {} loss {} strat {} hedge {} position {} position_b {}\n".format(self.last_bar, self.instrument, self.instrument_b, self.correlation , self.gain, self.loss , self.strat, self.hedge , self.position, self.position_b)             
            print(phrasing)

        # Trading windows
        def within_quiet_hours():
            return pd.to_datetime("20:45").time() <= now.time() <= pd.to_datetime("22:15").time()

        if len(positions) >= 1:
            timing = not within_quiet_hours()
            self.count = 0

            p0 = positions[0]
            self.latest_seen_position = 1 if p0.type == 0 else -1
            self.PL = p0.profit
            self.PL_tot = self.PL + self.PL_b
            self.position = 1 if p0.type == 0 else -1
            if self.price is None:
                self.price = p0.price_open

            if not timing:
                self.close_position(positions)
                self.price = self.close
                return
            
            if self.double_instrument>time_check_double and self.position!=0 and ((p0.type == 0 and self.objectif_reached_buy(self.price)) or (p0.type != 0 and self.objectif_reached_sell(self.price))):
                self.close_position(positions)
                self.price = self.close
                return       

            if self.beginning==1:
                self.beginning = -1
              
            if self.first_run!=0:
                self.first_run=0
            # closing logic mirrors original but uses cached values
            if p0.type == 0:  # BUY open
                cond_ok_spread = ((self.spread <= minimal_pip_multiplier*self.pip and self.spread_average < minimal_avg_pip_multiplier*self.pip) and self.position_b == -1) or self.position_b != -1
                if selection_condition_buy_sell==1:
                  cond_ok_buy_b = ((self.instrument_b_obj_reached_sell and self.close*self.inverse >= self.price*self.inverse and ((self.config_b == 1*self.strat_close and self.strat_close!=self.strat_close_b) or (self.config_b == -1*self.strat_close and self.strat_close==self.strat_close_b))) or self.close*self.inverse < self.price*self.inverse)
                else:
                  cond_ok_buy_b = ((self.config_b == 1*self.strat_close and self.strat_close!=self.strat_close_b) or (self.config_b == -1*self.strat_close and self.strat_close==self.strat_close_b)) and((self.instrument_b_obj_reached_sell and self.close*self.inverse >= self.price*self.inverse ) or self.close*self.inverse < self.price*self.inverse)

                if cond_ok_spread:
                    if (self.config == 1*self.strat_close and self.objectif_reached_buy(self.price) and cond_ok_buy_b and (self.position_b == -1 and self.safe == -1)):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = 1
                    elif (self.config == 1*self.strat_close and self.objectif_reached_buy(self.price) and (self.position_b != -1 or self.safe != -1)):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = 1
                    elif (self.objectif_reached_buy(self.price) and self.correlation == 0 and self.position_b == 0 and self.instrument_b == self.replacement_b):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = 1
            else:  # SELL open
                cond_ok_spread = ((self.spread <= minimal_pip_multiplier*self.pip and self.spread_average < minimal_avg_pip_multiplier*self.pip) and self.position_b == 1) or self.position_b != 1
                if selection_condition_buy_sell==1:
                  cond_ok_sell_b = ((self.instrument_b_obj_reached_buy and self.close*self.inverse <= self.price*self.inverse and ((self.config_b == -1*self.strat_close and self.strat_close!=self.strat_close_b) or (self.config_b == 1*self.strat_close and self.strat_close==self.strat_close_b))) or self.close*self.inverse > self.price*self.inverse)
                else:
                  cond_ok_sell_b = ((self.config_b == -1*self.strat_close and self.strat_close!=self.strat_close_b) or (self.config_b == 1*self.strat_close and self.strat_close==self.strat_close_b)) and ((self.instrument_b_obj_reached_buy and self.close*self.inverse <= self.price*self.inverse ) or self.close*self.inverse > self.price*self.inverse)
                  
                if cond_ok_spread:
                    if (self.config == -1*self.strat_close and self.objectif_reached_sell(self.price)  and cond_ok_sell_b and (self.position_b == 1 and self.safe == -1)):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = -1
                    elif (self.config == -1*self.strat_close and self.objectif_reached_sell(self.price) and (self.position_b != 1 or self.safe != -1)):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = -1
                    elif (self.objectif_reached_sell(self.price) and self.correlation == 0 and self.position_b == 0 and self.instrument_b == self.replacement_b):
                        self.price = self.close; self.count = 0; self.close_position(positions); 
                        if self.space == 0: self.previous_position = -1
        elif len(positions) ==0 :
            # no open position for this symbol
            self.count += 1
            timing = not (pd.to_datetime("20:45").time() < now.time() < pd.to_datetime("22:15").time())
            if self.price is None and self.close is not None:
                self.price = self.close
            if self.count > time_check_position:
                self.position = 0; self.PL = 0

            can_trade = (self.spread <= minimal_pip_multiplier*self.pip and self.spread_average < minimal_avg_pip_multiplier*self.pip and timing and self.correlation == 1 and self.emergency == 0 and self.double_instrument==0 and (not self.quota) and ((self.count > time_check_position and self.beginning != 1) or self.beginning == 1) and self.instrument!=self.instrument_b and self.position==0)
            if can_trade:
                # sell setup
                cond_sell = (((self.config == -1*self.strat and (self.previous_position != self.latest_seen_position or self.previous_position == 0)) or (self.previous_position == 1 and self.previous_position == self.latest_seen_position)) and (self.avg_space == 1 or apply_spread_avg == 0) and (self.beginning != 1)) or (self.beginning == 1 and self.position_b == 1) or (self.first_run==-1 and self.position_b == 0)
                cond_buy = (((self.config == 1*self.strat and (self.previous_position != self.latest_seen_position or self.previous_position == 0)) or (self.previous_position == -1 and self.previous_position == self.latest_seen_position)) and (self.avg_space == 1 or apply_spread_avg == 0) and (self.beginning != 1)) or (self.beginning == 1 and self.position_b == -1) or (self.first_run==1 and self.position_b == 0)
                far_enough = (abs(self.close - self.price) > self.space*val) or (self.initialize == 1)
                if cond_sell and far_enough:
                    self.sell_order(self.units)
                    self.price = self.close; self.val = val; self.beginning = -1; self.initialize = -1; self.count = 0; self.first_run=0
                elif cond_buy and far_enough:
                    self.buy_order(self.units)
                    self.price = self.close; self.val = val; self.beginning = -1; self.initialize = -1; self.count = 0; self.first_run=0

    # ---------- orders ----------
    def sell_order(self, value):
        tick = self.mm.get_tick(self.instrument)
        if not tick:
            return
        price = tick.bid
        self.price_sold = price
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.instrument,
            "volume": value,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
        }
        result = mt5.order_send(request)
        if result is None:
            print(f"Failed SELL {self.instrument}: {mt5.last_error()}")
        else:
            print(f"SELL ok {self.instrument}: ret={getattr(result,'retcode',None)}")

    def buy_order(self, value):
        tick = self.mm.get_tick(self.instrument)
        if not tick:
            return
        price = tick.ask
        self.price_bought = price
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.instrument,
            "volume": value,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "deviation": 20,
        }
        result = mt5.order_send(request)
        if result is None:
            print(f"Failed BUY {self.instrument}: {mt5.last_error()}")
        else:
            print(f"BUY ok {self.instrument}: ret={getattr(result,'retcode',None)}")

    def reset(self):
        self.beginning=self.beginning_origin      
        self.initialize=self.initialize_origin
        self.first_run=self.first_run_origin
      

    def close_position(self, positions):
        if not positions:
            return
        for p in positions:
            close_type = mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY
            tick = self.mm.get_tick(p.symbol)
            if not tick:
                continue
            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "type": close_type,
                "position": p.ticket,
                "volume": p.volume,
                "price": price,
                "magic": 0,
                "deviation": 20,
            }
            result = mt5.order_send(request)
            if getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
                print(f"Closed {p.ticket} {p.symbol}")
            else:
                print(f"Close failed {p.ticket} {p.symbol}: code={getattr(result,'retcode',None)} {getattr(result,'comment',None)}")

    # ---------- pairing & maintenance ----------
    def replace_instrument(self):
        # adjust decimals/pips when switching
        if self.position==0 and ((self.correlation==0) or (self.emergency==1)) and (self.replacement!=self.instrument) :
            temp = self.replacement
            if temp in ['USDJPY.pro','EURJPY.pro','AUDJPY.pro']:
                self.instrument = temp; self.decimal = 3; self.pip = 0.001
            elif temp in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
                self.instrument = temp; self.decimal = 3; self.pip = 0.002
            elif temp in ['CHFJPY.pro']:
                self.instrument = temp; self.decimal = 3; self.pip = 0.0025
            elif temp in ['EURCAD.pro']:
                self.instrument = temp; self.decimal = 5; self.pip = 0.000025
            elif temp in ['EURGBP.pro','EURCHF.pro']:
                self.instrument = temp; self.decimal = 5; self.pip = 0.000015
            else:
                self.instrument = temp; self.decimal = 5; self.pip = 0.00001
            self.replaced = 1
            self.raw_data_b = None
            self._last_corr_check  = 0.0
            if self.replacement_b != self.instrument_b:
                self.instrument_b = self.replacement_b

    def replace(self, instrument, instrument_b, ls):
        if (instrument not in ls) and instrument != instrument_b:
            self.replacement = instrument
            self.replacement_b = instrument_b

    def place_info(self, trader_b):
        self.position_b = trader_b.position
        self.raw_data_b = None if trader_b.raw_data is None else trader_b.raw_data.copy()
        self.strat_b = trader_b.strat
        self.strat_close_b = trader_b.strat_close
        self.close_b = trader_b.close
        self.price_b = trader_b.price
        self.hedge_b = trader_b.hedge
        self.pip_b = trader_b.pip
        self.decimal_b = trader_b.decimal
        self.config_b = trader_b.config
        self.score_b = trader_b.score
        self.loss_b = trader_b.loss
        self.gain_b = trader_b.gain
        self.PL_b = trader_b.PL
        self.instrument_b_obj_reached_buy = trader_b.objectif_reached_buy(trader_b.price)
        self.instrument_b_obj_reached_sell = trader_b.objectif_reached_sell(trader_b.price)
        if trader_b.instrument != self.instrument_b:
            self.instrument_b = trader_b.instrument
            self.replacement_b = trader_b.instrument

    def emergency_change_instrument(self, Watchlist, ls):
        if (self.instrument in ls) :
            self.emergency=1
            self.double_instrument+=1
            temp = random.choice(Watchlist)
            if temp not in ls:
                self.replacement = temp
                self.replace_instrument()
        elif (self.instrument==self.instrument_b) or (self.instrument not in Watchlist ) or (self.instrument_b not in Watchlist ):
            self.emergency=1
            temp = random.choice(Watchlist)
            if temp not in ls:
                self.replacement = temp
                self.replace_instrument()
        else:
            self.emergency=0
            self.double_instrument=0


    def random_change_instrument(self, Watchlist, ls):
        if self.position == 0:
            temp = random.choice(Watchlist)
            if temp not in ls:
                self.replacement = temp
                self.replace_instrument()


# =========================
# ===== CORRELATION =======
# =========================

def correlation_matrix(mm: MarketManager, trader1: ConTrader, trader2: ConTrader, ls, watchlist):
    """Compute correlation only once centrally using cached bars; minimize MT5 calls."""
    data = {}
    for symbol in watchlist:
        df = mm.get_rates(symbol)
        if df is None:
            # fallback: single refresh for missing symbol
            mm._refresh_rates_for_symbol(symbol)
            df = mm.get_rates(symbol)
        if df is not None and not df.empty:
            data[symbol] = df['c']
    if not data:
        return
    df_all = pd.DataFrame(data)
    corr = df_all.corr()

    # mask invalid pairs
    if corr_by_name==1:
        if correlation_inverse==1:
            for i in corr.index:
                for j in corr.columns:
                    if ((i[-7:] != j[-7:] and i[:3] != j[:3])) or (i in ls or j in ls):
                        corr.at[i, j] = np.nan

        else:
            for i in corr.index:
                for j in corr.columns:
                    if ((i[-7:] == j[-7:] or i[:3] == j[:3])) or (i in ls or j in ls):
                        corr.at[i, j] = np.nan    
    
                
    if correlation_inverse==1:
        if trader1.position == 0 and trader2.position == 0:
            max_corr_index = corr.where(corr < 1).stack().idxmax()
        elif trader1.position != 0 and trader2.position == 0:
            max_corr_index = (trader1.instrument, corr.loc[trader1.instrument].drop(trader1.instrument).idxmax())
        elif trader1.position == 0 and trader2.position != 0:
            max_corr_index = (corr.loc[trader2.instrument].drop(trader2.instrument).idxmax(), trader2.instrument)
        else:
            max_corr_index = corr.where(corr < 1).stack().idxmax()
    else:
        if trader1.position == 0 and trader2.position == 0:
            max_corr_index = corr.where(corr < 1).stack().idxmin()
        elif trader1.position != 0 and trader2.position == 0:
            max_corr_index = (trader1.instrument, corr.loc[trader1.instrument].drop(trader1.instrument).idxmin())
        elif trader1.position == 0 and trader2.position != 0:
            max_corr_index = (corr.loc[trader2.instrument].drop(trader2.instrument).idxmin(), trader2.instrument)
        else:
            max_corr_index = corr.where(corr < 1).stack().idxmin()

    trader1.replace(max_corr_index[0], max_corr_index[1], ls)
    trader2.replace(max_corr_index[1], max_corr_index[0], ls)


# =========================
# ======== MAIN ===========
# =========================

if __name__ == "__main__":
    if not mt5.initialize(login=nombre, password=pwd, server=server_name, path=path_name):
        print("initialize() failed")
        sys.exit(1)

    all_symbols = sorted(set(Watch_List + [
        trader1_instrument, trader2_instrument, trader3_instrument, trader4_instrument,
        trader5_instrument, trader6_instrument, trader7_instrument, trader8_instrument
    ]))

    mm = MarketManager(all_symbols)

    # Preload once to fill caches
    mm.pull_ticks(); mm.pull_positions(); mm.pull_rates()

    # Instantiate traders
    trader1 = ConTrader(mm, trader1_instrument, 0.001,3,  1,-1, gain_minus,  loss_minus,0, trader2_instrument,0.02,-1,1,1,-1,-1)
    trader2 = ConTrader(mm, trader2_instrument, 0.001,3, -1, 1, gain_plus , loss_plus,0, trader1_instrument,0.02, 1,1, -1,-1,-1)
    trader3 = ConTrader(mm, trader3_instrument, 0.001,3,  1,-1, gain_minus,  loss_minus,0, trader4_instrument,0.02,-1,1, -1,-1,-1)
    trader4 = ConTrader(mm, trader4_instrument, 0.001,3, -1, 1, gain_plus, loss_plus,0, trader3_instrument,0.02, 1,1,1,-1,-1)

    trader5 = ConTrader(mm, trader5_instrument, 0.00001,5, 1,-1, gain_minus,  loss_minus,0, trader6_instrument,0.02, 1,1,1,-1,-1)
    trader6 = ConTrader(mm, trader6_instrument, 0.00001,5,-1, 1, gain_plus , loss_plus,0, trader5_instrument,0.02,-1,1, -1,-1,-1)
    trader7 = ConTrader(mm, trader7_instrument, 0.00001,5, 1,-1, gain_minus,  loss_minus,0, trader8_instrument,0.02, 1,1, -1,-1,-1)
    trader8 = ConTrader(mm, trader8_instrument, 0.00001,5,-1, 1, gain_plus , loss_plus ,0, trader7_instrument,0.02,-1,1,1,-1,-1)

    traders = [trader1,trader2,trader3,trader4,trader5,trader6,trader7,trader8]

    assigned_symbols = []
    for t in traders:
        t.setUnits(Watch_List,assigned_symbols)

    # Prime correlation state with preloaded bars
    for t in traders:
        t.get_most_recent()
    # high correlation warmup
    for t in traders:
        t.highly_correlate()

    print("Warmup correlation:")
    for idx, t in enumerate(traders, start=1):
        print(f"Trader{idx} {t.instrument} corr={t.correlation}")

    # Main scheduler loop
    while True:
        now = datetime.now(timezone.utc)
        
        # stop conditions (same as original, but more compact)
        
        if pd.to_datetime("21:00").time() < now.time() < pd.to_datetime("22:00").time():
            for t in traders:
                t.reset()
                time.sleep(5.0)
            #break
        
                

        # keep MT5 session alive / re-init if needed
        if mt5.last_error()[0] != 1:
            mt5.initialize(login=nombre, password=pwd, server=server_name, path=path_name)

        try :             
            # central pulls
            mm.pull_ticks()
            mm.pull_positions()
            mm.pull_rates()

            # correlation maintenance (throttled via trader.corr_interval_s)
            # Pair 1
            if (trader1.correlation == 0 and trader1.replacement == trader1.instrument) or (trader2.correlation == 0 and trader2.replacement == trader2.instrument):
                correlation_matrix(mm, trader1, trader2, [trader3.instrument, trader4.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument], Watch_List)
            # Pair 2
            if (trader3.correlation == 0 and trader3.replacement == trader3.instrument) or (trader4.correlation == 0 and trader4.replacement == trader4.instrument):
                correlation_matrix(mm, trader3, trader4, [trader2.instrument, trader1.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument], Watch_List)
            # Pair 3
            if (trader5.correlation == 0 and trader5.replacement == trader5.instrument) or (trader6.correlation == 0 and trader6.replacement == trader6.instrument):
                correlation_matrix(mm, trader5, trader6, [trader7.instrument, trader8.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument], Watch_List)
            # Pair 4
            if (trader7.correlation == 0 and trader7.replacement == trader7.instrument) or (trader8.correlation == 0 and trader8.replacement == trader8.instrument):
                correlation_matrix(mm, trader7, trader8, [trader5.instrument, trader6.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument], Watch_List)

            # propagate counterpart info
            trader1.place_info(trader2); trader2.place_info(trader1)
            trader3.place_info(trader4); trader4.place_info(trader3)
            trader5.place_info(trader6); trader6.place_info(trader5)
            trader7.place_info(trader8); trader8.place_info(trader7)

            # emergency changes (use watchlists)
            trader1.emergency_change_instrument(Watch_List,[trader2.instrument,trader3.instrument,trader4.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument])
            trader2.emergency_change_instrument(Watch_List,[trader1.instrument,trader3.instrument,trader4.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument])
            trader3.emergency_change_instrument(Watch_List,[trader2.instrument,trader1.instrument,trader4.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument])
            trader4.emergency_change_instrument(Watch_List,[trader3.instrument,trader2.instrument,trader1.instrument,trader5.instrument,trader6.instrument,trader7.instrument,trader8.instrument])

            trader5.emergency_change_instrument(Watch_List,[trader6.instrument,trader7.instrument,trader8.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument])
            trader6.emergency_change_instrument(Watch_List,[trader5.instrument,trader7.instrument,trader8.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument])
            trader7.emergency_change_instrument(Watch_List,[trader5.instrument,trader6.instrument,trader8.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument])
            trader8.emergency_change_instrument(Watch_List,[trader5.instrument,trader6.instrument,trader7.instrument,trader1.instrument,trader2.instrument,trader3.instrument,trader4.instrument])

            # execute decisions
            for t in traders:
                t.execute_trades()

            # allow instrument replacement when safe
            for t in traders:
                t.replace_instrument()

            # pace with timeframe; 1s is enough for M1
            time.sleep(1.0)
        except:
            print("Trading not active")
            print(mt5.last_error())

    sys.exit(0)
