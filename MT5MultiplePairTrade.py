

"""
Created on Tue Nov  1 17:33:14 2022

@author: luck3
"""
import math
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from threading import Thread
import sys
import ta
import random
import time

from datetime import datetime,timezone

nombre =  62151134               
pwd = 'Sephiroth35*'
server_name = 'OANDATMS-MT5'
path_name = r'C:\Program Files\OANDA TMS MT5 Terminal\terminal64.exe'

thread_running=0
global_margin=0
global_equity=0
number_of_instrument = 8
total_gain=2
total_loss=1
hedge_factor=1.0
time_frame=mt5.TIMEFRAME_M1
base_currency='EUR'
#to allow change direction of hold position
#hold_invertible=False
add_spread=1
apply_quota=0
apply_spread_avg=0
global_leverage=16
global_inverse=1

value_spread_multiplier=10

minimal_pip_multiplier=20
minimal_avg_pip_multiplier=25

correlation_number=60
correlation_multiplier=4
correlation_divider=2

high_correlation_value=0.75
low_correlation_value=high_correlation_value/3

#advised to always use pairs with one currency always present in each of them
#not enough correct currencies to be used correctly
#GBPJPY and CHFJPY too expensive compared to EUR
Watch_List = ['AUDJPY.pro', 'EURJPY.pro','GBPJPY.pro', 'CHFJPY.pro',
              'USDJPY.pro','CADJPY.pro','NZDJPY.pro']

Watch_List_2 = ['AUDUSD.pro', 'EURUSD.pro','GBPUSD.pro', 'USDCHF.pro',
              'USDCAD.pro','NZDUSD.pro','EURGBP.pro','EURCAD.pro','EURCHF.pro']

trader1_instrument='EURJPY.pro'
trader2_instrument='USDJPY.pro'
trader3_instrument='CADJPY.pro'
trader4_instrument='AUDJPY.pro'


# Add more if needed but with a different watchlist if there is not enough pair to compare
trader5_instrument='EURUSD.pro'
trader6_instrument='GBPUSD.pro'
trader7_instrument='AUDUSD.pro'
trader8_instrument='NZDUSD.pro'



class ConTrader:
    def __init__(self, instrument,pip,decimal,strat,strat_close,gain,loss,space,instrument_b,pourcentage,hedge,initialize,beginning,safe):
        self.instrument = instrument
        self.instrument_b = instrument_b
        self.gain=gain
        self.loss=loss
        self.safe=safe
        self.gain_b=gain
        self.loss_b=loss

        self.gain_original=gain
        self.loss_original=loss
        self.pourcentage=pourcentage
        self.pip=pip
        self.pip_b=pip
        self.decimal=decimal
        self.decimal_b=decimal
        self.strat=strat
        self.strat_close=strat_close
        self.strat_org=strat
        self.strat_close_org=strat_close
        self.strat_b=strat
        self.strat_close_b=strat_close
        self.position = 0
        self.position_b = 0
        self.hedge = hedge
        self.hedge_b = hedge
        self.initialize=initialize
        self.initialize_origin=initialize
        self.beginning=beginning
        self.beginning_origin=beginning

        self.units=0.1
        self.initial_units=0.1 
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.raw_data_b = None
        self.data_b = None 
        self.last_bar = None
        self.config=0
        self.config_b=0
        #self.config_bol=0
        #self.config_b_bol=0
        self.PL=0
        self.PL_b=0
        self.PL_tot=0

        self.pip_loss = minimal_pip_multiplier*self.pip
        self.spread = minimal_pip_multiplier*self.pip
        self.spread_total = minimal_pip_multiplier*self.pip
        self.spread_count = 1
        self.spread_average = minimal_pip_multiplier*self.pip
        self.score=0
        self.score_b=0
        self.bid = 0
        self.ask = 0
        self.count=0        
        self.close=None
        self.close_b=None

        self.avg=None
        self.std=None
        #*****************add strategy-specific attributes here******************        

        self.strat_b=self.strat

        self.space=space
        self.leverage=global_leverage/number_of_instrument
        self.tolerance=0.001
    
        self.s=0
        self.max_level=None
        self.min_level=None
        self.atr=0  
        self.atr_avg=0         
        self.stop_loss=None
        self.take_profit=None
        self.val=value_spread_multiplier*minimal_pip_multiplier*self.pip
        self.val_instant=value_spread_multiplier*minimal_pip_multiplier*self.pip
        self.price=None
        self.avg_space=0

        self.quota=False
        self.original_balance=None
        self.global_equity=None

        self.correlation=1
        self.replacement=self.instrument
        self.replacement_b=self.instrument_b
        self.replaced=0
        self.price_bought=None
        self.price_sold=None

        self.instrument_b_obj_reached_sell=False        
        self.instrument_b_obj_reached_buy=False        

        #************************************************************************
                
 
    def setUnits(self):
        if self.instrument in [ 'USDJPY.pro' , 'EURJPY.pro','AUDJPY.pro']:
            self.decimal=3
            self.pip=0.001
        elif self.instrument in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
            self.decimal=3
            self.pip=0.002
        elif self.instrument in ['CHFJPY.pro']:
            self.decimal=3
            self.pip=0.0025
        elif self.instrument in ['EURCAD.pro']:
            self.decimal=5
            self.pip=0.000025
        elif self.instrument in ['EURGBP.pro','EURCHF.pro']:
            self.decimal=5
            self.pip=0.000015
        else:
            self.decimal=5
            self.pip=0.00001

        positions = mt5.positions_get()

        self.close_position(positions)

        account_info = mt5.account_info()
        balance = account_info.balance
        self.original_balance=balance

        hedge_fac=1
        if self.hedge==-1:
            hedge_fac=hedge_factor

        self.units = round(hedge_fac*max(round((math.floor(((balance / 100000)*self.leverage)*100))/100 ,2),0.01),2)
        self.initial_units = round(hedge_fac*max(round((math.floor(((balance / 100000)*self.leverage)*100))/100 ,2),0.01),2)
        print(self.instrument,self.units, "lots")


    def get_ask_bid(self):
        info= mt5.symbol_info_tick(self.instrument)
        self.ask = info.ask
        self.bid = info.bid
        self.close=round((self.bid+self.ask)/2,self.decimal)

        self.spread=abs(self.ask-self.bid)
        self.spread_total= self.spread_total + self.spread
        self.spread_count=self.spread_count + 1
        self.spread_average= self.spread_total / self.spread_count  

        self.score=self.atr_avg/self.spread_average


    def get_most_recent(self, number = correlation_number):

        rates = mt5.copy_rates_from_pos(self.instrument, time_frame, 0, number) 
        df = pd.DataFrame(rates)     
        df.rename(columns = {"close":"c"}, inplace = True)
        df.rename(columns = {"open":"o"}, inplace = True)
        df.rename(columns = {"low":"l"}, inplace = True)
        df.rename(columns = {"high":"h"}, inplace = True)
        self.get_ask_bid()        

        df["time"] = pd.to_datetime(df["time"], unit="s")

        df = df.set_index("time")     
        self.raw_data = df.copy()
 
        self.last_bar = self.raw_data.index[-1]
        self.s =  np.mean(df['h'] - df['l'])                


    def performTrade(self):

        self.get_most_recent()
        self.last_bar = self.raw_data.index[-1]
        self.execute_trades()
        previous_time=self.last_bar

        while True:

            try:
                self.replace_instrument()
                self.get_most_recent(correlation_number)                                                            
                self.execute_trades()

                if  previous_time!=self.last_bar  :   
                    previous_time= self.last_bar     
                    phrasing="\n {} {} {} correlation {} gain {} loss {} strat {} hedge {} \n".format(self.last_bar, self.instrument, self.instrument_b, self.correlation , self.gain, self.loss , self.strat, self.hedge)             
                    print(phrasing)
                    if (self.replacement!=self.instrument):
                        warning_phrase="\n {} still not replaced by {} \n".format(self.instrument, self.replacement)
                        print(warning_phrase)
                    self.spread = minimal_pip_multiplier*self.pip
                    self.spread_total = minimal_pip_multiplier*self.pip
                    self.spread_count = 1
                    self.spread_average = ((minimal_pip_multiplier+minimal_avg_pip_multiplier)/2)*self.pip
                    self.count=0
                time.sleep(0.1)
            except:
                error_message = mt5.last_error()
                print(f"Something went wrong : {error_message} \n")

                           
    def runTrade(self):
        
        self.thread= Thread(target=self.performTrade)
        self.thread.daemon=True   
        self.thread.start()

    def wwma(self,values, n):
        """
        J. Welles Wilder's EMA 
        """
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr_fct(self,df, n=14):
        data = df.copy()
        high = data["h"]
        low = data["l"]
        close = data["c"]
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        df["ATR"]  = self.wwma(tr, n)

    def correlate(self, number = correlation_number):
        data = {}
        if self.raw_data_b is None:
            rates_b = mt5.copy_rates_from_pos(self.instrument_b, time_frame, 0, number) 
            df_b = pd.DataFrame(rates_b) 
            df_b.rename(columns = {"close":"c"}, inplace = True)
            df_b.rename(columns = {"open":"o"}, inplace = True)
            df_b.rename(columns = {"low":"l"}, inplace = True)
            df_b.rename(columns = {"high":"h"}, inplace = True)
            df_b["time"] = pd.to_datetime(df_b["time"], unit="s")
            df_b = df_b.set_index("time")
        else:
            df_b=self.raw_data_b.copy()

        data[self.instrument_b] = df_b['c']
        data[self.instrument] =self.raw_data['c']   
        df = pd.DataFrame(data)
        correlation = df.corr()
        if not correlation.where(correlation < 1).stack().empty:

            max_corr_index = correlation.where(correlation < 1).stack().idxmax()
            corr = correlation.loc[max_corr_index]
            bool= (self.instrument!=self.instrument_b)
            bool_tot= bool
            
            if (corr>low_correlation_value) and bool_tot :
                self.correlation=1
            else:
                self.correlation=0 
 
               
    def highly_correlate(self, number = correlation_number):
        data = {}
        if self.raw_data_b is None:
            rates_b = mt5.copy_rates_from_pos(self.instrument_b, time_frame, 0, number) 
            df_b = pd.DataFrame(rates_b) 
            df_b.rename(columns = {"close":"c"}, inplace = True)
            df_b.rename(columns = {"open":"o"}, inplace = True)
            df_b.rename(columns = {"low":"l"}, inplace = True)
            df_b.rename(columns = {"high":"h"}, inplace = True)
            df_b["time"] = pd.to_datetime(df_b["time"], unit="s")

            df_b = df_b.set_index("time") 
        else:
            df_b=self.raw_data_b.copy()  
  
        data[self.instrument_b] = df_b['c']
        data[self.instrument] =self.raw_data['c']   

        df = pd.DataFrame(data)

        correlation = df.corr()
        if not correlation.where(correlation < 1).stack().empty:

            max_corr_index = correlation.where(correlation < 1).stack().idxmax()
            corr = correlation.loc[max_corr_index] 
            bool= (self.instrument!=self.instrument_b)
            bool_tot= bool
            
            if (corr>high_correlation_value) and bool_tot:
                self.correlation=1
            else:
                self.correlation=0

    def objectif_reached_buy(self,price):
        if self.close!=None and price!=None:
            return ((abs(self.close-price)>self.gain*max((self.val+(self.spread*add_spread)),self.atr) and self.close>price) or (abs(self.close-price)>self.loss*max(self.val,self.atr) and self.close<price))
        else:
            return False

    def objectif_reached_sell(self,price):
        if self.close!=None and price!=None:
            return ((abs(self.close-price)>self.gain*max((self.val+(self.spread*add_spread)),self.atr) and self.close<price) or (abs(self.close-price)>self.loss*max(self.val,self.atr) and self.close>price))
        else:
            return False
        
    def tenkan_sen(self,df):
        # (9-period high + 9-period low)/2))
        df['tenkan_high'] = df['h'].rolling(window=9).max()
        df['tenkan_low'] = df['l'].rolling(window=9).min()
        df['tenkan_sen'] = (df['tenkan_high'] + df['tenkan_low']) / 2

    def kijun_sen(self,df):
        # (26-period high + 26-period low)/2))
        df['kijun_high'] = df['h'].rolling(window=26).max()
        df['kijun_low'] = df['l'].rolling(window=26).min()
        df['kijun_sen'] = (df['kijun_high'] + df['kijun_low']) / 2

    def chikou_span(self,df):
        # Close shifted 26 days to the past
        df['chikou_span'] = df["c"].shift(-26)

    def a_senkou_span(self,df):
        #  (Tenkan Line + Kijun Line)/2))
        self.tenkan_sen(df)
        self.kijun_sen(df)
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    def b_senkou_span(self,df):
        #  (52-period high + 52-period low)/2))
        df['span_b_high'] = df['h'].rolling(window=52).max()
        df['span_b_low'] = df['l'].rolling(window=52).min()
        df['senkou_span_b'] = ((df['span_b_high'] + df['span_b_low']) / 2).shift(26)

    def ichimoku(self,df):
        self.chikou_span(df)
        self.a_senkou_span(df)
        self.b_senkou_span(df)

    def get_macd(self, df, slow, fast, smooth):
        exp1 = df["c"].ewm(span = fast, min_periods= fast).mean()
        exp2 = df["c"].ewm(span = slow, min_periods= slow).mean()
        macd = exp1-exp2
        signal = macd.ewm(span = smooth, min_periods= smooth).mean()
        hist = macd - signal
        df["macd"]=macd
        df["signal"]=signal
        df["hist"]=hist

    def getEMA(self,df):
        #df["SMA_bol"]=df['c'].rolling(20).mean()
        #df["Dist_bol"]=df['c'].rolling(20).std() * 2
        
        #df["Lower_bol"] = df["SMA_bol"] - df['c'].rolling(20).std()  # Lower Band -2 Std Dev
        #df["Upper_bol"] = df["SMA_bol"] + df['c'].rolling(20).std()  # Upper Band -2 Std Dev
        #df['config_bol'] = np.where((df['c']<df['Lower_bol'] ) ,1,0) 
        #df['config_bol'] = np.where((df['c']>df['Upper_bol'] ) ,-1,df['config_bol'] )  


        df['EMA_5'] = df["c"].ewm(span = 5, min_periods= 5).mean()
        df['EMA_15'] = df["c"].ewm(span = 10, min_periods= 10).mean()
        df['EMA_spread']=abs(df['EMA_5']-df['EMA_15'])
        df['EMA_spread_avg']=df["EMA_spread"].ewm(span = 5, min_periods= 5).mean()
        df['EMA_spread_bin']=np.where((df['EMA_spread']>df['EMA_spread_avg'] ),1,0) 

        #df['EMA_21'] = df["c"].ewm(span = 21, min_periods= 21).mean()
        df['config'] = np.where((df['EMA_5']>df['EMA_15'] ),1,0) 
        df['config'] = np.where((df['EMA_5']<df['EMA_15'] ),-1,df['config'] )                  
        df['ATR'] = ta.volatility.AverageTrueRange(df['h'],df['l'],df["c"],window=14,fillna=False).average_true_range()   
        self.avg=df["c"].mean()
        self.std=df["c"].std()

        self.atr=df["ATR"].iloc[-1] 
        self.atr_avg=  df["ATR"].mean() 
        self.config=  df["config"].iloc[-1]     
        #self.config_bol=  df["config_bol"].iloc[-1]   
        self.avg_space = df["EMA_spread_bin"].iloc[-1]

    def execute_trades(self):   
        df = self.raw_data
        self.correlate(correlation_number)
        #ls= list(df.index)   
        self.getEMA(df)
        val=max(value_spread_multiplier*self.spread,self.atr) 
        self.val_instant=val

        # get the list of open positions
        positions = mt5.positions_get(symbol=self.instrument)

        if apply_quota==1:
            account_info = mt5.account_info()
            self.global_equity = account_info.equity
            if self.original_balance==None :
                self.original_balance=account_info.balance
            elif self.global_equity==None:
                self.global_equity = account_info.equity

            if self.original_balance!=0 and self.original_balance!=None and self.global_equity!=None :
                if (self.global_equity/self.original_balance)>(1+(total_gain*self.pourcentage)) or (self.global_equity/self.original_balance)<(1-(total_loss*self.pourcentage))  :
                    self.quota=True
                elif (self.global_equity/self.original_balance)<((1+(total_gain*self.pourcentage))-self.tolerance) and (self.global_equity/self.original_balance)>((1-(total_loss*self.pourcentage))+self.tolerance):
                    self.quota=False

            if self.position==0 and self.position_b==0 and self.quota==True and len(positions)==0:
                self.original_balance=account_info.balance
                self.price=self.close
       
        #print(len(positions))
        now = datetime.now(timezone.utc)

        if self.replaced==1:
            self.correlation=0
            self.price=self.close
            self.replaced=0

        if len(positions) >= 1 : 
            timing=True
            avg=df["c"].mean()
            self.count=0

            if now.time() >= pd.to_datetime("20:45").time() and now.time() <= pd.to_datetime("22:15").time():
                timing=False
                self.close_position(positions)
                self.price=self.close
            
            if self.quota==True and ((self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip)) and apply_quota==1:
                self.close_position(positions)
                self.price=self.close
                self.initialize=self.initialize_origin
                self.beginning=self.beginning_origin
            
            if (positions[0].type==0) :
                self.PL=positions[0].profit
                self.PL_tot=self.PL+self.PL_b
                #originally buy position
                self.position=1
                if self.price==None :
                    self.price=positions[0].price_open 

                    self.max_level=df["h"].max()
                    self.min_level=df["l"].min() 
                    
                if  ((self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip) and self.position_b==-1) or self.position_b!=-1:


                    if  (self.config==1*self.strat_close)  and self.objectif_reached_buy(self.price) and self.config_b==1*self.strat_close and ((self.instrument_b_obj_reached_sell and self.close*global_inverse>self.price*global_inverse) or self.close*global_inverse<self.price*global_inverse) and (self.position_b==-1 and self.safe==-1):  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions)

                    elif  (self.config==1*self.strat_close)  and self.objectif_reached_buy(self.price)  and (self.position_b!=-1 ):  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions)  

                    elif  self.objectif_reached_buy(self.price) and self.correlation==0 and self.position_b==0 and self.instrument_b==self.replacement_b:  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions)  

            else :
                #originally sell position
                self.PL=positions[0].profit
                self.PL_tot=self.PL+self.PL_b
                self.position=-1
                if self.price==None :
                    self.price=positions[0].price_open 
                    self.max_level=df["h"].max()
                    self.min_level=df["l"].min()

                if  ((self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip) and self.position_b==1) or self.position_b!=1:
 
                    if  (self.config==-1*self.strat_close)  and self.objectif_reached_sell(self.price) and self.config_b==-1*self.strat_close and  ((self.instrument_b_obj_reached_buy and self.close*global_inverse<self.price*global_inverse) or self.close*global_inverse>self.price*global_inverse) and (self.position_b==1 and self.safe==-1):  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions)                                                         
                    #basically change hold position
                    
                    elif  (self.config==-1*self.strat_close)  and self.objectif_reached_sell(self.price)  and (self.position_b!=1 or self.safe==1):  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions) 

                    elif  self.objectif_reached_sell(self.price) and self.correlation==0 and self.position_b==0 and self.instrument_b==self.replacement_b:  
                        self.price=self.close
                        self.count=0
                        self.close_position(positions)                    

        elif len(positions) == 0:  
            self.position=0
            self.PL=0
            self.count+=1
            timing=True
            if now.time() > pd.to_datetime("20:45").time() and now.time() < pd.to_datetime("22:15").time() :
                timing=False

            if self.price==None and self.close!=None:
                self.price=self.close 
                self.max_level=df["h"].max()
                self.min_level=df["l"].min()
                
            if  self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip and timing and self.correlation==1 and self.quota==False and ((self.count>5 and self.beginning!=1) or self.beginning==1): 
                
                if  ((self.config==-1*self.strat and (self.avg_space==1 or apply_spread_avg==0) and (self.beginning!=1)) or (self.beginning==1 and self.position_b==1)) and (abs(self.close-self.price)>self.space*self.val or self.initialize==1) :
                    self.sell_order(self.units)
                    self.price=self.close 
                    self.val=val       
                    self.beginning=-1   
                    self.initialize=-1  
                    self.count=0     

                elif ((self.config==1*self.strat and (self.avg_space==1 or apply_spread_avg==0) and (self.beginning!=1)) or (self.beginning==1 and self.position_b==-1)) and (abs(self.close-self.price)>self.space*self.val or self.initialize==1):
                    self.buy_order(self.units)
                    self.price=self.close
                    self.val=val
                    self.beginning=-1    
                    self.initialize=-1   
                    self.count=0      
      

    def sell_order(self,value):

        # Check if symbol is available
        if not mt5.symbol_info(self.instrument).visible:
            print(f"Symbol {self.instrument} is not available")
            return

        # Get the current price of the symbol
        price = mt5.symbol_info_tick(self.instrument).bid
        self.price_sold=price 

        stop_loss=price+2*self.loss*self.val          

        deviation=20
        # Prepare a trade request

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.instrument,
            "volume": value,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            #"sl": stop_loss,  # Stop Loss (optional)
            "deviation": deviation,
        }
        
        # Execute the trade request
        result = mt5.order_send(request)
        if result is None:
            error_message = mt5.last_error()
            print(f"Failed to execute the trade: {error_message}")
        else:
            error_message = mt5.last_error()
            print(f"Message: {error_message}")
            print(f"Trade SELL executed successfully : {self.instrument}")        

    def buy_order(self,value):

        # Check if symbol is available
        if not mt5.symbol_info(self.instrument).visible:
            print(f"Symbol {self.instrument} is not available")
            return

        # Get the current price of the symbol
        price = mt5.symbol_info_tick(self.instrument).ask

        self.price_bought=price
 
        #take_profit=price+self.gain*val
        #stop_loss=price-self.loss*val    
        stop_loss=price-2*self.loss*self.val           
            # Prepare a trade request
        deviation=20          
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.instrument,
            "volume": value,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            #"sl": stop_loss,  # Stop Loss (optional)
            "deviation": deviation,
        }       
        # Execute the trade request
        result = mt5.order_send(request)
        if result is None:
            error_message = mt5.last_error()
            print(f"Failed to execute the trade: {error_message}")
        else:
            error_message = mt5.last_error()
            print(f"Message: {error_message}")
            print(f"Trade BUY executed successfully: {self.instrument}")

    
    def close_position(self,positions):

        # get the list of open positions
        if positions!=None:
            for position in positions:

                # close the position
                type=None

                if position.type==0 :
                    type=mt5.ORDER_TYPE_SELL
                else :
                    type=mt5.ORDER_TYPE_BUY
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "type": type,
                    "position": position.ticket,
                    "volume": position.volume,
                    "price": mt5.symbol_info_tick(position.symbol).bid,
                    "magic": 0,
                    "deviation": 20
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Position {position.ticket} closed successfully")
                else:
                    print(f"Failed to close position {position.ticket}. Error code: {result.retcode}, Error description: {result.comment}")

               

    def close_sell_position(self,positions):

        # get the list of open positions
        # iterate over the positions and close them
        for position in positions:
            # close the position
            type=None

            if position.type!=0 :
                type=mt5.ORDER_TYPE_BUY
            
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "type": type,
                    "position": position.ticket,
                    "volume": position.volume,
                    "price": mt5.symbol_info_tick(position.symbol).bid,
                    "magic": 0,
                    "deviation": 20
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Position {position.ticket} closed successfully")
                else:
                    print(f"Failed to close position {position.ticket}. Error code: {result.retcode}, Error description: {result.comment}")
                    
    def close_buy_position(self,positions):

        # get the list of open positions
        #positions = mt5.positions_get(symbol=self.instrument)
        cnt=0
        # iterate over the positions and close them
        for position in positions:
            # close the position
            type=None

            if position.type==0 :
                type=mt5.ORDER_TYPE_SELL
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "type": type,
                    "position": position.ticket,
                    "volume": position.volume,
                    "price": mt5.symbol_info_tick(position.symbol).bid,
                    "magic": 0,
                    "deviation": 20
                }
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Position {position.ticket} closed successfully")
                else:
                    print(f"Failed to close position {position.ticket}. Error code: {result.retcode}, Error description: {result.comment}")



    def replace_instrument(self):
        #if self.position==0  and self.replacement!=self.instrument and self.replacement_b!=self.instrument_b:               
        if self.position==0 and (self.correlation==0 )  and (self.replacement!=self.instrument) :               
            if self.replacement in [ 'USDJPY.pro' , 'EURJPY.pro','AUDJPY.pro']:
                self.instrument=self.replacement
                self.decimal=3
                self.pip=0.001
                self.replaced=1
                #self.correlation=0 
                self.raw_data_b=None

            elif self.replacement in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
                self.instrument=self.replacement
                self.decimal=3
                self.pip=0.002
                self.replaced=1 
                #self.correlation=0 
                self.raw_data_b=None

            elif self.replacement in ['CHFJPY.pro']:
                self.instrument=self.replacement
                self.decimal=3
                self.pip=0.0025
                self.replaced=1 
                #self.correlation=0
                self.raw_data_b=None

            elif self.replacement in ['EURCAD.pro']:
                self.instrument=self.replacement
                self.decimal=5
                self.pip=0.000025
                self.replaced=1 
                #self.correlation=0
                self.raw_data_b=None

            elif self.replacement in ['EURGBP.pro','EURCHF.pro']:
                self.instrument=self.replacement
                self.decimal=5
                self.pip=0.000015
                self.replaced=1 
                #self.correlation=0
                self.raw_data_b=None
            else:
                self.instrument=self.replacement
                self.decimal=5
                self.pip=0.00001
                self.replaced=1
                #self.correlation=0 
                self.raw_data_b=None

            if (self.replacement_b!=self.instrument_b):               
                self.instrument_b=self.replacement_b


    def replace(self,instrument,instrument_b,ls ):
        if (instrument not in ls) and instrument!=instrument_b:
            self.replacement=instrument
            self.replacement_b=instrument_b


    def place_info(self,trader):

        self.position_b=trader.position
        self.raw_data_b=trader.raw_data.copy()
        self.strat_b=trader.strat
        self.strat_close_b=trader.strat_close
        self.close_b=trader.close
        self.hedge_b=trader.hedge
        self.pip_b=trader.pip
        self.decimal_b=trader.decimal
        self.config_b=trader.config
        self.score_b=trader.score

        self.loss_b=trader.loss
        self.gain_b=trader.gain

        #self.config_b_bol=trader.config_bol
        self.PL_b=trader.PL
        self.instrument_b_obj_reached_buy=trader.objectif_reached_buy(trader.price)
        self.instrument_b_obj_reached_sell=trader.objectif_reached_sell(trader.price)
                
        if trader.instrument!=self.instrument_b:
            self.instrument_b=trader.instrument
            self.replacement_b=trader.instrument
        
        if (self.position==self.position_b and (self.strat==self.strat_b or self.strat_close==self.strat_close_b)) :
            self.strat=-self.strat_b
            self.strat_close=-self.strat_close_b
        else:
            self.strat=self.strat_org
            self.strat_close=self.strat_close_org
        

    
    def emergency_change_instrument(self,Watchlist,ls):
        if (self.instrument in ls) and self.position==0:
            temp=random.choice(Watchlist)
            if temp not in ls:
                if temp in [ 'USDJPY.pro' , 'EURJPY.pro','AUDJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.001
                    self.replaced=1
                elif temp in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.002
                    self.replaced=1   
                elif temp in ['CHFJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.0025
                    self.replaced=1  
                elif temp in ['EURGBP.pro','EURCHF.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.000015
                    self.replaced=1    
                elif temp in ['EURCAD.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.000025
                    self.replaced=1              
                else:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.00001
                    self.replaced=1

    def random_change_instrument(self,Watchlist,ls):
        if  self.position==0:
            temp=random.choice(Watchlist)
            if temp not in ls :
                if temp in [ 'USDJPY.pro' , 'EURJPY.pro','AUDJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.001
                    self.replaced=1
                elif temp in ['NZDJPY.pro','GBPJPY.pro','CADJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.002
                    self.replaced=1   
                elif temp in ['CHFJPY.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=3
                    self.pip=0.0025
                    self.replaced=1   
                elif temp in ['EURCAD.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.000025
                    self.replaced=1    
                elif temp in ['EURGBP.pro','EURCHF.pro']:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.000015
                    self.replaced=1             
                else:
                    self.instrument=temp
                    self.replacement=temp
                    self.decimal=5
                    self.pip=0.00001
                    self.replaced=1
        

"""
Modify code below to add new traders.

Don't forget to modify correlation matrix function to take into account all traders instrument name. 

Go to line 58 to add more instrument names

Seperate into several watch list if needed
"""

#not used in the end
def correlation_matrix(trader1,trader2,ls,watchlist):
    data = {}
    for symbol in watchlist:
        rates = mt5.copy_rates_from_pos(symbol, time_frame, 0, round(correlation_number*correlation_multiplier/correlation_divider))
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        data[symbol] = df['close']

    df = pd.DataFrame(data)
    correlation = df.corr()

    for i in correlation.index:
        for j in correlation.columns:
            if ((i[-7:] != j[-7:] and i[:3] != j[:3])) or (i in ls or j in ls):
                correlation.at[i, j] = np.nan
    
    if trader1.position==0 and trader2.position==0:
        max_corr_index = correlation.where(correlation < 1).stack().idxmax()
    elif trader1.position!=0 and trader2.position==0:
        max_corr_index = (trader1.instrument,correlation.loc[trader1.instrument].drop(trader1.instrument).idxmax())
    elif trader1.position==0 and trader2.position!=0:
        max_corr_index = (correlation.loc[trader2.instrument].drop(trader2.instrument).idxmax(),trader2.instrument)
    else:
        max_corr_index = correlation.where(correlation < 1).stack().idxmax()

    most_correlated_pair = max_corr_index[0], max_corr_index[1]
    correlation_value = correlation.loc[max_corr_index]
    trader1.replace(max_corr_index[0],max_corr_index[1],ls)
    trader2.replace(max_corr_index[1],max_corr_index[0],ls)

    

if __name__ == "__main__":
    
    if not mt5.initialize(login = nombre, password = pwd, server = server_name, path = path_name):
        print("initialize() failed")

    trader1 = ConTrader( trader1_instrument,  pip=0.001,decimal=3,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader2_instrument,pourcentage=0.02,hedge=-1,initialize=1,beginning=-1,safe=-1) 
    trader2 = ConTrader( trader2_instrument,  pip=0.001,decimal=3,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader1_instrument,pourcentage=0.02,hedge=1,initialize=1,beginning=1,safe=-1)
    trader3 = ConTrader( trader3_instrument,  pip=0.001,decimal=3,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader4_instrument,pourcentage=0.02,hedge=-1,initialize=1,beginning=1,safe=-1)
    trader4 = ConTrader( trader4_instrument,  pip=0.001,decimal=3,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader3_instrument,pourcentage=0.02,hedge=1,initialize=1,beginning=-1,safe=-1)
    
    trader5 = ConTrader( trader5_instrument,  pip=0.00001,decimal=5,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader6_instrument,pourcentage=0.02,hedge=1,initialize=1,beginning=-1,safe=-1) 
    trader6 = ConTrader( trader6_instrument,  pip=0.00001,decimal=5,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader5_instrument,pourcentage=0.02,hedge=-1,initialize=1,beginning=1,safe=-1)
    trader7 = ConTrader( trader7_instrument,  pip=0.00001,decimal=5,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader8_instrument,pourcentage=0.02,hedge=1,initialize=1,beginning=1,safe=-1)
    trader8 = ConTrader( trader8_instrument,  pip=0.00001,decimal=5,strat=-1,strat_close=1,gain=1,loss=1,space=0,instrument_b=trader7_instrument,pourcentage=0.02,hedge=-1,initialize=1,beginning=-1,safe=-1)
    #gain*mid_level and loss*mid_level -> ref
    trader1.setUnits()    
    trader2.setUnits()
    trader3.setUnits()
    trader4.setUnits()

    trader5.setUnits()    
    trader6.setUnits()
    trader7.setUnits()
    trader8.setUnits()
    
    trader1.get_most_recent(correlation_number*correlation_multiplier)    
    trader2.get_most_recent(correlation_number*correlation_multiplier)
    trader3.get_most_recent(correlation_number*correlation_multiplier)
    trader4.get_most_recent(correlation_number*correlation_multiplier)

    trader5.get_most_recent(correlation_number*correlation_multiplier)    
    trader6.get_most_recent(correlation_number*correlation_multiplier)
    trader7.get_most_recent(correlation_number*correlation_multiplier)
    trader8.get_most_recent(correlation_number*correlation_multiplier)
    

    trader1.highly_correlate(correlation_number*correlation_multiplier)    
    trader2.highly_correlate(correlation_number*correlation_multiplier)
    trader3.highly_correlate(correlation_number*correlation_multiplier)
    trader4.highly_correlate(correlation_number*correlation_multiplier)

    trader5.highly_correlate(correlation_number*correlation_multiplier)    
    trader6.highly_correlate(correlation_number*correlation_multiplier)
    trader7.highly_correlate(correlation_number*correlation_multiplier)
    trader8.highly_correlate(correlation_number*correlation_multiplier)
    

    print("correlation ",trader1.instrument,trader1.correlation)
    print("correlation ",trader2.instrument,trader2.correlation)
    print("correlation ",trader3.instrument,trader3.correlation)
    print("correlation ",trader4.instrument,trader4.correlation)
    
    print("correlation ",trader5.instrument,trader5.correlation)
    print("correlation ",trader6.instrument,trader6.correlation)
    print("correlation ",trader7.instrument,trader7.correlation)
    print("correlation ",trader8.instrument,trader8.correlation)
    

    if (trader1.correlation==0 and trader1.replacement==trader1.instrument and trader1.position==0) or (trader2.correlation==0 and trader2.replacement==trader2.instrument and trader2.position==0):
        correlation_matrix(trader1,trader2,[trader3_instrument,trader4_instrument],Watch_List)
        #trader1.random_change_instrument(trader2,trader3,Watch_List)
        trader1.replace_instrument()    
        trader2.replace_instrument() 
        trader1_instrument=trader1.instrument
        trader2_instrument=trader2.instrument  


    if (trader3.correlation==0 and trader3.replacement==trader3.instrument and trader3.position==0) or (trader4.correlation==0 and trader4.replacement==trader4.instrument and trader4.position==0):
        correlation_matrix(trader3,trader4,[trader2_instrument,trader1_instrument],Watch_List)
        #trader3.random_change_instrument(trader2,trader1,Watch_List)
        trader3.replace_instrument()    
        trader4.replace_instrument() 
        trader3_instrument=trader3.instrument
        trader4_instrument=trader4.instrument
    

    if (trader5.correlation==0 and trader5.replacement==trader5.instrument and trader5.position==0) or (trader6.correlation==0 and trader6.replacement==trader6.instrument and trader6.position==0):
        correlation_matrix(trader5,trader6,[trader7_instrument,trader8_instrument],Watch_List_2)
        #trader1.random_change_instrument(trader2,trader3,Watch_List)
        trader5.replace_instrument()    
        trader6.replace_instrument() 
        trader5_instrument=trader5.instrument
        trader6_instrument=trader6.instrument  


    if (trader7.correlation==0 and trader7.replacement==trader7.instrument and trader7.position==0) or (trader8.correlation==0 and trader8.replacement==trader8.instrument and trader8.position==0):
        correlation_matrix(trader7,trader8,[trader5_instrument,trader6_instrument],Watch_List_2)
        #trader3.random_change_instrument(trader2,trader1,Watch_List)
        trader7.replace_instrument()    
        trader8.replace_instrument() 
        trader7_instrument=trader7.instrument
        trader8_instrument=trader8.instrument
    

    thread1=trader1.runTrade()   
    thread2=trader2.runTrade()
    thread3=trader3.runTrade()
    thread4=trader4.runTrade()
    
    thread5=trader5.runTrade()   
    thread6=trader6.runTrade()
    thread7=trader7.runTrade()
    thread8=trader8.runTrade()
    
    
    while True:
        now = datetime.now(timezone.utc)
        """
        if now.time() > pd.to_datetime("21:00").time() and now.time() < pd.to_datetime("22:00").time():
            break 
        """
        
        if trader1.quota==True and trader2.quota==True and trader3.quota==True and trader4.quota==True:
            if trader1.position==0 and trader2.position==0 and trader3.position==0 and trader4.position==0:
                break
        
        if mt5.last_error()[0]!=1:
            mt5.initialize(login = nombre, password = pwd, server = server_name, path = path_name)
            print("initialize() failed")

        try :             
            thread_running = 1  

            if (trader1.correlation==0 and trader1.replacement==trader1.instrument) or (trader2.correlation==0 and trader2.replacement==trader2.instrument):
                #trader1.random_change_instrument(trader2,trader3,Watch_List)
                correlation_matrix(trader1,trader2,[trader3_instrument,trader4_instrument],Watch_List)
                #print("Replacement for trader 1 and 2 necessary")

            if (trader3.correlation==0 and trader3.replacement==trader3.instrument) or (trader4.correlation==0 and trader4.replacement==trader4.instrument) :
                #trader3.random_change_instrument(trader2,trader1,Watch_List)
                correlation_matrix(trader3,trader4,[trader2_instrument,trader1_instrument],Watch_List)
                #print("Replacement for trader 3 and 4 necessary")
            
            if (trader5.correlation==0 and trader5.replacement==trader1.instrument) or (trader6.correlation==0 and trader6.replacement==trader6.instrument):
                #trader1.random_change_instrument(trader2,trader3,Watch_List)
                correlation_matrix(trader5,trader6,[trader7_instrument,trader8_instrument],Watch_List_2)
                #print("Replacement for trader 1 and 2 necessary")

            if (trader7.correlation==0 and trader7.replacement==trader7.instrument) or (trader8.correlation==0 and trader8.replacement==trader8.instrument) :
                #trader3.random_change_instrument(trader2,trader1,Watch_List)
                correlation_matrix(trader7,trader8,[trader5_instrument,trader6_instrument],Watch_List_2)
                #print("Replacement for trader 3 and 4 necessary")
            

            trader1_instrument=trader1.instrument
            trader2_instrument=trader2.instrument
            trader3_instrument=trader3.instrument
            trader4_instrument=trader4.instrument
            
            trader5_instrument=trader5.instrument
            trader6_instrument=trader6.instrument
            trader7_instrument=trader7.instrument
            trader8_instrument=trader8.instrument
            

            trader1.place_info(trader2)
            trader2.place_info(trader1)
            trader3.place_info(trader4)
            trader4.place_info(trader3)
            
            trader5.place_info(trader6)
            trader6.place_info(trader5)
            trader7.place_info(trader8)
            trader8.place_info(trader7)
            

            trader1.emergency_change_instrument(Watch_List,[trader2_instrument,trader3_instrument,trader4.instrument,trader1.instrument_b])
            trader2.emergency_change_instrument(Watch_List,[trader1_instrument,trader3_instrument,trader4.instrument,trader2.instrument_b])
            trader3.emergency_change_instrument(Watch_List,[trader2_instrument,trader1_instrument,trader4.instrument,trader3.instrument_b])
            trader4.emergency_change_instrument(Watch_List,[trader3_instrument,trader2_instrument,trader1_instrument,trader4.instrument_b])
            
            trader5.emergency_change_instrument(Watch_List_2,[trader6_instrument,trader7_instrument,trader8.instrument,trader5.instrument_b])
            trader6.emergency_change_instrument(Watch_List_2,[trader5_instrument,trader7_instrument,trader8.instrument,trader6.instrument_b])
            trader7.emergency_change_instrument(Watch_List_2,[trader5_instrument,trader6_instrument,trader8.instrument,trader7.instrument_b])
            trader8.emergency_change_instrument(Watch_List_2,[trader5_instrument,trader6_instrument,trader7_instrument,trader8.instrument_b])
            
        except:
            print("Trading not active")
            print(mt5.last_error())
            thread_running =0
    sys.exit()
