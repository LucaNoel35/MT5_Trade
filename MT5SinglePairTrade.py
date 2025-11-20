"""
Created on Tue Nov  1 17:33:14 2022

@author: luck3
"""
import math
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import sys
import ta

from datetime import datetime,timezone

nombre =  62151134               
pwd = 'Sephiroth35*'
server_name = 'OANDATMS-MT5'

path_name = r'C:\Program Files\OANDA TMS MT5 Terminal\terminal64.exe'

global_margin=0
global_equity=0
number_of_instrument = 8
total_gain=2
total_loss=1

apply_quota=0
apply_quota_gain=0
apply_quota_loss=0
time_frame=mt5.TIMEFRAME_M1
hold_invertible=True
add_spread=1

value_spread_multiplier=10
max_positions=2
global_leverage=16/max_positions
global_percentage=(0.01/4)*global_leverage
minimal_pip_multiplier=20
minimal_avg_pip_multiplier=25

sample_number=60

# better use combination that includes three currencies like USDJPY and EURJPY (EUR and USD both linked to JPY)
trader1_instrument='USDJPY.pro'
trader2_instrument='EURJPY.pro'

trader3_instrument='EURCAD.pro'
trader4_instrument='USDCAD.pro'

trader5_instrument='AUDJPY.pro'
trader6_instrument='CADJPY.pro'

trader7_instrument='EURCHF.pro'
trader8_instrument='USDCHF.pro'

# adjust weight of each instrument so they have the same effect, basically do a ratio of instrument price on lowest of the instrument prices
# as a rule of thumb, each time we have a instrument with base currency that has higher value ex: (GBP)JPY and (EUR)JPY, we have to apply a weight to GBPJPY 0.5 (something inferior to 1)
#  ex: (GBP)USD and (EUR)USD, we have to apply a weight to GBPUSD 0.5 (something inferior to 1 to match gains and loss of other instruments)
# same thing if for exemple we have EUR(GBP) and EUR(USD), we have to apply a weight to EURGBP 0.5 especially since EUR and USD are both below GBP

Weight1=1
Weight2=1

Weight3=1
Weight4=1

Weight5=1
Weight6=1

Weight7=1
Weight8=1

class ConTrader:
    def __init__(self, instrument,pip,decimal,gain,loss,space,pourcentage,weight):
        self.instrument = instrument
        self.gain=gain
        self.loss=loss
        self.mid_value=(abs(gain)+abs(loss))/2
        self.space=space
        self.pourcentage=pourcentage
        self.pip=pip
        self.decimal=decimal
        self.weight=weight

        self.gain_original=self.gain
        self.loss_original=self.loss
        self.units=0.1
        self.initial_units=0.1 
        self.tick_data = pd.DataFrame()
        self.raw_data = None
        self.data = None 
        self.raw_data_higher = None
        self.last_bar = None

        self.pip_loss = minimal_pip_multiplier*self.pip
        self.spread = minimal_pip_multiplier*self.pip
        self.spread_total = minimal_pip_multiplier*self.pip
        self.spread_count = 1
        self.spread_average = minimal_pip_multiplier*self.pip
        self.bid = 0
        self.ask = 0
        self.count=0        
        self.close=None
        self.first_price=None
        self.price_bought=None
        self.price_sold=None
        self.counter_buy=0
        self.counter_sell=0
        self.add_spread=add_spread
        self.add_spread_origin=add_spread
        #*****************add strategy-specific attributes here******************        

        self.leverage=global_leverage/number_of_instrument
        self.tolerance=0.001
    
        self.s=0
        self.max_level=None
        self.min_level=None
        self.atr=0            
        self.stop_loss=None
        self.take_profit=None
        self.val=value_spread_multiplier*minimal_pip_multiplier*self.pip
        self.price_1=None
        self.price_2=None
        self.price_3=None
        self.price_4=None

        self.position_1=0
        self.position_2=0
        self.position_3=0
        self.position_4=0

        self.positions=None
        self.position=0
        self.initiate=1
        self.quota=False
        self.original_balance=None
        self.global_equity=None
        self.coherence_bool=True     
        self.hold_beginning=-1   
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
        positions = mt5.positions_get(symbol=self.instrument)

        if len(positions) == 0 :
            self.initiate=1
            account_info = mt5.account_info()
            balance = account_info.balance
            self.original_balance=balance
            self.units = max(round((math.floor(((balance / 100000)*self.leverage*self.weight)*100))/100 ,2),0.01)
            self.initial_units = max(round((math.floor(((balance / 100000)*self.leverage*self.weight)*100))/100 ,2),0.01)
        else:
            self.initiate=0
            self.hold_beginning=1
            for position in positions:
                self.units = position.volume
                self.initial_units = position.volume 
                if position.type==0:
                    self.counter_buy+=1
                elif position.type!=0:
                    self.counter_sell+=1
       


    def coherence(self):
        self.coherence_bool=True

        temp_pos=0
        temp_neg=0

        if self.position_1==1:
            temp_pos+=1
        if self.position_2==1:
            temp_pos+=1
        if self.position_3==1:
            temp_pos+=1
        if self.position_4==1:
            temp_pos+=1

        if self.position_1==-1:
            temp_neg+=1
        if self.position_2==-1:
            temp_neg+=1
        if self.position_3==-1:
            temp_neg+=1
        if self.position_4==-1:
            temp_neg+=1

        if temp_neg!= self.counter_sell or temp_pos!=self.counter_buy:
            self.coherence_bool=False
        else:
            self.coherence_bool=True
                    
        if self.coherence_bool==False:
            if self.counter_buy==0 and self.counter_sell==0:
                self.position_1=0
                self.position_2=0
                self.position_3=0
                self.position_4=0

            elif self.counter_buy==0 and self.counter_sell==1:
                self.position_1=-1
                self.position_2=0
                self.position_3=0
                self.position_4=0
                
            elif self.counter_buy==0 and self.counter_sell==2:
                self.position_1=-1
                self.position_2=-1
                self.position_3=0
                self.position_4=0

            #to delete if 2 pos
            elif self.counter_buy==0 and self.counter_sell==3:
                self.position_1=-1
                self.position_2=-1
                self.position_3=-1
                self.position_4=0

            #to delete if 2 pos
            elif self.counter_buy==0 and self.counter_sell==4:
                self.position_1=-1
                self.position_2=-1
                self.position_3=-1
                self.position_4=-1

            elif self.counter_buy==1 and self.counter_sell==0:
                self.position_1=1
                self.position_2=0
                self.position_3=0
                self.position_4=0

            elif self.counter_buy==1 and self.counter_sell==1:
                self.position_1=1
                self.position_2=-1
                self.position_3=0
                self.position_4=0

            #to delete if 2 pos
            elif self.counter_buy==1 and self.counter_sell==2:
                self.position_1=1
                self.position_2=-1
                self.position_3=-1
                self.position_4=0

            #to delete if 2 pos
            elif self.counter_buy==1 and self.counter_sell==3:
                self.position_1=1
                self.position_2=-1
                self.position_3=-1
                self.position_4=-1

            elif self.counter_buy==2 and self.counter_sell==0:
                self.position_1=1
                self.position_2=1
                self.position_3=0
                self.position_4=0

            #to delete if 2 pos
            elif self.counter_buy==2 and self.counter_sell==1:
                self.position_1=1
                self.position_2=1
                self.position_3=-1
                self.position_4=0
            #to delete if 2 pos
            elif self.counter_buy==2 and self.counter_sell==2:
                self.position_1=1
                self.position_2=1
                self.position_3=-1
                self.position_4=-1
            #to delete if 2 pos
            elif self.counter_buy==3 and self.counter_sell==0:
                self.position_1=1
                self.position_2=1
                self.position_3=1
                self.position_4=0
            #to delete if 2 pos
            elif self.counter_buy==3 and self.counter_sell==1:
                self.position_1=1
                self.position_2=1
                self.position_3=1
                self.position_4=-1

            #to delete if 2 pos
            elif self.counter_buy==4 and self.counter_sell==0:
                self.position_1=1
                self.position_2=1
                self.position_3=1
                self.position_4=1


    def get_ask_bid(self):
        info= mt5.symbol_info_tick(self.instrument)
        self.ask = info.ask
        self.bid = info.bid
        self.close=round((self.bid+self.ask)/2,self.decimal)
        self.spread=abs(self.ask-self.bid)
        self.spread_total= self.spread_total + self.spread
        self.spread_count=self.spread_count + 1
        self.spread_average= self.spread_total / self.spread_count  


    def get_most_recent(self, number = sample_number):


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

        self.last_bar = self.raw_data.index[-1]
        previous_time=self.last_bar
        try:
            self.get_most_recent(sample_number)                                                            
            self.prepare_data()
            val=max(value_spread_multiplier*self.spread,self.atr) 
            self.val=val

            if self.positions!=None:
                self.execute_trades(1,-1,"price_1","price_2","position_1","position_2",self.loss,self.gain,self.mid_value,self.loss,-1,1,-1,0)
                self.execute_trades(-1,1,"price_2","price_1","position_2","position_1",self.mid_value,self.loss,self.loss,self.gain,-1,1,1,0)        
                #self.execute_trades(-1,1,"price_3","position_3",self.mid_value,self.loss,-1,1)
                #self.execute_trades(-1,1,"price_4","position_4",self.gain,self.gain,-1,1)
            self.val=val
            if  previous_time!=self.last_bar  :   
                previous_time= self.last_bar     
                phrasing="\n {} {} price 1 {} price 2 {} price 3 {} price 4 {}  position 1 {} position 2 {} position 3 {} position 4 {} \n".format(self.last_bar, self.instrument,self.price_1,self.price_2,self.price_3,self.price_4,self.position_1,self.position_2,self.position_3,self.position_4)             
                print(phrasing)
                self.spread = minimal_pip_multiplier*self.pip
                self.spread_total = minimal_pip_multiplier*self.pip
                self.spread_count = 1
                self.spread_average = ((minimal_pip_multiplier+minimal_avg_pip_multiplier)/2)*self.pip
        except:
            error_message = mt5.last_error()
            print(f"Something went wrong : {error_message} \n")

                           

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


    def objectif_reached_buy(self,price,gain,loss,mode):
        if self.close!=None and price!=None and mode==1:
            return ((abs(self.close-price)>abs(gain)*max((self.val+(self.spread*self.add_spread)),self.atr) and self.close>price) or ((abs(self.close-price)>abs(loss)*max(self.val,self.atr) and self.close<price)))
        if self.close!=None and price!=None and mode==-1:
            return ((abs(self.close-price)>abs(self.space)*max((self.val+(self.spread*self.add_spread)),self.atr) and self.close>price) or (abs(self.close-price)>abs(self.space)*max(self.val,self.atr) and self.close<price))
        else:
            return False

    def objectif_reached_sell(self,price,gain,loss,mode):
        if self.close!=None and price!=None and mode==1:
            return ((abs(self.close-price)>abs(gain)*max((self.val+(self.spread*self.add_spread)),self.atr) and self.close<price) or ((abs(self.close-price)>abs(loss)*max(self.val,self.atr) and self.close>price)))
        if self.close!=None and price!=None and mode==-1:
            return ((abs(self.close-price)>abs(self.space)*max((self.val+(self.spread*self.add_spread)),self.atr) and self.close<price) or (abs(self.close-price)>abs(self.space)*max(self.val,self.atr) and self.close>price))
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

    def getEMA(self,df):
        #self.ichimoku(df)
        df['EMA_5'] = df["c"].ewm(span = 5, min_periods= 5).mean()
        df['EMA_10'] = df["c"].ewm(span = 10, min_periods= 10).mean()
        df['config'] = np.where((df['EMA_5']>df['EMA_10'] ) ,1,0) 
        df['config'] = np.where((df['EMA_5']<df['EMA_10'] ) ,-1,df['config'] )                   
        df['ATR'] = ta.volatility.AverageTrueRange(df['h'],df['l'],df["c"],window=14,fillna=False).average_true_range()   
        self.atr=df["ATR"].iloc[-1]                  

    def prepare_data(self):   
        df = self.raw_data
        self.getEMA(df)

        # get the list of open positions
        account_info = mt5.account_info()
        self.global_equity = account_info.equity
        if self.original_balance==None :
            self.original_balance=account_info.balance
        elif self.global_equity==None:
            self.global_equity = account_info.equity

        if self.original_balance!=0 and self.original_balance!=None and self.global_equity!=None and apply_quota==1:
            #add quota for loss if truely wanted
            if ((self.global_equity/self.original_balance)>(1+(total_gain*self.pourcentage)) and apply_quota_gain==1) or ((self.global_equity/self.original_balance)<(1-(total_loss*self.pourcentage)) and apply_quota_loss==1) :
                self.quota=True
            elif ((self.global_equity/self.original_balance)<((1+(total_gain*self.pourcentage))-self.tolerance) or apply_quota_gain==0) and ((self.global_equity/self.original_balance)>((1-(total_loss*self.pourcentage))+self.tolerance) or apply_quota_loss==0):
                self.quota=False
            

        #print(len(positions))
        self.data=df
        self.positions = mt5.positions_get(symbol=self.instrument)

        if self.quota==True and self.position_1==0 and self.position_2==0 and self.position_3==0 and self.position_4==0  and len(self.positions)==0:
            self.original_balance=account_info.balance
            self.price_1=self.close
            self.price_2=self.close
            self.price_3=self.close
            self.price_4=self.close
            self.setUnits()
            self.initiate=1
            self.hold_beginning=-1            

        if len(self.positions)>0: 
            self.initiate=0
            self.count=0
            self.counter_sell=0
            self.counter_buy=0
            for position in self.positions:
                if position.type==0:
                    self.counter_buy+=1
                else:
                    self.counter_sell+=1

        elif len(self.positions)==0:
            self.count=self.count+1
            if self.count>5:
                self.counter_sell=0
                self.counter_buy=0
                self.position_1=0                
                self.position_2=0                
                self.position_3=0                
                self.position_4=0 
        self.coherence()               

    def execute_trades(self,strat,strat_close,price_name,price_name_2,position_name,position_name_2,gain,loss,gain_2,loss_2,mode_placement,mode_close,safe,inverse):
        price=getattr(self, price_name)
        price_2=getattr(self, price_name_2)
        position_taken=getattr(self, position_name)
        position_taken_2=getattr(self, position_name_2)


        now = datetime.now(timezone.utc)   
        if self.positions==None:
            self.positions = mt5.positions_get(symbol=self.instrument)
        
        positions=self.positions
        df=self.data
        ls= list(df.index)   
        self.position=self.counter_buy+self.counter_sell

        timing=True
        if now.time() >= pd.to_datetime("20:45").time() and now.time() <= pd.to_datetime("22:15").time():
            timing=False
            self.close_position(positions)
            price=self.close
            position_taken=0
            #setattr(self,price_name,price)
            #setattr(self,position_name,position_taken) 
        """
        if self.quota==True and self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip and apply_quota==1:
        """
        if self.quota==True and apply_quota==1 and self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip: 
            self.close_position(positions)
            price=self.close
            position_taken=0
            setattr(self,price_name,price)
            setattr(self,position_name,position_taken) 
              

        if  price==None or self.max_level==None or self.min_level==None or self.first_price==None:               
            price=self.close
            if price_2==None:
                price_2=self.close
            self.first_price=self.close
            self.max_level=df["h"].max()
            self.min_level=df["l"].min()
            setattr(self,price_name,price)
                                                                                                                                    
        if len(positions) < max_positions:  
            """   
            if  self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip and timing : 
            """ 
            if  self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip and timing and self.quota==False : 
                self.max_level=df["h"].max()
                self.min_level=df["l"].min()

                temp=1                
                if (strat==strat_close):
                    temp=temp*self.hold_beginning
                
                
                if  df.at[ls[-1],"config"]==-1*strat*temp and ( (position_taken==0) ) :
                        
                    if (self.counter_buy+self.counter_sell<max_positions) and self.objectif_reached_buy(price,gain,loss,mode_placement):
                        self.sell_order(self.units)
                        self.counter_sell+=1   
                        price=self.close
                        position_taken=-1
                        if temp==-1:
                            self.hold_beginning=1
                        #setattr(self,price_name,price)
                        #setattr(self,position_name,position_taken)

                elif df.at[ls[-1],"config"]==1*strat*temp  and ((position_taken==0)):
                    if (self.counter_buy+self.counter_sell<max_positions) and self.objectif_reached_sell(price,gain,loss,mode_placement): 
                        self.buy_order(self.units)
                        self.counter_buy+=1      
                        price=self.close
                        position_taken=1
                        if temp==-1:
                            self.hold_beginning=-1
                        #setattr(self,price_name,price)
                        #setattr(self,position_name,position_taken)


        if len(positions) >= 1 : 
            if  (self.spread <= minimal_pip_multiplier*self.pip and self.spread_average<minimal_avg_pip_multiplier*self.pip and self.counter_buy==self.counter_sell) or self.counter_buy!=self.counter_sell : 
                for position in positions:                       
                    if (position.type==0) and (position_taken==1):
                    #originally buy position     
                   
                        if   (strat!=strat_close or hold_invertible==False) and (df.at[ls[-1],"config"]==1*strat_close) and (position_taken_2==-1 and safe==-1)  and ((self.close*inverse*strat_close>=inverse*price*strat_close and self.objectif_reached_sell(price_2,gain_2,loss_2,mode_close)) or self.close*inverse*strat_close<price*inverse*strat_close) and self.objectif_reached_buy(price,gain,loss,mode_close)  :  
                            #price=self.close
                            self.close_buy_position(position)   
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None
                        #basically change hold position

                        elif   (strat!=strat_close or hold_invertible==False) and (df.at[ls[-1],"config"]==1*strat_close)  and (position_taken_2!=-1 or safe==1) and self.objectif_reached_buy(price,gain,loss,mode_close)  :  
                            #price=self.close
                            self.close_buy_position(position)   
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None
                        #basically change hold position

                        elif   strat==strat_close and hold_invertible==True and (df.at[ls[-1],"config"]==-1*strat) and self.close*strat_close<self.first_price*strat_close  and self.objectif_reached_buy(self.first_price,gain,loss,mode_close)  :  
                            #price=self.close
                            self.close_buy_position(position)
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None


                    if (position.type!=0) and (position_taken==-1):
                    #originally sell position    

                        if  (strat!=strat_close or hold_invertible==False) and (df.at[ls[-1],"config"]==-1*strat_close)  and (position_taken_2==1 and safe==-1)  and ((self.close*inverse*strat_close<=price*inverse*strat_close and self.objectif_reached_buy(price_2,gain_2,loss_2,mode_close)) or self.close*inverse*strat_close>price*inverse*strat_close) and self.objectif_reached_sell(price,gain,loss,mode_close) :  
                            #price=self.close
                            self.close_sell_position(position)   
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None
                        elif  (strat!=strat_close or hold_invertible==False) and (df.at[ls[-1],"config"]==-1*strat_close)  and (position_taken_2!=1 or safe==1) and self.objectif_reached_sell(price,gain,loss,mode_close) :  
                            #price=self.close
                            self.close_sell_position(position)   
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None
                        #basically change hold position
                        elif  strat==strat_close and hold_invertible==True and (df.at[ls[-1],"config"]==1*strat) and self.close*strat_close>self.first_price*strat_close  and self.objectif_reached_sell(self.first_price,gain,loss,mode_close) :  
                            #price=self.close
                            self.close_sell_position(position)
                            position_taken=0
                            #setattr(self,price_name,price)
                            #setattr(self,position_name,position_taken)
                            self.positions=None

        setattr(self,price_name,price)
        setattr(self,position_name,position_taken)


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


    def close_sell_position(self,position):

        # get the list of open positions
        # iterate over the positions and close them
        counter=0
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
            counter+=1
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {position.ticket} closed successfully")
            else:
                print(f"Failed to close position {position.ticket}. Error code: {result.retcode}, Error description: {result.comment}")
                
    def close_buy_position(self,position):

        # get the list of open positions
        #positions = mt5.positions_get(symbol=self.instrument)
        counter=0
        # iterate over the positions and close them
            # close the position
            
        type=None
        if position.type==0:
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
            counter+=1
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {position.ticket} closed successfully")
            else:
                print(f"Failed to close position {position.ticket}. Error code: {result.retcode}, Error description: {result.comment}")
    
                  
if __name__ == "__main__":
    
    if not mt5.initialize(login = nombre, password = pwd, server = server_name, path = path_name):
        print("initialize() failed")

    trader1 = ConTrader( trader1_instrument, pip=0.001,decimal=3,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight1)    
    trader2 = ConTrader( trader2_instrument, pip=0.001,decimal=3,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight2) 

    trader3 = ConTrader( trader3_instrument, pip=0.00001,decimal=5,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight3)    
    trader4 = ConTrader( trader4_instrument, pip=0.00001,decimal=5,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight4)    

    trader5 = ConTrader( trader5_instrument, pip=0.001,decimal=3,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight5)    
    trader6 = ConTrader( trader6_instrument, pip=0.001,decimal=3,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight6) 

    trader7 = ConTrader( trader7_instrument, pip=0.00001,decimal=5,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight7)    
    trader8 = ConTrader( trader8_instrument, pip=0.00001,decimal=5,gain=2,loss=1,space=0,pourcentage=global_percentage,weight=Weight8)    

    trader1.setUnits()    
    trader2.setUnits() 

    trader3.setUnits()    
    trader4.setUnits()

    trader5.setUnits()    
    trader6.setUnits() 

    trader7.setUnits()    
    trader8.setUnits()


    trader1.get_most_recent(sample_number)    
    trader2.get_most_recent(sample_number)

    trader3.get_most_recent(sample_number)    
    trader4.get_most_recent(sample_number) 

    trader5.get_most_recent(sample_number)    
    trader6.get_most_recent(sample_number)

    trader7.get_most_recent(sample_number)    
    trader8.get_most_recent(sample_number) 


    while True:
        now = datetime.now(timezone.utc)
        
        if now.time() > pd.to_datetime("21:00").time() and now.time() < pd.to_datetime("22:00").time():
            if trader1.position==0 :
                break
        """
        if trader1.quota==True :
            if trader1.position==0 :
                break
        """

        try :         
            trader1.performTrade()   
            trader2.performTrade() 
            
            trader3.performTrade()   
            trader4.performTrade()  

            trader5.performTrade()   
            trader6.performTrade() 
            
            trader7.performTrade()   
            trader8.performTrade()
    
            if mt5.last_error()[0]!=1:
                mt5.initialize(login = nombre, password = pwd, server = server_name, path = path_name)
                print("initialize() failed")

        except:
            print("Trading not active")
            print(mt5.last_error())
    sys.exit()





