# region imports
from AlgorithmImports import *
from lstm_forecast import *
import tensorflow
from tensorflow.keras.models import model_from_json
import json
# endregion

class LSTMWeeklyPrediction(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2022, 12, 1)
        self.set_end_date(2024, 1, 1)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.CASH)
        self.set_cash(100000)
        self.mrk = self.add_equity("MRK", Resolution.DAILY).symbol
        self.set_benchmark(self.mrk)

        self.rolling_window = RollingWindow[float](4000)  
        self.set_warm_up(4000, Resolution.DAILY)

        self.model_name = 'lstm_model'
        self.model_trained = False
        self.prediction = None
        self.last_prediction = None
        
        stock_plot = Chart('Trade Plot')
        stock_plot.add_series(Series('BUY', SeriesType.SCATTER, '$', Color.Green, ScatterMarkerSymbol.TRIANGLE))
        stock_plot.add_series(Series('SELL', SeriesType.SCATTER, '$', Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))
        self.add_chart(stock_plot)
       
        self.train(self.date_rules.month_end(self.mrk), self.time_rules.midnight, self.train_model)
        self.schedule.on(self.date_rules.every_day(self.mrk), self.time_rules.after_market_open(self.mrk, 30), self.trade)

        
    def on_data(self, data: Slice): 
        self.rolling_window.add(self.securities[self.mrk].price)
        

    def trade(self):
        if self.is_warming_up or not self.rolling_window.is_ready or not self.model_trained:
            return
        
        price = self.securities[self.mrk].price
        mrk_df = pd.DataFrame(list(self.rolling_window)[::-1], columns=['close'])
        model = self.load_model(self.model_name)
        self.last_prediction = self.prediction
        self.prediction = forecast(self, mrk_df, model)

        if self.portfolio.invested:    
            if (self.prediction <= self.last_prediction):
                self.set_holdings(self.mrk, 0)
                self.plot('Trade Plot', 'SELL', price)
        
        else:                                                                  
            if (self.last_prediction is not None) and (self.prediction >= self.last_prediction):
                quantity = min(int(0.95 * self.portfolio.margin_remaining / price), int(0.95 * self.portfolio.total_portfolio_value / price))

                if quantity > 0:
                    self.market_order(self.mrk, quantity)
                    self.plot('Trade Plot', 'BUY', price)  

        self.plot('Trade Plot', 'PREDICTION', self.prediction)
        self.plot('Trade Plot', 'Benchmark', price)
     

    def train_model(self):
        if self.is_warming_up or not self.rolling_window.is_ready:
            return

        mrk_df = pd.DataFrame(list(self.rolling_window)[::-1], columns=['close'])
        retrain_model(self, mrk_df, self.model_name)
        self.model_trained = True
        
        
    def load_model(self, model_name='lstm_model'):
        if self.object_store.contains_key(model_name):
            json_config = self.ObjectStore.Read(model_name)
            model = model_from_json(json_config)
            return model
        return None
