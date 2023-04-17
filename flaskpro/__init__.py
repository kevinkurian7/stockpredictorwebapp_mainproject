import os
import requests
import csv

from flask import Flask, render_template, request, flash, redirect, url_for,jsonify
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf

import preprocessor as p
import re

from textblob import TextBlob
import constants as ct
from sklearn.linear_model import LinearRegression
import nltk
nltk.download('punkt')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_page')
def search_page():
    return render_template('search_page.html')   

#news



@app.route('/search')
def search():
 
    query = request.args.get('query')
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&outputsize=full&apikey=9HE15L51KC2VW3UQ'
    response = requests.get(url)
    data = response.json()
    results = []
    try:
        for item in data['bestMatches']:
            result = {
                'symbol': item['1. symbol'],
                'name': item['2. name']
            }
            results.append(result)
        return jsonify(results)
    except:
        return jsonify(results)
@app.route('/results', methods=['POST'])
def results():
    query = request.form['query']
    #news
    def get_company_news(query):
        url = f"https://newsapi.org/v2/everything?q={query}&apiKey=0defce38b4484b929a94879eb7184d17"
        response = requests.get(url)
        news = response.json()
        global articlesd
        articlesd = news["articles"][0:11]
        for article in articlesd:
            print(article["title"])
            print(article["url"])
            print()

    get_company_news(query)



    # query
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='9HE15L51KC2VW3UQ',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol=quote, outputsize='full')
            
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
           
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return
        #LSTM MODEL 



    def LSTM_ALGO(df):
        #Split data into training set and test set
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
       
        training_set=df.iloc[:,4:5].values
   
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))
        training_set_scaled=sc.fit_transform(training_set)
      
        X_train=[]
        y_train=[]#day i
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        #Convert list to numpy arrays
        X_train=np.array(X_train)
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
    
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
       
    
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
       
        regressor=Sequential()
        
        
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))

        regressor.add(Dropout(0.1))
        
       
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
     
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        
     
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        
   
        regressor.add(Dense(units=1))
        
     
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        
        #Training
        regressor.fit(X_train,y_train,epochs=25,batch_size=32 )
        
        real_stock_price=dataset_test.iloc[:,4:5].values
        
        
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        
        
       
        testing_set=sc.transform(testing_set)
       
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
          
        X_test=np.array(X_test)
        
       
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        

        predicted_stock_price=regressor.predict(X_test)
        
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('flaskpro/static/LSTM.PNG')
        plt.close(fig)
        
        
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        

        forecasted_stock_price=regressor.predict(X_forecast)
        
        
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")
        return lstm_pred,error_lstm

     #linear refression   
    def LIN_REG_ALGO(df):
       
        forecast_out = int(7)
        #Price after n days
        df['Close after n days'] = df['Close'].shift(-forecast_out)
       
        df_new=df[['Close','Close after n days']]

       
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
       
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
     
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])
        
       
        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]
        

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        
        #Training
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)
        
        #Testing
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test,label='Actual Price' )
        plt2.plot(y_test_pred,label='Predicted Price')
        
        plt2.legend(loc=4)
        plt2.savefig('flaskpro/static/LR.PNG')
        plt2.close(fig)
        
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        #Forecasting
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr
    

    
    quote=query
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:pass 
    else:
    
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()  
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2


        
        lstm_pred, error_lstm=LSTM_ALGO(df)
        df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        
        
        print("ERROR",error_lstm)
        data={"lstm_pred":round(lstm_pred,2),"error_lstm":round(error_lstm,2)}
        return render_template('results.html', open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),
                               adj_close=today_stock['Adj Close'].to_string(index=False),
                               
                               high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),
                               vol=today_stock['Volume'].to_string(index=False),
                               data=data,articlesd=articlesd,df=df, lr_pred=round(lr_pred,2),
                                forecast_set=forecast_set,quote=quote,
                                
                                  error_lr=round(error_lr,2))

   




@app.route('/ticker')
def ticker():
    with open('flaskpro/templates/ticker.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    return render_template('ticker.html', data=data)

       


if __name__ == '__main__':
    app.run()
   



 
