import talib
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class DesignMatrix:

    def __init__(self, df):
        self.X = self._calculateX(df)
        #Y's already shifted so we are predicting tomorrow's change
        self.Y = self._calculateY(df)

    def _calculateX(self, df):
      
        stoch = talib.abstract.STOCH(df).loc[:,"slowk"]
        ad = talib.abstract.ADOSC(df)
        adx = talib.abstract.ADX(df)
        macd = talib.abstract.MACD(df).loc[:,"macd"]

        stoch.name = "STOCH"
        ad.name = "AD"
        adx.name = "ADX"
        macd.name = "MACD"

        return pd.concat([stoch, ad, adx, macd], axis=1)

    def _calculateY(self, df):
        return ((df.loc[:, "close"] - df.loc[:, "open"]).div(df.loc[:,"close"])).shift(-1)

    #e.g. getSliceX("20110101", "20120101", ["MACD", "OBV"])
    def getSliceX(self, start, end, names):
        return self.X.ix[start:end, names]

    def getSliceY(self, start, end):
        return self.Y.ix[start:end]

class Linear():

    def __init__(self):
        self.positions = pd.Series()
        self.pred = pd.Series()

    def run(self, train, y, test):
        mm = StandardScaler() #scale training and test set
        train_trans = pd.DataFrame(mm.fit_transform(train), index=train.index)
        test_trans = pd.DataFrame(mm.transform(test), index=test.index)
        
        lr = LinearRegression()
        #fit to training
        lr.fit(train_trans, y)
        #predict test set
        pred = pd.Series(lr.predict(test_trans), index=test_trans.index)

        #position we will take the NEXT day
        longs = pred.ge(0).astype(int)
        shorts = pred.le(0).astype(int).mul(-1)

        self.pred = pd.concat([self.pred, pred])
        self.positions = pd.concat([self.positions, longs.add(shorts)])

class Tree():

    def __init__(self):
        self.positions = pd.Series()
        self.pred = pd.Series()

    def run(self, train, y, test):
        gb = GradientBoostingRegressor(loss="ls",
                                       n_estimators=200,
                                       learning_rate=.1,
                                       max_depth=3)
        #fit to training
        gb.fit(train, y)
        #predict test set
        pred = pd.Series(gb.predict(test),
                            index=test.index)

        #position we will take the NEXT day
        longs = pred.ge(0).astype(int)
        shorts = pred.le(0).astype(int).mul(-1)
        
        self.pred = pd.concat([self.pred, pred])
        self.positions = pd.concat([self.positions, longs.add(shorts)])


#calculate returns given daily positions, returned named series        
def _returns(changeSeries, positions, tiNames, modelName):
    #shift positions ahead to 1 day to take position tomorrow
    ret = changeSeries.mul(positions.shift(1))
    ret.name = modelName + "_" + "_".join(tiNames)
    return ret
    


#produce dataframe of returns for each model and a column of market returns
#kind can be one of:
#'raw' = daily profit in dollars (used for equity plot)
#'logper' = daily log returns (used for monte carlo permutation)
#'log_detrend' = daily log returns detrended
def returnsDF(trainDays=200, testDays=10, startDate="20000301", endDate="20141101", kind='raw'):

    #read in S&P data, start 60 days earlier since indicators have lag periods
    SP = web.get_data_yahoo("^GSPC", start=pd.to_datetime(startDate) - timedelta(days=60), end=pd.to_datetime(endDate))

    #rename columns to work nicely with talib module
    SP.rename(columns={"Close":"close", "Open":"open",
                       "High":"high", "Low":"low",
                       "Volume":"volume"}, inplace=True)

    #make sure volume is float not int in order to work with talib module
    SP.ix[:,"volume"] = SP.ix[:,"volume"].astype(float)

    #technical indicator combinations
    tiChoices = [["MACD", "STOCH"], ["MACD", "AD"],["MACD", "ADX"],
                  ["STOCH", "AD"], ["STOCH", "ADX"], ["AD", "ADX"],
                  ["MACD","STOCH","AD"],
                  ["STOCH","AD","ADX"],
                  ["AD","ADX","MACD"],
                  ["MACD","STOCH","AD","ADX"]]
    
    trainForward = timedelta(days=trainDays)
    testForward = timedelta(days=testDays)

    if(kind == 'log_detrend'):
        #detrended daily log returns
        change = np.log(1 + (SP.ix[startDate:endDate, 'close'] - SP.ix[startDate:endDate, 'open'])/SP.ix[startDate:endDate, 'open'])
        change = change - np.mean(change)
    elif(kind == 'logper'):
        #daily log returns
        change = np.log(1 + (SP.ix[startDate:endDate, 'close'] - SP.ix[startDate:endDate, 'open'])/SP.ix[startDate:endDate, 'open'])
    else:
        #daily dollar change
        change = (SP.ix[startDate:endDate, 'close'] - SP.ix[startDate:endDate, 'open'])
    
    
    change.name = "SP_Change"
    results = []

    #calculate design matrix including all 4 indicators
    data = DesignMatrix(SP)

    #iterate through the indicator combos and get returns for linear and tree models
    for tiNames in tiChoices:

        print("Calculating returns for {0}...".format((tiNames)))
        start = pd.to_datetime(startDate)
        end = pd.to_datetime(endDate)
        
        trainStart = start
        trainEnd = start + trainForward
        testStart = trainEnd 
        testEnd = testStart + testForward
        
        trainX = data.getSliceX(trainStart, trainEnd, tiNames)
        testX = data.getSliceX(testStart, testEnd, tiNames)
        Y = data.getSliceY(trainStart, trainEnd)

        #initialize linear model and boosted tree objects
        lin = Linear()
        tree = Tree()

        while((end - testEnd).days > 0):

            tree.run(trainX, Y, testX)
            lin.run(trainX, Y, testX)

            #step the time forward
            trainEnd = trainEnd + testForward
            trainStart = trainStart + testForward
            testStart = trainEnd
            testEnd = testStart + testForward

            #get new training and test data
            trainX = data.getSliceX(trainStart, trainEnd, tiNames)
            testX = data.getSliceX(testStart, testEnd, tiNames)
            Y = data.getSliceY(trainStart, trainEnd)

        results.append(_returns(change, lin.positions, tiNames, "Linear"))
        results.append(_returns(change, tree.positions, tiNames, "BoostTree"))
        
    #add in column for actual price changes
    results.append(change)
    return pd.concat(results, axis=1).dropna(how='any');



    
