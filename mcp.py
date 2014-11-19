
import pandas as pd
import numpy as np


#positionDF = data frame of the positions where each column corresponds to a model
#returnSeries = actual return series for the market
#n = number of iterations
def MCP(positionDF, returnSeries, n):

    posArr = positionDF.values
    #make copy since np.suffle() works in place
    retArr = returnSeries.copy(deep=True).values

    mcpRet = []

    while(n > 0):
        #shuffle positions and get return series for each
        np.random.shuffle(retArr)
        permRet = np.apply_along_axis(lambda(x): np.multiply(retArr,x) , 0, posArr)
        mcpRet.append(np.max(permRet.mean(axis=0)))
        n -= 1

    return mcpRet

#positionReturnDF = data frame of detrended log daily returns where each column corresponds to a model
#n = number of bootstrap samples
def boot(positionReturnDF, n):
    indices = positionReturnDF.index

    results = []

    means = positionReturnDF.mean(axis=0)
    
    while(n > 0):
        perm = np.random.choice(indices, replace=True, size=len(indices))
        m = positionReturnDF.ix[perm, :].mean(axis=0) - means
        return(m)
        results.append(np.max(m))
        n -= 1

    return results
    