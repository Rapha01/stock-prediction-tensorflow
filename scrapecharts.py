import json
from pprint import pprint
from urllib.request import urlopen
import numpy
from sklearn.preprocessing import MinMaxScaler

testSetSize = 8
maxEntries = 10000

#pricesJson = urlopen('http://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=spx&apikey=1RXOM5IRKK5OW1U9&outputsize=full').read()
#dict = json.loads(pricesJson)['Time Series (Daily)']
pricesJson = urlopen('https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=CNY&apikey=1RXOM5IRKK5OW1U9&outputsize=full').read()
dict = json.loads(pricesJson)['Time Series (Digital Currency Daily)']
#pprint(json.loads(pricesJson)['Time Series (Digital Currency Daily)']);



data = []
for key, value in dict.items():
    temp = [key,value]
    data.append(temp)

testSet = []
i = 0

for index in data[i:-1]:
    entry = []
    for time,prices in data[i:]:
        #entry.append(float(prices['4. close']))
        entry.append(float(prices['4b. close (USD)']))
        if (len(entry) >= testSetSize):
            scalemin = min(entry)
            scalemax = max(entry)

            for j in range(0,len(entry)):
                entry[j] = (entry[j] - scalemin) / (scalemax - scalemin)
                if entry[j] == 0:
                    entry[j] = 1/2**32
                if entry[j] == 1:
                    entry[j] = 1 - 1/2**32

            if(entry[-2] <= entry[-1]):
                entry[-1] = (1 - (1 / (4 + 8 * (entry[-1]/entry[-2])**8)))
                #entry[-1] = 1
            else:
                entry[-1] = 1 / (4 + 8 * (entry[-2]/entry[-1])**8)

            testSet.append(entry)
            break

    i = i + 1
    if (i > maxEntries):
        break

pprint(testSet)
pprint(len(data))
numpy.savetxt("stocks.csv", testSet, delimiter=",", fmt='%1.4f')
