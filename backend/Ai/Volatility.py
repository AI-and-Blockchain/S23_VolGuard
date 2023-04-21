import pandas as pd
import matplotlib.pyplot as plt

#vol = standard deviation/ number of periods in the time horizon

dataset = pd.read_csv("/../min_data.csv", index_col = 'timestamp', parse_dates=True)
print("Dataset loaded")
Volatility = []


index = 15    #every 15 min
while True:
	Volatility.append((dataset[:index]["close"].std()/15))
	index += 15
	if index > len(dataset):
		break
df = pd.DataFrame(Volatility,columns =["Volatility"])


plt.plot(df.index,df)
plt.title('Volatility')
plt.show()
