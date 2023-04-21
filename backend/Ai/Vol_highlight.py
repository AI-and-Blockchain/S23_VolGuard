import pandas as pd
import numpy as np


dataset = pd.read_csv("/Users/michelleyu/Desktop/Spring2023/CSCI4964/project/S23_VolGuard-main/Ai/hourly.csv")
print("Dataset loaded")

def confidence_interval(num,num_l):
	SE = np.std(num_l)/(len(num_l)**2)
	lower_bound = num - 1.96*SE
	upper_bound = num + 1.96*SE

	return lower_bound,upper_bound

def vol_highlight(dataset,datapoint):

	difference = []

	for d in range(len(dataset["close"])-1):
		diff = abs(dataset["close"].tolist()[d+1] - dataset["close"].tolist()[d])
		
		if len(difference) > datapoint :
			difference.pop()
			lower_bound,upper_bound = confidence_interval(np.mean(difference),difference)

			if diff > upper_bound:
				#print(dataset["close"].tolist()[d+1],dataset["close"].tolist()[d])
				print("TIME:",dataset["timestamp"].tolist()[d+1])
				print("Volatility",diff,"Average Volatility",np.mean(difference))
				print("							")

		difference.append(diff)

vol_highlight(dataset,168)
