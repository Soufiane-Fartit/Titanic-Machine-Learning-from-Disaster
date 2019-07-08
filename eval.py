import numpy as np
import pandas as pd

predicted = pd.read_csv("predicted.csv")
online = pd.read_csv("gender_submission.csv")

predicted = predicted[['Survived']].values
online = online[['Survived']].values

predicted = np.squeeze(predicted)
online = np.squeeze(online)

score = (len(predicted)-sum(abs(predicted-online)))/ float(len(predicted))
print("score : ", score)
