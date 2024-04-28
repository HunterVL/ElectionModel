import pandas as pd
import numpy as np
from itertools import combinations

# The number of polls in each sample
maxPollCount = 50

resDf = pd.read_csv(f"Data/Polling/Results.csv")
X_train = []
y_train = []
for electionID, electionData in resDf.groupby("Election"):
    # Load the dataframe for a given election
    ddf = pd.read_csv(f"Data/Polling/Frm{electionID}.csv")
    # We combine results from different parties and add them together for data augmentation
    for comboAmounts in range(1, 4):
        for partyIndicies in combinations(range(len(electionData)), min(comboAmounts, len(electionData)-1)):
            partyNames = electionData["Party"].iloc[list(partyIndicies)] # type: ignore
            X = np.zeros((maxPollCount, 3))
            X[:, 1] = -730 # Set the date to a far-away date
            y = 0
            nPolls = min(len(ddf), maxPollCount)
            X[-nPolls:, :2] = ddf[["Sample size", "T-Days"]].iloc[:maxPollCount]
            for partyIndex, partyName in zip(partyIndicies, partyNames):
                X[-nPolls:, 2] += ddf[partyName].iloc[:maxPollCount]
                y += electionData["Result"].iloc[partyIndex]
                
            if y >= 100: continue
            X_train += [np.expand_dims(X.astype(np.float32), axis=0)]
            y_train += [np.expand_dims(y, axis=0) / 100]

X_train = np.vstack(X_train)
y_train = np.vstack(y_train)

# Change the scale of the data
X_train[:, :, 0] = np.sqrt(X_train[:, :, 0])
X_train[:, :, 1] = 0.95**-X_train[:, :, 1]
X_train[:, :, 2] /= 100

np.save('Outputs/X.npy', X_train)
np.save('Outputs/y.npy', y_train)