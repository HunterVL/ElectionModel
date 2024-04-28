from keras.models import load_model
import pandas as pd
import numpy as np
from datetime import datetime

get_days = lambda dateStr: (datetime.strptime(dateStr, "%Y-%m-%d")-datetime.today()).days
clean_sample_size = lambda N: int(N.split(' ', 1)[0].replace(',', ''))

model = load_model('Outputs/my_model.keras')

poll_data = pd.read_csv("Data/CanadaNationalPolls.csv", sep="\t")
poll_data = poll_data.iloc[49::-1]
poll_data["T-Days"] = poll_data["Date"].map(get_days)
# Set the date to be the day after the last poll
poll_data["T-Days"] -= poll_data["T-Days"].max() + 1
poll_data["Sample size"] = poll_data["Sample size"].map(clean_sample_size)
parties = ["LPC", "CPC", "NDP", "BQ", "GPC", "PPC"]

X = np.zeros((6, 50, 3))
X[:, :, :2] = poll_data[["Sample size", "T-Days"]]
X[:, :, 2] = poll_data[parties].T

# Change the scale of the data
X[:, :, 0] = np.sqrt(X[:, :, 0])
X[:, :, 1] = 0.95**-X[:, :, 1]
X[:, :, 2] /= 100

# Forecast the popular vote of the next election
y_pred = model.predict(X, verbose=0) # type: ignore

# Ensure the popular vote adds up to 100%
next_election_pred = y_pred.T / y_pred.sum()

# Load the riding results of the last election
parties = ['Lib', 'Con', 'NDP', 'BQ', 'Green', 'PPC']
election_df = pd.read_csv("Data/2021ElectionResults.csv")
election_df = election_df[['Riding', 'Prov/Terr'] + parties]
election_df_cleaner = lambda x: int(str(x).replace(",","").replace("–","0") if x == x else "0")
for column in parties:
    election_df[column] = election_df[column].map(election_df_cleaner)

election_riding_results = np.array(election_df[parties])
election_national_results = np.array(election_df[parties]).sum(axis=0)

# Forecast the votes in each riding based on proportional swing
votes = election_riding_results * (next_election_pred / election_national_results)
votes /= votes.sum(axis=1, keepdims=1)

# Create a DataFrame of ridings and their winners
riding_victors = pd.DataFrame({
    "Riding": election_df["Riding"], 
    "Winner": np.array(parties)[votes.argmax(axis=1)]
})
riding_victors.to_csv("Outputs/RidingWinnerProjections.csv", index=False)

# Print and save the results
print(riding_victors)
print(next_election_pred)
np.save('Outputs/RidingVoteProjections.npy', votes)