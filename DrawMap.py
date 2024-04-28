import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd

hoc_map = gpd.read_file("Data/FED_CA_2021_EN.zip")

color_map = {"Lib": "#EA6D6A", "Con": "#6495ED", "NDP": "#F4A460", 
             "BQ": "#87CEFA", "Green": "#99C955", "PPC": "#6F5D9A"}

# Get the colours for each riding
projection_winners = pd.read_csv("Outputs/RidingWinnerProjections.csv")
projection_winners.index = projection_winners["Riding"] # type: ignore
projection_winners["colours"] = projection_winners["Winner"].map(color_map)
colour_codes = hoc_map["ED_NAMEE"].map(lambda x: projection_winners.loc[x, "colours"])

# Plot the seat results summary
partyText = ""
seatsText = ""
for party, seats in projection_winners["Winner"].value_counts().items():
    partyText += f"{party}:\n"
    seatsText += f"{seats}\n"

# Draw the map
hoc_map.plot(color=colour_codes)
plt.figtext(0.8, 0.5, partyText, fontsize="xx-large")
plt.figtext(0.95, 0.5, seatsText, fontsize="xx-large", multialignment="right")
plt.axis('off')
plt.savefig("Outputs/MapImg.svg", bbox_inches='tight', pad_inches=0.1)