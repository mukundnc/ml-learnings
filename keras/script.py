import pandas as pd

import matplotlib.pyplot as plt

white = pd.read_csv("./data/whitewine.csv", sep=";")

red = pd.read_csv("./data/redwine.csv", sep=";")

pd.isnull(red)

print(pd.isnull(red))
#fig, ax = plt.subplots(1, 2)

#ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
#ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

#fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
#ax[0].set_ylim([0, 1000])
#ax[0].set_xlabel("Alcohol in % Vol")
#ax[0].set_ylabel("Frequency")
#ax[1].set_xlabel("Alcohol in % Vol")
#ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
#fig.suptitle("Distribution of Alcohol in % Vol")

#plt.show()