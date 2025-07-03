import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data_root = "https://github.com/ageron/data/raw/main/"
life_sat = pd.read_csv(data_root + "lifesat/lifesat.csv")

X = life_sat[['GDP per capita (USD)']].values
Y = life_sat[['Life satisfaction']].values

life_sat.plot(
    kind='scatter',grid=True,
    x='GDP per capita (USD)',y='Life satisfaction'
)
plt.axis([23_500,62_500,4,9])
plt.savefig('plot.png')

model = LinearRegression()
model.fit(X,Y)

X_new = [[36_655.2]]
print(model.predict(X_new))

