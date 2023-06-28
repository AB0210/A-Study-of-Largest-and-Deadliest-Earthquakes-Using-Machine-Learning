# Decision Tree Deadliest
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv("DeadliestDT.csv")
df
  => Dataset
features = ['Country', 'Magnitude', 'Tsunami', 'Depth']
X = df[features]
X
  => Setting variable X
Y = df['Casualties']
Y
  => Setting variable Y
Labelencoder_X = LabelEncoder()
X = X.apply(LabelEncoder().fit_transform)
X
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)
tree.plot_tree(dtree, feature_names=features)
  => Output

# Do the same for Largest

# K-means for Deadliest PDC-1
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = {
    'Magnitude': [8.6, 6.5, 9, 7.3, 6.7, 7.4, 6.4, 6.7, 6.7, 7.3, 5.8, 6.4, 7, 6.1, 9.2, 7.4, 6.8, 6.6, 7.1, 6.4, 7.9, 6.9, 6.6, 7.6, 7.1, 6.7, 7.6, 7.5, 7.4, 8.2, 7.3, 7.1, 6.2, 6.6, 6.3, 8, 5.7, 7.1, 6.8, 5.3, 7.4, 6.8, 7.8, 6.3, 6.8, 6.9, 6.6, 7.3, 6.6, 7.6, 7.9, 7.7, 7.4, 6.6, 9.1, 7.6, 6.4, 8, 8, 7.6, 7, 9.1, 6.4, 7.7, 6.2, 7.8, 7.8, 7.3, 7.5, 6.4, 7, 7.2, 6, 7.8],
    'PDC 1': [-7, -10, -10, -6, -2, -3, -5, 5, 3, 1, 1, 2, 2, 3, -3, -4, 6, 5, 5, 3, -1.9, 1, -2, -3, -7, -7, -5, 0, 3, -3.9, 4, 9, 8, 5, 12, -14, 13, 5, -5, -10, -20, -32, -47, -54, -66, -77, -92, -95, -110, -98, -89, -90, -79, -73, -69, -68, -63, -59, -53, -44, -38, -40, -45, -40, -31, -39, -47, -51, -46, -35, -39, -42, -43, -42]
}
df = pd.DataFrame(data)
kmeans = KMeans(n_clusters=4).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['Magnitude'], df['PDC 1'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
  => Output

# Do the same for Deadliest PDC-2, Largest-1 and Largest-2

# Geographical Plotting for Deadliest
import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True)
import pandas as pd
df = pd.read_csv('Deadliest.csv')
df
  => Dataset
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        locations = df['Country'],
        locationmode = "country names",
        z = df['Depth'],
        text = df['Country'],
        colorbar = {'title' : 'Deadliest'},
      )
layout = dict(title = 'Depth for Deadliest Earthquakes',
              geo = dict(projection = {'type':'mercator'})
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
  => Output
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        locations = df['Country'],
        locationmode = "country names",
        z = df['Casualties'],
        text = df['Country'],
        colorbar = {'title' : 'Deadliest'},
      )
layout = dict(title = 'Casualties for Deadliest Earthquakes',
              geo = dict(projection = {'type':'mercator'})
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
  => Output
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        locations = df['Country'],
        locationmode = "country names",
        z = df['Magnitude'],
        text = df['Country'],
        colorbar = {'title' : 'Deadliest'},
      )
layout = dict(title = 'Magnitude for Deadliest Earthquakes',
              geo = dict(projection = {'type':'mercator'})
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
  => Output
data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        locations = df['Country'],
        locationmode = "country names",
        z = df['Tsunami'],
        text = df['Country'],
        colorbar = {'title' : 'Deadliest'},
      )
layout = dict(title = 'Tsunami for Deadliest Earthquakes',
              geo = dict(projection = {'type':'mercator'})
             )
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)
  => Output

# Do the same for Largest

# Matrix Visualization for Deadliest
import seaborn as sns
import pandas as pd
import numpy as np
%matplotlib inline
df=pd.read_csv("Deadliest.csv")
df
  => Dataset
df.corr()
  => Correlation
sns.heatmap(df.corr())
  => Correlation Heatmap
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
  => Output

# Do the same for Largest
