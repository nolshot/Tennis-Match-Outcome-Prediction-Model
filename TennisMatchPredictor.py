import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Load in csv file containing 2024 ATP data
url = "/Users/admin/Downloads/atp_matches_2024.csv"

# Store as a data frame
df = pd.read_csv(url)

# Winner Attributes
A_Seed = df.iloc[:,8]
A_Hand = df.iloc[:,11]
A_Ht = df.iloc[:,12]
A_Age = df.iloc[:,14]
A_Rank = df.iloc[:,45]
A_Rank_Points = df.iloc[:,46]
A_Ace = df.iloc[:,27]
A_Df = df.iloc[:,28]
A_svpt = df.iloc[:,29]
A_1stIn = df.iloc[:,30]
A_1stWon = df.iloc[:,31]
A_2ndWon = df.iloc[:,32]
A_SvGms = df.iloc[:,33]
A_bpSaved = df.iloc[:,34]
A_bpFaced = df.iloc[:,35]

# Loser Attributes
B_Seed = df.iloc[:,16]
B_Hand = df.iloc[:,19]
B_Ht = df.iloc[:,20]
B_Age = df.iloc[:,22]
B_Rank = df.iloc[:,47]
B_Rank_Points = df.iloc[:,48]
B_Ace = df.iloc[:,36]
B_Df = df.iloc[:,37]
B_svpt = df.iloc[:,38]
B_1stIn = df.iloc[:,39]
B_1stWon = df.iloc[:,40]
B_2ndWon = df.iloc[:,41]
B_SvGms = df.iloc[:,42]
B_bpSaved = df.iloc[:,43]
B_bpFaced = df.iloc[:,44]

# Match Context
surface = df.iloc[:,2]
best_of = df.iloc[:,24]

# Create full Data Frame
ndf = pd.DataFrame({

'A_Seed': A_Seed,
'A_Hand':A_Hand,
'A_Ht':A_Ht,
'A_Age':A_Age,
'A_Rank':A_Rank,
'A_Rank_Points':A_Rank_Points,
'A_Ace':A_Ace,
'A_Df':A_Df,
'A_svpt':A_svpt, 
'A_1stIn':A_1stIn, 
'A_1stWon':A_1stWon, 
'A_2ndWon':A_2ndWon, 
'A_SvGms':A_SvGms, 
'A_bpSaved':A_bpSaved, 
'A_bpFaced':A_bpFaced,


'B_Seed': B_Seed,
'B_Hand': B_Hand,
'B_Ht': B_Ht,
'B_Age': B_Age,
'B_Rank': B_Rank,
'B_Rank_Points': B_Rank_Points,
'B_Ace': B_Ace,
'B_Df': B_Df,
'B_svpt': B_svpt, 
'B_1stIn': B_1stIn, 
'B_1stWon': B_1stWon, 
'B_2ndWon': B_2ndWon, 
'B_SvGms': B_SvGms, 
'B_bpSaved': B_bpSaved, 
'B_bpFaced': B_bpFaced,

'surface':surface,
'best_of':best_of,
'target': pd.Series([1] * len(df))
})



# Create column of randomized booleans
mask = np.random.rand(len(ndf)) < 0.5

# Swap the winner and loser data for half of the rows in order to minimize bias
cols = ["Seed", 'Hand','Ht','Age','Rank','Rank_Points','Ace','Df','svpt','1stIn','1stWon','2ndWon','SvGms','bpSaved','bpFaced']

for c in cols:
    a = 'A_' + c
    b = 'B_' + c
    ndf.loc[mask,[a,b]] = ndf.loc[mask,[b,a]].values

ndf.loc[mask,'target'] = 0

# Make the surface numerical
ndf["surface"] = ndf["surface"].map({
    "Hard": 0,
    "Clay": 1
})

# Make the A players dominant hand numerical
ndf["A_Hand"] = ndf["A_Hand"].map({
    "R": 0,
    "L": 1
})

# Make the B players dominant hand numerical
ndf["B_Hand"] = ndf["B_Hand"].map({
    "R": 1,
    "L": 0
})


# Fill all nan values with the median for that column
for i in range(31):
    ndf.iloc[:,i] = ndf.iloc[:,i].fillna(ndf.iloc[:,i].median())


# X is Full Pandas Data
X = ndf.iloc[:, 0:32]

# Y is the target variable
y = ndf.iloc[:,32]

# Set split into 80% training data and 20% testing data
# Random state controls randomness so results are reproducable (same random shuffle every time)
# stratify=y. - same % of wins/losses and more stable scores
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state = 67,stratify=y)

model = LogisticRegression()

# Trains the model using training data
# (study these matches and learn patterns)
model.fit(X_train, y_train)

# Use the model to predict outcomes for unseen matches
y_pred = model.predict(X_test)

# Mean accuracy after machine learning
print(model.score(X_test, y_test))

# true positive, false positive, false negative, true negative
print(confusion_matrix(y_test, y_pred))

# Compares y_test and y_pred
# Precision - how often was the prediction correct (minimize false positives)
# Recall - how many of the true cases did the model find (minimize false negatives)
# F1-score- Single number tat balances precision and recall
# Support - how many true samples of this class exist in the test set
# Accuracy - overall fraction correct
# Macro avg - simple average across classes
# Weighted avg - average weighted by class frequency
print(classification_report(y_test, y_pred))
