from catboost import CatBoostClassifier, Pool, cv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

#(1)
# 1. In common sense, 'ID' and 'Name' have zero connections with label 'Medal'
# 2. 'Team' records full names of participating countries,  whereas NOC is the short code mapping to each country, thus we don't need 'Team'
# 3. 'Game' combines 'Year' and 'Season', we don't need it either
# 4. we will keep 'City' see how important will this feature be (home country effect)
df = df.drop(columns=['ID','Name','Team','Games'])
# print to console; now we have 11 columns of ['Sex', 'Age', 'Height', 'Weight', 'NOC', 'Year', 'Season', 'City', 'Sport', 'Event', 'Medal']
print(list(df.columns))
print(df.shape)

#(2)
# we want to drop the rows with NA value in features (since our dataset is large, we can drop samples with missing values)
df = df[~((df['Sex']=='NA')        | (np.isnan(df['Age'])) | (np.isnan(df['Height']))  | 
          (np.isnan(df['Weight'])) | (df['NOC']=='NA')     | (np.isnan(df['Year']))    | 
          (df['Season']=='NA')     | (df['City']=='NA')    | (df['Sport']=='NA')       | 
          (df['Event']=='NA'))]
# must reorder the index to avoid any mistake in step (4) and (5)
df = df.reset_index(drop=True)
print(df)
print(df.shape)

#(3)
# we want to examine some continous value distribution pattern by visualizing the data(eg. normal distribution, chi distribution.. )
# 'Age'
df.hist(column='Age'    , bins=50)
# 'Height'
df.hist(column='Height' , bins=50)
# 'Weight'
df.hist(column='Weight' , bins=50)
# 'Year'
df.hist(column='Year'   , bins=50)

#(4)
# except 'Year', 'Age' 'Height' 'Weight' all more or less conform to normal distribution
# 'qcut()' method from pandas will be adopted to transfer continous value to categorical ones, where quantiles set to be deciles(10).
# Quantiles means the dividing point that divide the data according to the density, higher the density, narrower the bin
# 'labels=False' label set to be false to return only integer indicators of the bins, which means 0~9 indicator
new_age     = pd.qcut(df.iloc[:, 1].tolist(),10, labels=False, retbins=True)
new_height  = pd.qcut(df.iloc[:, 2].tolist(),10, labels=False, retbins=True)
new_weight  = pd.qcut(df.iloc[:, 3].tolist(),10, labels=False, retbins=True)
new_year    = pd.qcut(df.iloc[:, 5].tolist(),10, labels=False, retbins=True)
# create new subset
new_df      = pd.DataFrame({'Age':list(new_age[0]), 'Height':list(new_height[0]), 'Weight':list(new_weight[0]), 'Year':list(new_year[0])})
print(new_df.shape)
# merge the new subset with the old set
df = df.drop(columns=['Age','Height','Weight','Year'])
print(df.shape)
df = new_df.join(df)
# print to console; now we have 11 columns of ['Age', 'Height', 'Weight', 'Year', 'Sex', 'NOC', 'Season', 'City', 'Sport', 'Event', 'Medal']
print(list(df.columns))
print(df.shape)

#(5)
# To set all other attributes from string type to numeric type. e.g. for 'Sex': M=1, F=0
df['Sex'] = pd.Categorical(df['Sex']).codes
df['NOC'] = pd.Categorical(df['NOC']).codes
df['Season'] = pd.Categorical(df['Season']).codes
df['City'] = pd.Categorical(df['City']).codes
df['Sport'] = pd.Categorical(df['Sport']).codes
df['Event'] = pd.Categorical(df['Event']).codes
df['Medal'] = pd.Categorical(df['Medal']).codes

#(6)
# Now the dataset are divided into two parts: training and testing. since the dataset is large, 8-2 or 9-1 ratio is enough for training purpose
train, test = train_test_split(df, test_size=0.1)
x           = train.drop(columns='Medal')
y           = train['Medal']
x_test      = test.drop(columns='Medal')
print(x.shape)
print(x.dtypes)
print(y.shape)
print(x_test.shape)
# Cross-validation is also necessary, thus we divide the training set into two: training, validating
x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.8)

#(7)
# Add model as CatBoostClassifier
model = CatBoostClassifier(
    custom_loss=['Accuracy'],
    use_best_model=True,
    logging_level='Silent'
)
# Train the model
model.fit(
    x_train, y_train,
    eval_set=(x_validation, y_validation),
    logging_level='Verbose',
    plot=True
); 

# Model Cross-Validation
cv_params = model.get_params()
cv_params.update({
    'loss_function': 'Logloss'
})
cv_data = cv(
    Pool(x, y),
    cv_params,
    verbose=True,
    plot=True
)
# Print the best accuracy score
print('Before cross validation:', model.score(x_train,y_train))
print('Precise validation accuracy score: {}'.format(np.max(cv_data['test-Accuracy-mean'])))
# Apply model to test set
predictions         = model.predict(x_test)
predictions_probs   = model.predict_proba(x_test)
print(predictions[:])
print(predictions_probs[:])

#(8)
# Evaluate Feature Importance
train_pool = Pool(x_train, y_train)
feature_importances = model.get_feature_importance(train_pool)
feature_names = x_train.columns
for score, name in sorted(zip(feature_importances, feature_names)):
    print('{}: {}'.format(name, score))
