from catboost import CatBoostClassifier,Pool
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("movie_rating.csv")

# Removing column here. if have multiple columns, syntax changes to columns=['a7','aX'..]
df = df.drop(columns=['a7','a8'])

# Treat some continous float value to categorical
df['a0'] = pd.qcut(list(df['a0']),10, labels=False, retbins=True)[0]
df['a1'] = pd.qcut(list(df['a1']),10, labels=False, retbins=True)[0]
df['a3'] = pd.qcut(list(df['a3']),10, labels=False, retbins=True)[0]
df['a4'] = pd.qcut(list(df['a4']), 5, labels=False, retbins=True)[0]
df['a6'] = pd.qcut(list(df['a6']),10, labels=False, retbins=True)[0]

train, test = train_test_split(df, test_size=0.2)

x           = train.drop(columns='labels')
y           = train['labels']
x_test      = test.drop(columns='labels')
y_test      = test['labels']

categorical_features_indices = np.where(x.dtypes != np.float)[0]

# Cross-validation is also necessary, thus we divide the training set into two: training, validating
x_train, x_validation, y_train, y_validation = train_test_split(x, y, train_size=0.9)

# Add model as CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000,
#    depth=8,
    learning_rate=0.1,
    random_seed=123,
    loss_function='MultiClass',
    eval_metric='MultiClass',
    use_best_model=True,                 
    od_type='Iter'
)

# Train the model
model.fit(
    x_train, y_train,
    eval_set=(x_validation, y_validation),
    cat_features=categorical_features_indices,
    logging_level='Verbose'
); 
        
## Print the best accuracy score
vali_prediction_prob    = model.predict(x_validation,prediction_type='Probability')
print('Validation set accuracy score:', accuracy_score(y_validation,np.rint(vali_prediction_prob.dot([1, 2, 3, 4, 5]))))

# Apply model to test set
test_prediction_prob    = model.predict(x_test,prediction_type='Probability')
predictions             = np.rint(test_prediction_prob.dot([1, 2, 3, 4, 5]))
# calculate and print the accuracy score for test samples
print('Test set accuracy score:',accuracy_score(y_test,predictions))


# Evaluate Feature Importance
train_pool = Pool(x_train, y_train, cat_features=categorical_features_indices)
feature_importances = model.get_feature_importance(train_pool)
feature_names = x_train.columns
print('Features Importance:')
for score, name in sorted(zip(feature_importances, feature_names),reverse=True):
    print('{}: {}'.format(name, score))

# Gather information for next tuning
print('treecount:',model.tree_count_)
print('learning rate:',model.learning_rate_)  
