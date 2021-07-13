import  pandas as pd , os as os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import label_binarize
from  sklearn.preprocessing import  LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

#################################

os.chdir('/Users/mahmoudi/Documents/Doc/2020/DuncanData/datafornetwork/LB_All/V1/ML/')
os.getcwd()
files = os.listdir()
lableencoder = LabelEncoder()
dffinal = []
data = pd.read_csv("/Users/mahmoudi/Documents/Doc/2020/DuncanData/datafornetwork/LB_All/V1/ML/dataRA.txt", sep='\t', header=0, index_col=0)#Relative abundance data
data = data[data['target'].str.contains('H1')== False]
data = data.dropna()
data["target"].value_counts()
data["targetN"] = lableencoder.fit_transform(data["target"])  #H = 0 , I = 1
data["targetN"].value_counts()
#data.head(5)
#data.shape


#X = data.drop(["target","targetN","ASV1_Mesorhizobium"], axis=1)  # remove last column lable from data
X = data.drop(["target","targetN"], axis=1)  # remove last column lable from data

y = np.array(data["targetN"]) #use numeric labaling for data
#X = SelectKBest(ch
# i2, k=40).fit_transform(X, y)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42,stratify=y)  # spleat train and test set, 70 % train and 30 percent test
#print("X_train:",X_train.shape,"X_test:",X_test.shape )
#rint("y_train:\n",np.array(np.unique(y_train, return_counts=True)).T,"\nX_test:\n",np.array(np.unique(y_test, return_counts=True)).T )
#data["targetN"].value_counts()
clf = RandomForestClassifier(n_estimators =1000)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = cross_val_predict(clf, X, y, cv=5)
conf_mat = confusion_matrix(y, y_pred)


scores1 = cross_validate(clf, X, y, cv=cv , scoring = 'accuracy', return_estimator =True)
scores1


#temp = []
'''df = pd.DataFrame(columns=["OTU" , "importance"])
for feat, importance in zip(X.columns, scores1['estimator']):
    print('feature: {f}, importance: {i}'.format(f=feat, i=importance))
    temp.append({'OTU' :feat,"score":importance})
df = pd.DataFrame(temp)
sort_by_score = df.sort_values('score',ascending = False)'''


##2 class
finalt = pd.DataFrame()
for idx,estimator in enumerate(scores1['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    f = pd.DataFrame(data = {'importance': estimator.feature_importances_,
            'cv':  idx} ,index = X.columns,
             columns=['importance' , 'cv']).sort_values('importance', ascending=False)
    finalt = pd.concat([finalt, f])

finalt.to_csv("Result/H2S2_RandomforestCV5Featureimportance.csv")










from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

clf = SVC(kernel='linear')
#clf1 = SVC(kernel='linear').fit(X,y)
cv = ShuffleSplit(n_splits=30, test_size=0.3, random_state=0)
scores = cross_val_score(clf, X, y, cv=13)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

cv_results = cross_validate(clf, X, y, cv=cv, return_estimator=True)
cv_results
cv_results

final = pd.DataFrame()
for idx,estimator in enumerate(cv_results['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    f = pd.DataFrame(data = np.transpose(estimator.coef_),
            index = X.columns,
             columns=["H2S2"])
    f["CV"] = idx
    final = pd.concat([final, f])


f2= final[final['CV'] ==3]
final1 = f2.sort_values("H2S2" , ascending=False)

final.to_csv("Result/H2S2_SVMCV5Featureimportance.csv")



from sklearn.svm import SVC
from sklearn.model_selection import cross_validate

clf = SVC(kernel='linear')
#clf1 = SVC(kernel='linear').fit(X,y)
cv = ShuffleSplit(n_splits=30, test_size=0.1, random_state=0)
scores = cross_val_score(clf, X, y, cv=13)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

cv_results = cross_validate(clf, X, y, cv=cv, return_estimator=True)
cv_results
cv_results

final = pd.DataFrame()
for idx,estimator in enumerate(cv_results['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    f = pd.DataFrame(data = np.transpose(estimator.coef_),
            index = X.columns,
             columns=["H2S2"])
    f["CV"] = idx
    final = pd.concat([final, f])


f2= final[final['CV'] ==3]
final1 = f2.sort_values("H2S2" , ascending=False)

final.to_csv("Result/H1S2_SVMCV5Featureimportance.csv")


cv = ShuffleSplit(n_splits=30, test_size=0.1, random_state=0)
#scores = cross_val_score(clf, X, y, cv=10)
#scores

cm_holder = []
conf_matrix_list_of_arrays = []
kf = KFold(n_splits=10, random_state=42, shuffle=True)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
       # X_train, X_test = X[[train_index]], X[[test_index]]
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       y_train, y_test = y[train_index], y[test_index]
       clf.fit(X_train, y_train)
       clf.coef_
       print(confusion_matrix(y_test, clf.predict(X_test)))
       print(classification_report(y_test,clf.predict(X_test)))

       cm_holder.append(confusion_matrix(y_test,clf.predict(X_test)))
       conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
       conf_matrix_list_of_arrays.append(conf_matrix)

mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)



from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)
cm_holder = []
conf_matrix_list_of_arrays = []

for train_index, test_index in loo.split(X):
       # X_train, X_test = X[[train_index]], X[[test_index]]
       print("TRAIN:", train_index, "TEST:", test_index)
       X_train, X_test = X.iloc[train_index], X.iloc[test_index]
       y_train, y_test = y[train_index], y[test_index]
       clf.fit(X_train, y_train)
       clf.coef_
       print(confusion_matrix(y_test, clf.predict(X_test)))
       print(classification_report(y_test,clf.predict(X_test)))

       cm_holder.append(confusion_matrix(y_test,clf.predict(X_test)))
       conf_matrix = confusion_matrix(y_test, clf.predict(X_test))
       conf_matrix_list_of_arrays.append(conf_matrix)

mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)



































clf1.intercept_

#3 class
final = pd.DataFrame()
for idx,estimator in enumerate(cv_results['estimator']):
    print("Features sorted by their score for estimator {}:".format(idx))
    f = pd.DataFrame(data = np.transpose(estimator.coef_),
            index = X.columns,
             columns=["H1","H2","S2"])
    f["CV"] = idx
    final = pd.concat([final, f])
n2= final[final['CV'] ==5]
n3 = n2.sort_values("H1" , ascending=False)

print(clf1.classes_)  # ['A', 'B', 'C', 'D']
print(clf1.coef_.shape)



for model in cv_results['estimator']:
    print(model.coef_)

nn = estimator.coef_




feature_importances = pd.DataFrame(da,
                                       index = X.columns,
                                        columns=['importance' , 'cv']).sort_values('importance', ascending=False)

    f = pd.concat(f,feature_importances)

    print(feature_importances)




import numpy as np
from sklearn.svm import SVC

svc = SVC(kernel='linear').fit(X, y)

print(svc.classes_)  # ['A', 'B', 'C', 'D']
print(svc.coef_.shape)


clf.fit(X, y)
print('w = ',svc.coef_)
print('b = ',svc.intercept_)
print('Indices of support vectors = ', svc.support_)
print('Support vectors = ', svc.support_vectors_)
print('Number of support vectors for each class = ', svc.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(svc.dual_coef_))


