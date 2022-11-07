# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:22:40 2022

@author: faikm
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve

df = pd.read_excel('stress_data.xlsx', header=None)

df.columns=['Target', 'ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']

X_train, X_test, y_train, y_test = train_test_split(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']], df['Target'], shuffle=True,
                                                        test_size=0.25, random_state=42)

minmax_scale = preprocessing.MinMaxScaler().fit(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']])
df_minmax = minmax_scale.transform(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']])

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(df_minmax, df["Target"],
                                                                        shuffle=True, test_size=0.25, random_state=42)
def plot():
    plt.figure(figsize=(8, 6))
    
    plt.scatter(df['Hand GSR(mV)'], df['HR(bpm)'],
                color= 'green', label='Giriş Skalası', alpha=0.3)
    
    plt.scatter(df_minmax[:, 0], df_minmax[:, 1],
                color= 'blue', label='min-max Ölçekledirme', alpha=0.6)
    
    plt.title('Fizyolojik veri setinin GSR ve HR içeriği')
    plt.xlabel('El GSR')
    plt.ylabel('HR')
    plt.legend(loc='lower right')
    plt.grid()
    
    plt.tight_layout()
    
plot()
plt.show()

model = DecisionTreeClassifier(criterion='entropy',splitter= 'best', max_leaf_nodes=3)
fit = model.fit (X_train, y_train)

model_norm = DecisionTreeClassifier(max_leaf_nodes=3)
fit_norm = model_norm.fit(X_train_norm, y_train)

pred_train = model.predict(X_train)

pred_test = model.predict(X_test)

print('Karar Ağacı normalize yapılmadan kesinlik ölçütü:')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

pred_train_norm = model_norm.predict(X_train_norm)

print('Normalizasyon uygulayarak yapılan Kesinlik Ölçütü:')
pred_test_norm = model_norm.predict(X_test_norm)
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_norm)))

# Tahminlerimiz
pred = model.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
pred1 = model.predict([[-0.005,0.49,8.257,9.853,66.142,10.998]])
pred2 = model_norm.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
pred3 = model_norm.predict([[0.005,0.49,8.257,5.853,80.142,45.998]])
tahminlerimiz = [pred, pred1, pred2, pred3]
tahmin_sayimiz  = 0

for tahmin in tahminlerimiz:
    tahmin_sayimiz += 1
    if tahmin  ==  0 :
        print(f'{tahmin_sayimiz}. tahmin sonucumuz: Stresli Değilsiniz!')
    else:
        print(f'{tahmin_sayimiz}. tahmin sonucumuz: Streslisiniz!!!')
        

# Grafiklerimizi

fpr, tpr, _= roc_curve(y_test, pred_test_norm)
roc = metrics.auc(fpr, tpr)
plt.title("Roc Eğrisi")
plt.plot(fpr, tpr, 'b', label= 'AUC = %0.2f' % roc)
plt.legend(loc='lower right')
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.grid()
plt.show()



