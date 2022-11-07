# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:56:06 2022

@author: faikm
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_curve

df = pd.read_excel('Stress_data.xlsx', header = None)

df.columns=['Target', 'ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']
X_train, X_test, y_train, y_test = train_test_split(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']], df['Target'],
    test_size=0.25, random_state=42, shuffle=True)

minmax_skala = preprocessing.MinMaxScaler().fit(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']])
df_minmax = minmax_skala.transform(df[['ECG(mV)', 'EMG(mV)','Foot GSR(mV)','Hand GSR(mV)', 'HR(bpm)','RESP(mV)']])

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(df_minmax, df.iloc[:, 0], test_size=0.25, shuffle=True, random_state=42)


def plot():
    plt.figure(figsize=(8, 6))
    
    plt.scatter(df['Hand GSR(mV)'], df['HR(bpm)'], 
                color='green', label='Giriş Skalası', alpha=0.5)
    
    plt.scatter(df_minmax[:, 3], df_minmax[:, -2],
                color='blue', label='Min-Max Skalası', alpha= 0.3)
    
    plt.title("Hand GSR ve  HR fizyolojik veri girişleri")
    plt.xlabel("Hand GSR")
    plt.ylabel("Hr")
    plt.legend(loc=True)
    plt.grid()

    plt.tight_layout()

plot()
plt.show()        

# Normalize olmadan Gauss ile Uygulama

gnb = GaussianNB() 
fit = gnb.fit(X_train, y_train)    

# Normalize ile Gauss Uygulama
gnb_norm = GaussianNB()
fit_norm = gnb_norm.fit(X_train_norm, y_train_norm)

pred_train = gnb.predict(X_train)
pred_test = gnb.predict(X_test)

# Ölçümler

print('Kesinlik ölçüsü:')
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test)))

print('Accuracy Normalize Tahmin')
pred_test_norm = gnb_norm.predict(X_test_norm)
print('{:.2%}\n'.format(metrics.accuracy_score(y_test, pred_test_norm)))

# Tahminlerin Karşılaştırılması
print('Target doğrusu:\n', y_test.values[0:25])
print('Tahmin Sonuçlarımız:\n', pred_test_norm[0:25])

#Karmaşılık Matrisi

print(metrics.confusion_matrix(y_test, pred_test_norm))
print("True", y_test.values[0:25])
print("Pred:", pred_test_norm[0:25])

confusion = metrics.confusion_matrix(y_test, pred_test_norm)
print(confusion)
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print(f'Classification Accuracy: {metrics.accuracy_score(y_test,pred_test_norm)} \n')
print(f'Classification Hatası: 1-metrics.accuracy_score(y_test, pred_test_norm)\n')

#Doğruluk 1 ise doğruluk oranı
print(f"Duyarlılık:{metrics.recall_score(y_test, pred_test_norm)}\n")

# Target 0 ise doğruluk oranı
print(f"Özgüllük: {TN /float(TN+FP)}\n")

# FP Oranı

print(f"FP Oranı:{FP/float(TN+FP)}\n")

#Precision: tahminimiz 1 ise tahmin oranı
print(f'Precision:{metrics.precision_score(y_test, pred_test_norm)}\n')


pred  = gnb.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
pred1 = gnb.predict([[-0.005,0.49,8.257,9.853,66.142,10.998]])
pred2 = gnb_norm.predict([[0.001,0.931,5.91,19.773,99.065,35.59]])
pred3 = gnb_norm.predict([[0.005,0.49,8.257,5.853,80.142,45.998]])
tahminlerimiz = [pred, pred1, pred2, pred3]
tahmin_sayimiz  = 0

for tahmin in tahminlerimiz:
    tahmin_sayimiz += 1
    if tahmin  ==  0 :
        print(f'{tahmin_sayimiz}. tahmin sonucumuz: Stresli Değilsiniz!')
    else:
        print(f'{tahmin_sayimiz}. tahmin sonucumuz: Streslisiniz!!!')





#Grafiklerimiz

fpr, tpr, _ = roc_curve(y_test, pred_test_norm)

roc = metrics.auc(fpr, tpr)
plt.title("Roc Eğrisi")
plt.plot(fpr, tpr, 'b', label='AUC %0.2f' % roc)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc=True)
plt.grid()
plt.show()









                       