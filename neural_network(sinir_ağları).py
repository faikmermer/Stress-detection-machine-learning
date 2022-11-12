# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:44:53 2022

@author: faikm
"""
import numpy as np
from tensorflow import  keras
from keras.datasets import boston_housing
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_excel("stress_data.xlsx")

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

train_data, test_data, train_targets, test_targets = train_test_split(X, y, random_state=42, test_size=.15)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="rmsprop",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    
    return model

k= 5
num_val_samples = len(train_data) // 5

num_epocs = 200
hata_payı_val_durumu = []
loss_durumu = []
accuracy_durumu = []

for i in range(k):
    print(f'Validation katlama işleniyor... #{i}')
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]], axis= 0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]], axis= 0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epocs, batch_size=16, verbose=0)
    hist_dict = history.history
    
    val_loss_hist =  hist_dict["val_loss"]
    hata_payı_val_durumu.append(val_loss_hist)
    
    loss_hist = hist_dict["loss"]
    loss_durumu.append(loss_hist)
    
    accuracy_hist = hist_dict["val_accuracy"]
    accuracy_durumu.append(accuracy_hist)
    
    model_basarısı = model.evaluate(test_data, test_targets, batch_size=16)
    print(f'Model Başarımız: {model_basarısı[1]}')


#Sütunların ortlaması sonrası grafik çıkarma
average_loss_hist = [
    np.mean([x[i] for x in loss_durumu]) for i in range(num_epocs)]

average_val_hist = [
    np.mean([x[i] for x in hata_payı_val_durumu]) for i in range(num_epocs)]

average_accuracy_hist = [
    np.mean([x[i] for x in accuracy_durumu]) for i in range(num_epocs)]




plt.plot(range(1, len(average_val_hist) + 1), average_val_hist)
plt.title("Validation Data Loss")
plt.xlabel("Epochs")
plt.ylabel("Validation Loss")    
plt.show()

plt.plot(range(1, len(average_loss_hist) + 1), average_loss_hist)
plt.title("Loss Grafiği")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(range(1, len(average_accuracy_hist) + 1), average_accuracy_hist)
plt.title("Kesinlik Grafiği")
plt.xlabel("Epochs")
plt.ylabel("Kesinlik")
plt.grid()
plt.show()
    
