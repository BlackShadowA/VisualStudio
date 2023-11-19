import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

negatives_train = np.random.normal(5,1,90_000)
positives_train = np.random.normal(8,1,10_000)
x_train = np.concatenate([negatives_train, positives_train])
y_train = np.array([0] * len(negatives_train) + [1] * len(positives_train))
positives_train_os = np.random.choice(positives_train, size=len(negatives_train))
x_train_os = np.concatenate([negatives_train, positives_train_os])
y_train_os = np.array([0] * len(negatives_train) + [1] * len(positives_train_os))
negatives_serve = np.random.normal(5,1,8_800)
positives_serve = np.random.normal(8,1,1_200)
y_serve = np.array([0] * len(negatives_serve) + [1] * len(positives_serve))
x_serve = np.concatenate([negatives_serve, positives_serve])


df_train = pd.DataFrame({'x': x_train, 'y': y_train})
df_train_os = pd.DataFrame({'x': x_train_os, 'y': y_train_os})
df_train_serve = pd.DataFrame({'x': x_serve, 'y': y_serve})

# log loss
# trainig set
ts = np.linspace(np.min(x_train), np.max(x_train), 100)
log_loss_train = [log_loss(y_train, x_train >= t) for t in ts]
best_thres_train = ts[np.argmin(log_loss_train)]
df_log_loss_train = pd.DataFrame({'x': ts, 'y': log_loss_train})

# trainig set os
ts_os = np.linspace(np.min(x_train_os), np.max(x_train_os), 100)
log_loss_train_os = [log_loss(y_train_os, x_train_os >= t) for t in ts_os]
best_thres_train_os = ts[np.argmin(log_loss_train_os)]
df_log_loss_train_os = pd.DataFrame({'x': ts_os, 'y': log_loss_train_os})

# trainig set serv
ts_serv = np.linspace(np.min(x_serve), np.max(x_serve), 100)
log_loss_train_serv = [log_loss(y_serve, x_serve >= t) for t in ts_os]
best_thres_train_serv = ts[np.argmin(log_loss_train_serv)]
df_log_loss_train_serv = pd.DataFrame({'x': ts_serv, 'y': log_loss_train_serv})



figure, axs = plt.subplots(2, 3, sharex=True, figsize=(8,7))
figure.suptitle('Distribuzione variabile target')

sns.histplot(df_train, x="x", hue="y", multiple="stack", ax=axs[0, 0])
axs[0, 0].set_title(f"Train data = {x_train.shape[0]} , prevalence = {(np.count_nonzero(y_train == 1) / x_train.shape[0]) * 100}", fontsize = 8)
sns.histplot(df_train_os, x="x", hue="y", multiple="stack", ax=axs[0, 1])
axs[0, 1].set_title(f"Train data os  = {x_train_os.shape[0]}, prevalence = {(np.count_nonzero(y_train_os == 1) / x_train_os.shape[0]) * 100}", fontsize = 8)
sns.histplot(df_train_serve, x="x", hue="y", multiple="stack", ax=axs[0, 2])
axs[0, 2].set_title(f"Train data serv = {x_serve.shape[0]} , prevalence = {(np.count_nonzero(y_serve == 1) / x_serve.shape[0]) * 100}", fontsize = 8)
plt.figtext(0.5, 0.5, 'Log - Loss', ha='center', va='center')
sns.lineplot(x="x", y="y", data=df_log_loss_train, ax=axs[1, 0])
axs[1, 0].axvline(best_thres_train, color='r')
axs[1, 0].set_title(f"Train data, Best threshold ={best_thres_train}", fontsize = 8)
sns.lineplot(x="x", y="y", data=df_log_loss_train_os, ax=axs[1, 1])
axs[1, 1].axvline(best_thres_train_os, color='r')
axs[1, 1].set_title(f"Train data os, Best threshold ={best_thres_train_os}", fontsize = 8)
sns.lineplot(x="x", y="y", data=df_log_loss_train_serv, ax=axs[1, 2])
axs[1, 2].axvline(best_thres_train_serv, color='r')
axs[1, 2].set_title(f"Train data serv , Best threshold ={best_thres_train_serv}", fontsize = 8)
plt.show()
