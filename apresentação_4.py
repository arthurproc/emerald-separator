# -*- coding: utf-8 -*-

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
import seaborn as sns

selecao = #carregue aqui seu dataset
selecao_df = pd.DataFrame(selecao.data, columns=selecao.feature_names)

print ("Linhas e colunas do dataset: ", selecao_df.shape)

esmeraldas = len(selecao.data[selecao.target==1])
print ("Número de esmeraldas no dataset: ", esmeraldas)

X_train, X_test, Y_train, Y_test = train_test_split(selecao.data, selecao.target, test_size=0.25, 
                                                    stratify=selecao.target, random_state=30)

print ("Matriz de treinamento: ", X_train.shape)
print ("Matriz de testes: ", X_test.shape)

selecao_df_labels = selecao_df.copy()
selecao_df_labels['labels'] = selecao.target
selecao_df_labels.tail(3)

scaler1 = StandardScaler()
scaler1.fit(selecao.data)
feature_scaled = scaler1.transform(selecao.data)

pca1 = PCA(n_components=4)
pca1.fit(feature_scaled)
feature_scaled_pca = pca1.transform(feature_scaled)
print("Formato da matriz após o PCA: ", np.shape(feature_scaled_pca))

feat_var = np.var(feature_scaled_pca, axis=0)
feat_var_rat = feat_var/(np.sum(feat_var))

print ("Variância dos 4 componentes principais extraídos pelo PCA: ", feat_var_rat)

selecao_target_list = selecao.target.tolist()

feature_scaled_pca_X0 = feature_scaled_pca[:, 0]
feature_scaled_pca_X1 = feature_scaled_pca[:, 1]
feature_scaled_pca_X2 = feature_scaled_pca[:, 2]
feature_scaled_pca_X3 = feature_scaled_pca[:, 3]

labels = selecao_target_list
colordict = {0:'brown', 1:'darkslategray'}
piclabel = {0:'Outros', 1:'Esmeralda'}
markers = {0:'o', 1:'*'}
alphas = {0:0.3, 1:0.4}

fig = plt.figure(figsize=(12, 7))
plt.subplot(1,2,1)
for l in np.unique(labels):
    ix = np.where(labels==l)
    plt.scatter(feature_scaled_pca_X0[ix], feature_scaled_pca_X1[ix], c=colordict[l], 
               label=piclabel[l], s=40, marker=markers[l], alpha=alphas[l])
plt.xlabel("First Principal Component", fontsize=15)
plt.ylabel("Second Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.subplot(1,2,2)
for l1 in np.unique(labels):
    ix1 = np.where(labels==l1)
    plt.scatter(feature_scaled_pca_X2[ix1], feature_scaled_pca_X3[ix1], c=colordict[l1], 
               label=piclabel[l1], s=40, marker=markers[l1], alpha=alphas[l1])
plt.xlabel("Third Principal Component", fontsize=15)
plt.ylabel("Fourth Principal Component", fontsize=15)

plt.legend(fontsize=15)

plt.savefig('Cancer_labels_PCAs.png', dpi=200)
plt.show()

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe_pca_steps = [('scaler', StandardScaler()), ('pca', PCA()), ('SupVM', SVC(kernel='rbf'))]
pipe_steps = [('SupVM', SVC(kernel='rbf'))]

check_pca_params= {
    'pca__n_components': [2], 
    'SupVM__C': [0.1, 0.5, 1, 10,30, 40, 50, 75, 100, 500, 1000], 
    'SupVM__gamma' : [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}
check_params= {
    'SupVM__C': [0.1, 0.5, 1, 10,30, 40, 50, 75, 100, 500, 1000], 
    'SupVM__gamma' : [0.001, 0.005, 0.01, 0.05, 0.07, 0.1, 0.5, 1, 5, 10, 50]
}

pipelinepca = Pipeline(pipe_pca_steps)
pipeline = Pipeline(pipe_steps)

"""Teste sem o uso do PCA"""

from tqdm import tqdm_notebook as tqdm 
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")



print ("Iniciado o treinamento de parâmetros")
for cv in tqdm(range(4,6)):
    create_grid = GridSearchCV(pipeline, param_grid=check_params, cv=cv)
    create_grid.fit(X_train, Y_train)
    print ("pontuação para %d fold CV := %3.2f" %(cv, create_grid.score(X_test, Y_test)))
    print ("!!!!!!!! Melhores parâmetros para os dados de treino SEM PCA !!!!!!!!!!!!!!")
    print (create_grid.best_params_)

print ("saiu do loop")

"""Com o uso do PCA"""

from tqdm import tqdm_notebook as tqdm 
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")



print ("Iniciado o treinamento de parâmetros")
for cv in tqdm(range(4,6)):
    create_gridpca = GridSearchCV(pipelinepca, param_grid=check_pca_params, cv=cv)
    create_gridpca.fit(X_train, Y_train)
    print ("pontuação para %d fold CV := %3.2f" %(cv, create_gridpca.score(X_test, Y_test)))
    print ("!!!!!!!! Melhores parâmetros para os dados de treino COM PCA !!!!!!!!!!!!!!")
    print (create_gridpca.best_params_)

print ("saiu do loop")

from sklearn.metrics import confusion_matrix

Y_pred = create_grid.predict(X_test)
cm = confusion_matrix(Y_test, Y_pred)
print("Matriz de confusão sem PCA: \n")
print(cm)

Y_predpca = create_gridpca.predict(X_test)
cm = confusion_matrix(Y_test, Y_predpca)
print("Matriz de confusão com PCA: \n")
print(cm)

scaler1 = StandardScaler()
scaler1.fit(X_test)
X_test_scaled = scaler1.transform(X_test)


pca1 = PCA(n_components=2)
X_test_scaled_reduced = pca1.fit_transform(X_test_scaled)


svm_model = SVC(kernel='rbf', C=float(create_gridpca.best_params_['SupVM__C']),
                gamma=float(create_gridpca.best_params_['SupVM__gamma']))

classify = svm_model.fit(X_test_scaled_reduced, Y_test)

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

def make_meshgrid(x, y, h=.1):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))#,
                         #np.arange(z_min, z_max, h))
    return xx, yy

X0, X1 = X_test_scaled_reduced[:, 0], X_test_scaled_reduced[:, 1]
xx, yy = make_meshgrid(X0, X1)

fig, ax = plt.subplots(figsize=(12,9))
fig.patch.set_facecolor('white')
cdict1={0:'deeppink',1:'lime'}

Y_tar_list = Y_test.tolist()
yl1= [int(target1) for target1 in Y_tar_list]
labels1=yl1
 
labl1={0:'Outros',1:'Esmeralda'}
marker1={0:'d',1:'*'}
alpha1={0:.8, 1:0.5}

for l1 in np.unique(labels1):
    ix1=np.where(labels1==l1)
    ax.scatter(X0[ix1],X1[ix1], c=cdict1[l1],label=labl1[l1],s=70,marker=marker1[l1],alpha=alpha1[l1])

ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1], s=40, facecolors='none', 
           edgecolors='midnightblue', label='Vetores de suporte')

plot_contours(ax, classify, xx, yy,cmap='PRGn', alpha=0.4)
plt.legend(fontsize=15)

plt.xlabel("1º Componente principal",fontsize=14)
plt.ylabel("2º Componente principal",fontsize=14)

plt.show()