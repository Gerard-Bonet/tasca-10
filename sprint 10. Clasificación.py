#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import math

import statistics as stat
from sklearn.neural_network import MLPRegressor


# Seguimos el hilo argumental del [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) y  [ejercicio 9.2](https://github.com/Gerard-Bonet/sprint9-tasca2-Aprenantatge-supervisat.git)
# para trabajarcon el ejercicio 10 . Primero cargamos el Data Set. 

# In[3]:


df = pd.read_csv("DelayedFlights.csv") # este es el conjunto de datos proporcionado en el ejercicio 
df.head(10)


# 0.  **Tratamiento de variables**. 
# 
# En este apartado vamos a hacer tratamiento de variables para luego aplicar a programas de clasificación en función de la variable ArrDelay
# 
# Hemos dejado los enlaces del [ejercicio 9.1](https://github.com/Gerard-Bonet/Sprint9Tasca1.git) y  [ejercicio 9.2](https://github.com/Gerard-Bonet/sprint9-tasca2-Aprenantatge-supervisat.git) en que se explica los razonamientos para sleccionar una variable un otra, 
# 
# aunque volveremos a exolicar los motivos por los que se seleccionan variables, unas sí u otras no, no entraremos tan en detalle
# 

# **De las notas de 9.1 y 9.2:**
# 
# a) Variable Unnamed 0 y Year, básicamente son un índice y el año de vuelos del 2008. Year es una constante. 
# Así que las eliminamos. 
# 
# b) las variables "UniqueCarrier', 'FlightNum','TailNum, 'Month', 'DayofMonth', 'DayOfWeek', 'Cancelled', 'CancellationCode', "Diverted", "Origin", "Dest" tamopco tenían ningún peso de influencia en la variable ArrDelay
# - UniqueCarrier es la compañía que hace el vuelo. No tiene piese alguno en la variable ArrDelay,no se puede afirmar que ninguna compañía vaya a tener más retrasos qur otra
# - Flightnum(número de vuelo) y TailNum(Número de avión), que bien, sí podríam darnos información acerca de retrasos, nos 
# generaría por Dummies centenares de variables nuevas. Ya que un avión que puede hacer X viajes en un dia o dos, nos podría indicar si va con retraso, si los vuelos anteriores van con retraso ya que no llegaría a la hora para el embarco, y suponiendo que el avión  no fuera sustituído. Pero para poder hacer esta previsión se debería tener en el train la información del vuelo, de las horas anteriores o incluso del día anterior para vuelos de varia horas o que empezaron el día anterior. Se debería tener una secuencia de los vuelos anteriores, o al menos del anterior. Viendo que se necesita cierta información previa y que nos genera un exceso de variables independientes también las eliminamos.
# - Los vuelos cancelados, el código de cancelación y desviados, no los ponemos en el Dataset ya que los vuelos no llegan a producirse. Se podría poner, a nivel conceptual arrdelay= $\infty$ , pero nos generaría outliers, así que tampoco contamos contamos con ello
# - 'Month', 'DayofMonth', 'DayOfWeek. Son el Mes, día del mes o de la semana en la que se produce el vuelo. Como vimos, tampoco tiene ningún peso matemático con ArrDelay, así como el Origen o del Destino. No hay días del año o aeropuertos en que haya más probabilidad de tener un retraso. 
# -También eliminamos 'ActualElapsedTime', 'CRSElapsedTime', 'AirTime' ya que tienen un 0.95 de correlación o más  con Distance y aportan la misma información. ActualElapsed es el tiempo esperado total del vuelo( desembarco, salida, vuelo, más atterizaje), 
# CRSElapsedTime es el mismo tiempo previsto, mientras que AirTime es el tiempo que el avión está en el aire y Distance la 
# distancia recorrida en millas. 
# 
# 
# 

# In[4]:


df[["ActualElapsedTime", 'CRSElapsedTime', 'AirTime' , "Distance" ]].corr()


# In[5]:


df1= df.drop(["Unnamed: 0", "Year", "UniqueCarrier", 'FlightNum',"TailNum", "Month", "DayofMonth", 
              'DayOfWeek', 'Cancelled', 'CancellationCode', "Diverted", "Origin", "Dest","ActualElapsedTime", 'CRSElapsedTime', 'AirTime'], axis =1)
df1.head(10)


# Como vamos a empezar a transformar variables, vamos a hacer previamente un muestreo. 
# 

# In[6]:


df2= df1.sample (390000,random_state=55)


# Cómo vimos en los ejercicios 9.1 y 9.2, los valores NaN de Arrdelay, así como de otras variables  coincidían con aquellos que el vuelo había desviado o cancelado, valores imposibles de deducir ( los NaN) por el mismo concepto de cancelación o desvió. 
# por lo que eliminamos los valores NaN de ArrDelay

# In[7]:


df3=df2.dropna( subset=["ArrDelay"]).reset_index(drop=True)
df3.isna().sum()


# In[8]:


df3.shape


# 0.1. **Transformación de variables horarias**
# 
# En esta parte vamos a convertir las variables DepTime,	CRSDepTime,	ArrTime,	CRSArrTime en la función cíclica. 
# 
# Estas cuatro variables vienen en formato horario  hh:mm. Lo que haremos será contar todos los minutos transcurridos durante 
# 
# el día, siendo 0 minutos a las 00:00 y 1440 los minutos transcurridos durante el día a las 23:59.
# 
# Más adelante, en el apartado 0.4, transformaremos estas variables en cíclicas. 
# 

# In[9]:


# Primero de todo convierto las variables horarias en formato hora y para eso tienen que haber 4 digítos, que los relleno por la
#izquierda con ceros
# Primero tengo que convertir en entero las variables DepTime y ArrTime en enteros para evitar los decimales


df3['DepTime'] = df3['DepTime'].astype(int)


# In[10]:



df3['ArrTime'] = df3['ArrTime'].astype(int)


# In[11]:


#relleno por la izquierda con ceros
df3['DepTime'] = df3['DepTime'].astype(str).str.zfill(4)
df3['CRSDepTime'] = df3['CRSDepTime'].astype(str).str.zfill(4)
df3['ArrTime'] = df3['ArrTime'].astype(str).str .zfill(4)
df3['CRSArrTime'] = df3['CRSArrTime'].astype(str).str.zfill(4)
df3.head()


# In[12]:


# las convierto en formato horario( Nota: en un principio lo pasé a formato horario por si lo necesitaba para datetime, pero 
# al final opté por otro tipo de conversión)
df3['DepTime'] = df3['DepTime'].astype(str).str[:2]  + ':' + df3['DepTime'].astype(str).str[2:4] + ':00' 
df3['CRSDepTime'] = df3['CRSDepTime'].astype(str).str[:2] + ':' + df3['CRSDepTime'].astype(str).str[2:4] + ':00' 
df3['ArrTime'] = df3['ArrTime'].astype(str).str[:2] + ':' + df3['ArrTime'].astype(str).str[2:4]  + ':00'
df3['CRSArrTime'] = df3['CRSArrTime'].astype(str).str[:2] + ':' + df3['CRSArrTime'].astype(str).str[2:4] + ':00'


df3


# In[13]:


# creamos la función minutos, que divide la hora hh:mm con un Split, en una lista ("hh","mm"), reconvierte hh y mm en enteros,
# para luego pasarlos a minutos, y con la reconverión ya comentada aplica la función minutos()
def minutos(x):    
    x=x.split( sep=":")
    seg= 60*(int(x[0]))+(int(x[1]))
    
    return seg



dfhoras= df3[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]]


# In[14]:


dfhoras_DT= dfhoras["DepTime"].apply(minutos)
dfhoras_CRSD=dfhoras["CRSDepTime"].apply(minutos)
dfhoras_AT=dfhoras["ArrTime"].apply(minutos)
dfhoras_CRSA=dfhoras["CRSArrTime"].apply(minutos)


# In[15]:


df4= df3.drop([ 'DepTime',
       'CRSDepTime', 'ArrTime', 'CRSArrTime'], axis=1) 


# In[16]:


# ahora añadimos las cuatro columnas nuevas

df5= pd.concat([df4, dfhoras_DT,dfhoras_CRSD , dfhoras_AT, dfhoras_CRSA], axis=1)
df5.columns


# In[17]:


df5[["DepTime", "CRSDepTime", "ArrTime", "CRSArrTime"]].describe()# miramos como quedan para ver si hay alguna anomalía en 
# los máximos y mínimos


# In[18]:


df5.isna().sum()


# 0.2 **Valores NaN** 
# 
# En esta sección vamos a completar las variables del motivo del retraso que tiene varios NaN. 
# Como ya pudimos observar en el ejercicio 9.2, los vuelos con retrasos de menos de 14 minutos, tienen valores Nan
# 
# Carrier Delay es el retraso de la compañía 
# 
# WeatherDelay es el retraso por las condiciones climatológicas
# SecurityDelay es el retraso por cuestiones de seguridad
# 
# LateAircraftDelay es el retraso de la misma aeronave. 
# 
# Nas delay son los retraso causado por el Sistema Nacional del Espacio Aéreo (NAS)
# 
# por lo que vamos a asigarn 0 a los valores Nan
# 

# In[19]:


df_delay= df5[['ArrDelay','DepDelay','CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay',]]
df5_0= df5.drop(  ["CarrierDelay", 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay'], axis=1 )
delay_not_NAN= df_delay[['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay','LateAircraftDelay']].fillna(0.0)


# In[20]:


df5= pd.concat([df5_0, delay_not_NAN], axis =1)
df5.head(10)


# In[21]:


df5.isna().sum()


# 0.3 **dos nuevas variables.**
# 
# Se puede ver a simple vista que ArrDelay y DepDelay se obtienen de la resta entre (DepTime-CRSDepTime)y	(ArrTime-CRSArrTime), así que calculamos dos nuevas variables, para reducir la dimensionalidad. 
# 
# 
# Primero miramos el valor más bajo de ArrDelay, para poder hacer los cálculo correctamente 

# In[22]:


minimo=df[df["ArrDelay"]<0].sort_values("ArrDelay")# miramos los valores más bajos de ArrDelay
zmin=minimo["ArrDelay"].min()
zmin


# In[23]:


def rest(z):
    x=z[0]
    y=z[1]
    if (x < y) & ((x-y)<zmin): 
        t= (1440+x)-y
        return t
    else:
        t= x-y
        return t  
    
x11= df5[["DepTime","CRSDepTime" ]].apply(rest, axis=1)
x10= df5[["ArrTime","CRSArrTime" ]].apply(rest,axis=1)

x10=x10.rename("X10")
x11=x11.rename("X11")
x10.describe()


# In[24]:


x11.describe()


# Sean **x10** y **x11** definidas por 
# 
# **x10= df6["ArrTime"]-df6["CRSArrTime"]**
# 
# **x11= df6["DepTime"]-df6["CRSDepTime"]**
# y tienen una relación  lineal con ArrDelay y DepDeplay respectivamente 
# 

# In[25]:


x1= pd.concat([x10,x11,df5["ArrDelay"],df5["DepDelay"] ],axis=1)
x1.corr()


# In[26]:


# remodelamos el data set con las dos nuevas variables 
df5b=df5.drop(['DepTime', 'CRSDepTime', 'ArrTime', 'CRSArrTime'], axis=1)
df6=pd.concat([df5b, x10,x11], axis =1)
df6.head(10)


# In[27]:


df6.shape


# En el siguiente paso vamos a analizar la multicolinealidad. Para ver si alguna variable tiene un multicolinealidad muy elevada
# VIFi = 1/ (1 -$Ri^2$) donde Ri es el coeficiente de  determinación de la regresión lineal 

# In[28]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(X):
    vifDF = pd.DataFrame()
    vifDF["variables"] = X.columns
    vifDF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vifDF
Xvif=df6.drop(["ArrDelay"], axis=1)
round(vif(Xvif),2)


# Podemos observar en las variables independientes que DepDelay tiene un "factor de agrandamiento de la varianza "(VIF) muy elevado
# Así que procedemos a eliminar DepDelay y ver como queda el VIF

# In[29]:


Xvif2=df6.drop(["ArrDelay", "DepDelay"], axis=1)

round(vif(Xvif2),2)


# Podemos observar que el vif ha mejorado notablemente que pero deberíamos tener un Factor de agrandamiento de la varianza por 
# debajo de 5, así que eliminamos la siguiente variable con más VIF que es X11
# 
# 

# In[30]:


Xvif3=df6.drop(["ArrDelay", "DepDelay", "X11"], axis=1)

round(vif(Xvif3),2)


# Estas podrían ser una buena selección de variables pero vamos a ver que nos dice el método Anova para selección de variables en 
# clasificación
# 

# In[31]:


ypreseleccion= df6["ArrDelay"]
def categorizar ( x):
    if x<0 : 
        t= 1
        return t
    else: 
        t=0
        return t
Y0= ypreseleccion.apply( categorizar)
Y0.value_counts()


# In[32]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
Xpreseleccion= df6.drop(["ArrDelay"], axis=1)

for i in range(1,12): # vamos a mirar como f_selection va escogiendo las variables en función del estadístico F
    fs = SelectKBest(score_func=f_classif, k=i)
    XS=fs.fit(Xpreseleccion,Y0)
    

    filter=fs.get_support()
    variables=np.array(Xpreseleccion.columns)
    print(variables[filter])
    print(XS.scores_[filter])
    print ("\n")


# Podemos ver que el método Anova puntúa para la clasificación, de mejor manera las variables DepDelay y X11, obviamente con 
# más correlación lineal que el resto de variables, y para evitar la multicolinealidad y el ruido, evitaremos usarlas. 
# Probamos a ver, si eliminando alguna variable con poco peso para el test de Anova, conseguimos reducir el VIF, tras varia 
# 
# 

# In[33]:


Xvif4=df6.drop(["ArrDelay", "DepDelay", "X11",  "CarrierDelay"], axis=1)

round(vif(Xvif4),2)


# Podemos observar que elimiando Carrier Delay, que es la octava mejor variable, de once que hay, para el problema de
# clasificación, con el método de Anova,  nos queda una VIF bueno para ajuste.  

# In[34]:


Y0= Y0.rename("ArrDelay")# renombramos Y0 como ArrDelay
scaler = StandardScaler()

df70= pd.DataFrame(scaler.fit_transform(Xvif4), columns=Xvif4.columns )# estandarizamos las variables independientes

df7= pd.concat( [df70,Y0 ], axis=1)
df7.head(10)


# Vamos a ver si la clase ArrDelay está desvalanceada

# In[35]:



print(Y0.value_counts()[0]/(Y0.value_counts()[0]+Y0.value_counts()[1]))
print(Y0.value_counts()[1]/(Y0.value_counts()[0]+Y0.value_counts()[1]))


# In[36]:


df7.describe().round(2)


# Aunque la clase está desbalanceada, no es por debajo del 5%, así que no vamos a manipular el desbalanceo. Por último hacemos la separación en Train y Test

# In[37]:


X_= df7.drop(["ArrDelay"], axis=1)

y_= df7["ArrDelay"]

Xtrain, Xtest, ytrain, ytest = train_test_split( X_, y_, test_size=0.30, random_state=42, shuffle=True)


# - **Ejercicio 1**. 
# 
# Crea al menos tres modelos de clasificación, Haremos una regresión logística, y dos predicciones por Random Forest y red neuronal 

# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
log= LogisticRegression()
rft= RandomForestClassifier()
mlp= MLPClassifier()


# Empezamos por la regresión logística

# In[39]:


log.fit(Xtrain,ytrain)
fxlog=log.predict(Xtest)


# In[40]:


log.score(Xtest, ytest)# devuelve Accuracy


# Seguimos por el Bosque

# In[41]:


rft.fit(Xtrain, ytrain)
fxrft= rft.predict(Xtest)


# In[42]:


rft.score(Xtest, ytest)


# Acabamos con el la red neuronal

# In[43]:


mlp.fit(Xtrain, ytrain)
fxmlp= mlp.predict(Xtest)


# In[44]:


mlp.score(Xtest, ytest)


# Podemos observar que tanto la Red Neuronal como el Bosque aleatorio tienen mejor precisión (accuracy) que la regresión logística
# 
# Vamos a intentar analizar mejor el error con otros estimadores

# 2. Cálculo de métricas. 
# 
# calcula la accuracy, la matriz de confianza , y otras métricas más avanzadas

# In[45]:


from sklearn.metrics import accuracy_score


# Calculamos la accuracy para los tres modelos de clasificación

# In[46]:


print ( " precisión para la regresión logísticas es : ", accuracy_score( ytest, fxlog))
print ( " precisión para el Bosque aleatorio es : ", accuracy_score( ytest, fxrft))
print ( " precisión para la red neuronal es : ", accuracy_score( ytest, fxmlp))


# Que son los resultados obtenidos previamente con score(x,y)

# In[47]:


from sklearn.metrics import confusion_matrix


# -Empezamos por la regresión logística

# In[48]:


cmlog= confusion_matrix(ytest, fxlog)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# Observamos que detecta mucho mejor los vuelos retrasados( clase mayoritaria ), que los que llegan puntuales, clase minoritaria.
# 

# Calculamos la sensibilidad y la especificidad.

# In[49]:


print(" sensibilidad =  ", cmlog[0][0]/(cmlog[0][0]+cmlog[0][1]))
print(" especificidad =  ", cmlog[1][1]/(cmlog[1][1]+cmlog[1][0]))


# Podemos ver que a la regresión logística le cuesta acertar los verdaderos negativos, la clase minoritaria, no llega ni a acertar
# el 60% de los vuelos que llegarán puntuales.  Si para la compañía, es más importante conocer los vuelos que van a llegar tarde, para preveer los costes generados por los retrasos, es una buena clasificación. 

# - Bosque Aleatorio

# In[50]:


cmrft= confusion_matrix(ytest, fxrft)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmrft.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmrft, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# Observamos que el índice de acierto es casi perfecto, con sensibilidad y especificad igual a uno(aprox) 
# 

# - Red Neuronal 

# In[51]:


cmmlp= confusion_matrix(ytest, fxmlp)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmmlp.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmmlp, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# Cómo en el bosque aleatorio la sensibilidad casi igual a 1 y la especificidad igual a 1

# **Curva Roc y Area bajo al curva**
# 
# Vamos a comrpobar la precisión de la curva roc y el área bajo la curva 

# - Regresión logística

# In[56]:


probalog= log.predict_proba(Xtest)
from sklearn import metrics
FP, TP, thresholds = metrics.roc_curve(ytest, probalog[::,1])

plt.plot(FP, TP)
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.2])

plt.title('curva ROC para vuelos con retrasos ')
plt.xlabel('Tasa de falsos positivos)')
plt.ylabel('Sensibilidad ')
plt.grid(True)


# In[53]:


# calculamos el área bajo la curva 
area= metrics.roc_auc_score(ytest,probalog[:,1])
print ("el área bajo la curva es igual a : " , area)


# In[54]:


probarft= rft.predict_proba(Xtest)
from sklearn import metrics
FP, TP, thresholds = metrics.roc_curve(ytest, probarft[:,1])

plt.plot(FP, TP)
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.2])

plt.title('curva ROC para vuelos con retrasos ')
plt.xlabel('Tasa de falsos positivos)')
plt.ylabel('Sensibilidad ')
plt.grid(True)


# In[55]:


probamlp= mlp.predict_proba(Xtest)
from sklearn import metrics
FP, TP, thresholds = metrics.roc_curve(ytest, probamlp[:,1])

plt.plot(FP, TP)
plt.xlim([-0.1, 1.0])
plt.ylim([0.0, 1.2])

plt.title('curva ROC para vuelos con retrasos ')
plt.xlabel('Tasa de falsos positivos)')
plt.ylabel('Sensibilidad ')
plt.grid(True)


# En los casos de del Bosque Aleatorio y de la Red Neuronal, el Área bajo la curva es 1 ya que los falsos Positivos son cero

# - Ejercicio 3.
# 
# Entrena los modelos usando distintos parámetros. 

# - **Regresión logística.** 
# 
# En este caso voy a intentar mejorar el rendimiento, usando el cálculo completo de la matriz Hessiana mediante el método de 
# Newton, ya que los datos no están normalizados, y por bibliografía encontrada, parece que el método de del cálculo del gradiente
# funciona de manera más optima. 
# 
# Pondría una penalización L1, ya que sabemos que no hay multicolinealidad, pero no es admisible con Newton. 
# 
# 

# In[58]:


log1= LogisticRegression(solver= "newton-cg")


# In[59]:


log1.fit(Xtrain,ytrain)
fxlog1=log1.predict(Xtest)


# In[60]:


# miramos la matriz de confusión 
cmlog1= confusion_matrix(ytest, fxlog1)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog1.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog1, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# In[64]:


print(" sensibilidad =  ", cmlog1[0][0]/(cmlog1[0][0]+cmlog1[0][1]))
print(" especificidad =  ", cmlog1[1][1]/(cmlog1[1][1]+cmlog1[1][0]))


# In[76]:


print(Xtrain.columns)
print (log1.coef_)


# Sacamos los mismo resultado. Así que vamos a intentar mejorar el resultado cambiando a una regularización L1, 

# In[96]:


log2= LogisticRegression( solver = "saga",penalty="elasticnet", l1_ratio =0.1, random_state=21, max_iter=2000)


# In[97]:


log2.fit(Xtrain,ytrain)
fxlog2=log2.predict(Xtest)


# In[123]:


# miramos la matriz de confusión 
cmlog2= confusion_matrix(ytest, fxlog2)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog2.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog2, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# In[99]:


print(" sensibilidad =  ", cmlog2[0][0]/(cmlog2[0][0]+cmlog2[0][1]))
print(" especificidad =  ", cmlog2[1][1]/(cmlog2[1][1]+cmlog2[1][0]))


# Para intentar otra solución, vamos a intentar penalizar el modelo, aumentando el  sobreajuste de los datos
# 

# In[162]:


log4= LogisticRegression(  C= 100000)


# In[164]:


log4.fit(Xtrain,ytrain)
fxlog4=log4.predict(Xtest)


# In[165]:


# miramos la matriz de confusión 
cmlog4= confusion_matrix(ytest, fxlog4)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog4.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog4, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# Podemos ver que tampoco hemos mejorado mucho la precisión de los falsos positivos incluso con sobreajuste

# In[121]:


log5= LogisticRegression(solver= "newton-cg", multi_class= "multinomial")


# In[122]:


log5.fit(Xtrain,ytrain)
fxlog5=log5.predict(Xtest)


# In[155]:


# miramos la matriz de confusión 
cmlog5= confusion_matrix(ytest, fxlog5)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog5.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog5, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# Sospecho que una regularización del tipo L1 es más importante que el método de solución(Newton, Saga,etc...), penalizando la variables poco importantes con L1

# In[168]:


log6= LogisticRegression( solver = "saga",penalty="l1", random_state=21, max_iter=3000, C=100)


# In[169]:


log6.fit(Xtrain,ytrain)
fxlog6=log6.predict(Xtest)


# In[170]:


# miramos la matriz de confusión 
cmlog6= confusion_matrix(ytest, fxlog6)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmlog6.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmlog6, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# In[171]:


print(" sensibilidad =  ", cmlog6[0][0]/(cmlog6[0][0]+cmlog6[0][1]))
print(" especificidad =  ", cmlog6[1][1]/(cmlog6[1][1]+cmlog6[1][0]))


# Se mejora ligeramente la especificidad, pero nada destacable. 
# 

# - Bosque Aleatorio

# Al igual que en la red neuronal, el bosque aleatorio funciona muy bien. Mi idea principal, sería modificar un parámetro para cada uno, ya que mejorar la sensibilidad y la especificidad, hasta encontrar el modelo que detecte el único falso negativo( contemos que ambos casos tiene sólo un falso negativo), lo veo innecesario. Al final, tal cómo he dejado el Data Set, con que el modelo detecte si X10 es mayor o menor que cero, es suficiente para que funcione y clasificamente correctamene. 
# Pero bien podría ser que el modelo funcionosae bien para esos datos de test, así que vamos a adelanter el ejercicio 4 y vamos 
# a hacer una validación cruzada. 

# - Ejercicio 4. 
# 
# Comprobamos la validación cruzada en el train. con cuatro métricas

# In[183]:


from sklearn.model_selection import cross_validate


# In[176]:


metric = ["accuracy", "precision", "recall", "roc_auc"] # las 4 métricas


# In[177]:


CVlog = cross_validate( log , Xtrain, ytrain, cv=5, scoring=metric, return_train_score=True)
        


# In[178]:


sorted(CVlog.keys())


# In[181]:


test = ["test_accuracy","test_precision","test_recall" , "test_roc_auc"]
for k in test:
    print ( k , ":", CVlog[k])


# Observamos unos valores coherentes con lo predicho anteriormente, y ninguna parte del train falla

# - Bosque Aleatorio 

# In[182]:


CVrft = cross_validate( rft , Xtrain, ytrain, cv=5, scoring=metric, return_train_score=True)


# In[184]:



for k in test:
    print ( k , ":", CVrft[k])


# Podemos observar lo esperado, así como en la **red neuronal** que viene tras esto. 

# In[185]:


CVmlp = cross_validate( mlp , Xtrain, ytrain, cv=5, scoring=metric, return_train_score=True)


# In[186]:



for k in test:
    print ( k , ":", CVmlp[k])


# Seguimos con el ejercicio 3. Haremos un par de probaturas en la red neuronal y el bosque Aleatorio. Vamos a intentar empeorar el rendimiento de ambos, ya que mejorarlo epodría llevar mucho trabajo.
# 
# En el caso del Bosque aleatorio optaremos por menos estimadores, el método de convergencia por entropía y límite de profundidad.
# En el caso de la Red Neuronal, usaremos relu para activación, y la convergencia lbfgs, apta para datasets pequeños. 

# In[188]:


rft2= RandomForestClassifier(n_estimators= 40, criterion="entropy", max_depth= 3)
mlp2= MLPClassifier(hidden_layer_sizes= (50), activation="relu", solver="lbfgs")


# In[189]:


rft2.fit(Xtrain, ytrain)
fxrft2= rft2.predict(Xtest)


# In[190]:


mlp2.fit(Xtrain, ytrain)
fxmlp2= mlp2.predict(Xtest)


# In[191]:


cmrft2= confusion_matrix(ytest, fxrft2)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmrft2.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmrft2, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# In[192]:


cmmlp2= confusion_matrix(ytest, fxmlp2)

group_names = ["TP","FN","FP","TN"]
group_counts = ["{0:0.0f}".format(value) for value in
                cmmlp2.flatten()]
labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_names,group_counts)]
labels = np.asarray(labels).reshape(2,2)
ax = sns.heatmap(cmmlp2, annot=labels, fmt="" ,cmap='Blues')

ax.set_title('Confusion Matrix ');
ax.set_xlabel('\n Valores predichos')
ax.set_ylabel('Valores reales');




ax.xaxis.set_ticklabels(['True','False'])
ax.yaxis.set_ticklabels(['True','False'])

plt.show()


# In[193]:


Xtest.shape


# Conclusiones. 
# 
# Primero: El dataset, una vez transformado, y observado que ArrDelay es igua a la resta de ArrTime y CRSArrtime, X10, el problema de clasificación se reduce a clasificar en función de signo de X10. 
# 
# Segundo: La Red Neuronal y el Bosque aleatorio han clasificado todas bien, menos un falso negativo. Incluso empeorando ambos modelos a través de los parámetros,  el fallo ha estado en 22 contra 116508. 
# 
# Tercero. La regresión logística no ha clasificado tan bien un problema de clasificación tan simple al clasificar muchos falsos positivos. Incluso discriminando variables poco importantes para la clasificación, usando la regularización L1, ha seguido clasificando muchos falsos positivos.  Seguramente un data Set más limpio de variables habría obtenido mejores resultados. 

# In[ ]:





# In[ ]:




