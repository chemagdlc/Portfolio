# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:35:28 2021

@author: Chemagdlc
"""

reset -f 

import os
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype #For definition of custom categorical data types (ordinal if necesary)
import matplotlib.pyplot as plt
import seaborn as sns  # For hi level, Pandas oriented, graphics
import scipy.stats as stats  # For statistical inference 
from statsmodels.formula.api import ols



os.chdir(r'C:\Users\Chemagdlc\Desktop\Chema\Estudios\WineQuality')
os.getcwd()

db = pd.read_csv ("winequality-red.csv")

db.head()
db.quality.describe() 


#RECODE
db = db.rename(columns={"fixed acidity": "fixed_acidity", "volatile acidity": "volatile_acidity", "citric acid": "citric_acid", "residual sugar": "residual_sugar",
                   "free sulfur dioxide": "free_sulfur_dioxide", "total sulfur dioxide": "total_sulfur_dioxide", "total sulfur dioxid" : "total_sulfur_dioxid"})





'''
TARGET = QUALITY OF THE WINW (1-10)
'''



#Histograma de frecuencia de la calidad
x=db['quality']

plt.hist(x,edgecolor='black',bins=20)
plt.xticks(np.arange(3,9,step=1))
plt.title("figura 1. registro numerico calidad del vino")
plt.ylabel('frecuencia')
plt.xlabel('Notas de calidad')
plt.show()


#Estudio nominal
mytable = db.groupby(['quality']).size()
n= (mytable.sum())
mytable2= (mytable/n)*100

mytable3 = round(mytable2,1)



plt.bar( mytable3.index, mytable3, edgecolor='black') 
plt.ylabel('percentaje')
plt.title('percentaje of rates of quality')
props = dict(boxstyle='round', facecolor='white', lw=0.5)       #caja para leyenda o información
textstr = '$\mathrm{n}=%.0f$'%(n)                               #Texto que aparece en la caja
plt.text (7.5, 40, textstr , bbox=props) 
plt.show()


#Comparación quality con distintas variables


## quality vs fixed acidity

x=db.quality 
y=db.fixed_acidity
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('fixed_acidity')

plt.xlabel('Quality (1-10)')
plt.show()


## quality vs citric acid

x=db.quality 
y=db.citric_acid
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('citric_acid')

plt.xlabel('Quality (1-10)')
plt.show()



## quality vs residual sugar

x=db.quality 
y=db.residual_sugar
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('residual_sugar')

plt.xlabel('Quality (1-10)')
plt.show()


## quality vs chlorides

x=db.quality 
y=db.chlorides
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('chlorides')

plt.xlabel('Quality (1-10)')
plt.show()



## quality vs free sulfur dioxide

x=db.quality 
y=db.free_sulfur_dioxide
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('free_sulfur_dioxide')

plt.xlabel('Quality (1-10)')
plt.show()




## quality vs total sulfur dioxid

x=db.quality 
y=db.total_sulfur_dioxide
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('total_sulfur_dioxide')

plt.xlabel('Quality (1-10)')
plt.show()



## quality vs density

x=db.quality 
y=db.density
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('density')

plt.xlabel('Quality (1-10)')
plt.show()



## quality vs pH

x=db.quality 
y=db.pH
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('pH')

plt.xlabel('Quality (1-10)')
plt.show()






## quality vs sulphates

x=db.quality 
y=db.sulphates
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('sulphates')

plt.xlabel('Quality (1-10)')
plt.show()




## quality vs alcohol

x=db.quality 
y=db.alcohol
tit='Figure .Quality.''\n'
plt.figure(figsize=(5,5))

plt.scatter(x,y, s=20, facecolors='none', edgecolors='C0')
plt.ylabel('alcohol')

plt.xlabel('Quality (1-10)')
plt.show()




#ESTUDIO DE REGRESIÓN 


model1= ols('quality ~ fixed_acidity  ', data=db).fit()

print(model1.summary2())


model2 = ols('quality ~ fixed_acidity + citric_acid', data=db).fit()

print(model2.summary2())



model3 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar', data=db).fit()

print(model3.summary2())



model4 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides', data=db).fit()

print(model4.summary2())




model5 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide', data=db).fit()

print(model5.summary2())



model6 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide', data=db).fit()

print(model6.summary2())




model7 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density', data=db).fit()

print(model7.summary2())




model8 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH', data=db).fit()

print(model8.summary2())




model9 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates', data=db).fit()

print(model9.summary2())




model10 = ols('quality ~ fixed_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol', data=db).fit()

print(model10.summary2())




from stargazer.stargazer import Stargazer                                       #Librelia para generar y comparar modelos de regresión

stargazer = Stargazer([model1, model2, model3, model4, model5, model6, model7, model8, model9, model10])                 #Genera una tabla en html

stargazer.render_html()



