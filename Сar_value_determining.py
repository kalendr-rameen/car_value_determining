#!/usr/bin/env python
# coding: utf-8

# # Определение стоимости автомобилей

# Сервис по продаже автомобилей с пробегом «Не бит, не крашен» разрабатывает приложение для привлечения новых клиентов. В нём можно быстро узнать рыночную стоимость своего автомобиля. В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 
# 
# Заказчику важны:
# 
# - качество предсказания;
# - скорость предсказания;
# - время обучения.

# ## Подготовка данных

# In[1]:


import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor


# In[2]:


df = pd.read_csv('./autos.csv')


# In[3]:


df = df.rename(columns=str.lower)


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df.isna().any()


# In[7]:


df['price'].min()


# Итак как мы можем заметить цена может быть нулевой, что вряд ли является правдой, поэтому давайте сделаем некую отсечку например в 100 евро которую примем за минимальную стоимость. Хотя как я посмотрел средняя масса машина варьируется от 1-1.5 тонны, а стоимость в европе 1кг стали(по большей части из которого состтоит машина) составляет 7евро. Что означает что даже 100 евро очень мало, но так как мы иначе потеряем слишком много данных при отсечке в 700 евро например, что является приблизительной стоимость если сдать машину на металлолом, поэтому я не уверен что отсечка в 100 евро является верной но как видно на боксплоте такие данные не выбиваются от нормальных значений

# In[8]:


df['price'].plot.box()


# In[9]:


df['price'].describe()


# In[10]:


df.drop(df[df['price'] < 100].index, inplace=True)


# Видно что данные о регистрации машины неправильные попытаемся проанализировать данные чтобы решитть эту проблему

# In[11]:


df['registrationyear'].min(), df['registrationyear'].max()


# In[12]:


df


# In[13]:


df['lastseen']= pd.to_datetime(df['lastseen'], format='%Y-%m-%d %H:%M:%S')


# In[14]:


pd.DatetimeIndex(df['lastseen']).year.max()


# Видно что дата последнего посещения это 2016 год поэтому сделаем отсечку по максимуму по этому значению,  минимума по очень старым значений годов например 1930 год когда были уже машины и теоретически такие могли быть выложены как что то очень раритетное

# In[15]:


df['datecrawled']= pd.to_datetime(df['datecrawled'], format='%Y-%m-%d %H:%M:%S')


# In[16]:


pd.DatetimeIndex(df['datecrawled']).year.max()


# In[17]:


df[df['registrationyear'] > 2016].info()


# Можно заметить что если делать отсечку после 2016 года то кол-во пропущенных значений в таких колонках как `vehicletype` `fueltype`, `notrepaired` достаточно велико, что может нас склонять к тому что данные заполненые после 2016 года являются скорее неким выбросом нежели реальными данными, поэтому я все таки решил оставить этот год как значение по которому мы производим отсечку

# In[18]:


df[df['registrationyear'] < 1930]['registrationyear'].values


# In[19]:


df[df['registrationyear'] < 1930].info()


# In[20]:


df.drop(index=df[(df['registrationyear'] > 2016) | (df['registrationyear'] < 1930)].index, inplace=True)


# In[21]:


df.reset_index(inplace=True, drop=True)


# In[22]:


df


# Итак со странностями в данных 'справились' теперь посморим как заполнить пропущенные значения в колонках. Будем заполнять пропущенные значения самыми повторяющимися значения из сводных таблиц или таблиц полученых методом `.groupby()`. Начнём с колонки `model`

# ### `model`-col

# Итак чтобы заполнить колонку `model` я попытался разбить все данные с помощью столбцов `brand, registrationyear, power, price` при помощи построения свободной таблицы, где значения внутри этой таблицы заполняются самым часто встречающимся значением</div>

# In[23]:


p_1 = pd.pivot_table(df, values='model', index=['brand','registrationyear', 'power'],columns='price', aggfunc=pd.Series.mode)


# Затем я создал функцию суть которой состоит в том чтобы разрешать конфликтные моменты если например у меня в выдаче получилось, вместо одного значения модели (напр. c-class), два и более значений (напр. [c-class, e-class,b-class]) или вовсе значение отсутсвует. Данная функция будет выбирать значение с индексом 0 и создавать в конечном итоге список из таких моделей. По сути эта функция аналогична функции `transform()` где данные заполняются в зависимости от выбранной функции для 'трансформирования данных', вообщем надеюсь так понятно)</div>

# In[24]:


def model_determine(list):
    model_list = []
    num = 0
    for i in list:
        if type(p_1.loc[i[0], i[2], i[3]][i[1]]) == np.ndarray:
            num += 1
            try:
                model_list.append(p_1.loc[i[0], i[2], i[3]][i[1]][0])
            except:
                model_list.append('')
        else:
            model_list.append(p_1.loc[i[0], i[2], i[3]][i[1]])
    print(num)
    return model_list    


# При вызове строки снизу показывается число как раз таких конфликтных моментов и создается обьект типа list которым мы будем запонять пропуски в колонке `model`. 

# In[25]:


a = model_determine(df[['brand','price','registrationyear', 'power']].values)


# In[26]:


df['model'] = df['model'].fillna(pd.Series(a))


# In[27]:


df.drop(index = df[df['model'] == ''].index, inplace=True)


# In[28]:


df.reset_index(drop=True, inplace=True)


# In[29]:


df.isna().any()


# Здесь чтобы заполнить колонку `vehicletype` я решил что можно сгрупоировать значения по модели машины ведь скорее всего модели в основном определяют тип кузова(как мне по крайней мере всегда казалось) и выбирать также самое частое значение. Сама же функцию выполняет полностью аналогичную роль с функцией `model_determine`

# ### `vehicletype`-col

# In[30]:


g_1 = df.groupby('model')['vehicletype'].agg(pd.Series.mode)


# In[31]:


def vehicletype_determine(list):
    vehicle_type = []
    num = 0 
    for i in list:
        if type(g_1[i]) == np.ndarray:
            num += 1 
            try:
                vehicle_type.append(g_1.loc[i][0])
            except:
                vehicle_type.append('')
        else:
            vehicle_type.append(g_1.loc[i])
    print(num)        
    return vehicle_type


# In[32]:


a  = pd.Series(vehicletype_determine(df['model'].values))


# In[33]:


df['vehicletype'].fillna(a,inplace=True)


# In[34]:


df.drop(index = df[df['vehicletype'] == ''].index, inplace=True)


# In[35]:


df.reset_index(drop=True, inplace=True)


# In[36]:


df.isna().any()


# ### `gearbox`- col

# In[37]:


p_2 = df.pivot_table(values='gearbox', index='model', columns='registrationyear', aggfunc=pd.Series.mode)


# In[38]:


def gearbox_determine(list):
    model_list = []
    num = 0
    for i in list:
        if type(p_2.loc[i[0]][i[1]]) == np.ndarray:
            num +=1
            try:
                model_list.append(p_2.loc[i[0]][i[1]][0])
            except:
                model_list.append('')
        else:
            model_list.append(p_2.loc[i[0]][i[1]])
    print(num)            
    return model_list    


# In[39]:


a = gearbox_determine(df[['model','registrationyear']].values)


# In[40]:


df['gearbox'] = df['gearbox'].fillna(pd.Series(a))


# In[41]:


df.drop(index = df[df['gearbox'] == ''].index, inplace=True)


# In[42]:


df.reset_index(drop=True, inplace=True)


# In[43]:


df.isna().any()


# ### `notrepaired`-col

# In[44]:


p_3 = df.pivot_table(values='notrepaired', index='kilometer', columns='registrationyear', aggfunc=pd.Series.mode)


# In[45]:


def notrepaired_determine(list):
    new_list = []
    num = 0 
    for i in list:
        if type(p_3.loc[i[0]][i[1]]) == np.ndarray:
            num += 1 
            try:
                new_list.append(p_3.loc[i[0]][i[1]][0])
            except:
                new_list.append('')
        else:
            new_list.append(p_3.loc[i[0]][i[1]])
    print(num)            
    return new_list    


# In[46]:


a = notrepaired_determine(df[['kilometer','registrationyear']].values)


# In[47]:


df['notrepaired'] = df['notrepaired'].fillna(pd.Series(a))


# In[48]:


df.drop(index = df[df['notrepaired'] == ''].index, inplace=True)


# In[49]:


df.reset_index(drop=True, inplace=True)


# In[50]:


df.isna().any()


# ### `fueltype`-col

# In[51]:


p_4 = pd.pivot_table(df, values='fueltype', index=['model', 'power'], columns='registrationyear', aggfunc=pd.Series.mode)


# In[52]:


def fueltype_determine(list):
    new_list = []
    num = 0
    for i in list:
        if type(p_4.loc[i[0], i[2]][i[1]]) == np.ndarray:
            num +=1
            try:
                new_list.append(p_4.loc[i[0], i[2]][i[1]][0])
            except:
                new_list.append('')
        else:
            new_list.append(p_4.loc[i[0], i[2]][i[1]])
    print(num)            
    return new_list    


# In[53]:


a = fueltype_determine(df[['model','registrationyear', 'power']].values)


# In[54]:


df['fueltype'] = df['fueltype'].fillna(pd.Series(a))


# In[55]:


df.drop(index = df[df['fueltype'] == ''].index, inplace=True)


# In[56]:


df.reset_index(drop=True, inplace=True)


# In[57]:


df.isna().any()


# ## Обучение моделей

# Итак сначала уберем из наших таблиц данные который точно не должны повлиять на значение цены(которая является нашим таргетом)

# In[58]:


needless_col = [x.lower() for x in ['DateCrawled', 'DateCreated', 'NumberOfPictures', 'LastSeen']]
df.drop(columns=needless_col, inplace=True)


# Теперь применим технику порядкового кодирования, чтобы перевести категориальные данные в числовые(По правде сказать изначально я сделал этот перевод потому что не мог обучить с категориальными данными модель LightGBM, но после того как перевел заметил что значения на Catboost значительно улучшились, так например RMSE был 1500 на catboost, с категориальными данными, а после стал 300, поэтому решил полностью ко всем моделям это применить, но не совсем понимаю почему результат получается лучше, ведь обе модели рассчитаны на работу с категориальными данными)

# In[59]:


df.info()


# In[60]:


col = list(df.loc[:, df.dtypes == 'object'].columns)


# In[61]:


enc = OrdinalEncoder()
enc.fit(df[col])
df[col] = enc.transform(df[col])


# In[62]:


df_train, df_other = train_test_split(df, test_size=0.4, random_state=12345)


# In[63]:


df_valid, df_test = train_test_split(df_other, test_size=0.5, random_state=12345)


# In[64]:


features_train = df_train.drop(columns='price')
target_train = df_train.price


# In[65]:


features_valid = df_valid.drop(columns='price')
target_valid = df_valid.price


# In[66]:


features_test = df_test.drop(columns='price')
target_test = df_test.price


# Теперь обучим модели на полученныз данных. Возьмем 3 модели: Catboost, LightGBM, LinearRegression. Начнём с первой

# ### Catboost

# In[67]:


model = CatBoostRegressor(iterations=2000, learning_rate=0.05, depth=10, loss_function = 'MAE', eval_metric = 'RMSE')


# In[68]:


start_time = time.time()
model.fit(features_train, target_train, use_best_model=True, silent=True, eval_set=(features_valid, target_valid))
catboost_exec_time = time.time() - start_time


# In[69]:


model.best_score_


# In[70]:


mse_catboost = mean_squared_error(target_test, model.predict(features_test))
rmse_catboost = mse_catboost ** 0.5


# In[71]:


rmse_catboost


# ### LGBMmse_catboost

# In[72]:


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    "num_iterations": 1000,
    "n_estimators": 10000
}


# In[73]:


lgb_train = lgb.Dataset(features_train, target_train)
lgb_eval = lgb.Dataset(features_test, target_test, reference=lgb_train)


# In[74]:


start_time = time.time()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)
lgbm_exec_time = time.time() - start_time


# In[75]:


y_pred = gbm.predict(features_test, num_iteration=gbm.best_iteration)


# In[76]:


mse_lgbm = mean_squared_error(target_test, y_pred)
rmse_lgbm = mse_lgbm ** 0.5


# In[77]:


rmse_lgbm


# ### LinearRegression

# In[78]:


model = LinearRegression(fit_intercept=True, normalize=True)


# In[79]:


start_time = time.time()
model.fit(features_train, target_train)
linear_exec_time = time.time() - start_time


# In[80]:


model.predict(features_test)


# In[81]:


mse_linear = mean_squared_error(target_test, model.predict(features_test))
rmse_linear = mse_linear ** 0.5


# In[82]:


rmse_linear


# ## Анализ моделей

# In[83]:


RMSE = [rmse_catboost, rmse_lgbm, rmse_linear]


# In[84]:


fit_time = [catboost_exec_time, lgbm_exec_time, linear_exec_time]


# In[85]:


pd.DataFrame([RMSE, fit_time], index = ['RMSE', 'Fit time'], columns=['Catboost', 'LightGBM', 'LinearRegression'])


# Видно что лучше всех справился Catboost, но при этом он дольше всех работал, хуже всех себя показала линейная регрессия, но она оказалась самой быстрой. При этом самым оптимальным выбором между качество и временем является LightGBM
