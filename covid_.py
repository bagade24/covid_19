import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from fbprophet import Prophet

df = pd.read_csv (r'/home/vaishu/PycharmProjects/pythonProject/venv/DSet/time_series_covid_19_confirmed.csv')
column_list = list(df)
column_list.remove('Province/State')
column_list.remove('Country/Region')
column_list.remove('Lat')
column_list.remove('Long')
#df['sum'] = df[column_list].sum(axis=1)
column_list=list(df)
column_list.remove('Province/State')
column_list.remove('Lat')
column_list.remove('Long')
df = pd.DataFrame(df, columns=column_list)
df = df.groupby('Country/Region').sum().reset_index('Country/Region')
select_color = df.loc[df['Country/Region'] == 'India']

#select_color = df.loc[['China']]
select_color=select_color.melt(id_vars=['Country/Region'],var_name="Date",value_name="Value")
column_list=list(select_color)
column_list.remove('Country/Region')
select_color = pd.DataFrame(select_color, columns=column_list)
select_color.rename({'Date': 'ds', 'Value': 'y'},axis='columns',inplace =True)
m = Prophet()
m.fit(select_color)
future = m.make_future_dataframe(periods=30)
future.tail()
forecast1 = m.predict(future)
fig1 = m.plot(forecast1)


select_color = df.loc[df['Country/Region'] == 'China']
select_color=select_color.melt(id_vars=['Country/Region'],var_name="Date",value_name="Value")
column_list=list(select_color)
column_list.remove('Country/Region')
select_color = pd.DataFrame(select_color, columns=column_list)
select_color.rename({'Date': 'ds', 'Value': 'y'},axis='columns',inplace =True)
m = Prophet()
m.fit(select_color)
future = m.make_future_dataframe(periods=30)
future.tail()
forecast2 = m.predict(future)
fig2 = m.plot(forecast2)
plt.show()


#df = pd.DataFrame(df, columns= ['Province/State','Confirmed'])





