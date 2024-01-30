import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

weather = pd.read_csv('C:\\Users\\ossri\\Downloads\\kolweather.csv')
weather.columns = weather.columns.str.lower()
print(weather.apply(pd.isnull).sum())

weather['datetime_column']=pd.to_datetime(weather['datetime'])
weather['year'] = weather['datetime_column'].dt.year
weather['month'] = weather['datetime_column'].dt.month
weather['day'] = weather['datetime_column'].dt.day

grouped_data = weather.groupby(['year', weather['datetime_column'].dt.day])['mean'].mean()
# Iterate through each group and plot
for year_month, group_data in grouped_data.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label=str(year_month), marker='o')
# Add labels
plt.xlabel('Day of the Year')
plt.ylabel('Mean Temperature')
plt.title('Daily Mean Temperature Grouped by Year')
plt.legend(title='Year')
plt.show()

weather=weather.ffill()
rr=Ridge(alpha=.1)
weather["targetmin"]=weather.shift(-1)["min"]
weather["targetmax"]=weather.shift(-1)["max"]
weather["targetmean"]=weather.shift(-1)["mean"]
weather["targethum"]=weather.shift(-1)["humidity"]
weather["targetppt"]=weather.shift(-1)["precipitation"]
predictors=weather.columns[~weather.columns.isin(['datetime','year','datetime_column','targetmin', 'targetmax', 'targetmean', 'targethum', 'targetppt'])]

target_cols = ['targetmin', 'targetmax', 'targetmean', 'targethum', 'targetppt']
def backtest(weather,model,predictors,target_cols,start=365, step=30):
    all_predictions=[]

    for i in range(start,weather.shape[0],step):
            train=weather.iloc[:i,:]
            test=weather.iloc[i:(i+step),:]

            model.fit(train[predictors],train[target_col])
            preds=model.predict(test[predictors])
            preds=pd.Series(preds,index=test.index)
            combined=pd.concat([test[target_col],preds],axis=1)
            combined.columns=["actual","prediction"]
            combined["prediction"]=combined["prediction"].abs()
            combined["diff"]=(combined["prediction"]-combined["actual"]).abs()
            all_predictions.append(combined)
    return(pd.concat(all_predictions))
for target_col in  target_cols :
    backtest(weather, rr, predictors, target_cols)



def pct_diff(old,new):
    return((new-old)/old)

def compute_rolling(weather,horizon,col):
    label=f"rolling_{horizon}_{col}"
    weather[label]=weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"]=pct_diff(weather[col],weather[label])
    return weather
rolling_horizon=[5,7]
for horizon in rolling_horizon:
    for col in ['min']:
        weather=compute_rolling(weather,horizon,col)
# print(weather)
weather=weather.iloc[14:,:]
weather=weather.fillna(0)
def expand_mean(weather):
    return weather.expanding(1).mean()
for col in['targetmin', 'targetmax', 'targetmean']:
    weather[f"month_avg{col}"]=weather[col].groupby(weather.month,group_keys=False).apply(expand_mean)
    weather[f"day_avg{col}"]=weather[col].groupby(weather.day,group_keys=False).apply(expand_mean)

predictors=weather.columns[~weather.columns.isin(['datetime','year','datetime_column','targetmin', 'targetmax', 'targetmean', 'targethum', 'targetppt'])]
for target_col in  target_cols :
    predictions=backtest(weather, rr, predictors, target_cols)
    print('The mean absolute error in ',target_col,'is',mean_absolute_error(predictions["actual"],predictions["prediction"]))

    if target_col=='targetmean':
        # print(predictions.sort_values("diff",ascending=False))
        # print(predictions['diff'].round().value_counts())

        column_series = pd.Series(predictions['actual'])

        # Check if any entry in the column is 0
        if (column_series == 0).any():
            print("There is at least one entry with value 0 in the column.")
        else:
            print("There are no entries with value 0 in the column.")
# Plotting a graph for only 2022
grouped_data_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['mean'].mean()
grouped_data1_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['targetmean'].mean()

for year_month, group_data in grouped_data_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Actual', color='b', marker='o')
for year_month, group_data in grouped_data1_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Predicted', color='r', marker='o')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Actual vs. Prediction Over Time for the Mean Temperature of 2022')
plt.legend()
plt.show()

grouped_data_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['max'].mean()
grouped_data1_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['targetmax'].mean()

for year_month, group_data in grouped_data_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Actual', color='b', marker='o')
for year_month, group_data in grouped_data1_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Predicted', color='r', marker='o')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Actual vs. Prediction Over Time for the Max Temperature of 2022')
plt.legend()
plt.show()

grouped_data_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['min'].mean()
grouped_data1_2022 = weather[weather['year'] == 2022].groupby(['year', weather['datetime_column'].dt.day])['targetmin'].mean()

for year_month, group_data in grouped_data_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Actual', color='b', marker='o')
for year_month, group_data in grouped_data1_2022.groupby(level=0):
    plt.plot(group_data.index.get_level_values('datetime_column'), group_data.values, label='Predicted', color='r', marker='o')
plt.xlabel('Date')
plt.ylabel('Values')
plt.title('Actual vs. Prediction Over Time for the Min Temperature of 2022')
plt.legend()
plt.show()