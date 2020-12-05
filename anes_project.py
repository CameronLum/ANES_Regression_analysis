# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:54:23 2020

@author: Cameron
"""
import pandas as pd
import numpy as np
import pyreadstat as pr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf



# To convert dta file to csv and save it to folder, run the following two lines 
# data = pd.io.stata.read_stata('anes_timeseries_2016.dta')
# data.to_csv('anes_timeseries_2016.csv')

#loading csv with only needed columns
df = pd.read_csv('anes_timeseries_2016.csv', usecols = ['V161310x', 'V161342', 'V161126', 'V161158x', 'V161267', 
'V161270', 'V161361x', 'V161003', 'V161004', 'V161008', 'V161009', 'V161114x', 'V161194x', 'V161204x', 'V161233x', 
'V162150x', 'V162176x', 'V162295x', 'V162345', 'V162346', 'V162347', 'V162348', 'V162349', 'V162350', 'V162351', 
'V162352', 'V162314', 'V162312', 'V162311', 'V162310', 'V160102'])

#renaming columns 
df = df.rename(columns = {'V161310x': 'race', 'V161342': 'gender', 'V161126': 'ideology', 'V161158x': 'party_id', 
'V161267': 'age', 'V161270': 'education', 'V161361x': 'income', 'V161003': 'attn_politics_and_elections', 
'V161004': 'campaing_interest', 'V161008': 'days_watch_news', 'V161009': 'attn_news', 'V161114x': 'health_care_reform_opinion', 
'V161194x': 'birthright_citizenship_opinion', 'V161204x': 'affirmative_action_opinion', 'V161233x': 'death_penalty_opinion', 
'V162150x': 'gender_equal_pay_opinion', 'V162176x': 'free_trade_opinion', 'V162295x': 'torture_opinion', 
'V162345': 'stereotype_whites_hardworking', 'V162346': 'stereotype_blacks_hardworking', 'V162347': 'stereotype_hispanics_hardworking', 
'V162348': 'stereotype_asians_hardworking', 'V162349': 'stereotype_whites_violent', 'V162350': 'stereotype_blacks_violent', 
'V162351': 'stereotype_hispanics_violent', 'V162352': 'stereotype_asians_violent', 'V162314': 'feel_therm_whites', 'V162312': 'feel_therm_blacks', 
'V162311': 'feel_therm_hispanics', 'V162310': 'feel_therm_asians', 'V160102': 'weights'})

# masking by white respondents
df = df[df['race'].str.startswith('1.', na = False)]



#DATA CLEANING*********************************************************************************************************

#use to drop invalid values
def drop_values(column_name, str_starts_with):
    global df
    df = df[~df[column_name].str.startswith(str_starts_with)]
    
    
drop_values('stereotype_whites_hardworking', '-')
drop_values('stereotype_blacks_hardworking', '-')
drop_values('stereotype_hispanics_hardworking', '-')
drop_values('stereotype_asians_hardworking', '-')
drop_values('stereotype_whites_violent', '-')
drop_values('stereotype_blacks_violent', '-')
drop_values('stereotype_hispanics_violent', '-')
drop_values('stereotype_asians_violent', '-')
drop_values('feel_therm_whites', '-') #rows 3038 -> 2592
drop_values('feel_therm_blacks', '-') #rows 2592 -> 2589
drop_values('feel_therm_hispanics', '-')
drop_values('feel_therm_asians', '-')#rows 2586 -> 2583
drop_values('health_care_reform_opinion', '-')
drop_values('ideology', '-')
drop_values('party_id', '-')
drop_values('birthright_citizenship_opinion', '-')
drop_values('affirmative_action_opinion', '-')
drop_values('death_penalty_opinion', '-')
drop_values('age', '-')
drop_values('education', '-')
drop_values('gender', '-')
drop_values('income', '-')
drop_values('gender_equal_pay_opinion', '-')
drop_values('free_trade_opinion', '-')
drop_values('torture_opinion', '-')


#MISC CLEANING
df = df[~df['gender'].str.contains('Other')]


# removing everything but the first character in each string so that it can be converted to a float
def str_index_on_df_column(column_name, i, j):
    df[column_name] = df[column_name].str[i:j]

str_index_on_df_column('health_care_reform_opinion', 0, 1)
str_index_on_df_column('ideology', 0, 1)
str_index_on_df_column('party_id', 0, 1)
str_index_on_df_column('birthright_citizenship_opinion', 0, 1)
str_index_on_df_column('affirmative_action_opinion', 0, 1)
str_index_on_df_column('death_penalty_opinion', 0, 1)
str_index_on_df_column('age', 0, 4)
str_index_on_df_column('education', 0, 2)
str_index_on_df_column('gender', 0, 1)
str_index_on_df_column('gender_equal_pay_opinion', 0, 1)
str_index_on_df_column('free_trade_opinion', 0, 1)
str_index_on_df_column('torture_opinion', 0, 1)
str_index_on_df_column('income', 0, 1)

# recoding values for stereotype questions so they can be converted to floats
def recode_stereotype_scale(column_name):
    for i in df[column_name]:
        if i == '1. Hard-working':
            df[column_name] = df[column_name].replace({'1. Hard-working': 1.0})
        elif i == '7. Lazy':
            df[column_name] = df[column_name].replace({'7. Lazy': 7.0})
        elif i == '1. Peaceful':
            df[column_name] = df[column_name].replace({'1. Peaceful': 1.0})
        elif i == '7. Violent':
            df[column_name] = df[column_name].replace({'7. Violent': 7.0})

recode_stereotype_scale('stereotype_whites_hardworking')
recode_stereotype_scale('stereotype_blacks_hardworking')
recode_stereotype_scale('stereotype_hispanics_hardworking')
recode_stereotype_scale('stereotype_asians_hardworking')
recode_stereotype_scale('stereotype_whites_violent')
recode_stereotype_scale('stereotype_blacks_violent')
recode_stereotype_scale('stereotype_hispanics_violent')
recode_stereotype_scale('stereotype_asians_violent')        
 
#converting strings to floats, have to do this because values were all imported as strings       
def convert_to_float(column_name):
    df[column_name] = df[column_name].astype(float)

convert_to_float('feel_therm_whites')
convert_to_float('feel_therm_blacks')
convert_to_float('feel_therm_hispanics')
convert_to_float('feel_therm_asians')
convert_to_float('stereotype_whites_hardworking')
convert_to_float('stereotype_blacks_hardworking')
convert_to_float('stereotype_hispanics_hardworking')
convert_to_float('stereotype_asians_hardworking')
convert_to_float('stereotype_whites_violent')
convert_to_float('stereotype_blacks_violent')
convert_to_float('stereotype_hispanics_violent')
convert_to_float('stereotype_asians_violent')

convert_to_float('health_care_reform_opinion')
convert_to_float('ideology')
convert_to_float('party_id')
convert_to_float('birthright_citizenship_opinion')
convert_to_float('affirmative_action_opinion')
convert_to_float('death_penalty_opinion')
convert_to_float('age')
convert_to_float('income')
convert_to_float('education')
convert_to_float('gender')
convert_to_float('gender_equal_pay_opinion')
convert_to_float('free_trade_opinion')
convert_to_float('torture_opinion')

# CONVERTS SCALE FROM (1 to 7) to (-3 to 3)
def shift_seven_point_scale(column_name):
    df[column_name] = -(df[column_name]) + 4

shift_seven_point_scale('stereotype_whites_hardworking')
shift_seven_point_scale('stereotype_blacks_hardworking')
shift_seven_point_scale('stereotype_hispanics_hardworking')
shift_seven_point_scale('stereotype_asians_hardworking')
shift_seven_point_scale('stereotype_whites_violent')
shift_seven_point_scale('stereotype_blacks_violent')
shift_seven_point_scale('stereotype_hispanics_violent')
shift_seven_point_scale('stereotype_asians_violent')
shift_seven_point_scale('health_care_reform_opinion')
shift_seven_point_scale('birthright_citizenship_opinion')
shift_seven_point_scale('affirmative_action_opinion')
shift_seven_point_scale('gender_equal_pay_opinion')
shift_seven_point_scale('free_trade_opinion')
shift_seven_point_scale('torture_opinion')



df = df[df['education'] < 20]

#CONVERT PARTY ID INTO REPULICAN AND DEMOCRAT AND INDEPENDENT 1-3 dem, 5-7 R
df.loc[df['party_id'] == 1.0, 'party_id'] = 0
df.loc[df['party_id'] == 2.0, 'party_id'] = 0
df.loc[df['party_id'] == 3.0, 'party_id'] = 0

df.loc[df['party_id'] == 4.0, 'party_id'] = 1 #indep

df.loc[df['party_id'] == 5.0, 'party_id'] = 2
df.loc[df['party_id'] == 6.0, 'party_id'] = 2
df.loc[df['party_id'] == 7.0, 'party_id'] = 2

 
# MAKING GENDER A DUMMY VARIABLE, Male = 1 and Female = 0 
df['gender'] = -(df['gender']) + 2

#CALCULATING OUR FIRST ETHNOCENTRISM SCALE, E, USING STEREOTYPE ANSWERS
df['E'] = ((df.stereotype_whites_hardworking - 
            ((df.stereotype_blacks_hardworking + df.stereotype_hispanics_hardworking + df.stereotype_asians_hardworking) / 3)
            + (df.stereotype_whites_violent - ((df.stereotype_blacks_violent + df.stereotype_hispanics_violent + df.stereotype_asians_violent)/ 3))) / 2)

# SCALING E DOWN TO -1  +1 (+1 -> most ethnocentric)
df['E'] = df['E'] / 6

# CALCULATING SECOND ETHNOCENTRISM SCALE, E* USING FEELING THERMOMETER SCORES
#TRANSFORMED -> RANGE 7-13 13 REPRESENTING MOST ETHNOCENTRIC
df['E*'] = (df.feel_therm_whites - ((df.feel_therm_blacks + df.feel_therm_hispanics + df.feel_therm_asians) / 3))
df['E*_transformed'] = (df['E*'] + 74).apply(np.sqrt)





#PLOTTING********************************************************************************************************

df_feel_therm = df[['feel_therm_whites','feel_therm_blacks','feel_therm_hispanics','feel_therm_asians']]
df_stereotype_hardworking = df[['stereotype_whites_hardworking','stereotype_blacks_hardworking','stereotype_hispanics_hardworking','stereotype_asians_hardworking']]
df_stereotype_violent = df[['stereotype_whites_violent','stereotype_blacks_violent','stereotype_hispanics_violent','stereotype_asians_violent']]

# X = E*TRANSFORMED , Y = INCOME, ANOTHER Y = IDEOLOGY, ANOTHER Y = PARTY_ID AND HEALTHCARE REFORM
p1 = sns.regplot('E*_transformed', 'health_care_reform_opinion', data = df, scatter=True, fit_reg=True)
plt.title('E* vs HealthCare Reform Opinion')
p2 = sns.lmplot('E*_transformed', 'income', data = df,  scatter=True, fit_reg=True)
plt.title('E* vs Income')
p3 = sns.lmplot('E*_transformed', 'ideology', data = df, hue='party_id', scatter=True, fit_reg=True)
plt.title('E* vs Ideology')
p4 = sns.lmplot('E*_transformed', 'health_care_reform_opinion', hue='gender', data = df, scatter=True, fit_reg=True)
plt.title('E* vs HealthCare Reform opinion by Gender')
p5 = sns.lmplot('E*_transformed', 'health_care_reform_opinion', hue='party_id', data = df, scatter=True, fit_reg=True)
plt.title('E* vs Healthcare Reform Opinion by party ID: 0=Dem 2=REP, 1=INDEP')
p6 = sns.lmplot('E*_transformed', 'education', hue='party_id', data = df, scatter=True, fit_reg=True)
plt.title('E* vs Education X partyID: 0=Dem 2=REP, 1=INDEP')

plt.show(p1)
plt.show(p2)
plt.show(p3)
plt.show(p4)
plt.show(p5)
plt.show(p6)

#PIE CHARTS*************************************************************************************************************
import plotly.io as pio
import plotly.express as px
pio.renderers.default = "svg" #sets to open as svg in plots tab instead of browser

health_labels = {-3: 'strongly oppose', -2: 'oppose', -1: 'slightly oppose', 0: 'neutral', 1: 'slightly favor', 2: 'favor', 3: 'favor strongly'}

SO = df.loc[df.health_care_reform_opinion == -3, 'health_care_reform_opinion'].count()
O = df.loc[df.health_care_reform_opinion == -2, 'health_care_reform_opinion'].count()
SLO = df.loc[df.health_care_reform_opinion == -1, 'health_care_reform_opinion'].count()
N = df.loc[df.health_care_reform_opinion == 0, 'health_care_reform_opinion'].count()
SLF = df.loc[df.health_care_reform_opinion == 1, 'health_care_reform_opinion'].count()
F = df.loc[df.health_care_reform_opinion == 2, 'health_care_reform_opinion'].count()
SF = df.loc[df.health_care_reform_opinion == 3, 'health_care_reform_opinion'].count()
#dict_health = {'strongly oppose': SO,'oppose':O,'slightly oppose':SLO,'neutral':O,'slightly favor':SF, 'favor':F, 'favor strongly':SF}   
#df_health = pd.DataFrame(data=dict_health.values(), columns=dict_health.keys()) 
h_values = np.array([SO,O,SLO,N,SLF,F,SF])
h_keys = ['SO', 'O', 'SLO','N','SLF','F','SF']
pie = px.pie(data_frame=h_values, names=h_keys,  title='Distribution of Healthcare Opinion')
pie.show()

plt.pie(h_values, labels=h_keys)
plt.title('Distribution of Healthcare Reform Answers')
plt.show()

sns.barplot(x=h_keys,y=h_values)
plt.title('Distribution of Healthcare Reform Answers')



#DO THE THING****************************************************************************************************
from sklearn.linear_model import LinearRegression

#drop NaN from df on E*_transformed
df = df[df['E*_transformed'].notna()]

X = df[['E*_transformed', 'party_id', 'ideology', 'education', 'age', 'gender', 'income']]
X1= df['E*_transformed']
df_E = df[['E', 'party_id', 'ideology', 'education', 'age', 'gender', 'income']]

#divide data into train and test MULTIVARIABLE LINEAR REG
from sklearn.model_selection import train_test_split

y = df['health_care_reform_opinion']

X_train, X_test,y_train,y_test = train_test_split(X,y, test_size=.20)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#statsmodelss
Xc = sm.add_constant(1)
regressor1 = sm.OLS(y, X).fit()
print(regressor1.summary())



#output descriptive statistics
stat_E = X.describe()
stat_feel = df_feel_therm.describe()
stat_hard_working = df_stereotype_hardworking.describe()
stat_violent = df_stereotype_violent.describe()
stat_stereotype = pd.concat([stat_hard_working, stat_violent], axis = 1) 
print('E calc and descriptive statistics:')
print(stat_E)
print('')
print('Feeling Thermometer and Stereotype question statisitics')
print(stat_stereotype)


#output the regressor coefficients
reg_score = regressor.score(X_test, y_test)
train_score = regressor.score(X_train, y_train)
df_reg_coef = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])


print("Regression Train Intercept:")
print(regressor.intercept_)
print("")
print("Regression Test score:")
print(regressor.score(X_test, y_test))
print("")
print('Regression variable Coefficients:')
print(df_reg_coef)




#save figures

df_reg_coef.to_csv('RegressionCoefficients.csv')
stat_E.to_csv('EthnocentrismSummary.csv')
stat_stereotype.to_csv('StereotypeSummary.csv')
df_stat = df.describe()
df.to_csv('CleanedDF.csv')
df_stat.to_csv('DFStatistics.csv')


























#plotting
# plt.scatter(X_train['E'], y_train, color = "red")
# plt.plot(X_train['E'], regressor.predict(X_train), color = 'green')
# plt.title('Multiple Regression E Predicting Health Care Reform')
# plt.xlabel(' E*') # , party_id, ideology, education, age, gender, income
# plt.ylabel("Health Care Reform Opinion")
# plt.show()

# X = E*TRANSFORMED , Y = INCOME, ANOTHER Y = IDEOLOGY, ANOTHER Y = PARTY_ID AND HEALTHCARE REFORM
#df_plot = df[['E', 'E*_transformed', 'health_care_reform_opinion', 'party_id', 'ideology', 'education', 'age', 'gender', 'income']]
#sns.pairplot(df_plot, dropna="True", kind='reg', diag_kind = 'auto', )  # kind : {'scatter', 'reg'}, diag_kind : {'auto', 'hist', 'kde', None},

#fig, axes = plt.subplots(ncols= 8)
#sns.pairplot(df_healthcare, df['E*_transformed'])


#sns.lmplot(x = X['E*_transformed'], y = y, data = y)




#clean political awareness cols
#days_watch_news = 1. - 7. being days per week watch news
#attn_news: 1-4 4 = a little 1 = A great Deal
#campaining_interest: 1-3 1=very much interested, 3=NotMuch Interested
#attn_politics_and_elections: 5=never,4=some of the time, 1= Always

#person control variables 
#education1-16 16 phd, gender:0,1 1Male, income 1-28, age, 
#df['income_num'] = str_index_on_df_column('income', 0,2)

#reg:H1 higher Ethnocentrism scores correlate to greater Healthcare reform opposition
#health opinion = Y = ddependet = var we are trying to predict
#X = E and E*, Independ, var using to make predictions
#sns.pairplot()


# #linear regression between 'health_care_reform_opinion' and E, then E*_transformed
# #sns.regplot(x = df['E'], y = df['health_care_reform_opinion'])
# df_plot = df[['E*_transformed', 'health_care_reform_opinion']]
# #sns.pairplot(df_plot)

# df = df.dropna(subset=['E*_transformed'])


# from sklearn import linear_model
# #X = df_E
# #Y = df_healthcare
# X = df['E*_transformed']
# Y = df['health_care_reform_opinion']
# # print(np.any(np.isnan(df_healthcare)))
# #print(np.any(np.isnan(df_E)))
# lm = linear_model.LinearRegression()
# #model = lm.fit(X,Y)

# #predictions = lm.predict(X) #predicts Y using lm
# #print('R^2 value for the model without weights: ', model.score(X,Y)) 
# #print('R^2 value for the model with weights: ', model.score(X,Y, df['weights']))
# #print(predictions) 


# #STATSMODELS OLS REGRESSION PRINTOUT


# #bivariate correlation
# correlationMatrix = df[['E*_transformed','gender','ideology','party_id','age','education','health_care_reform_opinion','birthright_citizenship_opinion','affirmative_action_opinion',
#                        'gender_equal_pay_opinion','torture_opinion','free_trade_opinion','death_penalty_opinion']]
# p = correlationMatrix.corr()
# print(p)
# sns.heatmap(p, xticklabels=p.columns.values, yticklabels=p.columns.values)
# #convert the correlation matrix into sorted series
# p_sorted = p.unstack()
# print('pSorted')
# print(p_sorted)



























