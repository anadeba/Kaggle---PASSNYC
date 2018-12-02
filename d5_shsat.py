import pandas as pd

d5_shsat_data = pd.read_csv(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\D5 SHSAT Registrations and Testers.csv')

def check_missing_values(df):
   df2 = pd.DataFrame()
   for field in list(df):
       df2 = df2.append(df[[field]].isnull().sum().reset_index())
   df2[0] = (df2[0]/df.shape[0])*100
   df2.columns = ['Fields', '% of missing values']
   df2.reset_index(drop = True, inplace = True)
   return df2.sort_values(by = '% of missing values', ascending = False)

def calculate_cagr(ser):
    n = len(ser)
    cagr = ((ser.iloc[-1]/ser.iloc[0])**(1/n)) - 1
    return cagr

parameters = {
    'Enrollment on 10/31' : calculate_cagr,
    'Number of students who registered for the SHSAT' : calculate_cagr,
    'Number of students who took the SHSAT' : calculate_cagr
    }

d5_shsat_data[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']] = d5_shsat_data[['Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']].replace(0, 0.1)
cagr_d5_shsat = d5_shsat_data[['DBN', 'Grade level', 'Year of SHST', 'Enrollment on 10/31', 'Number of students who registered for the SHSAT', 'Number of students who took the SHSAT']].groupby(['DBN', 'Grade level']).agg(parameters).reset_index(drop = False)
