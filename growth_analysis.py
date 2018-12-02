import pandas as pd

school_results = pd.read_csv(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\additional data\School ELA Results 2013-2017 (Public)_consolidated.csv')
school_data = pd.read_excel(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\2016 School Explorer.xlsx')
d5_shsat_data = pd.read_csv(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\D5 SHSAT Registrations and Testers.csv')

school_results['Mean Scale Score'] = school_results['Mean Scale Score'].apply(lambda x: int(x) if x != 's' else 's')
school_results['Level 1%'] = school_results['Level 1%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 2%'] = school_results['Level 2%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 3%'] = school_results['Level 3%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 4%'] = school_results['Level 4%'].apply(lambda x: float(x) if x != 's' else 0.1)
school_results['Level 3+4%'] = school_results['Level 3+4%'].apply(lambda x: float(x) if x != 's' else 0.1)

school_results[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']] = school_results[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].replace(0, 0.1)

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
    'Number Tested' : calculate_cagr,
    'Level 1%' : calculate_cagr,
    'Level 2%' : calculate_cagr,
    'Level 3%' : calculate_cagr,
    'Level 4%' : calculate_cagr,
    'Level 3+4%' : calculate_cagr
    }

df = school_results[['Category', 'DBN', 'Grade', 'Number Tested', 'Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].groupby(['Category','DBN', 'Grade']).agg(parameters).reset_index(drop = False)
df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%', 'Number Tested']] = df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%', 'Number Tested']]*100

# 1: substract mean
df_norm_row=df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].sub(df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].mean(axis=1), axis=0)
# 2: divide by standard dev
df_norm_row=df_norm_row.div( df[['Level 1%', 'Level 2%', 'Level 3%', 'Level 4%', 'Level 3+4%']].std(axis=1), axis=0 )



