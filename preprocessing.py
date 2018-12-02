import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors

school_data = pd.read_excel(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\2016 School Explorer.xlsx')
school_data.drop(['Adjusted Grade', 'New?', 'Other Location Code in LCGMS'], axis = 1, inplace = True)

def check_missing_values(df):
   df2 = pd.DataFrame()
   for field in list(df):
       df2 = df2.append(df[[field]].isnull().sum().reset_index())
   df2[0] = (df2[0]/df.shape[0])*100
   df2.columns = ['Fields', '% of missing values']
   df2.reset_index(drop = True, inplace = True)
   return df2.sort_values(by = '% of missing values', ascending = False)

def string2decimal(s):
    if isinstance(s, str):
        s = s.strip()
        s = s.replace(",", "")
        s = s.replace("%", "")
        return float(s[1:])
    else:
        return s

##### School service categories

school_data['Grade Low'] = school_data['Grade Low'].astype(str)
school_data['Grade High'] = school_data['Grade High'].astype(str)
service_grade_categories = school_data[['SED Code', 'Grade Low', 'Grade High']].pivot_table(values = 'SED Code', index = 'Grade Low', columns = 'Grade High', aggfunc = np.size, fill_value = 0)
sns.heatmap(service_grade_categories, annot = True, fmt = 'd', linewidths=.2, cmap="YlGnBu")


school_data['Early_Childhood'] = 0
school_data['Elementary'] = 0
school_data['Middle'] = 0
school_data['High'] = 0

school_data['Early_Childhood']  = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['0K', '2', '3', '4']), 1, \
                                           np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['5', '6']), 1, \
                                                    np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['8']), 1, \
                                                             np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['Elementary'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['5', '6']), 1, \
                                    np.where(school_data['Grade Low'].isin(['1', '2', '3', '4']) & school_data['Grade High'].isin(['5', '6']), 1, \
                                            np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['8']), 1, \
                                                    np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['Middle'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['8']), 1, \
                                 np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, \
                                          np.where(school_data['Grade Low'].isin(['4', '5', '6']) & school_data['Grade High'].isin(['8']), 1, \
                                                   np.where(school_data['Grade Low'].isin(['5', '6', '7']) & school_data['Grade High'].isin(['12']), 1, 0))))

school_data['High'] = np.where(school_data['Grade Low'].isin(['PK', '0K']) & school_data['Grade High'].isin(['12']), 1, \
                                np.where(school_data['Grade Low'].isin(['5', '6', '7']) & school_data['Grade High'].isin(['10', '12']), 1,
                                         np.where(school_data['Grade Low'].isin(['9']) & school_data['Grade High'].isin(['12']), 1, 0)))


############## Economic Index analysis #################
sns.kdeplot(school_data['Economic Need Index'].fillna(0), cut = 0, shade  = True, color = 'r')
sns.regplot(x="Economic Need Index", y="School Income Estimate", data=school_data[['Economic Need Index', 'School Income Estimate']])

############## Regression Plot Economic Need Index Vs. Average ELA Proficiency ################
############## Regression Plot Economic Need Index Vs. Average Math Proficiency ################
sns.regplot(x="Economic Need Index", y="Average ELA Proficiency", data=school_data[['Economic Need Index', 'Average ELA Proficiency']])
sns.regplot(x="Economic Need Index", y="Average Math Proficiency", data=school_data[['Economic Need Index', 'Average Math Proficiency']])


###### % schools with different ENI levels

pct_of_schools = []
ecn_indx = []
for ei in range(0,100):
    pct_of_schools.append(100*sum(school_data['Economic Need Index'] >= ei/100)/1272)
    ecn_indx.append(ei)

df = pd.DataFrame(ecn_indx, pct_of_schools).reset_index(drop = False)
df.columns = ['ENI', 'percent of schools']

plt.plot(df['ENI'], df['percent of schools'])


###### Schools with poor socioeconomic index but good student performance ######
school_data_high_eni_ela_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average ELA Proficiency'] >= 3)]
school_data_low_eni_ela_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average ELA Proficiency'] >= 3)]

school_data_high_eni_math_proficient = school_data[(school_data['Economic Need Index'] >= 0.65) & (school_data['Average Math Proficiency'] >= 3)]
school_data_low_eni_math_proficient = school_data[(school_data['Economic Need Index'] < 0.65) & (school_data['Average Math Proficiency'] >= 3)]


school_data['proficiency'] = np.where((school_data['Average ELA Proficiency'] >= 3) & (school_data['Average Math Proficiency'] >= 3), 1, \
                                np.where((school_data['Average ELA Proficiency'] >= 3) & (school_data['Average Math Proficiency'] < 3), 2, \
                                    np.where((school_data['Average ELA Proficiency'] < 3) & (school_data['Average Math Proficiency'] >= 3), 3, 4)))


school_data['Community'] = np.where(school_data['Community School?'] == 'Yes', 1, 0)
plt.pyplot(school_data['Average ELA Proficiency'], school_data['Average Math Proficiency'])


##### knn conversion model

# Missing values
school_data_with_missing_values = pd.DataFrame()
school_data_with_missing_values = school_data_with_missing_values.append([school_data[~pd.notnull(school_data['Economic Need Index'])]])
#school_data_with_missing_values.to_csv(r'X:\Hackathon\Kaggle - Data Science For Good\Kaggle - PASSNYC\data\data-science-for-good\school_data_with_missing_values.csv', index = False)

# Remove records with good number of missing values
school_data = school_data[pd.notnull(school_data['Economic Need Index'])]
#school_data = school_data[pd.notnull(school_data['Student Achievement Rating'])]
#school_data = school_data[pd.notnull(school_data['Supportive Environment Rating'])]

### Treat other missing values by imputation
school_data['Student Achievement Rating'] = school_data['Student Achievement Rating'].fillna('missing')
school_data['Supportive Environment Rating'] = school_data['Supportive Environment Rating'].fillna('missing')
school_data['Collaborative Teachers Rating'] = school_data['Collaborative Teachers Rating'].fillna('missing')
school_data['Rigorous Instruction Rating'] = school_data['Rigorous Instruction Rating'].fillna('missing')
school_data['Trust Rating'] = school_data['Trust Rating'].fillna('missing')
school_data['Strong Family-Community Ties Rating'] = school_data['Strong Family-Community Ties Rating'].fillna('missing')
school_data['Effective School Leadership Rating'] = school_data['Effective School Leadership Rating'].fillna('missing')
school_data['Average Math Proficiency'] = school_data['Average Math Proficiency'].fillna(school_data['Average Math Proficiency'].mean())
school_data['Average ELA Proficiency'] = school_data['Average ELA Proficiency'].fillna(school_data['Average ELA Proficiency'].mean())

school_data.reset_index(drop = True, inplace=True)

### Get Dummys
def categorical_variable_encoding(df):
    return pd.get_dummies(df, columns=list(df))

for field in ['Rigorous Instruction Rating', 'Collaborative Teachers Rating', 'Supportive Environment Rating', 'Effective School Leadership Rating', 'Strong Family-Community Ties Rating', 'Trust Rating', 'Student Achievement Rating']:
    df = categorical_variable_encoding(school_data[field])
    df.columns = [field + ' ' + s for s in list(df)]
    school_data = school_data.join(df)


group1 = school_data[school_data['proficiency'] == 1].copy()
group2 = school_data[school_data['proficiency'] == 3].copy()
group3 = school_data[school_data['proficiency'] == 4].copy()

group1.reset_index(drop=True, inplace=True)
group2.reset_index(drop=True, inplace=True)
group3.reset_index(drop=True, inplace=True)

### Group2 --> Group1 conversion

X_group2 = np.array(group2[[    'Early_Childhood',
                                'Elementary',
                                'Middle',
                                'High',
                                'proficiency',
                                'Community',
                                'Economic Need Index',
                                'Percent ELL',
                                'Percent Asian',
                                'Percent Black',
                                'Percent Hispanic',
                                'Percent Black / Hispanic',
                                'Percent White',
                                'Student Attendance Rate',
                                'Percent of Students Chronically Absent',
                                'Rigorous Instruction %',
                                'Collaborative Teachers %',
                                'Supportive Environment %',
                                'Effective School Leadership %',
                                'Strong Family-Community Ties %',
                                'Trust %',
                                'Rigorous Instruction Rating Approaching Target',
                                'Rigorous Instruction Rating Exceeding Target',
                                'Rigorous Instruction Rating Meeting Target',
                                'Rigorous Instruction Rating Not Meeting Target',
                                'Rigorous Instruction Rating missing',
                                'Collaborative Teachers Rating Approaching Target',
                                'Collaborative Teachers Rating Exceeding Target',
                                'Collaborative Teachers Rating Meeting Target',
                                'Collaborative Teachers Rating Not Meeting Target',
                                'Collaborative Teachers Rating missing',
                                'Supportive Environment Rating Approaching Target',
                                'Supportive Environment Rating Exceeding Target',
                                'Supportive Environment Rating Meeting Target',
                                'Supportive Environment Rating Not Meeting Target',
                                'Supportive Environment Rating missing',
                                'Effective School Leadership Rating Approaching Target',
                                'Effective School Leadership Rating Exceeding Target',
                                'Effective School Leadership Rating Meeting Target',
                                'Effective School Leadership Rating Not Meeting Target',
                                'Effective School Leadership Rating missing',
                                'Strong Family-Community Ties Rating Approaching Target',
                                'Strong Family-Community Ties Rating Exceeding Target',
                                'Strong Family-Community Ties Rating Meeting Target',
                                'Strong Family-Community Ties Rating Not Meeting Target',
                                'Strong Family-Community Ties Rating missing',
                                'Trust Rating Approaching Target',
                                'Trust Rating Exceeding Target',
                                'Trust Rating Meeting Target',
                                'Trust Rating Not Meeting Target',
                                'Trust Rating missing',
                                'Student Achievement Rating Approaching Target',
                                'Student Achievement Rating Exceeding Target',
                                'Student Achievement Rating Meeting Target',
                                'Student Achievement Rating Not Meeting Target',
                                'Student Achievement Rating missing',
                                'Average ELA Proficiency',
                                'Average Math Proficiency']])

X_group1 = np.array(group1[[    'Early_Childhood',
                                'Elementary',
                                'Middle',
                                'High',
                                'proficiency',
                                'Community',
                                'Economic Need Index',
                                'Percent ELL',
                                'Percent Asian',
                                'Percent Black',
                                'Percent Hispanic',
                                'Percent Black / Hispanic',
                                'Percent White',
                                'Student Attendance Rate',
                                'Percent of Students Chronically Absent',
                                'Rigorous Instruction %',
                                'Collaborative Teachers %',
                                'Supportive Environment %',
                                'Effective School Leadership %',
                                'Strong Family-Community Ties %',
                                'Trust %',
                                'Rigorous Instruction Rating Approaching Target',
                                'Rigorous Instruction Rating Exceeding Target',
                                'Rigorous Instruction Rating Meeting Target',
                                'Rigorous Instruction Rating Not Meeting Target',
                                'Rigorous Instruction Rating missing',
                                'Collaborative Teachers Rating Approaching Target',
                                'Collaborative Teachers Rating Exceeding Target',
                                'Collaborative Teachers Rating Meeting Target',
                                'Collaborative Teachers Rating Not Meeting Target',
                                'Collaborative Teachers Rating missing',
                                'Supportive Environment Rating Approaching Target',
                                'Supportive Environment Rating Exceeding Target',
                                'Supportive Environment Rating Meeting Target',
                                'Supportive Environment Rating Not Meeting Target',
                                'Supportive Environment Rating missing',
                                'Effective School Leadership Rating Approaching Target',
                                'Effective School Leadership Rating Exceeding Target',
                                'Effective School Leadership Rating Meeting Target',
                                'Effective School Leadership Rating Not Meeting Target',
                                'Effective School Leadership Rating missing',
                                'Strong Family-Community Ties Rating Approaching Target',
                                'Strong Family-Community Ties Rating Exceeding Target',
                                'Strong Family-Community Ties Rating Meeting Target',
                                'Strong Family-Community Ties Rating Not Meeting Target',
                                'Strong Family-Community Ties Rating missing',
                                'Trust Rating Approaching Target',
                                'Trust Rating Exceeding Target',
                                'Trust Rating Meeting Target',
                                'Trust Rating Not Meeting Target',
                                'Trust Rating missing',
                                'Student Achievement Rating Approaching Target',
                                'Student Achievement Rating Exceeding Target',
                                'Student Achievement Rating Meeting Target',
                                'Student Achievement Rating Not Meeting Target',
                                'Student Achievement Rating missing',
                                'Average ELA Proficiency',
                                'Average Math Proficiency']])

nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(X_group2)

kneighbour_matrix = nbrs.kneighbors_graph(X_group1).toarray()