import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#importing the dataset csv file
patient=pd.read_csv('heart_attack_dataset.csv')
#print(patient)

#patient['Treatment'].value_counts()

predictor_df=patient.drop(['Treatment'],axis=1)
# print(predictor_df)

target_df=patient['Treatment']
# print(target_df)

enc=LabelEncoder()
target_df_encoded=enc.fit_transform(target_df)
#print(target_df_encoded)

# #Encoding the features in the predictor
ordinal_list=['Never','Former','Current']
ordinal_list1=['Asymptomatic','Non-anginal Pain','Atypical Angina','Typical Angina']
ct=ColumnTransformer([('ohe',OneHotEncoder(drop='first'),['Gender','Has Diabetes']),
                      ('oe',OrdinalEncoder(categories=[ordinal_list,ordinal_list1]),['Smoking Status','Chest Pain Type'])
                     ],remainder='passthrough')

predictor_encoded=ct.fit_transform(predictor_df)
#print(predictor_encoded)


#Splitting the  dataset into training data and testing data

x_train, x_test, y_train, y_test = train_test_split(predictor_encoded,target_df_encoded,
                                                     test_size=0.2, random_state=2)

# print(len(x_train))
# print(x_test)
# print(y_test)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
#print(x_test)
def pre_process_data(predictor_data):
    predictor_data_new=predictor_data.drop('Treatment',axis=1)
    predictor_data_encoded=ct.transform(predictor_data_new)
    x_data_final=sc.transform(predictor_data_encoded)
    return x_data_final

#print(x_test)

