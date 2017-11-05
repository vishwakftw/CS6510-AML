import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale

train_data = np.genfromtxt('trainData.csv', delimiter=',', dtype=str)
m = train_data.shape[0] - 1
d = train_data.shape[1] - 1
train_input_raw, train_output = train_data[1:,0:d].tolist(), train_data[1:,d].tolist()

for i in range(0, m):
    for j in range(0, d-1):
        if train_input_raw[i][j].find("\"") != -1:
            train_input_raw[i][j] = train_input_raw[i][j][1:-1] # removing double quotes in pairs
        else:
            train_input_raw[i][j] = float(train_input_raw[i][j])
    train_output[i] = float(train_output[i])
train_output = np.array(train_output).astype(float)

test_input_raw = np.genfromtxt('testData.csv', delimiter=',', dtype=str)[1:,1:].tolist()
for i in range(0, len(test_input_raw)):
    for j in range(0, d-1):
        if test_input_raw[i][j].find("\"") != -1:
            test_input_raw[i][j] = test_input_raw[i][j][1:-1] # removing double quotes in pairs
        else:
            test_input_raw[i][j] = float(test_input_raw[i][j])

train_input_raw = np.array(train_input_raw).astype(str)
test_input_raw = np.array(test_input_raw).astype(str)

m_train, m_test = 24712, 16476
all_inputs = np.vstack((train_input_raw, test_input_raw))
# Age
age = all_inputs[:,0].astype(float)

text_enc = LabelEncoder()
text_enc.fit(['admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'])
# Job
job = text_enc.transform(all_inputs[:,1]).astype(float)
one_hot_enc = OneHotEncoder()
job_ohe = one_hot_enc.fit_transform(job.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit([ 'divorced','married','single','unknown'])
# Marital Status
marital = text_enc.transform(all_inputs[:,2]).astype(float)
one_hot_enc = OneHotEncoder()
marital_ohe = one_hot_enc.fit_transform(marital.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit(['basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'])
# Education
education = text_enc.transform(all_inputs[:,3]).astype(float)
one_hot_enc = OneHotEncoder()
education_ohe = one_hot_enc.fit_transform(education.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit(['no','yes','unknown'])
# Default
default = text_enc.transform(all_inputs[:,4]).astype(float)
one_hot_enc = OneHotEncoder()
default_ohe = one_hot_enc.fit_transform(default.reshape(-1, 1).astype(int)).toarray()

# Housing
housing = text_enc.transform(all_inputs[:,5]).astype(float)
one_hot_enc = OneHotEncoder()
housing_ohe = one_hot_enc.fit_transform(housing.reshape(-1, 1).astype(int)).toarray()

# Loan
loan = text_enc.transform(all_inputs[:,6]).astype(float)
one_hot_enc = OneHotEncoder()
loan_ohe = one_hot_enc.fit_transform(loan.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit(['cellular', 'telephone'])
# Contact
contact = text_enc.transform(all_inputs[:,7]).astype(float)
one_hot_enc = OneHotEncoder()
contact_ohe = one_hot_enc.fit_transform(contact.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit(['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
# Month
month = text_enc.transform(all_inputs[:,8]).astype(float)
one_hot_enc = OneHotEncoder()
month_ohe = one_hot_enc.fit_transform(month.reshape(-1, 1).astype(int)).toarray()

text_enc = LabelEncoder()
text_enc.fit(['mon','tue','wed','thu','fri'])
# Day
day = text_enc.transform(all_inputs[:,9]).astype(float)
one_hot_enc = OneHotEncoder()
day_ohe = one_hot_enc.fit_transform(day.reshape(-1, 1).astype(int)).toarray()

# Campaign
campaign = all_inputs[:,10].astype(float)

# Pdays
pdays = all_inputs[:,11].astype(float)

# Previous
previous = all_inputs[:,12].astype(float)

text_enc = LabelEncoder()
text_enc.fit(['failure', 'success', 'nonexistent'])
# Poutcome
poutcome = text_enc.transform(all_inputs[:,13]).astype(float)
one_hot_enc = OneHotEncoder()
poutcome_ohe = one_hot_enc.fit_transform(poutcome.reshape(-1, 1).astype(int)).toarray()

# Emp Var Rate
emp_var_rate = all_inputs[:,14].astype(float)

# Consumer Price Index
cons_price_idx = all_inputs[:,15].astype(float)

# Consumer Confidence Index
cons_conf_idx = all_inputs[:,16].astype(float)

# Euribor 3 months
euribor3m = all_inputs[:,17].astype(float)

# Number employed
nm_employ = all_inputs[:,18].astype(float)

scaled_numeric = []
for attr in [age, campaign, pdays, previous, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nm_employ]:
    scaled_numeric.append(scale(attr))
    
age_scaled = scaled_numeric[0]
campaign_scaled = scaled_numeric[1]
pdays_scaled = scaled_numeric[2]
previous_scaled = scaled_numeric[3]
emp_var_rate_scaled = scaled_numeric[4]
cons_price_idx_scaled = scaled_numeric[5]
cons_conf_idx_scaled = scaled_numeric[6]
euribor3m_scaled = scaled_numeric[7]
nm_employ_scaled = scaled_numeric[8]

all_data_1 = np.hstack((age.reshape(-1, 1), contact_ohe, month_ohe, pdays_scaled.reshape(-1, 1), 
                        campaign_scaled.reshape(-1, 1), poutcome_ohe, euribor3m_scaled.reshape(-1, 1), 
                        nm_employ_scaled.reshape(-1, 1), job_ohe, default_ohe
                      ))
train_1 = all_data_1[:m_train]
test_1 = all_data_1[m_train:]

gradbooster = GradientBoostingClassifier(min_samples_leaf=0.002, n_estimators=500, max_depth=5)
gradbooster.fit(train_1, train_output)
sub_1_pred = gradbooster.predict_proba(test_1)
sub_file = open('sub_file_1.csv', 'w')
sub_file.write('Id,Class\n')
for i in range(0, sub_1_pred.shape[0]):
    sub_file.write('{0},{1}\n'.format(i+1, sub_1_pred[i,1]))
sub_file.close()

all_data_2 = np.hstack((age_scaled.reshape(-1, 1), job_ohe, marital_ohe, education_ohe, default_ohe, housing_ohe, loan_ohe, contact_ohe,
                        month_ohe, day_ohe, campaign_scaled.reshape(-1, 1), pdays_scaled.reshape(-1, 1), previous_scaled.reshape(-1, 1),
                        poutcome_ohe, emp_var_rate_scaled.reshape(-1, 1), cons_price_idx_scaled.reshape(-1, 1),
                        cons_conf_idx_scaled.reshape(-1, 1), euribor3m_scaled.reshape(-1, 1), nm_employ_scaled.reshape(-1, 1)
                      ))
train_2 = all_data_2[:m_train]
test_2 = all_data_2[m_train:]

rnd_frst = RandomForestClassifier(n_estimators=500, min_samples_leaf=0.001, class_weight='balanced')
rnd_frst.fit(train_2, train_output)
sub_2_pred = rnd_frst.predict_proba(test_2)
sub_file = open('sub_file_2.csv', 'w')
sub_file.write('Id,Class\n')
for i in range(0, sub_2_pred.shape[0]):
    sub_file.write('{0},{1}\n'.format(i+1, sub_2_pred[i,1]))
sub_file.close()
