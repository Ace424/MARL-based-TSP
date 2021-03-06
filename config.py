"""config parameters
each test should input 4 params:
1.mode ('classify' or 'regression')
2. c_columns: continous columns names type: list of strs
3. d_columns: discrete columns names  type: list of strs
4. target: data target name type: list of strs
eg like configs['credit_dataset1688.csv']"""

configs = {
    'employee_train': {'mode': 'classify',
                       'c_columns': ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2',
                                     'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY',
                                     'ROLE_CODE'],
                       'd_columns': [],
                       'target': "ACTION",
                       "model": "lr",
                       "dataset_path": "data/employee_train.csv",
                       "metric": "auc"},
    'credit_dataset': {'mode': 'classify',
                       'c_columns': ['YEARS_EMPLOYED', 'AGE', 'INCOME', 'BEGIN_MONTH'],
                       'd_columns': ['GENDER', 'CAR', 'REALITY', 'NO_OF_CHILD', 'INCOME_TYPE', 'EDUCATION_TYPE',
                                     'FAMILY_TYPE', 'HOUSE_TYPE', 'WORK_PHONE', 'PHONE', 'E_MAIL',
                                     'FAMILY SIZE'],
                       'target': "TARGET",
                       "model": "lr",
                       "dataset_path": "data/credit_dataset.csv",
                       "metric": "auc"},
    'Dry_Bean_Dataset': {'mode': 'classify',
                         'c_columns': ['Area', 'Perimeter', 'MajorAxisLength',
                                       'MinorAxisLength', 'AspectRation', 'Eccentricity',
                                       'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity',
                                       'roundness', 'Compactness', 'ShapeFactor1',
                                       'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'],
                         'd_columns': [],
                         'target': 'Class',
                         "model": "lr",
                         "dataset_path": "data/Dry_Bean_Dataset.csv",
                         "metric": "auc"},
    "hour": {"mode": "regression",
             "c_columns": ['temp', 'atemp', 'hum', 'windspeed', 'hr'],
             "d_columns": ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit'],
             "target": "cnt"
             },
    "adult": {"mode": "classify",
              "c_columns": ['age', 'fnlwgt', 'earnings', 'loss', 'hour', 'edu_nums'],
              "d_columns": ['work_cate', 'education', 'marital', 'profession', 'relation', 'race', 'gender',
                            'motherland'],
              "target": "label",
              "model": "lr",
              "dataset_path": "data/adult.csv",
              "metric": "auc"},
    'winequality-red': {'mode': 'classify',
                        'c_columns': ['fixed acidity', 'volatile acidity', 'citric acid',
                                      'residual sugar', 'chlorides', 'free sulfur dioxide',
                                      'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                      'alcohol'],
                        'd_columns': [],
                        'target': 'quality',
                        "model": "lr",
                        "dataset_path": "data/winequality-red.csv",
                        "metric": "f1"},
    'winequality-white': {'mode': 'classify',
                          'c_columns': ['fixed acidity', 'volatile acidity', 'citric acid',
                                        'residual sugar', 'chlorides', 'free sulfur dioxide',
                                        'total sulfur dioxide', 'density', 'pH', 'sulphates',
                                        'alcohol'],
                          'd_columns': [],
                          'target': 'quality',
                          "model": "lr",
                          "dataset_path": "data/winequality-white.csv",
                          "metric": "f1"},
    'creditcard': {'mode': 'classify',
                   'c_columns': ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],
                   'd_columns': [],
                   'target': 'Class',
                   "model": "lr",
                   "dataset_path": "data/creditcard.csv",
                   "metric": "auc"},
    'SPECTF': {'mode': 'classify',
               'c_columns': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13',
                             'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26',
                             'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V39',
                             'V40', 'V41', 'V42', 'V43'],
               'd_columns': [],
               'target': 'label',
               "model": "lr",
               "dataset_path": "data/SPECTF.csv",
               "metric": "f1"},
    "titanic": {"mode": "classify",
                "c_columns": ["Age", "Fare"],
                "d_columns": ["Pclass", "Sex", "SibSp", "Parch", "Embarked"],
                "target": "Survived",
                "model": "lr",
                "dataset_path": "data/titanic.csv",
                "metric": "auc"},
    "water_potability": {"mode": "classify",
                         "c_columns": ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
                                       'Organic_carbon', 'Trihalomethanes', 'Turbidity'],
                         "d_columns": [],
                         "target": "Potability",
                         "model": "lr",
                         "dataset_path": "data/water_potability.csv",
                         "metric": "auc"},
    "mobile_pricerange_train": {"mode": "classify",
                                "c_columns": ['battery_power', 'clock_speed', 'int_memory', 'm_dep', 'mobile_wt',
                                              'px_height', 'px_width', 'ram', 'fc', 'pc', 'sc_h', 'sc_w'],
                                "d_columns": ['blue', 'dual_sim', 'four_g', 'n_cores', 'talk_time', 'three_g',
                                              'touch_screen', 'wifi'],
                                "target": "price_range"},
    "Tamil_Nadu_State_Elections_2021_Details": {"mode": "classify",
                                                "c_columns": ['EVM_Votes', 'Postal_Votes', 'Total_Votes',
                                                              '%_of_Votes',
                                                              'Tot_Constituency_votes_polled',
                                                              'Tot_votes_by_parties', 'Winning_votes'],
                                                "d_columns": ['Constituency', 'Candidate', 'Party'],
                                                "target": "Win_Lost_Flag"},
    "club_loan": {"mode": "classify",
                  "c_columns": ['int.rate', 'installment', 'log.annual.inc', 'dti',
                                'fico', 'days.with.cr.line', 'revol.bal', 'revol.util'],
                  "d_columns": ['credit.policy', 'purpose', 'inq.last.6mths',
                                'delinq.2yrs', 'pub.rec'],
                  "target": "not.fully.paid"},
    "bank_add": {"mode": "classify",
                 "c_columns": ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'age',
                               'duration'],
                 "d_columns": ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                               'day_of_week', 'campaign', 'pdays', 'previous', 'poutcome'],
                 "target": "y"},
    'healthcare-dataset-stroke-data': {"mode": "classify",
                                       "d_columns": ['gender', 'hypertension', 'heart_disease', 'ever_married',
                                                     'work_type', 'Residence_type', 'smoking_status'],
                                       "c_columns": ['age', 'avg_glucose_level', 'bmi'],
                                       "target": 'stroke'},
    'hzd_amend': {'mode': 'classify',
                  'c_columns': ['??????', '??????6???????????????????????????', '??????6??????????????????????????????', '????????????6??????????????????', '????????????????????????', '????????????', '???????????????',
                                '??????????????????', '????????????'],
                  'd_columns': ["??????????????????", '?????????????????????', '??????????????????', '??????????????????', '????????????', '?????????????????????',
                                '????????????', '????????????'],
                  'target': 'rst',
                  "model": "lr",
                  "dataset_path": "data/hzd_amend.csv",
                  "metric": "auc"},
    "default_credit_card": {'mode': 'classify',
                            'c_columns': ['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2',
                                          'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                                          'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4',
                                          'PAY_AMT5', 'PAY_AMT6'],
                            'd_columns': ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2',
                                          'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6'],
                            'target': 'target',
                            "model": "rf",
                            "dataset_path": "data/default_credit_card.csv",
                            "metric": "auc"},
    "credit-a": {'mode': 'classify',
                 'c_columns': ['A2', 'A3', 'A8', 'A11', 'A14', 'A15'],
                 'd_columns': ['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'],
                 'target': 'label',
                 "model": "rf",
                 "dataset_path": "data/credit-a.csv",
                 "metric": "f1"},
    "ionosphere": {'mode': 'classify',
                   'c_columns': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22', 'C23',
                                 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32'],
                   'd_columns': ['D1', 'D2'],
                   'target': 'label',
                   "model": "lr",
                   "dataset_path": "data/ionosphere.csv",
                   "metric": "f1"},
    "messidor_features": {'mode': 'classify',
                          'c_columns': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
                                        'C13', 'C14', 'C15', 'C16'],
                          'd_columns': ['D1', 'D2', 'D3'],
                          'target': 'label',
                          "model": "rf",
                          "dataset_path": "data/messidor_features.csv",
                          "metric": "f1"},
    "PimaIndian": {'mode': 'classify',
                   'c_columns': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
                   'd_columns': [],
                   'target': 'label',
                   "model": "lr",
                   "dataset_path": "data/PimaIndian.csv",
                   "metric": "f1"},
    'german_credit_24': {'mode': 'classify',
                         'c_columns': ['C0', 'C1', 'C2'],
                         'd_columns': ['V0', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',
                                       'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20'],
                         'target': 'label',
                         "model": "lr",
                         "dataset_path": "data/german_credit_24.csv",
                         "metric": "f1"
                         },
    #############################################TS
    'car_sale_prediction': {'mode': 'regression',
                            'target': 'car_sale.csv',
                            'covariates': ['car_comment.csv', 'car_popularity.csv', 'car_reply.csv'],
                            'dataset_path': './data/car_sale_prediction/'
                            }
}
