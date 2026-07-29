"""Datasets, classifiers, seeds, sampling fractions, and split configuration."""

# Each entry is: [dataset directory, columns excluded from modeling, target].
dataset_list = [
    ['BoTNeTIoT-L01', ['Device_Name', 'Attack', 'Attack_subType'], 'label'],  # 7062606 x 27
    ['DDOS-ANL', [], 'PKT_CLASS'],  # 2160668 x 28#
    ['X-IIoTID', ['Date', 'Timestamp', 'class1', 'class2'], 'class3'],  # 820834 x 68
    ['IoTID20', ['Flow_ID', 'Cat', 'Sub_Cat'], 'Label'],  # 625783 x 86
    ['5G_Slicing', [], 'Slice Type (Output)'],  # 466739, 9
    ['IoT-DNL', [], 'normality'],  # 477426 x 14
    ['NSL-KDD', [], 'class'],  # 148517 x 42
    ['RT-IoT2022', ['no'], 'Attack_type'],  # 123117, 84
    ['QoS-QoE', ['RebufferingRatio', 'AvgVideoBitRate', 'AvgVideoQualityVariation'], 'StallLabel'],  # 69129 x 38
    ['DeepSlice', ['no'], 'slice Type'],  # 63167 x 10
    ['NSR', [], 'slice Type'],  # 31583 x 17
    ['IoT-APD', ['second'], 'label'],  # 10845 x 17
    ['ASNM-CDX-2009', ['id', 'label_poly'], 'label_2'],  # 5771 x 877
    ['UNAC', ['file'], 'output'],  # 389 x 23
    ['KPI-KQI', [], 'Service'],  # 165 x 14
    ['CDC Diabetes Health Indicators', [], 'Diabetes_binary'],  # 253680 x 22
    ['Rain in Australia', [], 'RainTomorrow'],  # 145460 x 23
    ['Airline Passenger Satisfaction', [], 'satisfaction'],  # 129880 x 25
    ['Secondary Mushroom', [], 'class'],  # 61069 x 21
    ['Mushroom', [], 'class'],  # 54035 x 9
    ['Bank Marketing', [], 'y'],  # 41188 x 21
    ['NATICUSdroid', [], 'Result'],  # 29332 x 87
    ['MAGIC Gamma Telescope', [], 'class'],  # 19020 x 11
    ['Pulsar', [], 'target_class'],  # 17898 x 9
    ['World Air Quality Index by City and Coordinates', [], 'AQI Category'],  # 16695 x 14
    ['Eye State', [], 'eyeDetection'],  # 14980 x 15
    ['Body performance', [], 'class'],  # 13393 x 12
    ['Customer Segmentation Classification', [], 'Segmentation'],  # 10695 x 11
    ['Bank Dataset', [], 'Exited'],  # 10000 x 14
    ['Car Insurance Data', [], 'OUTCOME'],  # 10000 x 19
    ['Employee Future', [], 'LeaveOrNot'],  # 4653 x 9
    ['TUNADROMD', [], 'Label'],  # 4465 x 242
    ['Predict Students Dropout and Academic Success', [], 'Target'],  # 4424 x 37
    ['Breast Cancer', [], 'Status'],  # 4024 x 16
    ['Apple Quality', [], 'Quality'],  # 4001 x 9
    ['Water Quality and Potability', [], 'Potability'],  # 3276 x 10
    ['Gender Recognition by Voice', [], 'label'],  # 3168 x 21
    ['Engineering Placements', [], 'PlacedOrNot'],  # 2966 x 8
    ['Students Performance Dataset', [], 'GradeClass'],  # 2392 x 15
    ['NHANES', [], 'age_group'],  # 2278 x 10
    ['Auction Verification', [], 'verification.result'],  # 2043 x 9
    ['Car Evaluation', [], 'class'],  # 1728 x 7
    ['Pistachio types detection', [], 'Class'],  # 1718 x 17
    ['Depression', [], 'depressed'],  # 1429 x 23
    ['Student Stress Factors', [], 'stress_level'],  # 1100 x 21
    ['Milk Quality Prediction', [], 'Grade'],  # 1059 x 8
    ['Home Loan Approval', [], 'Loan_Status'],  # 981 x 13
    ['Mammographic Mass', [], 'Severity'],  # 961 x 6
    ['Tic-Tac-Toe Endgame', [], 'Class'],  # 958 x 10
    ['Startup Success Prediction', [], 'status'],  # 923 x 49
    ['Heart Failure Prediction', [], 'HeartDisease'],  # 918 x 12
    ['Diabetes', [], 'Outcome'],  # 768 x 9
    ['Balance Scale', [], 'Class Name'],  # 625 x 5
    ['Congressional Voting', [], 'Class Name'],  # 435 x 17
    ['Cirrhosis Patient Survival Prediction', [], 'Status'],  # 418 x 20
    ['Chronic kidney disease', [], 'Class'],  # 400 x 14
    ['Social Network Ads', [], 'Purchased'],  # 400 x 3
    ['Differentiated Thyroid Cancer Recurrence', [], 'Recurred'],  # 383 x 17
    ['Dermatology', [], 'class'],  # 366 x 35
    ['Disease Symptoms and Patient Profile', [], 'Outcome Variable'],  # 349 x 10
    ['Haberman Survival', [], 'Survival status'],  # 306 x 4
    ['Heart attack possibility', [], 'target'],  # 303 x 14
    ['Z-Alizadeh Sani', [], 'Cath'],  # 303 x 56
    ['Iris', [], 'Species'],  # 150 x 6
    ['Cryotherapy', [], 'Result_of_Treatment'],  # 90 x 7
]

model_list = [
    'SGD',
    'LR',
    'Ridge',
    'Perceptron',
    'DT',
    'ExtraTree',
    'LinearSVC',
    'GaussianNB',
    'BernoulliNB',
    'RF',
    'ExtraTrees',
    'AdaBoost',
    'HistGradientBoosting',
    'Bagging',
    'MLP',
    'DNN'
    ]

seed_list = [5, 7, 11, 13, 42]

dataset_percentage_list = [0.2, 0.4, 0.6, 0.8, 1.0]

# Held-out fraction for each base classification task.
test_size = 0.2
