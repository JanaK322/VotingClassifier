import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import pickle

# Load your data
df = pd.read_csv(r"C:\Users\janas\Downloads\Telegram Desktop\neo_asteroids_data.csv")

# Create diameter_avg
df['diameter_avg'] = (df['diameter_km_min'] + df['diameter_km_max']) / 2

def remove_outliers_iqr_iterative(df, columns):
    prev_shape = None
    while prev_shape != df.shape:
        prev_shape = df.shape
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

# Apply to your DataFrame
df = remove_outliers_iqr_iterative(df, ['diameter_avg', 'velocity_km_s', 'miss_distance_km'])


# Prepare features and label
X = df[['velocity_km_s', 'miss_distance_km', 'diameter_avg']]
y = df['hazardous'].astype(int)  # 1 for hazardous, 0 for not

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Rebuild the best model explicitly
best_lr = LogisticRegression(C=0.1)
best_knn = KNeighborsClassifier(n_neighbors= 7)
best_svm = SVC(C=10, kernel= 'rbf', probability=True)

voting = VotingClassifier(estimators=[
    ('lr', best_lr),
    ('knn', best_knn),
    ('svm', best_svm)
], voting='hard')

# Fit and save
voting.fit(X_train, y_train)

# Save to pickle
pickle.dump(voting, open('voting_clas.pkl', 'wb'))

