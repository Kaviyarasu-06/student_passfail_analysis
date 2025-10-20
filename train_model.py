import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score
import joblib


def main(csv_path='student_study_habits.csv', models_dir='models'):
    os.makedirs(models_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # basic preprocessing matching the notebook
    num_cols = ['study_hours_per_week', 'sleep_hours_per_day', 'attendance_percentage', 'assignments_completed']

    # cap outliers similar to notebook
    def cap_outliers(df, columns):
        df_capped = df.copy()
        for col in columns:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mean_value = df_capped[col].mean()
            df_capped[col] = df_capped[col].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)
        return df_capped

    df = cap_outliers(df, ['study_hours_per_week','sleep_hours_per_day','attendance_percentage'])

    df['result'] = pd.cut(df['final_grade'], bins=[0, 55, 100], labels=['Fail','Pass'])
    le_target = LabelEncoder()
    df['result'] = le_target.fit_transform(df['result'])

    # features used by RandomForest in notebook: X = df.drop(columns=['result', 'final_grade'])
    X = df.drop(columns=['result', 'final_grade'])
    y = df['result']

    # For Naive Bayes notebook used numeric columns only
    X_num = df[num_cols]

    # handle categorical columns by simple encoding (if any)
    # find object columns
    obj_cols = X.select_dtypes(include=['object']).columns.tolist()
    if obj_cols:
        X = pd.get_dummies(X, columns=obj_cols, drop_first=True)

    # align columns between training and later use by saving column lists

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    rf_prec = precision_score(y_test, y_pred_rf, average='weighted')

    # Naive Bayes on numeric features
    Xn_train, Xn_test, yn_train, yn_test = train_test_split(X_num, y, test_size=0.2, random_state=42)
    nb = GaussianNB()
    nb.fit(Xn_train, yn_train)
    y_pred_nb = nb.predict(Xn_test)
    nb_acc = accuracy_score(yn_test, y_pred_nb)
    nb_prec = precision_score(yn_test, y_pred_nb, average='weighted')

    # Save models and metadata
    joblib.dump(rf, os.path.join(models_dir, 'rf_model.pkl'))
    joblib.dump(nb, os.path.join(models_dir, 'nb_model.pkl'))

    metadata = {
        'rf_features': X.columns.tolist(),
        'nb_features': X_num.columns.tolist(),
        'label_encoder_classes': le_target.classes_.tolist(),
        'rf_accuracy': rf_acc,
        'rf_precision': rf_prec,
        'nb_accuracy': nb_acc,
        'nb_precision': nb_prec,
    }
    joblib.dump(metadata, os.path.join(models_dir, 'metadata.pkl'))

    print('Training complete')
    print(f'RF acc: {rf_acc:.3f}, prec: {rf_prec:.3f}')
    print(f'NB acc: {nb_acc:.3f}, prec: {nb_prec:.3f}')


if __name__ == '__main__':
    main()
