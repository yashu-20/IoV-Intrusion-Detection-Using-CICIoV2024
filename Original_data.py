import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, accuracy_score, matthews_corrcoef
)

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

#Data Loading
benign = pd.read_csv("decimal/decimal_benign.csv")
dos = pd.read_csv("decimal/decimal_DoS.csv")
spoofing_gas = pd.read_csv("decimal/decimal_spoofing-GAS.csv")
spoofing_rpm = pd.read_csv("decimal/decimal_spoofing-RPM.csv")
spoofing_speed = pd.read_csv("decimal/decimal_spoofing-SPEED.csv")
spoofing_steering = pd.read_csv("decimal/decimal_spoofing-STEERING_WHEEL.csv")

benign['source'] = 'BENIGN'
dos['source'] = 'DoS'
spoofing_gas['source'] = 'Spoofing-GAS'
spoofing_rpm['source'] = 'Spoofing-RPM'
spoofing_speed['source'] = 'Spoofing-SPEED'
spoofing_steering['source'] = 'Spoofing-STEERING'

#combining all data files
data = [benign, dos, spoofing_gas, spoofing_rpm, spoofing_speed, spoofing_steering]
combined_data = pd.concat(data, ignore_index=True)

features = ['DATA_0', 'DATA_1', 'DATA_2', 'DATA_3']
for feature in features:
    plt.figure(figsize=(10, 5))
    sns.kdeplot(data=combined_data, x=feature, hue='source', fill=True, warn_singular=False)
    plt.title(f'Distribution of {feature} by Dataset')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

source_counts = combined_data['source'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=source_counts.index, y=source_counts.values, palette="Set2")
plt.title("Dataset Size per Category")
plt.ylabel("Number of Rows")
plt.xlabel("Dataset")
plt.xticks(rotation=15)
plt.tight_layout()       
plt.show()

print(f"Original Dataset Size: {combined_data.shape}")
print("Original Label Distribution:\n", combined_data['label'].value_counts())
sns.countplot(data=combined_data, x='label', order=combined_data['label'].value_counts().index)
plt.title('Class Distribution Before Label Encoding')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

sns.heatmap(combined_data.select_dtypes(include='number').corr(), cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Features)')
plt.show()

print("First 10 rows of the dataset:")
print(combined_data.head(10))
combined_data.iloc[-1]
print("\nLast row of the combined dataset:")
print(combined_data.iloc[-1].to_frame().T)


from tabulate import tabulate
print(tabulate(combined_data.head(10), headers='keys', tablefmt='grid'))
print("\nLast row of the combined dataset:")
print(combined_data.iloc[-1].to_frame().T)

#Data Preprocessing
total_duplicates_before = combined_data.duplicated().sum()
duplicates_df = combined_data[combined_data.duplicated()]
print(f"Total Duplicates: {total_duplicates_before}")
combined_data = combined_data.drop_duplicates()
print(f"Dataset size after removing duplicates: {combined_data.shape}")

combined_data['label'] = combined_data['label'].map({'BENIGN': 0, 'ATTACK': 1})
X = combined_data.drop(['label', 'category', 'specific_class', 'source'], axis=1)
y = combined_data['label']

benign_count = (combined_data['label'] == 0).sum()
attack_count = (combined_data['label'] == 1).sum()

print(f"Benign samples: {benign_count}")
print(f"Attack samples: {attack_count}")

sns.countplot(x='label', data=combined_data)
plt.title("Label Distribution After Removing Duplicates (0 = BENIGN, 1 = ATTACK)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Modeling-Approach

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()

import seaborn as sns
import matplotlib.pyplot as plt

# Correlation Heatmap - original data
plt.figure(figsize=(10, 8))
sns.heatmap(combined_data.select_dtypes(include='number').corr(), annot=True, fmt=".1f", cmap='rocket', vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Without Synthetic Data (Before Removing Duplicates)')
plt.tight_layout()
plt.savefig("heatmap_without_synthetic_before.png")
plt.show()

# Correlation Heatmap - original data - after preprocessing
plt.figure(figsize=(10, 8))
sns.heatmap(combined_data.select_dtypes(include='number').corr(), annot=True, fmt=".1f", cmap='magma', vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Without Synthetic Data (After Removing Duplicates)')
plt.tight_layout()
plt.savefig("heatmap_without_synthetic_after.png")
plt.show()

scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)


#model Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# 7.Supervised model training
random_forest = RandomForestClassifier(class_weight='balanced', random_state=42)
param_grid_random_forest = {'n_estimators': [50, 100], 'max_depth': [10, None]}
grid_random_forest = GridSearchCV(random_forest, param_grid_random_forest, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
grid_random_forest.fit(X_train_resampled, y_train_resampled)
evaluate_model(grid_random_forest.best_estimator_, X_test, y_test)

xgboost = XGBClassifier(scale_pos_weight=(len(y_train_resampled) / sum(y_train_resampled)), random_state=42, eval_metric='logloss')
param_grid_xgboost = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.1]}
grid_xgboost = GridSearchCV(xgboost, param_grid_xgboost, cv=StratifiedKFold(n_splits=5), scoring='f1', n_jobs=-1)
grid_xgboost.fit(X_train_resampled, y_train_resampled)
evaluate_model(grid_xgboost.best_estimator_, X_test, y_test)

svm = SVC(probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train_resampled, y_train_resampled)
print("SVM Classifier Report:")
evaluate_model(svm, X_test, y_test)

naive_bayes = GaussianNB()
naive_bayes.fit(X_train_resampled, y_train_resampled)
print("Naive Bayes Classifier Report:")
evaluate_model(naive_bayes, X_test, y_test)

voting_classifier = VotingClassifier(estimators=[
    ('rf', grid_random_forest.best_estimator_),
    ('xgb', grid_xgboost.best_estimator_),
    ('lr', LogisticRegression())
], voting='soft', weights=[2, 1, 1])
voting_classifier.fit(X_train_resampled, y_train_resampled)
evaluate_model(voting_classifier, X_test, y_test)

#Final Metrics
y_pred = voting_classifier.predict(X_test)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred))
print("MCC:", matthews_corrcoef(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


#Confusion Matrix
def plot_confusion_matrix_custom(model, X_test, y_test, model_name):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix_custom(grid_random_forest.best_estimator_, X_test, y_test, "Random Forest")
plot_confusion_matrix_custom(grid_xgboost.best_estimator_, X_test, y_test, "XGBoost")
plot_confusion_matrix_custom(voting_classifier, X_test, y_test, "Voting Classifier")
plot_confusion_matrix_custom(svm, X_test, y_test, "SVM Classifier")
plot_confusion_matrix_custom(naive_bayes, X_test, y_test, "Naive Bayes Classifier")


#ROC-AUC
#Precision-Recall Curves
def plot_roc_curve(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()
    plt.show()

def plot_precision_recall_curve(model, X_test, y_test, model_name):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f'{model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.legend()
    plt.show()

plot_roc_curve(grid_random_forest.best_estimator_, X_test, y_test, "Random Forest")
plot_roc_curve(grid_xgboost.best_estimator_, X_test, y_test, "XGBoost")
plot_roc_curve(voting_classifier, X_test, y_test, "Voting Classifier")
plot_roc_curve(svm, X_test, y_test, "SVM Classifier")
plot_roc_curve(naive_bayes, X_test, y_test, "Naive Bayes Classifier")

plot_precision_recall_curve(grid_random_forest.best_estimator_, X_test, y_test, "Random Forest")
plot_precision_recall_curve(grid_xgboost.best_estimator_, X_test, y_test, "XGBoost")
plot_precision_recall_curve(voting_classifier, X_test, y_test, "Voting Classifier")
plot_precision_recall_curve(svm, X_test, y_test, "SVM Classifier")
plot_precision_recall_curve(naive_bayes, X_test, y_test, "Naive Bayes Classifier")


#Cross Validation - 5 fold
print("Voting Classifier CV F1:", np.mean(cross_val_score(voting_classifier, X_train_resampled, y_train_resampled, cv=5, scoring='f1')))
print("Random Forest CV F1:", np.mean(cross_val_score(grid_random_forest.best_estimator_, X_train_resampled, y_train_resampled, cv=5, scoring='f1')))
print("XGBoost CV F1:", np.mean(cross_val_score(grid_xgboost.best_estimator_, X_train_resampled, y_train_resampled, cv=5, scoring='f1')))
print("SVM CV F1:", np.mean(cross_val_score(svm, X_train_resampled, y_train_resampled, cv=5, scoring='f1')))


train_sizes, train_scores, val_scores = learning_curve(grid_random_forest.best_estimator_, X_train_resampled, y_train_resampled, cv=5)
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Training score")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation score")
plt.xlabel("Training size")
plt.ylabel("Score")
plt.title("Learning Curve - Random Forest")
plt.legend()
plt.show()

#Feature Importance
rf_importances = pd.DataFrame({'Feature': X.columns, 'Importance': grid_random_forest.best_estimator_.feature_importances_})
xgb_importances = pd.DataFrame({'Feature': X.columns, 'Importance': grid_xgboost.best_estimator_.feature_importances_})

sns.barplot(data=rf_importances.sort_values(by='Importance', ascending=False).head(20), y='Feature', x='Importance')
plt.title('Top 20 Feature Importances - Random Forest')
plt.show()

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#Predictions
rf_preds = grid_random_forest.best_estimator_.predict(X_test)
xgb_preds = grid_xgboost.best_estimator_.predict(X_test)
voting_preds = voting_classifier.predict(X_test)
svm_preds = svm.predict(X_test)
nb_preds = naive_bayes.predict(X_test)

#Compare accuracy
accuracy_data = {
    "Model": ["Random Forest", "XGBoost", "Voting Classifier", "SVM", "Naive Bayes"],
    "Accuracy": [
        accuracy_score(y_test, rf_preds) * 100,
        accuracy_score(y_test, xgb_preds) * 100,
        accuracy_score(y_test, voting_preds) * 100,
        accuracy_score(y_test, svm_preds) * 100,
        accuracy_score(y_test, nb_preds) * 100
    ]
}

df_accuracy = pd.DataFrame(accuracy_data)

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='Accuracy', data=df_accuracy, palette="viridis")


for index, row in df_accuracy.iterrows():
    plt.text(index, row.Accuracy + 0.2, f"{row.Accuracy:.2f}%", ha='center', va='bottom', fontsize=10)

plt.title('Supervised Learning Models - Accuracy Comparison (Auto-Computed)', fontsize=16)
plt.ylabel('Accuracy (%)')
plt.ylim(90, 100)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# PU Learning - Semi Supervised


#Separate Positive and Unlabeled from Training Set
y_train_pos_mask = (y_train_resampled == 1)

X_pos = X_train_resampled[y_train_pos_mask]
y_pos = y_train_resampled[y_train_pos_mask] 

X_unlabeled = X_train_resampled[~y_train_pos_mask]
y_unlabeled = np.zeros(X_unlabeled.shape[0], dtype=int)  # Mark unlabeled as 0 for PU

X_pu = np.vstack((X_pos, X_unlabeled))
y_pu = np.concatenate((y_pos, y_unlabeled))


pu_df = pd.DataFrame(X_pu, columns=X.columns)
pu_df['PU_Label'] = y_pu 
print(pu_df.head())   


print("PU Dataset Shape:", X_pu.shape)
print("Label distribution in PU data:")
print(pd.Series(y_pu).value_counts()) 

sns.countplot(x=y_pu)
plt.title("PU Dataset Label Distribution (1 = Positive, 0 = Unlabeled)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.show()


#Combination of positive + unlabeled data
X_pu = np.vstack((X_pos, X_unlabeled))
y_pu = np.concatenate((y_pos, y_unlabeled))

class ElkanotoPU(GaussianNB):
    def predict_proba(self, X):
        positive_prob = super().predict_proba(X)
        return positive_prob * (self.class_prior_ / positive_prob.sum(axis=1, keepdims=True))
    
    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

#Training
elkanoto_pu = ElkanotoPU()
elkanoto_pu.fit(X_pu, y_pu)
evaluate_model(elkanoto_pu, X_test, y_test)

bagging_pu = BaggingClassifier(estimator=RandomForestClassifier(), n_estimators=10, random_state=42)
bagging_pu.fit(X_pu, y_pu)
evaluate_model(bagging_pu, X_test, y_test)

pu_svm = SVC(probability=True, class_weight='balanced', random_state=42)
pu_svm.fit(X_pu, y_pu)
evaluate_model(pu_svm, X_test, y_test)

def combined_roc_curve(models, X_test, y_test, model_names):
    plt.figure(figsize=(8, 6))
    for model, name in zip(models, model_names):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve - PU Learning Models')
    plt.legend()
    plt.grid(True)
    plt.show()


#Two-Step PU Learning (with Random Forest classifier)
X_pos = X_train_resampled[y_train_resampled == 1]
X_unlabeled = X_train_resampled[y_train_resampled == 0]


X_temp = np.vstack((X_pos, X_unlabeled))
y_temp = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_unlabeled))))

#Train temporary classifier
temp_clf = RandomForestClassifier(class_weight='balanced', random_state=42)
temp_clf.fit(X_temp, y_temp)

#Predict probabilities on unlabeled data
unlabeled_probs = temp_clf.predict_proba(X_unlabeled)[:, 1]

#Choose Reliable Negatives: low positive probability
threshold = 0.2  
rn_mask = unlabeled_probs < threshold
X_rn = X_unlabeled[rn_mask]

print(f"Reliable Negatives Identified: {X_rn.shape[0]} out of {X_unlabeled.shape[0]}")


#Step 2: Train final classifier
X_final = np.vstack((X_pos, X_rn))
y_final = np.concatenate((np.ones(len(X_pos)), np.zeros(len(X_rn))))

rf_two_step = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_two_step.fit(X_final, y_final)


#ROC-AUC
combined_roc_curve([rf_two_step,elkanoto_pu, bagging_pu, pu_svm], X_test, y_test, ["Two-Step PU RF", "Elkanoto PU", "Bagging PU", "PU SVM"])

print("\nTwo-Step PU Learning Evaluation")
evaluate_model(rf_two_step, X_test, y_test)

#Predictions 
def store_predictions(model, X_test):
    preds = model.predict(X_test)
    return np.sum(preds), len(preds) - np.sum(preds)

models_pu = [rf_two_step, voting_classifier]
labels_pu = ['Two-Step PU RF','Voting Classifier']
positives, negatives = [], []

for model in models_pu:
    pos, neg = store_predictions(model, X_test)
    positives.append(pos)
    negatives.append(neg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(labels_pu, positives, label='Predicted Positives', color='teal')
ax.bar(labels_pu, negatives, bottom=positives, label='Predicted Negatives', color='black')
ax.set_title("Model Predictions on Test Data")
ax.set_xlabel("Model")
ax.set_ylabel("Count")
ax.legend()
plt.show()

#Confusion Matrix - PU
plot_confusion_matrix_custom(elkanoto_pu, X_test, y_test, "Elkanoto PU")
plot_confusion_matrix_custom(bagging_pu, X_test, y_test, "Bagging PU")
plot_confusion_matrix_custom(pu_svm, X_test, y_test, "PU SVM")
plot_confusion_matrix_custom(rf_two_step, X_test, y_test, "Two-Step PU RF")

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

models_pu = [rf_two_step, elkanoto_pu, bagging_pu, pu_svm]
model_names_pu = ["Two-Step PU RF", "Elkanoto PU", "Bagging PU", "PU SVM"]

plt.figure(figsize=(8, 6))
for model, name in zip(models_pu, model_names_pu):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve – PU Learning Models')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

#Precision–Recall Curves
plt.figure(figsize=(8, 6))
for model, name in zip(models_pu, model_names_pu):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap_score = average_precision_score(y_test, y_pred_proba)
    plt.plot(recall, precision, label=f"{name} (AP = {ap_score:.2f})")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision–Recall Curve – PU Learning Models')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

#predictions
pu_preds_dict = {
    "Two-Step PU RF": rf_two_step.predict(X_test),
    "Elkanoto PU": elkanoto_pu.predict(X_test),
    "Bagging PU": bagging_pu.predict(X_test),
    "PU SVM": pu_svm.predict(X_test)
}


pu_accuracy_data = {
    "Model": [],
    "Accuracy": []
}

for model_name, preds in pu_preds_dict.items():
    acc = accuracy_score(y_test, preds) * 100
    pu_accuracy_data["Model"].append(model_name)
    pu_accuracy_data["Accuracy"].append(acc)

df_pu_accuracy = pd.DataFrame(pu_accuracy_data)

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=df_pu_accuracy, palette="viridis")

for index, row in df_pu_accuracy.iterrows():
    plt.text(index, row.Accuracy + 0.2, f"{row.Accuracy:.2f}%", ha='center', va='bottom', fontsize=10)

plt.title('PU Learning Models - Accuracy Comparison (Auto-Computed)', fontsize=16)
plt.ylabel('Accuracy (%)')
plt.ylim(90, 100)
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_3d_distribution_without_synthetic(df):
    df['label'] = df['label'].map({0: 'Benign', 1: 'Attack'})
    label_counts = df['label'].value_counts()
    category_counts = df[df['label'] == 'Attack']['category'].value_counts()
    specific_counts = df[df['label'] == 'Attack']['specific_class'].value_counts()

    # Data for plotting
    levels = ['Label', 'Category', 'Specific Class']
    subgroups = [
        ['Benign', 'Malicious/DoS'],
        ['Benign', 'Malicious/DoS', 'Malicious/Spoofing'],
        ['Benign', 'Malicious/Spoofing/Gas', 'Malicious/Spoofing/Steering', 'Malicious/Spoofing/RPM']
    ]
    values = [
        [label_counts.get('Benign', 0), label_counts.get('Attack', 0)],
        [label_counts.get('Benign', 0), category_counts.get('DoS', 0), category_counts.get('Spoofing', 0)],
        [
            label_counts.get('Benign', 0),
            specific_counts.get('Spoofing-GAS', 0),
            specific_counts.get('Spoofing-STEERING', 0),
            specific_counts.get('Spoofing-RPM', 0)
        ]
    ]

    # Plot setup
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    xpos, ypos, zpos, dx, dy, dz, colors = [], [], [], [], [], [], []
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, group in enumerate(values):  
        for j, val in enumerate(group):  
            xpos.append(i * 3)
            ypos.append(j)
            zpos.append(0)
            dx.append(0.9)
            dy.append(0.9)
            dz.append(val)
            colors.append(color_palette[j % len(color_palette)])

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

    ax.set_xticks([i * 3 for i in range(len(levels))])
    ax.set_xticklabels(levels)
    ax.set_yticks(range(5))
    ax.set_yticklabels(['Benign', 'Malicious/DoS', 'Malicious/Spoofing/Gas', 'Malicious/Spoofing/Steering', 'Malicious/Spoofing/RPM'])
    ax.set_zlabel("Number of Instances")
    ax.set_title("3D Distribution Plot - Without Synthetic Data")
    plt.tight_layout()
    plt.show()
plot_3d_distribution_without_synthetic(combined_data)



from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# All models (supervised + PU)
models_combined = {
    'Random Forest': grid_random_forest.best_estimator_,
    'XGBoost': grid_xgboost.best_estimator_,
    'SVM': svm,
    'Naive Bayes': naive_bayes,
    'Voting Classifier': voting_classifier,
    'Two-Step PU RF': rf_two_step,
    'PU SVM': pu_svm,
    'Bagging PU': bagging_pu
}

# Compute F1 and Positives
model_names = []
f1_scores = []
positives_predicted = []

for name, model in models_combined.items():
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    pos = np.sum(preds)
    
    model_names.append(name)
    f1_scores.append(f1)
    positives_predicted.append(pos)

# Create DataFrame
df_performance = pd.DataFrame({
    'Model': model_names,
    'F1-Score': f1_scores,
    'Positives Predicted': positives_predicted
})

df_performance.to_csv("df_performance.csv", index=False)

#PLOT
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for F1-Scores
sns.barplot(x='Model', y='F1-Score', data=df_performance, ax=ax1, color='skyblue', label='F1-Score')
ax1.set_ylabel('F1-Score', color='blue')
ax1.set_ylim(0, 1.05)
ax1.tick_params(axis='y', labelcolor='blue')
plt.xticks(rotation=20)

# Line plot for Positives Predicted
ax2 = ax1.twinx()
sns.lineplot(x='Model', y='Positives Predicted', data=df_performance, ax=ax2, color='red', marker='o', label='Positives Predicted')
ax2.set_ylabel('Positives Predicted', color='red')
ax2.tick_params(axis='y', labelcolor='red')

plt.title('Fig. 6A: Performance without Synthetic Data (Auto-Computed)')
fig.tight_layout()
plt.savefig('Fig_6A_No_Synthetic.png')
plt.show()
