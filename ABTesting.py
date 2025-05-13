import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel

#Load Performance Results
data_original = pd.read_csv("df_performance.csv")        
data_synthetic = pd.read_csv("df_performance_synth.csv")        

#Sort Models
data_original = data_original.sort_values('Model').reset_index(drop=True)
data_synthetic = data_synthetic.sort_values('Model').reset_index(drop=True)

#Dataset Labels
data_original['Dataset'] = 'Without Synthetic'
data_synthetic['Dataset'] = 'With Synthetic'

#Combined Data
data_ab = pd.concat([data_original, data_synthetic], ignore_index=True)

plt.figure(figsize=(14, 6))
sns.barplot(data=data_ab, x='Model', y='F1-Score', hue='Dataset')
plt.title("F1-Score Comparison: With vs Without Synthetic Data")
plt.ylabel("F1-Score")
plt.xticks(rotation=20)
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("abtest_f1_comparison.png")
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(data=data_ab, x='Model', y='Positives Predicted', hue='Dataset')
plt.title("Predicted Positives: With vs Without Synthetic Data")
plt.ylabel("Predicted Positives")
plt.xticks(rotation=20)
plt.legend(title="Dataset")
plt.tight_layout()
plt.savefig("abtest_predicted_positives.png")
plt.show()

#A/B Testing: Paired t-tests
# F1-Score t-test
t_stat_f1, p_val_f1 = ttest_rel(data_original['F1-Score'], data_synthetic['F1-Score'])

print("Paired t-test on F1-Scores")
print(f"T-statistic: {t_stat_f1:.4f}")
print(f"P-value: {p_val_f1:.4f}")
if p_val_f1 < 0.05:
    print("Statistically significant difference (p < 0.05). Synthetic data had an impact on F1.")
else:
    print("No statistically significant difference in F1-scores.")

# Predicted Positives t-test
t_stat_pos, p_val_pos = ttest_rel(data_original['Positives Predicted'], data_synthetic['Positives Predicted'])

print("\nPaired t-test on Predicted Positives")
print(f"T-statistic: {t_stat_pos:.4f}")
print(f"P-value: {p_val_pos:.4f}")
if p_val_pos < 0.05:
    print("Statistically significant difference in predicted positives.")
else:
    print("No statistically significant difference in predicted positives.")

print("\nModel-wise F1 Improvements")
diff_data = pd.DataFrame({
    'Model': data_original['Model'],
    'F1_Without_Synthetic': data_original['F1-Score'],
    'F1_With_Synthetic': data_synthetic['F1-Score'],
    'F1_Difference': data_synthetic['F1-Score'] - data_original['F1-Score']
})
print(diff_data)

data_ab.to_csv("ab_test_combined_results.csv", index=False)
diff_data.to_csv("ab_test_f1_diff_table.csv", index=False)
