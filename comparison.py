import matplotlib.pyplot as plt
import numpy as np

# [F1_no_synth, F1_with_synth, Prec_no_synth, Prec_with_synth, Recall_no_synth, Recall_with_synth, Acc_no_synth, Acc_with_synth]
results = {
    'Random Forest':     [0.86, 0.9868, 1.00, 0.9891, 0.75, 0.9845, 99.72, 98.84],
    'XGBoost':           [0.8571, 0.9866, 0.75, 0.9760, 1.00, 0.9974, 99.63, 98.72],
    'SVM':               [0.8333, 0.9943, 0.8333, 0.9943, 0.8333, 0.9943, 99.63, 98.61],
    'Naive Bayes':       [0.2105, 0.9940, 0.1250, 0.9941, 0.7500, 0.9939, 93.87, 97.93],
    'Voting Classifier': [0.9167, 0.9943, 0.9167, 0.9945, 0.9167, 0.9941, 99.01, 98.87],
    'Two-Step PU RF':    [0.88, 0.9918, 0.93, 0.9919, 0.9167, 0.9918, 99.72, 98.69],
    'Bagging PU':        [0.80, 0.9909, 1.00, 0.9917, 0.6667, 0.9902, 99.63, 98.73],
    'PU SVM':            [0.8333, 0.9943, 0.8333, 0.9943, 0.8333, 0.9943, 99.63, 98.52],
    'Elkanoto PU':       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 98.89, 52.26]
}

models = list(results.keys())
metrics = ['F1 Score', 'Precision', 'Recall', 'Accuracy']

for i, metric in enumerate(metrics):
    without_synth = [results[model][i * 2] for model in models]
    with_synth = [results[model][i * 2 + 1] for model in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, without_synth, width, label='Without Synthetic', color='skyblue')
    bars2 = ax.bar(x + width/2, with_synth, width, label='With Synthetic', color='orange')

    ax.set_ylabel('Score (%)' if metric == 'Accuracy' else metric)
    ax.set_title(f'{metric} Comparison: With vs Without Synthetic Data', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, 105 if metric == 'Accuracy' else 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    def annotate_bars(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    annotate_bars(bars1)
    annotate_bars(bars2)

    plt.tight_layout()
    plt.show()
