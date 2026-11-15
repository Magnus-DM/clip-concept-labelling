import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

# --- SETTINGS ---
dataset = ['cub200', 'celebA'][1]  # Options: 'cub200', 'celebA'
prompt = 5  # Options: 1, 2
attempt = 1  # Options: 1, 2

all_concepts_csv = f'{dataset}_attribute_accuracies.csv'  # CSV with columns: attribute_number, balanced_accuracy
selected_concepts_csv = f'{dataset}_results/{dataset}_prompt{prompt}_try{attempt}.csv'  # CSV with column: attribute_number

# --- LOAD DATA ---
all_df = pd.read_csv(all_concepts_csv)
selected_df = pd.read_csv(selected_concepts_csv)

# Mark if each concept was selected (new column: 'Selected')
all_df['Selected'] = all_df['attribute_number'].isin(selected_df['attribute_number'])

# Check basic counts
print('Total concepts:', len(all_df))
print('Selected concepts:', all_df['Selected'].sum())

# --- CALCULATE PAIRWISE AUC ---
# First argument: binary indicator (Selected)
# Second argument: continuous score (balanced_accuracy)
auc_score = roc_auc_score(all_df['Selected'], all_df['balanced_accuracy'])
print(f'Pairwise AUC (sklearn method): {auc_score:.3f}')

# --- GET ROC CURVE POINTS ---
fpr, tpr, thresholds = roc_curve(all_df['Selected'], all_df['balanced_accuracy'])

# --- PLOT ROC CURVE ---
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for LLM Attribute Selection')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(f'{dataset}_results/ROC_{dataset}_prompt{prompt}_try{attempt}.svg')
plt.show()

# --- PLOT DISTRIBUTION OF ACCURACIES ---
plt.figure(figsize=(8, 5))

plt.hist(all_df[all_df['Selected']]['balanced_accuracy'], bins=15, alpha=0.7,
         label='Selected', edgecolor='black')
plt.hist(all_df[~all_df['Selected']]['balanced_accuracy'], bins=15, alpha=0.7,
         label='Not selected', edgecolor='black')
plt.xlabel('Balanced Accuracy')
plt.ylabel('Count')
plt.title(f'Distribution of Balanced Accuracy\n(Selected vs. Not Selected)') #\nAUC = {auc_score:.3f}')
plt.legend()
plt.tight_layout()
plt.savefig(f'{dataset}_results/BarChart_{dataset}_prompt{prompt}_try{attempt}.svg')
plt.show()
