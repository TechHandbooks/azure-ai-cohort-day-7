# Import SMOTE and necessary libraries
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
import numpy as np

# Create an imbalanced dataset
X, y = make_classification(n_classes=2, class_sep=2, 
                           weights=[0.9, 0.1], n_informative=3, 
                           n_redundant=1, flip_y=0, 
                           n_features=5, n_clusters_per_class=1, 
                           n_samples=1000, random_state=42)

print(f"Original class distribution: {np.bincount(y)}")

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Resampled class distribution: {np.bincount(y_resampled)}")
