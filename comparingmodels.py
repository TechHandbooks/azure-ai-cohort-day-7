# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Train Logistic Regression
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
log_reg_preds = log_reg.predict(X_test)

# Train Random Forest
rand_forest = RandomForestClassifier(random_state=42)
rand_forest.fit(X_train, y_train)
rand_forest_preds = rand_forest.predict(X_test)

# Compare accuracies
log_reg_acc = accuracy_score(y_test, log_reg_preds)
rand_forest_acc = accuracy_score(y_test, rand_forest_preds)

print(f"Logistic Regression Accuracy: {log_reg_acc}")
print(f"Random Forest Accuracy: {rand_forest_acc}")
