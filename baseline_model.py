"""
Baseline sentiment analysis model
Uses TF-IDF vectorization and Logistic Regression
"""

import os
import pandas as pd
import json
import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineModel:
    """Baseline sentiment analysis model"""
    
    def __init__(self):
        """Initialize the baseline pipeline"""
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8,
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                solver='liblinear',
                class_weight='balanced',
                random_state=42
            ))
        ])
        self.metrics = {}
        print("✅ Baseline model initialized")
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        
        self.pipeline.fit(X_train, y_train)
        print("✅ Model training complete!")
        
        # Get feature importance
        tfidf = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['clf']
        
        feature_names = np.array(tfidf.get_feature_names_out())
        coefficients = clf.coef_[0]
        
        # Top positive and negative features
        top_positive_idx = np.argsort(coefficients)[-10:]
        top_negative_idx = np.argsort(coefficients)[:10]
        
        print("\nTop Positive Features:")
        for idx in reversed(top_positive_idx):
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
        
        print("\nTop Negative Features:")
        for idx in reversed(top_negative_idx):
            print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    def evaluate(self, X_test, y_test, split_name='test'):
        """Evaluate model on test set"""
        print(f"\n{'='*60}")
        print(f"EVALUATING ON {split_name.upper()}")
        print(f"{'='*60}")
        
        # Predictions
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        self.metrics[split_name] = metrics
        
        # Print metrics
        print("\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper():12}: {value:.4f}")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Negative', 'Positive']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
        
        return metrics, y_pred, y_pred_proba
    
    def save(self, model_path='models/baseline_model.pkl'):
        """Save trained model"""
        os.makedirs('models', exist_ok=True)
        pickle.dump(self.pipeline, open(model_path, 'wb'))
        print(f"\n✅ Model saved to {model_path}")
    
    def load(self, model_path='models/baseline_model.pkl'):
        """Load trained model"""
        self.pipeline = pickle.load(open(model_path, 'rb'))
        print(f"✅ Model loaded from {model_path}")
    
    def predict(self, text):
        """Make prediction on new text"""
        prediction = self.pipeline.predict([text])[0]
        probability = self.pipeline.predict_proba([text])[0]
        
        return {
            'sentiment': 'positive' if prediction == 1 else 'negative',
            'confidence': float(probability[prediction]),
            'probabilities': {
                'negative': float(probability[0]),
                'positive': float(probability[1])
            }
        }


def plot_confusion_matrix(y_test, y_pred, split_name='test'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {split_name.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/confusion_matrix_{split_name}_baseline.png', 
                dpi=300, bbox_inches='tight')
    print(f"✅ Saved confusion matrix plot")
    plt.close()


def main():
    print("\n" + "="*60)
    print("🎬 BASELINE SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('data/train_data.csv')
    val_df = pd.read_csv('data/val_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    # Initialize and train model
    model = BaselineModel()
    model.train(train_df['text'].values, train_df['label'].values)
    
    # Evaluate on validation set
    val_metrics, _, _ = model.evaluate(
        val_df['text'].values, 
        val_df['label'].values, 
        'val'
    )
    
    # Evaluate on test set
    test_metrics, y_pred, y_pred_proba = model.evaluate(
        test_df['text'].values, 
        test_df['label'].values, 
        'test'
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(test_df['label'].values, y_pred, 'test')
    
    # Save model
    model.save()
    
    # Save metrics
    os.makedirs('results', exist_ok=True)
    with open('results/baseline_metrics.json', 'w') as f:
        json.dump(model.metrics, f, indent=4)
    print(f"✅ Metrics saved to results/baseline_metrics.json")
    
    print("\n" + "="*60)
    print("✅ BASELINE MODEL TRAINING COMPLETE!")
    print("="*60)
    
    return model


if __name__ == "__main__":
    model = main()
