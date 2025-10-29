#!/usr/bin/env python3
"""
Training script for name classification model
Uses sklearn Pipeline for modular preprocessing and model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import warnings
warnings.filterwarnings('ignore')

from feature_transformers import DataPreprocessor
from model_registry import register_champion_model


def load_data():
    """Load the preprocessed training and test data"""
    train_path = Path("data/train_processed.csv")
    test_path = Path("data/test_processed.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Processed data files not found. Please run preprocessing.py first.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    return train_df, test_df


def create_models():
    """Create models optimized for high-dimensional features (including embeddings)"""
    # Random Forest with parameters optimized for high-dimensional data
    rf = RandomForestClassifier(
        n_estimators=200,  # Increased for better performance with more features
        max_depth=25,      # Increased to handle more complex feature space
        min_samples_split=3,  # Reduced to allow more splits
        min_samples_leaf=1,   # Reduced for better granularity
        max_features='sqrt',  # Use sqrt for high-dimensional data
        random_state=42,
        n_jobs=-1
    )
    
    # Logistic Regression with regularization for high-dimensional data
    lr = LogisticRegression(
        random_state=42,
        max_iter=2000,  # Increased for convergence with more features
        C=0.1,          # Regularization for high-dimensional data
        solver='liblinear'  # Good for high-dimensional data
    )

    # Ensemble model: Voting classifier combining top performers
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('lr', lr)
        ],
        voting='soft',  # Use probability-based voting
        n_jobs=-1
    )

    return {
        'Random Forest': rf,
        'Logistic Regression': lr,
        'Ensemble (RF+LR)': ensemble
    }


def evaluate_model_comprehensive(model, X_train, y_train, X_test, y_test, model_name):
    """Comprehensive evaluation of a single model"""
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics for train set
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='weighted')
    train_recall = recall_score(y_train, y_train_pred, average='weighted')
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    
    # Calculate metrics for test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
    
    return {
        'model': model,
        'model_name': model_name,
        'train_metrics': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        },
        'test_metrics': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        },
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_test_pred,
        'probabilities': model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    }


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate multiple models and return comprehensive results"""
    console = Console()
    results = {}
    
    console.print(Panel.fit("🤖 Training and Evaluating Models", style="bold blue"))
    
    for name, model in track(models.items(), description="Training models..."):
        # Comprehensive evaluation
        result = evaluate_model_comprehensive(model, X_train, y_train, X_test, y_test, name)
        results[name] = result
        
        console.print(f"✅ {name}: Test Accuracy = {result['test_metrics']['accuracy']:.4f}, "
                     f"Test F1 = {result['test_metrics']['f1']:.4f}, "
                     f"CV F1 = {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    
    return results


def display_comprehensive_results(results, y_test, label_mapping):
    """Display comprehensive results using Rich tables"""
    console = Console()
    
    # Comprehensive model comparison table
    comparison_table = Table(title="🏆 Comprehensive Model Performance Comparison", show_header=True, header_style="bold magenta")
    comparison_table.add_column("Model", style="cyan", width=20)
    comparison_table.add_column("Train Acc", style="green", width=10)
    comparison_table.add_column("Test Acc", style="green", width=10)
    comparison_table.add_column("Train F1", style="yellow", width=10)
    comparison_table.add_column("Test F1", style="yellow", width=10)
    comparison_table.add_column("CV F1", style="blue", width=10)
    comparison_table.add_column("CV Std", style="red", width=10)
    
    for name, result in results.items():
        comparison_table.add_row(
            name,
            f"{result['train_metrics']['accuracy']:.4f}",
            f"{result['test_metrics']['accuracy']:.4f}",
            f"{result['train_metrics']['f1']:.4f}",
            f"{result['test_metrics']['f1']:.4f}",
            f"{result['cv_mean']:.4f}",
            f"{result['cv_std']:.4f}"
        )
    
    console.print(comparison_table)
    
    # Detailed metrics table for each model
    console.print(Panel.fit("📊 Detailed Model Metrics", style="bold blue"))
    
    for name, result in results.items():
        model_table = Table(title=f"📈 {name} - Detailed Metrics", show_header=True, header_style="bold cyan")
        model_table.add_column("Metric", style="cyan", width=15)
        model_table.add_column("Train Set", style="green", width=12)
        model_table.add_column("Test Set", style="yellow", width=12)
        model_table.add_column("Overfitting", style="red", width=12)
        
        # Calculate overfitting indicators
        acc_diff = result['train_metrics']['accuracy'] - result['test_metrics']['accuracy']
        f1_diff = result['train_metrics']['f1'] - result['test_metrics']['f1']
        
        model_table.add_row(
            "Accuracy",
            f"{result['train_metrics']['accuracy']:.4f}",
            f"{result['test_metrics']['accuracy']:.4f}",
            f"{acc_diff:+.4f}"
        )
        model_table.add_row(
            "Precision",
            f"{result['train_metrics']['precision']:.4f}",
            f"{result['test_metrics']['precision']:.4f}",
            f"{result['train_metrics']['precision'] - result['test_metrics']['precision']:+.4f}"
        )
        model_table.add_row(
            "Recall",
            f"{result['train_metrics']['recall']:.4f}",
            f"{result['test_metrics']['recall']:.4f}",
            f"{result['train_metrics']['recall'] - result['test_metrics']['recall']:+.4f}"
        )
        model_table.add_row(
            "F1-Score",
            f"{result['train_metrics']['f1']:.4f}",
            f"{result['test_metrics']['f1']:.4f}",
            f"{f1_diff:+.4f}"
        )
        
        console.print(model_table)
        console.print()  # Add spacing between models
    
    # Champion model selection
    console.print(Panel.fit("🏆 Champion Model Selection", style="bold green"))
    
    # Select champion based on test F1 score (balanced metric)
    champion_name = max(results.keys(), key=lambda x: results[x]['test_metrics']['f1'])
    champion_result = results[champion_name]
    
    console.print(f"🥇 Champion Model: {champion_name}")
    console.print(f"   Selection Criteria: Highest Test F1-Score")
    console.print(f"   Test F1-Score: {champion_result['test_metrics']['f1']:.4f}")
    console.print(f"   Test Accuracy: {champion_result['test_metrics']['accuracy']:.4f}")
    console.print(f"   Overfitting (Acc): {champion_result['train_metrics']['accuracy'] - champion_result['test_metrics']['accuracy']:+.4f}")
    
    # Classification report for champion
    class_names = list(label_mapping.keys())
    report = classification_report(y_test, champion_result['predictions'], target_names=class_names, output_dict=True)
    
    report_table = Table(title=f"📊 Champion Model ({champion_name}) - Classification Report", show_header=True, header_style="bold magenta")
    report_table.add_column("Class", style="cyan")
    report_table.add_column("Precision", style="green")
    report_table.add_column("Recall", style="yellow")
    report_table.add_column("F1-Score", style="blue")
    report_table.add_column("Support", style="red")
    
    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            report_table.add_row(
                class_name,
                f"{metrics['precision']:.3f}",
                f"{metrics['recall']:.3f}",
                f"{metrics['f1-score']:.3f}",
                str(int(metrics['support']))
            )
    
    # Add macro and weighted averages
    report_table.add_row("", "", "", "", "")
    report_table.add_row(
        "Macro Avg",
        f"{report['macro avg']['precision']:.3f}",
        f"{report['macro avg']['recall']:.3f}",
        f"{report['macro avg']['f1-score']:.3f}",
        str(int(report['macro avg']['support']))
    )
    report_table.add_row(
        "Weighted Avg",
        f"{report['weighted avg']['precision']:.3f}",
        f"{report['weighted avg']['recall']:.3f}",
        f"{report['weighted avg']['f1-score']:.3f}",
        str(int(report['weighted avg']['support']))
    )
    
    console.print(report_table)
    
    return champion_name, champion_result['model']


def display_confusion_matrix(y_true, y_pred, label_mapping, model_name):
    """Display confusion matrix in a formatted table"""
    console = Console()

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_names = list(label_mapping.keys())

    # Create table for confusion matrix
    cm_table = Table(title=f"🎯 Confusion Matrix - {model_name}", show_header=True, header_style="bold cyan")
    cm_table.add_column("True \\ Pred", style="cyan", width=12)

    for class_name in class_names:
        cm_table.add_column(class_name, style="yellow", width=10, justify="center")

    # Add rows
    for i, true_class in enumerate(class_names):
        row = [true_class]
        for j in range(len(class_names)):
            # Highlight diagonal (correct predictions)
            if i == j:
                row.append(f"[bold green]{cm[i][j]}[/bold green]")
            else:
                row.append(str(cm[i][j]))
        cm_table.add_row(*row)

    console.print(cm_table)

    # Calculate and display per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    accuracy_table = Table(title=f"📊 Per-Class Metrics - {model_name}", show_header=True, header_style="bold magenta")
    accuracy_table.add_column("Class", style="cyan", width=15)
    accuracy_table.add_column("Accuracy", style="green", width=12)
    accuracy_table.add_column("Total Samples", style="yellow", width=15)

    for i, class_name in enumerate(class_names):
        accuracy_table.add_row(
            class_name,
            f"{per_class_accuracy[i]:.4f}",
            str(cm.sum(axis=1)[i])
        )

    console.print(accuracy_table)


def hyperparameter_tuning(X_train, y_train, model_name, model):
    """Perform hyperparameter tuning for the best model"""
    console = Console()
    
    console.print(Panel.fit(f"🔧 Hyperparameter Tuning for {model_name}", style="bold blue"))
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
    }
    
    if model_name in param_grids:
        grid_search = GridSearchCV(
            model,
            param_grids[model_name],
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        console.print("Searching for best parameters...")
        grid_search.fit(X_train, y_train)
        
        console.print(f"Best parameters: {grid_search.best_params_}")
        console.print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    else:
        console.print(f"No hyperparameter tuning defined for {model_name}")
        return model


def save_model_and_preprocessor(model, preprocessor, model_name, output_dir="models"):
    """Save the trained model and preprocessor for inference"""
    console = Console()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save model
    model_path = output_path / f"{model_name.lower().replace(' ', '_')}_model.joblib"
    joblib.dump(model, model_path)
    
    # Save preprocessor
    preprocessor_path = output_path / f"{model_name.lower().replace(' ', '_')}_preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save label mappings
    label_mapping_path = output_path / f"{model_name.lower().replace(' ', '_')}_label_mapping.joblib"
    label_mappings = {
        'label_mapping': preprocessor.get_label_mapping(),
        'inverse_mapping': preprocessor.get_inverse_label_mapping()
    }
    joblib.dump(label_mappings, label_mapping_path)
    
    console.print(Panel.fit(f"💾 Model and preprocessor saved to {output_path}", style="bold green"))
    console.print(f"  Model: {model_path}")
    console.print(f"  Preprocessor: {preprocessor_path}")
    console.print(f"  Label mappings: {label_mapping_path}")


def run():
    """Main training function"""
    console = Console()
    
    console.print(Panel.fit("🚀 NAME CLASSIFICATION MODEL TRAINING", style="bold blue"))
    
    try:
        # Load data
        console.print("📂 Loading data...")
        train_df, test_df = load_data()
        
        console.print(f"  Training set: {len(train_df):,} samples")
        console.print(f"  Test set: {len(test_df):,} samples")
        
        # Initialize preprocessor with embeddings
        console.print("🔧 Initializing preprocessor with embeddings...")
        preprocessor = DataPreprocessor(
            use_char_ngrams=True,
            use_embeddings=True,  # Enable embeddings
            embedding_model='all-MiniLM-L6-v2'
        )
        
        # Prepare data
        X_train = train_df[['dirty_name']]
        y_train = train_df['dirty_label']
        X_test = test_df[['dirty_name']]
        y_test = test_df['dirty_label']
        
        # Fit preprocessor on training data (includes embedding model download/fitting)
        console.print("🧹 Preprocessing training data with embeddings...")
        console.print("   This may take a few minutes for first-time embedding model download...")
        X_train_processed, y_train_encoded = preprocessor.fit_transform(X_train, y_train)
        
        # Transform test data
        console.print("🧹 Preprocessing test data with embeddings...")
        X_test_processed, y_test_encoded = preprocessor.transform(X_test, y_test)
        
        console.print(f"  Processed training set: {X_train_processed.shape}")
        console.print(f"  Processed test set: {X_test_processed.shape}")
        console.print(f"  Features: {list(X_train_processed.columns)}")
        
        # Get label mapping for display
        label_mapping = preprocessor.get_label_mapping()
        console.print(f"  Label mapping: {label_mapping}")
        
        # Create and evaluate models
        models = create_models()
        results = evaluate_models(X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, models)
        
        # Display comprehensive results
        best_model_name, best_model = display_comprehensive_results(results, y_test_encoded, label_mapping)

        # Display confusion matrix for champion model
        console.print("\n" + "="*60)
        display_confusion_matrix(y_test_encoded, results[best_model_name]['predictions'], label_mapping, best_model_name)

        # Hyperparameter tuning
        console.print("\n" + "="*60)
        tuned_model = hyperparameter_tuning(X_train_processed, y_train_encoded, best_model_name, best_model)
        
        # Final evaluation of tuned model
        console.print("\n" + "="*60)
        console.print(Panel.fit("🎯 Final Champion Model Evaluation", style="bold blue"))
        
        # Comprehensive evaluation of tuned model
        final_result = evaluate_model_comprehensive(tuned_model, X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, f"{best_model_name} (Tuned)")
        
        # Display final results
        final_table = Table(title="🏆 Final Champion Model Performance", show_header=True, header_style="bold green")
        final_table.add_column("Metric", style="cyan", width=15)
        final_table.add_column("Train Set", style="green", width=12)
        final_table.add_column("Test Set", style="yellow", width=12)
        final_table.add_column("Overfitting", style="red", width=12)
        
        acc_diff = final_result['train_metrics']['accuracy'] - final_result['test_metrics']['accuracy']
        f1_diff = final_result['train_metrics']['f1'] - final_result['test_metrics']['f1']
        
        final_table.add_row(
            "Accuracy",
            f"{final_result['train_metrics']['accuracy']:.4f}",
            f"{final_result['test_metrics']['accuracy']:.4f}",
            f"{acc_diff:+.4f}"
        )
        final_table.add_row(
            "Precision",
            f"{final_result['train_metrics']['precision']:.4f}",
            f"{final_result['test_metrics']['precision']:.4f}",
            f"{final_result['train_metrics']['precision'] - final_result['test_metrics']['precision']:+.4f}"
        )
        final_table.add_row(
            "Recall",
            f"{final_result['train_metrics']['recall']:.4f}",
            f"{final_result['test_metrics']['recall']:.4f}",
            f"{final_result['train_metrics']['recall'] - final_result['test_metrics']['recall']:+.4f}"
        )
        final_table.add_row(
            "F1-Score",
            f"{final_result['train_metrics']['f1']:.4f}",
            f"{final_result['test_metrics']['f1']:.4f}",
            f"{f1_diff:+.4f}"
        )
        
        console.print(final_table)

        # Display confusion matrix for final tuned model
        console.print("\n" + "="*60)
        display_confusion_matrix(y_test_encoded, final_result['predictions'], label_mapping, f"{best_model_name} (Tuned)")

        # Save model and preprocessor
        save_model_and_preprocessor(tuned_model, preprocessor, best_model_name)
        
        # Register champion model in registry
        console.print(Panel.fit("📝 Registering Champion Model", style="bold blue"))
        
        champion_metrics = {
            "test_accuracy": final_result['test_metrics']['accuracy'],
            "test_f1": final_result['test_metrics']['f1'],
            "test_precision": final_result['test_metrics']['precision'],
            "test_recall": final_result['test_metrics']['recall'],
            "train_accuracy": final_result['train_metrics']['accuracy'],
            "train_f1": final_result['train_metrics']['f1'],
            "cv_f1_mean": final_result['cv_mean'],
            "cv_f1_std": final_result['cv_std'],
            "overfitting_accuracy": acc_diff,
            "overfitting_f1": f1_diff
        }
        
        model_id = register_champion_model(
            model_name=best_model_name.lower().replace(' ', '_'),
            model_type=best_model_name,
            metrics=champion_metrics
        )
        
        console.print(f"✅ Champion model registered with ID: {model_id}")
        console.print(f"   Model: {best_model_name}")
        console.print(f"   Test F1-Score: {champion_metrics['test_f1']:.4f}")
        console.print(f"   Test Accuracy: {champion_metrics['test_accuracy']:.4f}")
        
        console.print(Panel.fit("✅ Training Complete!", style="bold green"))
        
    except Exception as e:
        console.print(Panel.fit(f"❌ Error: {str(e)}", style="bold red"))
        raise


if __name__ == "__main__":
    run()
