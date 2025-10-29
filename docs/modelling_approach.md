# Modelling Approach

## Overview

The name classification system uses a **hybrid machine learning approach** that combines domain-specific feature engineering with character-level pattern recognition and a Random Forest classifier. This approach achieves 93.14% test accuracy while maintaining fast inference (<10ms) on CPU-only hardware.

## Architecture

```
Input Name → Text Cleaning → Feature Extraction → Random Forest → Classification
                                    ↓
                         65 Features Total:
                         - 9 basic text features
                         - 6 domain-specific features
                         - 50 character n-gram features
```

## Feature Engineering

### 1. Basic Text Features (9 features)
- **Length metrics**: Character count, word count
- **Pattern detection**: Prefixes (Dr., Prof.), suffixes (Ltd, Inc, Corp)
- **Character analysis**: Uppercase flag, numbers, special characters
- **Vowel patterns**: Starts/ends with vowel

### 2. Domain-Specific Features (6 features)
- **University keywords**: Detects "university", "college", "institute", "school", "academy"
- **Company suffixes**: Extended detection for "pty", "ltd", "inc", "corp", "llc", "limited", "plc", "gmbh"
- **Acronym patterns**: Consecutive capitals (IBM, UCLA, MIT)
- **Capital ratio**: Proportion of uppercase letters
- **Word length patterns**: First/last word length (e.g., "The University", "Smith LLC")

### 3. Character N-gram TF-IDF (50 features)
- **N-gram range**: 3-4 character sequences
- **Purpose**: Captures subword patterns ("niv" in University, "Inc" in companies)
- **Advantage**: Works for abbreviations and cross-language names
- **Constraint**: Limited to 50 features for memory efficiency

## Model Selection: Random Forest

### Configuration
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1
)
```

### Why Random Forest?
- ✅ Handles mixed feature types (binary, numeric, TF-IDF)
- ✅ Fast CPU inference (~10ms per prediction)
- ✅ No feature scaling required
- ✅ Robust to overfitting (1.27% gap in our model)
- ✅ Small model size (~11MB)
- ✅ Fits resource constraints (1 vCPU, 2GB RAM)

## Performance Metrics

```
Champion Model: Random Forest (Tuned)
├─ Test Accuracy: 93.14%
├─ Test F1-Score: 0.9248
├─ Training Accuracy: 94.41%
├─ Overfitting Gap: 1.27% (excellent generalization)
└─ Per-Class Performance:
   ├─ Person: 99.6% precision
   ├─ University: 95.0% precision
   └─ Company: 60.3% precision
```

## Why "Hybrid"?

The approach is hybrid because it combines three paradigms:

1. **Rule-based**: Explicit domain knowledge (e.g., "university" keyword → University class)
2. **Statistical**: Data-driven character patterns learned via TF-IDF
3. **Machine Learning**: Random Forest learns optimal feature combinations

This provides both **interpretability** (we understand the rules) and **learning power** (ML handles edge cases).

## Design Constraints

The model was designed to meet strict production requirements:

| Constraint | Target | Actual |
|------------|--------|--------|
| CPU | Max 1 vCPU | ✅ Single core |
| Memory | Max 2GB RAM | ✅ ~500MB peak |
| Image Size | <6GB | ✅ ~1.5GB |
| Inference Time | <100ms | ✅ ~10ms |
| Accuracy | >85% | ✅ 93.14% |

## Alternatives Considered

| Approach | Accuracy | Why Not Selected |
|----------|----------|------------------|
| NLTK NER | ~70-80% | No "University" category, lower accuracy |
| spaCy Custom NER | ~85-90% | Requires more training data, GPU recommended |
| DistilBERT | ~95%+ | >1GB RAM per request, slow on CPU, large model |
| Rule-based only | ~75% | Brittle, misses edge cases |

## Model Registry & Champion Selection

The system uses a **model registry** (`models/model_registry.json`) to track all trained models:

- Automatically logs metrics for each trained model
- Selects champion based on test F1-score
- Enables easy model comparison and rollback
- Supports A/B testing of different approaches

## Future Improvements

Potential enhancements for higher accuracy:

1. **Active learning**: Label misclassified examples to improve Company class (currently 60.3%)
2. **Ensemble methods**: Combine Random Forest with Gradient Boosting
3. **Domain expansion**: Add features for detecting founders' names in companies
4. **Cross-validation tuning**: Further hyperparameter optimization
5. **Data augmentation**: Generate synthetic training examples for underrepresented classes

## References

- Training implementation: `src/training/train_model.py`
- Feature engineering: `src/training/transformers.py`
- Inference pipeline: `src/training/inference.py`
- Model registry: `src/training/model_registry.py`
