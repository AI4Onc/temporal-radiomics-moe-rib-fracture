import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, Multiply, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers.legacy import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras import backend as K


def merge_radiomics_data(df):
    # Create a dictionary to store merged data
    merged_data = []
    grouped = df.groupby(['PID', 'Rib'])
    
    for (pid, rib), group in grouped:
        merged_row = {'PID': pid, 'Rib': rib, 'Fracture': group['Fracture'].iloc[0]}
        # Merge features from Time 1, 2, 3
        for time in [1, 2, 3]:
            time_group = group[group['Time'] == time]
            if not time_group.empty:
                for col in time_group.columns:
                    if col not in ['PID', 'Time', 'Rib', 'Fracture']:
                        merged_row[f"{col}_{time}"] = time_group[col].values[0]
        
        merged_data.append(merged_row)
    merged_df = pd.DataFrame(merged_data)
    return merged_df

def generate_case_specific_knowledge(
    X_df,
    feature_to_embedding,
    unique_groups,
    use_abs=True,
    normalize=False,
    eps=1e-8
):
    knowledge_dim = len(unique_groups)

    feature_embeddings = []
    valid_features = []

    for feature in X_df.columns:
        if feature in feature_to_embedding:
            feature_embeddings.append(feature_to_embedding[feature])
            valid_features.append(feature)

    if len(valid_features) == 0:
        return np.zeros((X_df.shape[0], knowledge_dim), dtype=np.float32)

    feature_embedding_matrix = np.stack(feature_embeddings).astype(np.float32)
    X_values = X_df[valid_features].values.astype(np.float32)
    X_values = np.nan_to_num(X_values, nan=0.0, posinf=0.0, neginf=0.0)

    if use_abs:
        X_values = np.abs(X_values)

    # Sum feature values within each group
    group_sums = np.matmul(X_values, feature_embedding_matrix)

    # Count number of features in each group
    group_counts = feature_embedding_matrix.sum(axis=0, keepdims=True)

    # Group-wise average radiomic magnitude
    knowledge_matrix = group_sums / (group_counts + eps)

    # Optional row normalization; set False if matching manuscript exactly
    if normalize:
        row_sums = knowledge_matrix.sum(axis=1, keepdims=True)
        knowledge_matrix = knowledge_matrix / (row_sums + eps)

    return knowledge_matrix.astype(np.float32)

