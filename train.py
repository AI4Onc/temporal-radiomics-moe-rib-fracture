from utility import merge_radiomics_data, generate_case_specific_knowledge
from model import TimeMoEWithKAN
import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


def main():
    data_path = "train_930.xlsx"
    df = pd.read_excel(data_path)

    # Merge same rib
    merged_df = merge_radiomics_data(df)

    # Prepare Features and Labels
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan).dropna()
    X = merged_df.drop(columns=['PID', 'Rib', 'Fracture'])
    y = merged_df['Fracture']

    # Step 2.2: Load Testing Data
    test_data_path = "test.xlsx"
    df_test = pd.read_excel(test_data_path)

    # Merge same rib
    merged_df_test = merge_radiomics_data(df_test)

    # Prepare Features and Labels for testing
    merged_df_test = merged_df_test.replace([np.inf, -np.inf], np.nan).dropna()
    X_test = merged_df_test.drop(columns=['PID', 'Rib', 'Fracture'])
    y_test = merged_df_test['Fracture']


    undersampler = RandomUnderSampler(sampling_strategy={0:528, 1: 528}, random_state=42)
    X_resampled, y_resampled = X[:1800],y[:1800]

    num_radiomics = 1051
    X_1 = X_resampled.iloc[:, :num_radiomics]
    X_2 = X_resampled.iloc[:, num_radiomics:2*num_radiomics]
    X_3 = X_resampled.iloc[:, 2*num_radiomics:]

    print(y_resampled.value_counts()[1], y_resampled.value_counts()[0])

    # Step 5: Generate Knowledge Embeddings
    # Define feature groups based on renamed feature names
    feature_groups = {
        'Shape': [col for col in X.columns if 'shape' in col.lower()],
        'FirstOrder': [col for col in X.columns if 'firstorder' in col.lower()],
        'GLCM': [col for col in X.columns if 'glcm' in col.lower()],
        'GLRLM': [col for col in X.columns if 'glrlm' in col.lower()],
        'GLSZM': [col for col in X.columns if 'glszm' in col.lower()],
        'GLDM': [col for col in X.columns if 'gldm' in col.lower()],
        'NGTDM': [col for col in X.columns if 'ngtdm' in col.lower()],
        'Original': [col for col in X.columns if 'original' in col.lower()],
        'log3': [col for col in X.columns if 'log-sigma-3' in col.lower()],
        'log5': [col for col in X.columns if 'log-sigma-5' in col.lower()],
    }

    # Assign group index
    unique_groups = list(feature_groups.keys())
    group_to_idx = {group: i for i, group in enumerate(unique_groups)}

    # Create knowledge embeddings for each feature without overwriting
    # Each feature can have a multi-hot embedding if it belongs to multiple groups
    feature_to_embedding = {}

    for group, features in feature_groups.items():
        group_idx = group_to_idx[group]
        for feature in features:
            # If this feature has not appeared before, initialize a zero vector
            if feature not in feature_to_embedding:
                feature_to_embedding[feature] = np.zeros(len(unique_groups), dtype=np.float32)
            # Add current group information without overwriting previous groups
            feature_to_embedding[feature][group_idx] = 1.0

    # Aggregate knowledge embeddings for each timestamp
    # ============================================================
    # Step 6: Prepare case-specific knowledge inputs
    # ============================================================

    knowledge_1 = generate_case_specific_knowledge(X_1, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_2 = generate_case_specific_knowledge(X_2, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_3 = generate_case_specific_knowledge(X_3, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)

    X_train = [X_1, X_2, X_3, knowledge_1, knowledge_2, knowledge_3]

    #initial train
    X_train_slice = X[:963]
    y_train_data = y[:963]

    X_train_1 = X_train_slice.iloc[:, :num_radiomics]
    X_train_2 = X_train_slice.iloc[:, num_radiomics:2*num_radiomics]
    X_train_3 = X_train_slice.iloc[:, 2*num_radiomics:]

    # Generate Knowledge Embeddings for test data
    knowledge_train_1 = generate_case_specific_knowledge(X_train_1, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_train_2 = generate_case_specific_knowledge(X_train_2, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_train_3 = generate_case_specific_knowledge(X_train_3, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)

    X_train_data = [X_train_1, X_train_2, X_train_3, knowledge_train_1, knowledge_train_2, knowledge_train_3]
    #validation
    X_test_1 = X_test.iloc[:, :num_radiomics]
    X_test_2 = X_test.iloc[:, num_radiomics:2*num_radiomics]
    X_test_3 = X_test.iloc[:, 2*num_radiomics:]

    # Generate Knowledge Embeddings for test data
    knowledge_test_1 = generate_case_specific_knowledge(X_test_1, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_test_2 = generate_case_specific_knowledge(X_test_2, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)
    knowledge_test_3 = generate_case_specific_knowledge(X_test_3, feature_to_embedding=feature_to_embedding, unique_groups=unique_groups, use_abs=True)

    X_test_data = [X_test_1, X_test_2, X_test_3, knowledge_test_1, knowledge_test_2, knowledge_test_3]

    X_train = [
        X_1.values.astype(np.float32),
        X_2.values.astype(np.float32),
        X_3.values.astype(np.float32),
        knowledge_1.astype(np.float32),
        knowledge_2.astype(np.float32),
        knowledge_3.astype(np.float32)
    ]

    X_train_data = [
        X_train_1.values.astype(np.float32),
        X_train_2.values.astype(np.float32),
        X_train_3.values.astype(np.float32),
        knowledge_train_1.astype(np.float32),
        knowledge_train_2.astype(np.float32),
        knowledge_train_3.astype(np.float32)
    ]

    X_test_data = [
        X_test_1.values.astype(np.float32),
        X_test_2.values.astype(np.float32),
        X_test_3.values.astype(np.float32),
        knowledge_test_1.astype(np.float32),
        knowledge_test_2.astype(np.float32),
        knowledge_test_3.astype(np.float32)
    ]

    w = 6
    class_weights = {0: 1, 1: w}

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    time_moe_kan = TimeMoEWithKAN(num_radiomics=num_radiomics, knowledge_dim=len(unique_groups))

    history = time_moe_kan.train(X_train, y_resampled, X_train_data, y_train_data, epochs=80, batch_size=8, class_weight=class_weights)

    model_save_path = "KA-TMoE.h5"
    time_moe_kan.model.save(model_save_path)
    K.clear_session()
