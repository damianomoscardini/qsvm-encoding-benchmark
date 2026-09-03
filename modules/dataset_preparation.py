from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ------------------------------------------------------------------------------------------------

# PCA APPLICATION FUNCTION.

def apply_PCA(X_dataset_1, X_dataset_2, number_features):

    scaler = StandardScaler()
    X_dataset_1_scaled = scaler.fit_transform(X_dataset_1)
    X_dataset_2_scaled = scaler.transform(X_dataset_2)

    pca = PCA(n_components=number_features)
    X_dataset_1_PCA = pca.fit_transform(X_dataset_1_scaled)
    X_dataset_2_PCA = pca.transform(X_dataset_2_scaled)

    return X_dataset_1_PCA, X_dataset_2_PCA

# ------------------------------------------------------------------------------------------------

# DATASET PREPARATION FUNCTION.

def prepare_dataset(dataset, number_samples, number_features, fraction_train):

    # LOADING THE DATA.

    X_dataset, y_dataset = dataset

    # UNDERSAMPLING.

    # If number_samples is greater than the number of the samples of the dataset, the code must give an error.

    if number_samples > X_dataset.shape[0]:
        raise ValueError(f"The number of samples requested, {number_samples}, is greater than the number of samples of the dataset, {X_dataset.shape[0]}.")

    # If number_samples is equal to the number of the samples of the dataset, we do nothing. Otherwise, we must undersample the original dataset.

    if number_samples < X_dataset.shape[0]:
        X_dataset, _, y_dataset, _ = train_test_split(X_dataset, y_dataset,
                                                        train_size=number_samples,
                                                        stratify=y_dataset,
                                                        random_state=42)

    # SPLITTING OF THE DATASET.

    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset,
                                                          train_size=fraction_train,
                                                          stratify=y_dataset,
                                                          random_state=42)

    # CHECK FEATURES.

    # We cap number_features at the dataset's actual dimensionality.
    # number_features is treated as an upper bound, not a fixed target: if the dataset
    # has fewer raw features than requested, we simply use all of them.

    effective_number_features = min(number_features, X_dataset.shape[1])

    if effective_number_features < number_features:
        print(f"\nNOTE: requested {number_features} features, but \"{X_dataset.shape[1]}\" is the maximum available. Using {effective_number_features}.\n")

    # RESULTS.

    train_dataset = (X_train, y_train)
    test_dataset = (X_test, y_test)

    return train_dataset, test_dataset, effective_number_features