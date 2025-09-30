from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def get_preprocessing_pipeline():
    """
    Creates and returns a scikit-learn ColumnTransformer for preprocessing the client data.

    This pipeline is designed to handle the mixed data types (categorical and numerical)
    in the consulting churn dataset. It applies One-Hot Encoding to categorical
    features and leaves numerical features untouched.

    Returns:
        sklearn.compose.ColumnTransformer: A configured preprocessing object ready to be
        used in a model pipeline (e.g., with `fit_transform` or `transform`).
    """
    
    # Define which columns are categorical.
    # The model needs these text-based columns converted into a numerical format.
    categorical_features = [
        'industry',
        'company_size',
        'service_level',
        'payment_history'
    ]
    
    # All other columns in the feature set (X) are numerical and will be passed through without changes.

    # Create a transformer for the categorical features using OneHotEncoder.
    # - OneHotEncoder converts categories like 'Tech' or 'Finance' into binary columns (0s and 1s).
    # - `handle_unknown='ignore'`: This is a critical setting for production. If the model
    #   encounters a new category in the future (e.g., a new industry not seen during training),
    #   it will not throw an error. Instead, it will encode it as all zeros.
    # - `sparse_output=False`: Ensures the output is a standard dense numpy array, which is
    #   easier to work with.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    # Use ColumnTransformer to apply the specified transformations to the correct columns.
    # It takes a list of tuples, where each tuple defines:
    # 1. A name for the step ('cat' for categorical).
    # 2. The transformer object to use (our OneHotEncoder).
    # 3. A list of column names to apply it to.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ],
        # `remainder='passthrough'` is a key instruction. It tells the ColumnTransformer
        # to leave all other columns (i.e., the numerical ones) unchanged. If we didn't
        # include this, the numerical columns would be dropped.
        remainder='passthrough'
    )
    
    return preprocessor
