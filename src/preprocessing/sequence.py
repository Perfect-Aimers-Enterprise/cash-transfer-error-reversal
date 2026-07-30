import numpy as np


def create_sequences(
    df,
    feature_columns,
    target_column,
    sequence_length=10,
):
    """
    Create chronological GRU sequences across the
    entire transaction stream.
    """

    sequences = []
    labels = []

    # -----------------------------
    # Sort transactions by time
    # -----------------------------

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # -----------------------------
    # Extract features and labels
    # -----------------------------

    features = df[
        feature_columns
    ].to_numpy(dtype=np.float32)

    targets = df[
        target_column
    ].to_numpy()

    # -----------------------------
    # Sliding Window
    # -----------------------------

    for i in range(
        len(df) - sequence_length
    ):

        sequences.append(
            features[
                i : i + sequence_length
            ]
        )

        labels.append(
            targets[
                i + sequence_length
            ]
        )

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        np.asarray(labels),
    )