from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ordinal_feature:
    name: str
    mapping: dict[str:int]


class Preprocess:
    _non_numerical_ordinal_features = [
        ordinal_feature("internet_access", {"no": 0, "yes": 1}),
        ordinal_feature("sleep_quality", {"poor": 0, "average": 1, "good": 2}),
        ordinal_feature("facility_rating", {"low": 0, "medium": 1, "high": 2}),
        ordinal_feature("exam_difficulty", {"easy": 0, "moderate": 1, "hard": 2}),
    ]

    _nominal_features = ["gender", "course", "study_method"]

    def __init__(self, data: pd.DataFrame, target_col: str = "exam_score"):
        self.data = data
        self.target_col = target_col
        self.scaler = StandardScaler()

    def transform_ordinal_features(self):
        for ordinal_feature in self._non_numerical_ordinal_features:
            self.data[ordinal_feature.name] = (
                self.data[ordinal_feature.name]
                .map(ordinal_feature.mapping)
                .astype("Int64")  # Use nullable integer type to handle NaN values
            )

    def transform_nominal_features(self):
        encoded_features = pd.get_dummies(
            self.data[self._nominal_features],
            prefix=self._nominal_features,
            drop_first=False,
        )

        self.data = self.data.drop(columns=self._nominal_features)
        self.data = pd.concat([self.data, encoded_features], axis=1)

    def preprocess(self):
        """Apply all preprocessed transformations."""
        self.transform_ordinal_features()
        self.transform_nominal_features()
        return self.data

    def get_train_test_split(self, test_size=0.2, random_state=42, scale=True):
        """
        Split the data into train and test sets and optionally apply standard scaling.

        Returns:
            X_train, X_test, y_train, y_test
        """
        # Apply preprocessed
        self.preprocess()

        # Separate features and target
        X = self.data.drop(
            columns=(
                [self.target_col, "id"]
                if "id" in self.data.columns
                else [self.target_col]
            )
        )
        y = self.data[self.target_col]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Apply standard scaling if requested
        if scale:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )

        return X_train, X_test, y_train, y_test

    def export_processed_df(
        self, path: str = "../../data/processed/student_processed.parquet"
    ):
        """Export the fully processed dataframe to a parquet file."""
        self.preprocess()
        self.data.to_parquet(path, index=False)


if __name__ == "__main__":
    train_df = pd.read_csv("data/raw/playground-series-s6e1/train.csv")

    # Export full processed dataset
    prep = Preprocess(train_df.copy())
    prep.export_processed_df("data/processed/student_processed.parquet")

    # Create train/test split
    prep2 = Preprocess(train_df.copy())
    X_train, X_test, y_train, y_test = prep2.get_train_test_split()

    X_train.to_parquet("data/processed/X_train.parquet", index=False)
    X_test.to_parquet("data/processed/X_test.parquet", index=False)
    y_train.to_frame().to_parquet("data/processed/y_train.parquet", index=False)
    y_test.to_frame().to_parquet("data/processed/y_test.parquet", index=False)

    print(f"Successfully exported all datasets to data/processed/")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
