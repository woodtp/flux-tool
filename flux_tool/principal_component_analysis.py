import numpy as np
import pandas as pd


class PCA:
    """Dataclass to carry out a principal component analysis.

    Attributes:
        threshold : float
            Threshold value for the cumulative sum of eigenvalues, above which components are discarded.
            Defaults to 2.0, which keeps all components.
        eigenvectors : np.ndarray
            2-D array containing all eigenvectors as rows, sorted by eigenvalue.
        eigenvalues : np.ndarray
            Array containing the list of eigenvalues in descending order.
        fractional_eigenvalues : np.ndarray
            Array containing the fraction of the variance of the dataset explained by the eigenvalues.
        cumulative_sum : np.ndarray
            Array containing the cumulative sums of fractional_eigenvalues.
        eigenvalues_df : pd.DataFrame
            Pandas DataFrame representation of eigenvalues.
            Also contains columns for fractional_eigenvalues and cumulative_sum.
        principal_component_df : pd.DataFrame
            Pandas DataFrame representation of eigenvectors weighted by eigenvalue.
        new_covariance_matrix: np.ndarray
            Array representing the new covariance matrix reconstructed from the principal components.
    """

    __slots__ = (
        "covariance_matrix",
        "threshold",
        "eigenvalues",
        "eigenvectors",
        "eigenvalues_df",
        "principal_component_df",
    )

    def __init__(self, covariance_matrix: pd.DataFrame, threshold: float = 2) -> None:
        self.covariance_matrix = covariance_matrix
        self.threshold = threshold

    def fit(self) -> None:
        """Perform eigendecomposition of covariance_matrix and sort resulting eigenvectors by largest eigenvalues.

        Parameters:
                covariance_matrix: np.ndarray | pd.DataFrame
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix)

        # reversing the order so that largest eigenvalue comes first
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]

        fractional_eigenvalues = eigenvalues / eigenvalues.sum()

        cumulative_sum = np.cumsum(fractional_eigenvalues)

        selected_components = (cumulative_sum < self.threshold) & (eigenvalues > 0)

        self.eigenvectors = eigenvectors[:, selected_components]
        self.eigenvalues = eigenvalues[selected_components]

        index = self.covariance_matrix.index

        self.principal_component_df = pd.concat(
            [
                pd.DataFrame(
                    self.eigenvectors,
                    index=index,
                ),
                pd.DataFrame(
                    np.sqrt(self.eigenvalues) * self.eigenvectors,
                    index=index,
                ),
            ],
            keys=["evec", "evec_scaled"],
            names=["scale"] + index.names,
        )

        self.eigenvalues_df = pd.DataFrame(
            self.eigenvalues,
            columns=["eigenvalue"],
        )

        self.eigenvalues_df.index.name = "principal_component"

        self.eigenvalues_df["fractional_eigenvalue"] = fractional_eigenvalues[
            selected_components
        ]

        self.eigenvalues_df["cumulative_sum"] = cumulative_sum[selected_components]

    @property
    def new_covariance_matrix(self) -> np.ndarray:
        return self.eigenvectors @ np.diag(self.eigenvalues) @ self.eigenvectors.T
