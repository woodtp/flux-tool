import numpy as np
import pandas as pd
import pytest

from flux_tool.principal_component_analysis import PCA


@pytest.fixture
def sample_pca():
    covariance_matrix = pd.DataFrame(np.array([[1.0, 0.5], [0.5, 1.0]]))
    threshold = 1.0
    pca = PCA(covariance_matrix, threshold)
    pca.fit()
    return pca


class TestPCA:
    def test_eigenvalues_and_vectors_are_correct_shape(self, sample_pca):
        print(sample_pca.eigenvalues)
        assert len(sample_pca.eigenvalues) == 2
        assert sample_pca.eigenvectors.shape == (2, 2)

    def test_reconstructed_covariance_matrix_is_equivalent_to_the_input(self, sample_pca):
        input_matrix = pd.DataFrame(np.array([[1.0, 0.5], [0.5, 1.0]]))
        assert np.allclose(input_matrix, sample_pca.new_covariance_matrix, atol=1e-10)
