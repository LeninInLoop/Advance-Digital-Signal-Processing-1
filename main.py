from typing import Tuple
import numpy as np

class Helper:
    @staticmethod
    def generate_random_matrix(shape: Tuple[int, int]) -> np.ndarray:
        return np.random.rand(*shape)

    @staticmethod
    def print_matrix(matrix: np.ndarray, precision: int = 4):
        float_format = f"{{: 0.{precision}f}}"

        rows = []
        for row in matrix:
            formatted_row = " ".join([float_format.format(x) for x in row])
            rows.append(f"| {formatted_row} |")

        print("\n".join(rows))

class SVDCalculator:
    @staticmethod
    def calculate_phi_svd(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f_transpose_f = matrix.T @ matrix
        eigen_value, eigen_vectors = np.linalg.eig(f_transpose_f)

        singular_values, phi = np.sqrt(np.maximum(eigen_value, 0)), eigen_vectors  # Avoid negative values
        sorted_indices = np.argsort(singular_values)[::-1]

        singular_values_sorted = singular_values[sorted_indices]
        phi_sorted = phi[:, sorted_indices]

        return np.diag(singular_values_sorted), phi_sorted

    @staticmethod
    def calculate_si_from_phi(matrix: np.ndarray, singular_values: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inverse_singular_values = np.linalg.pinv(singular_values)
        si_sorted = matrix @ phi @ inverse_singular_values
        return si_sorted

    @staticmethod
    def calculate_original_matrix(singular_value: np.ndarray, phi: np.ndarray, si: np.ndarray) -> np.ndarray:
        return si @ singular_value @ phi.T

def main():
    random_matrix = Helper.generate_random_matrix(shape = (16, 16))
    print("Original Random Matrix:")
    Helper.print_matrix(random_matrix)

    singular_value_from_phi, phi = SVDCalculator.calculate_phi_svd(random_matrix)

    si = SVDCalculator.calculate_si_from_phi(
        matrix=random_matrix,
        singular_values=singular_value_from_phi,
        phi=phi
    )

    print("\n\n", 50 * "=" , "\nSingular Values:")
    Helper.print_matrix(singular_value_from_phi)

    print("\n\n", 50 * "=" , "\nPhi Matrix:")
    Helper.print_matrix(phi)

    print("\n\n", 50 * "=" , "\nSi Matrix:")
    Helper.print_matrix(si)

    recalculated_matrix = SVDCalculator.calculate_original_matrix(singular_value_from_phi, phi, si)
    print("\n\n", 50 * "=" , "\nRecalculated Matrix :")
    Helper.print_matrix(recalculated_matrix)

    print(f"\nAre the matrices close? {np.allclose(random_matrix, recalculated_matrix)}")

if __name__ == '__main__':
    main()