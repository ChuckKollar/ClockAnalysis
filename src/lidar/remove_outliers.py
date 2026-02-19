import numpy as np
from scipy import stats


def remove_outliers_zscore(data, column_index, threshold=3):
    """
    Removes outliers from an array of tuples based on the Z-score of a specific column.

    Args:
        data (list): A list of tuples.
        column_index (int): The index of the value in the tuple to check for outliers.
        threshold (float): The Z-score threshold for identifying an outlier (default is 3).

    Returns:
        list: A new list of tuples with outliers removed.
    """
    # 1. Extract the values from the specified column
    values = np.array([item[column_index] for item in data])

    # 2. Calculate the absolute Z-scores for each value
    z_scores = np.abs(stats.zscore(values))

    # 3. Create a boolean mask for non-outliers
    # Points with an absolute Z-score <= threshold are kept
    not_outlier_mask = z_scores <= threshold

    # 4. Filter the original data using the mask
    # Convert list of tuples to numpy array for efficient boolean indexing
    data_np = np.array(data, dtype=object)  # Use dtype=object if tuples contain mixed types
    filtered_data_np = data_np[not_outlier_mask]

    # 5. Convert the result back to a list of tuples
    filtered_data = [tuple(row) for row in filtered_data_np.tolist()]
    outlier_data = list(set(data) - set(filtered_data))

    return filtered_data, outlier_data


if __name__ == '__main__':
    # Example Usage:
    data = [('A', 10), ('B', 12), ('C', 15), ('D', 100), ('E', 13), ('F', 8), ('G', 11)]
    column_to_check = 1  # Check the second element (numerical value)

    filtered_list, outlier_data = remove_outliers_zscore(data, column_to_check, threshold=2)
    print("Original data:", data)
    print("Filtered data (Z-score < 2):", filtered_list)
    print("Outlier data:", outlier_data)
