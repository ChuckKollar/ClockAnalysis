import numpy as np
from scipy import stats


def remove_outliers_zscore(data, column_index, threshold=2.25):
    """
    Removes outliers from an array of tuples based on the Z-score of a specific column.

    Args:
        data (list): A list of tuples.
        column_index (int): The index of the value in the tuple to check for outliers.
        threshold (float): The Z-score threshold for identifying an outlier.

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


import math


def remove_outliers_iqr(data, column_index):
    """
    Removes tuples where a specific member (by column_index) is an outlier
    using the IQR method.
    """
    # Extract the values from the specified column
    values = sorted([item[column_index] for item in data])

    # Calculate the first (Q1) and third (Q3) quartiles
    q1_idx = math.floor(len(values) * 0.25)
    q3_idx = math.ceil(len(values) * 0.75)
    q1 = values[q1_idx]
    q3 = values[q3_idx]

    # Calculate the Interquartile Range (IQR)
    iqr = q3 - q1

    # Define bounds for outliers (commonly 1.5 * IQR)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter the original data list, keeping only non-outliers
    cleaned_data = [item for item in data if lower_bound <= item[column_index] <= upper_bound]

    outlier_data = list(set(data) - set(cleaned_data))

    return cleaned_data, outlier_data

if __name__ == '__main__':
    # Example Usage:
    #data = [('A', 10), ('B', 20), ('C', 25), ('D', 100), ('E', 30), ('F', 22), ('G', 15), ('H', 200)]
    data = [('A', 10), ('B', 12), ('C', 15), ('D', 100), ('E', 13), ('F', 8), ('G', 11), ('H', 75)]
    column_to_check = 1  # Check the second element (numerical value)

    filtered_list, outlier_data = remove_outliers_zscore(data, column_to_check, threshold=1.25)
    print("Original data:", data)
    print("Filtered data (Z-score < 2):", filtered_list)
    print("Outlier data:", outlier_data)

    cleaned_list_iqr, outlier_list_iqr = remove_outliers_iqr(data, column_index=column_to_check)
    print("Cleaned list (IQR):", cleaned_list_iqr)
    print("Outlier list:", outlier_list_iqr)
