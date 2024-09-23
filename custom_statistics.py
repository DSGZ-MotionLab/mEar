import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import pingouin as pg
import matplotlib.pyplot as plt
import configuration
import utilities


def bland_altman_statistics(method_a, method_b, plot=False):
    """
    Calculate Bland-Altman statistics and optionally plot the Bland-Altman plot.

    Parameters:
    - method_a: Array-like, measurements from method A.
    - method_b: Array-like, measurements from method B.
    - plot: Boolean, if True, plots the Bland-Altman plot.

    Returns:
    - bias: Mean difference between the methods.
    - rpc: Reproducibility coefficient (1.96 * standard deviation of differences).
    - cv: Coefficient of variation.
    """

    # Calculating differences and means
    differences = method_b - method_a
    mean_difference = np.mean(differences)
    std_dev_difference = np.std(differences, ddof=1)
    means = (method_a + method_b) / 2

    # Bias (Mean Difference)
    bias = mean_difference

    # Reproducibility Coefficient (RPC)
    rpc = 1.96 * std_dev_difference

    # Coefficient of Variation (CV)
    cv = (std_dev_difference / abs(bias)) * 100

    # Bland-Altman Plot
    if plot:
        plt.scatter(means, differences)
        plt.axhline(mean_difference, color="gray", linestyle="--")
        plt.axhline(mean_difference + rpc, color="gray", linestyle="--")
        plt.axhline(mean_difference - rpc, color="gray", linestyle="--")
        plt.title("Bland-Altman Plot")
        plt.xlabel("Mean of Two Methods")
        plt.ylabel("Difference Between Methods")
        plt.show()

        # Printing the calculated values
        print(f"Bias (Mean Difference): {bias}")
        print(f"Reproducibility Coefficient (RPC): {rpc}")
        print(f"Coefficient of Variation (CV): {cv:.2f}%")

    return bias, rpc, cv


def icc_statistics(method_a, method_b):
    """
    Calculate Intraclass Correlation Coefficient (ICC) between two methods.

    Parameters:
    - method_a: Array-like, measurements from method A.
    - method_b: Array-like, measurements from method B.

    Returns:
    - icc_results: DataFrame with ICC results.
    """

    subjects = list(range(1, len(method_a) + 1))

    # Prepare the DataFrame
    data = pd.DataFrame(
        {
            "Subject_ID": subjects + subjects,  # Repeat subject IDs for each method
            "Measurement": list(method_a) + list(method_b),  # Combine measurements
            "Method": ["A"] * len(method_a) + ["B"] * len(method_b),  # Label methods
        }
    )

    # Convert "Method" to categorical
    data["Method"] = pd.Categorical(data["Method"])

    # Calculate ICCs
    icc_results = pg.intraclass_corr(
        data=data, targets="Subject_ID", raters="Method", ratings="Measurement"
    )

    return icc_results


def uniform_statistics(method_gt, method_pd):
    """
    Calculate Pearson correlation coefficient, Bland-Altman statistics, and ICC between two methods.

    Parameters:
    - method_gt: Array-like, measurements from method A (ground truth).
    - method_pd: Array-like, measurements from method B (new measurement system).

    Returns:
    - correlation_coefficient: Pearson correlation coefficient between the two methods.
    - p_value: P-value for the Pearson correlation.
    - bias: Mean difference between the methods.
    - rpc: Reproducibility coefficient.
    - cv: Coefficient of variation.
    - icc_results: DataFrame with ICC results.
    """

    method_gt = np.array(method_gt)
    method_pd = np.array(method_pd)

    # Filter arrays for NaNs
    method_gt, method_pd = utilities.remove_nan_positions(method_gt, method_pd)
    
    # absolute error
    absolute_error = np.mean(np.abs(method_gt - method_pd))
    # relative error
    relative_error = np.mean(np.abs((method_gt - method_pd) / method_gt)) * 100

    # Calculate RMSE
    rmse = np.mean(np.sqrt(np.mean((method_gt - method_pd) ** 2)))
    # relative RMSE
    relative_rmse = (rmse / np.mean(method_gt))

    mean_a = np.nanmean(method_gt)
    mean_b = np.nanmean(method_pd)
    std_a = np.nanstd(method_gt)
    std_b = np.nanstd(method_pd)

    # Calculate Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = pearsonr(method_gt, method_pd)

    # Calculate Bland-Altman statistics
    bias, rpc, cv = bland_altman_statistics(method_gt, method_pd)

    # Calculate ICC
    icc_results = icc_statistics(method_gt, method_pd)

    # Create a formatted table
    table = (
        f"Mean GT: {mean_a:.4f}\n"
        f"Mean PD: {mean_b:.4f}\n"
        f"Std GT: {std_a:.4f}\n"
        f"Std PD: {std_b:.4f}\n"
        f"Absolute Error: {absolute_error:.4f}\n"
        f"Relative Error: {relative_error:.4f}\n"
        f"RMSE: {rmse:.4f}\n"
        f"Relative RMSE: {relative_rmse:.4f}\n"
        f"Correlation Coefficient: {correlation_coefficient:.4f}\n"
        f"P-value: {p_value:.4f}\n"
        f"Bias (Mean Difference): {bias:.4f}\n"
        f"Reproducibility Coefficient (RPC): {rpc:.4f}\n"
        f"Coefficient of Variation (CV): {cv:.4f}\n"
        f"\nICC Results:\n{icc_results}"
    )

    return table


def calculate_metrics_with_tolerance(
    predictions, ground_truth, tolerance=configuration.positive_tolerance
):

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Track where are ones in respect to tolerance
    tolerance_ones = set()

    # First check for True positive
    for i, gt in enumerate(ground_truth):

        if gt == 1:
            found_match = False

            # Add the window to the tolerance ones for masking later also
            for j in range(i - tolerance, i + tolerance -1 ):
                tolerance_ones.add(j)
                
            tolerance0 = 0
            if i-tolerance < 0:
                tolerance0 = 0
            else :
                tolerance0 = i-tolerance
            
            if i + tolerance - 1 > len(predictions):
                tolerance1 = len(predictions-1)
            else:
                tolerance1 = i + tolerance

            for j in range(tolerance0, tolerance1):

                if predictions[j] == 1:
                    TP += 1
                    found_match = True
                    break

            if not found_match:
                FN += 1

    # Second check for no event in respect to the tolerance window
    # this loop is blind inside the tolerance window
    for i, gt in enumerate(ground_truth):

        if predictions[i] == 0:
            TN += 1

        elif i not in tolerance_ones:  # Ensure this FP hasn't been counted as TP
            # Also check if there is no positive Event nearby the tolerance

            FP += 1

    # Calculating metrics
    accuracy = (TP + TN) / len(predictions)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return accuracy, precision, recall, f1_score, TP, TN, FP, FN
