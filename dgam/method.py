"""The Double Gaussian Approximation Method (DGAM).

Input for one concept: a list of ``(cosine_similarity, label)`` tuples, one per image.
The labels are only used for evaluation; the method itself never looks at them.

Steps 2-4 of the method live here:
    2. fit a single and a double Gaussian to the histogram of cosine similarities,
       optionally correct the double Gaussian when one component models a tail;
    3. score how well the two components separate (used to rank concepts);
    4. derive a concept-specific threshold from the fitted Gaussians.
"""

import math

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.special import erf


def _gaussian(x, amp, mu, sigma):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def _r_squared(counts, fitted):
    ss_res = np.sum((counts - fitted) ** 2)
    ss_tot = np.sum((counts - np.mean(counts)) ** 2)
    return 1 - (ss_res / ss_tot)


def single_gaussian_approximation(labelled_data):
    """Fit one Gaussian to the histogram of the cosine similarities."""
    data = [x[0] for x in labelled_data]

    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_guess = [np.max(counts), np.mean(data), np.std(data)]
    lower_bounds = [0, min(data), 0]
    upper_bounds = [np.inf, max(data), np.inf]

    try:
        solution_parameters, _ = curve_fit(
            _gaussian,
            bin_centers,
            counts,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )
        r_squared = _r_squared(counts, _gaussian(bin_centers, *solution_parameters))
    except RuntimeError:
        print("Error - single curve_fit failed")
        solution_parameters = initial_guess
        r_squared = float("inf")

    return solution_parameters, r_squared


def _constrained_double_gaussian(x, amp1, mu1, sigma1, amp2, mu2, sigma2):
    """Sum of two Gaussians plus penalty terms that keep the fit plausible."""
    g1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
    g2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))

    penalty = 0.0

    # Constraint 1: the sigmas shouldn't differ too much.
    if sigma1 > 3 * sigma2:
        penalty += 1e6 * (sigma1 - 3 * sigma2)
    if sigma2 > 3 * sigma1:
        penalty += 1e6 * (sigma2 - 3 * sigma1)

    # Constraint 2: if one Gaussian is much larger, it should have the larger sigma.
    if amp1 > 2 * amp2 and sigma1 < sigma2:
        amp1_2 = amp1 / amp2
        if sigma1 * (1 + amp1_2 / 16) < sigma2:
            penalty += 1e3 * (sigma2 - sigma1 * (1 + amp1_2 / 16))
    if amp2 > 2 * amp1 and sigma2 < sigma1:
        amp2_1 = amp2 / amp1
        if sigma2 * (1 + amp2_1 / 16) < sigma1:
            penalty += 1e3 * (sigma1 - sigma2 * (1 + amp2_1 / 16))

    return g1 + g2 + penalty


def double_gaussian_approximation(labelled_data):
    """Fit a sum of two Gaussians to the histogram of the cosine similarities.

    Tries a base guess first, then three extra guesses, then seeded random
    perturbations, and keeps the best successful fit.
    """
    data = [x[0] for x in labelled_data]
    max_retries = 5

    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    amp_guess = np.max(counts)
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)

    base_guess = [
        amp_guess * 0.67,
        mu_guess - sigma_guess / 2,
        sigma_guess,
        amp_guess * 0.33,
        mu_guess + sigma_guess / 2,
        sigma_guess,
    ]
    simple_guess = [amp_guess * 0.5, mu_guess, sigma_guess, amp_guess * 0.5, mu_guess, sigma_guess]
    extreme_guess_1 = [
        amp_guess * 0.25,
        mu_guess - 2.5 * sigma_guess,
        sigma_guess,
        amp_guess * 0.75,
        mu_guess + 2.5 * sigma_guess,
        sigma_guess,
    ]
    extreme_guess_2 = [
        amp_guess * 0.75,
        mu_guess - 2.5 * sigma_guess,
        sigma_guess,
        amp_guess * 0.25,
        mu_guess + 2.5 * sigma_guess,
        sigma_guess,
    ]

    extra_guess_list = [simple_guess, extreme_guess_1, extreme_guess_2]
    extra_guess_r_list = []
    extra_guess_parameters_list = []
    random_guess_r_list = []
    random_guess_parameters_list = []

    lower_bounds = [0, min(data), 0, 0, min(data), 0]
    upper_bounds = [np.inf, max(data), np.inf, np.inf, max(data), np.inf]

    success = False
    for attempt in range(max_retries + len(extra_guess_list)):
        try:
            if attempt == 0:
                guess = base_guess
            elif attempt <= len(extra_guess_list):
                guess = extra_guess_list[attempt - 1]
            else:
                # Small random perturbations with a fixed seed for reproducibility.
                np.random.seed(69)
                guess = [
                    p * np.random.uniform(0.5, 1.5)
                    if i % 3 != 1
                    else p + np.random.normal(0, sigma_guess * 0.5)
                    for i, p in enumerate(simple_guess)
                ]

            solution_parameters, _ = curve_fit(
                _constrained_double_gaussian,
                bin_centers,
                counts,
                p0=guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=2000,
            )
            success = True
            r_squared = _r_squared(
                counts, _constrained_double_gaussian(bin_centers, *solution_parameters)
            )

            if attempt == 0:
                break
            if attempt <= len(extra_guess_list):
                extra_guess_r_list.append(r_squared)
                extra_guess_parameters_list.append(solution_parameters)
            else:
                random_guess_r_list.append(r_squared)
                random_guess_parameters_list.append(solution_parameters)

            if attempt == len(extra_guess_list):
                solution_parameters = extra_guess_parameters_list[
                    extra_guess_r_list.index(max(extra_guess_r_list))
                ]
                break
            if attempt == max_retries + len(extra_guess_list) - 1:
                solution_parameters = random_guess_parameters_list[
                    random_guess_r_list.index(max(random_guess_r_list))
                ]
                break

        except RuntimeError:
            if 0 < attempt <= len(extra_guess_list):
                extra_guess_r_list.append(-1)
                extra_guess_parameters_list.append(None)
            elif attempt > len(extra_guess_list):
                random_guess_r_list.append(-1)
                random_guess_parameters_list.append(None)

            if attempt == len(extra_guess_list) and max(extra_guess_r_list) != -1:
                solution_parameters = extra_guess_parameters_list[
                    extra_guess_r_list.index(max(extra_guess_r_list))
                ]
                break
            if (
                attempt == max_retries + len(extra_guess_list) - 1
                and max(random_guess_r_list) != -1
            ):
                solution_parameters = random_guess_parameters_list[
                    random_guess_r_list.index(max(random_guess_r_list))
                ]
                break

    if not success:
        solution_parameters = base_guess

    r_squared = _r_squared(counts, _constrained_double_gaussian(bin_centers, *solution_parameters))
    return solution_parameters, r_squared


def double_gaussian_approximation_given_one_gaussian(labelled_data, single_gaussian_parameters):
    """Fit a double Gaussian where one of the two components is already fixed."""
    data = [x[0] for x in labelled_data]
    amp2, mu2, sigma2 = single_gaussian_parameters

    def constrained_double_gaussian_with_one_fixed(x, amp1, mu1, sigma1):
        g1 = amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))
        g2 = amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))

        penalty = 0.0
        # Constraint 1: the sigmas shouldn't differ too much.
        if sigma1 > 2 * sigma2:
            penalty += 1e6 * (sigma1 - 3 * sigma2)
        if sigma2 > 2 * sigma1:
            penalty += 1e6 * (sigma2 - 3 * sigma1)

        return g1 + g2 + penalty

    counts, bin_edges = np.histogram(data, bins=50)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    initial_guess = [np.max(counts), np.mean(data), np.std(data)]
    lower_bounds = [0, min(data), 0]
    upper_bounds = [np.inf, max(data), np.inf]

    try:
        solution_parameters, _ = curve_fit(
            constrained_double_gaussian_with_one_fixed,
            bin_centers,
            counts,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
        )
    except RuntimeError:
        solution_parameters = initial_guess

    r_squared = _r_squared(
        counts, constrained_double_gaussian_with_one_fixed(bin_centers, *solution_parameters)
    )
    return solution_parameters, r_squared


def double_gaussian_correction(
    labelled_data, double_gaussian_parameters, r_squared, per_attribute=True
):
    """Refine the fit when one Gaussian models the main body and the other a tail.

    If one component is much larger and the other much wider, the tail on the side
    of the small component is mirrored, re-fitted with a single Gaussian, and the
    other component is re-fitted with that one fixed.
    """
    data = [x[0] for x in labelled_data]
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters
    sigma_scale = 1.5 if per_attribute else 1.25

    if (amp1 > 1.5 * amp2 and sigma2 > sigma_scale * sigma1) or (
        amp2 > 1.5 * amp1 and sigma1 > sigma_scale * sigma2
    ):
        old_amp1 = amp1
        old_amp2 = amp2
        sigma_good = min(sigma1, sigma2)
        mu_good = mu1 if amp1 > amp2 else mu2
        mu_bad = mu2 if amp1 > amp2 else mu1
        if mu_bad > mu_good:
            safe_max = np.mean(sorted(data, reverse=True)[:10]) if len(data) > 10 else max(data)
            right_end_data = [x for x in data if x >= safe_max - 3 * sigma_good]
            min_right_end_data = min(right_end_data)
            mirrored_right_end_data = [
                -(x - min_right_end_data) + min_right_end_data for x in right_end_data
            ]
            double_end_data = mirrored_right_end_data[:-1] + right_end_data
        else:
            safe_min = np.mean(sorted(data)[:10]) if len(data) > 10 else min(data)
            left_end_data = [x for x in data if x <= safe_min + 3 * sigma_good]
            max_left_end_data = max(left_end_data)
            mirrored_left_end_data = [
                -(x - max_left_end_data) + max_left_end_data for x in left_end_data
            ]
            double_end_data = left_end_data + mirrored_left_end_data[1:]

        counts, bin_edges = np.histogram(double_end_data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        amp_guess = np.max(counts)
        mu_guess = np.mean(double_end_data)
        sigma_guess = sigma_good

        try:
            popt, _ = curve_fit(
                _gaussian, bin_centers, counts, p0=[amp_guess, mu_guess, sigma_guess]
            )
            amp_fit, mu_fit, sigma_fit = popt
        except RuntimeError:
            amp_fit, mu_fit, sigma_fit = amp_guess, mu_guess, sigma_guess

        if amp1 == max(amp1, amp2):
            amp2 = amp_fit
            mu2 = mu_fit
            sigma2 = sigma_fit
        else:
            # The original code leaves sigma1 unchanged here (kept as-is).
            amp1 = amp_fit
            mu1 = mu_fit

        if old_amp1 == max(old_amp1, old_amp2):
            [amp1, mu1, sigma1], r_squared = double_gaussian_approximation_given_one_gaussian(
                labelled_data, [amp2, mu2, sigma2]
            )
        else:
            [amp2, mu2, sigma2], r_squared = double_gaussian_approximation_given_one_gaussian(
                labelled_data, [amp1, mu1, sigma1]
            )

    return [amp1, mu1, sigma1, amp2, mu2, sigma2], r_squared


def double_gaussian_intersection(double_gaussian_parameters):
    """Return (x, y) of the intersection of the two Gaussians used for scoring."""
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters

    def gaussian1(x):
        return amp1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1**2))

    def gaussian2(x):
        return amp2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2**2))

    def difference(x):
        return gaussian1(x) - gaussian2(x)

    lower_bound = min(mu1, mu2) - 1 * max(sigma1, sigma2)
    upper_bound = max(mu1, mu2) + 1 * max(sigma1, sigma2)
    x_vals = np.linspace(lower_bound, upper_bound, 2000)
    y_vals = [difference(x) for x in x_vals]

    crossings = []
    for i in range(len(x_vals) - 1):
        if y_vals[i] * y_vals[i + 1] < 0:
            x_cross = x_vals[i] - y_vals[i] * (x_vals[i + 1] - x_vals[i]) / (
                y_vals[i + 1] - y_vals[i]
            )
            crossings.append(x_cross)

    if crossings:
        intersection_x = max(crossings, key=lambda x: gaussian1(x))
        intersection_y = gaussian1(intersection_x)
    else:
        intersection_x = (mu1 + mu2) / 2 - 2 * max(sigma1, sigma2)
        max_gaussian1 = max(gaussian1(x) for x in x_vals)
        max_gaussian2 = max(gaussian2(x) for x in x_vals)
        intersection_y = (max_gaussian1 + max_gaussian2) / 2

    return intersection_x, intersection_y


def get_score(
    single_gaussian_parameters, double_gaussian_parameters, single_r_squared, double_r_squared
):
    """Heuristic score of how well CLIP separates this concept (higher is better)."""
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters
    intersection_x, intersection_y = double_gaussian_intersection(double_gaussian_parameters)

    mu_difference = abs(mu2 - mu1)
    relative_intersection_height = intersection_y / max(amp1, amp2)
    double_over_single_advantage = double_r_squared / single_r_squared - 1
    intersection_to_mean_of_peaks = abs(intersection_x - (mu1 + mu2) / 2)

    return (
        750 * mu_difference
        - 20 * relative_intersection_height
        + 500 * double_over_single_advantage
        - 150 * intersection_to_mean_of_peaks
    )


def gaussian_cdf(x, mu, sigma):
    """Cumulative area from -inf to x of a normalised Gaussian."""
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + erf(z))


def optimal_threshold_for_balanced_accuracy(mu1, sigma1, mu2, sigma2, intersection_x):
    """Threshold that maximises the theoretical balanced accuracy of the two Gaussians."""
    if mu1 < mu2:
        mu_left, sigma_left = mu1, sigma1
        mu_right, sigma_right = mu2, sigma2
    else:
        mu_left, sigma_left = mu2, sigma2
        mu_right, sigma_right = mu1, sigma1

    def objective(threshold):
        perc_left = gaussian_cdf(threshold, mu_left, sigma_left)
        perc_right = 1 - gaussian_cdf(threshold, mu_right, sigma_right)
        return -(perc_left + perc_right)

    result = minimize_scalar(
        objective,
        bounds=(
            intersection_x - 2 * max(sigma1, sigma2),
            intersection_x + 2 * max(sigma1, sigma2),
        ),
        method="bounded",
    )
    return result.x, -result.fun


def get_threshold(double_gaussian_parameters, balanced_accuracy_threshold=True):
    """Concept-specific threshold: optimised for balanced accuracy, or the intersection."""
    amp1, mu1, sigma1, amp2, mu2, sigma2 = double_gaussian_parameters
    intersection_x, _ = double_gaussian_intersection(double_gaussian_parameters)
    if not balanced_accuracy_threshold:
        return intersection_x
    threshold, _ = optimal_threshold_for_balanced_accuracy(mu1, sigma1, mu2, sigma2, intersection_x)
    return threshold


def fit_concept(cosine_similarity_list, use_correction=True, balanced_accuracy_threshold=True):
    """Run steps 2-4 for one concept. Returns (score, threshold)."""
    single_parameters, single_r_squared = single_gaussian_approximation(cosine_similarity_list)
    double_parameters, double_r_squared = double_gaussian_approximation(cosine_similarity_list)
    if use_correction:
        double_parameters, double_r_squared = double_gaussian_correction(
            cosine_similarity_list, double_parameters, double_r_squared
        )
    score = get_score(single_parameters, double_parameters, single_r_squared, double_r_squared)
    threshold = get_threshold(double_parameters, balanced_accuracy_threshold)
    return score, threshold


def get_balanced_accuracy(cosine_similarity_list_dict, threshold_dict, score_dict, cutoff):
    """Pooled balanced accuracy over the top-``cutoff`` concepts of the ranking.

    ``score_dict`` maps concept -> [concept, score]; concepts are ranked by score.
    All image-concept pairs of the selected concepts are pooled into one
    confusion matrix. Returns (balanced_accuracy, tp, tn, fp, fn).
    """
    tp = tn = fp = fn = 0

    sorted_scores = [[attribute, score] for attribute, score in score_dict.values()]
    sorted_scores.sort(key=lambda x: x[1], reverse=True)
    if cutoff <= len(sorted_scores):
        sorted_scores = sorted_scores[:cutoff]

    for attribute, _ in sorted_scores:
        if attribute not in threshold_dict:
            raise ValueError(f"Attribute {attribute} not found in threshold_dict")
        threshold = threshold_dict[attribute]
        for cosine_similarity, label in cosine_similarity_list_dict[attribute]:
            if cosine_similarity > threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
            elif label == 0:
                tn += 1
            else:
                fn += 1

    balanced_accuracy = (
        (tp / (tp + fn) + tn / (tn + fp)) / 2 if (tp + fn) > 0 and (tn + fp) > 0 else 0
    )
    return balanced_accuracy, tp, tn, fp, fn
