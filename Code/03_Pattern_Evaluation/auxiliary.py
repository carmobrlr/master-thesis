import pandas as pd
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Add the parent directory to the Python path

from DISAtool import DISA
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def retrive_patterns(file_path, print_information):
    biclusters = []

    with open(file_path) as file:
        j = 0
        for line in file:
            if j < 13:
                j+=1
                continue
            else:
                aux = line.split("\t")
                nr_r = int(aux[2])
                nr_c = int(aux[2+nr_r+1])
                rows = []
                for i in range(3, 3+nr_r):
                    rows.append(int(aux[i])-1)

                cols = []
                for i in range(2+nr_r+2, 2+nr_r+2+nr_c):
                    cols.append(int(aux[i])-1)

                biclusters.append({
                    "lines":rows,
                    "columns":cols
                })

    number_of_cols = []
    number_of_rows = []

    for i in range(len(biclusters)):
        number_of_cols.append(len(biclusters[i]['columns']))
        number_of_rows.append(len(biclusters[i]['lines']))

    if print_information:
        print("Total number of bics")
        print(len(biclusters))
        print("Average number of columns")
        print(np.average(number_of_cols))
        print("Standard deviation of columns")
        print(np.std(number_of_cols))
        print("Average number of rows")
        print(np.average(number_of_rows))
        print("Standard deviation of rows")
        print(np.std(number_of_rows))

    return biclusters

def filter_patterns_support(patterns, number_of_lines, support):
    """This function filters the biclusters found using the support metric
    
    Parameters:
    patterns: list of dictionaries with lines and columns of the biclusters found
    number_of_lines: number of lines in the total dataset
    support: the percentage of the support we want to apply
    
    Returns:
    filtered_patterns: list of dictionaries (each dictionary is a bicluster) that satisfy the given support
    filtered_patterns_idxs: list of the original indexes of the filtered patterns
    """

    filtered_patterns = []
    filtered_patterns_idxs = []
    a = number_of_lines * support/100

    for i in range(len(patterns)):
        if len(patterns[i]['lines']) >= a:
            patterns[i]['original pattern'] = i+1
            filtered_patterns_idxs.append(i+1)
            filtered_patterns.append(patterns[i])

    number_of_cols = []
    number_of_rows = []

    for i in range(len(filtered_patterns)):
        number_of_cols.append(len(filtered_patterns[i]['columns']))
        number_of_rows.append(len(filtered_patterns[i]['lines']))

    print()
    print('After filtering the biclusters with support >= ', support,'%:')
    print()
    print("Total number of bics")
    print(len(filtered_patterns))
    print("Average number of columns")
    print(np.average(number_of_cols))
    print("Standard deviation of columns")
    print(np.std(number_of_cols))
    print("Average number of rows")
    print(np.average(number_of_rows))
    print("Standard deviation of rows")
    print(np.std(number_of_rows))

    return filtered_patterns, filtered_patterns_idxs

def stats(data, patterns, class_information, output_configurations, list_original_patterns):
    discriminative_scores = DISA(data, patterns, class_information, output_configurations).assess_patterns(list_original_patterns)

    # information_gain = []
    # gini_index = []
    # chi_squared = []
    lift = []
    std_lift = []
    stat_sig = []
    for dictionary in discriminative_scores:
        # information_gain.append(dictionary["Information Gain"])
        # gini_index.append(dictionary["Gini index"])
        # chi_squared.append(dictionary["Chi-squared"])
        lift.append(dictionary["Lift"])
        std_lift.append(dictionary["Standardised Lift"])
        # stat_sig.append(dictionary["Statistical Significance"])

    # print("Average Information Gain")
    # print(np.average(information_gain))
    # print("Standard deviation of Information Gain")
    # print(np.std(information_gain))
    # print("Average Gini Index")
    # print(np.average(gini_index))
    # print("Standard deviation of Gini Index")
    # print(np.std(gini_index))
    # print("Average Chi-Squared")
    # print(np.average(chi_squared))
    # print("Standard deviation of Chi-Squared")
    # print(np.std(chi_squared))
    print("Average Lift")
    print(np.average(lift))
    print("Standard deviation of Lift")
    print(np.std(lift))
    print("Average Standardised Lift")
    print(np.average(std_lift))
    print("Standard deviation of Standardised Lift")
    print(np.std(std_lift))

    # print("Average Statistical Significance")
    #print(np.average(stat_sig))
    #print("Standard deviation of Statistical Significance")
    #print(np.std(stat_sig))

