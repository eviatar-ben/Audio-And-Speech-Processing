# In this exercise you will implement the CTC loss in Python. CTC calculates the probability of a specific labeling
# given the model’s output distribution over phonemes
import sys

import numpy


def ctc(matrx_distribution, labeling, possible_output_tokens):
    labeling = ["#" + s for s in labeling]
    labeling.append("#")
    labeling = "".join(labeling)

    mapping = {possible_output_tokens[i]: i+1 for i in range(len(possible_output_tokens))}
    mapping["#"] = 0
    alpha_1_1 = matrx_distribution[0][0]
    alpha_2_1 = matrx_distribution[mapping[labeling[1]]][0]

    prev_column = numpy.zeros(len(labeling))
    prev_column[0] = alpha_1_1
    prev_column[1] = alpha_2_1

    # next_column = numpy.zeros(len(labeling))
    for t in range(1, len(matrx_distribution[0])):
        next_column = numpy.zeros(len(labeling))
        for i, s in enumerate(labeling):
            y_t_zs = matrx_distribution[mapping[s]][t]
            if i == 0:
                alpha_s_t = (prev_column[i]) * y_t_zs
            elif i == 1:
                alpha_s_t = (prev_column[i - 1] + prev_column[i]) * y_t_zs
            elif (i > 2 and labeling[i] == labeling[i - 2]) or s == "#":
                alpha_s_t = (prev_column[i - 1] + prev_column[i]) * y_t_zs
            else:
                alpha_s_t = (prev_column[i - 2] + prev_column[i - 1] + prev_column[i]) * y_t_zs
            next_column[i] = alpha_s_t
        prev_column = next_column

    return prev_column[-1] + prev_column[-2]


def print_p(p: float):
    print("%.3f" % p)


def main(path="example_matrix.npy", labeling="aaabb", possible_output_tokens="abc"):
    matrx_distribution = numpy.load(path).T
    p = ctc(matrx_distribution, labeling, possible_output_tokens)
    print_p(p)


if __name__ == '__main__':
    args = sys.argv
    main(args[1], args[2], args[3])
    # main()
