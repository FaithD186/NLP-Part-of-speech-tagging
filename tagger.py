import sys
import argparse
import numpy


def tagging_words(word_list, tag_list, initial_prob, transition_prob, observ_prob, test_words):
    """tagging words in the test file based on the training files"""

    prob_matrix = viterbi(word_list, tag_list, initial_prob, transition_prob, observ_prob, test_words)
    prediction_tags = prediction(prob_matrix[0], prob_matrix[1], tag_list, test_words)

    final_pred = []

    for word, tag in zip(test_words[:], prediction_tags[:]):
        final_pred.append((word, tag))

    return final_pred


def prediction(prob, hidden_states, tag_list, test_word_list):
    """return list of predicted tags based on viterbi algorithm"""
    tag_dict = {i: tag for i, tag in enumerate(tag_list)}
    test_wordlist_len = len(test_word_list)
    tag_predictions = numpy.empty(len(test_word_list), dtype=object)
    temp_list = numpy.zeros(test_wordlist_len, dtype=int)
    tag_num = prob.shape[0]
    # tag_dict = {}
    # h = 0
    # for tag in tag_list:
    #     tag_dict[h] = tag
    #     h += 1
    prev_max = float("-inf") # initialize as a very low number
    for j in range(tag_num):
        if prob[j, test_wordlist_len - 1] > prev_max:
            prev_max = prob[j, test_wordlist_len - 1] # update prev_max
            temp_list[test_wordlist_len - 1] = j
    tag_predictions[test_wordlist_len - 1] = tag_dict[temp_list[test_wordlist_len - 1]]

    for i in range(test_wordlist_len - 1, 0, -1):
        temp_list[i - 1] = hidden_states[int(temp_list[i]), i]
        tag_predictions[i - 1] = tag_dict[temp_list[i - 1]]
    return tag_predictions


def viterbi(word_list, tag_list, initial_prob, transition_prob, observ_prob, test_words):
    """viterbi algorithm"""
    tag_list_length = len(tag_list)
    prob = numpy.zeros((tag_list_length, len(test_words)))
    prev = numpy.zeros((tag_list_length, len(test_words)))
    word_dict = {word: i for i, word in enumerate(word_list)}

    # initializing the first column
    for i in range(tag_list_length):
        if test_words[0] not in word_dict:
            observation_probability = 1/len(word_list)
        else:
            observation_probability = observ_prob[i, word_dict[test_words[0]]]
        # if it is unknown word, observ_prob will throw a key error
        # if tag_list[i] in observ_prob and test_words[0] in observ_prob[tag_list[i]]:
        #     observation_probability = observ_prob[tag_list[i]][test_words[0]]
        # else:
        #     observation_probability = 1/len(word_list)
        prob[i, 0] = initial_prob[tag_list[i]] * observation_probability
        prev[i, 0] = 0

        for t in range(1, len(test_words)):
            if test_words[t] in word_dict:
                observation_prob = observ_prob[:, word_dict[test_words[t]]]
                probability = prob[:, t-1, None] * transition_prob * observation_prob[None, :]
            else:
                probability = prob[:, t-1, None] * transition_prob * 1/len(word_list)
            best_state = numpy.argmax(probability, axis=0)
            prob[:, t] = numpy.max(probability, axis=0)
            prev[:, t] = best_state

    return prob, prev

    # for t in range(1, len(test_words)):
    #     for i in range(tag_list_length):
    #         best_prob = float("-inf")
    #         best_state = None
    #         for j in range(tag_list_length):
    #             if test_words[t] not in word_dict:
    #                 observation_probability = 1/len(word_list)
    #             else:
    #                 observation_probability = observ_prob[i, word_dict[test_words[t]]]
    #             # if tag_list[i] in observ_prob and test_words[t] in observ_prob[tag_list[i]]:
    #             #     observation_probability = observ_prob[tag_list[i]][test_words[t]]
    #             # else:
    #             #     observation_probability = 1/len(word_list)
    #             probability = prob[j, t-1] * transition_prob[j, i] * observation_probability
    #             if probability > best_prob:
    #                 best_prob = probability
    #                 best_state = j
    #
    #         prob[i, t] = best_prob
    #         prev[i, t] = best_state

    # return prob, prev


def initial_probabilities(tag_list, first_freq, num_sentences):
    """return dictionary of initial probabilities
     which is how likely each POS tag appears at the beginning of, a sentence
    """
    initial_prob = {}
    for tag in tag_list:
        if tag in first_freq:
            num_first = first_freq[tag]
            initial_prob[tag] = num_first / num_sentences
        else:
            initial_prob[tag] = 0
    return initial_prob


def transition_probabilities(tag_transitions, tag_list):
    """return dictionary of transition probabilities
    probability of a certain tag followed by another given tag """

    transitions_dict = numpy.zeros((len(tag_list), len(tag_list)))
    for i in range(len(tag_list)):
        for j in range(len(tag_list)):
            if tag_list[i] in tag_transitions and tag_list[j] in tag_transitions[tag_list[i]]:
                num_starting_prev = sum(tag_transitions[tag_list[i]].values())
                probability = tag_transitions[tag_list[i]][tag_list[j]] / num_starting_prev
            else:
                probability = 0
            transitions_dict[i, j] = probability
    return transitions_dict


def observation_probabilities(tag_word, word_list, tag_list):
    """return dictionary of observation probabilities
    probability of a tag being associated with a word"""
    observation_prob = numpy.zeros((len(tag_list), len(word_list)))
    for i in range(len(tag_list)):
        for j in range(len(word_list)):
            if tag_list[i] in tag_word and word_list[j] in tag_word[tag_list[i]]:
                num_tags = sum(tag_word[tag_list[i]].values())
                probability = tag_word[tag_list[i]][word_list[j]] / num_tags
            else:
                probability = 0
            observation_prob[i, j] = probability
    return observation_prob


def read_test_file(filename):
    """return list of sentences from test file"""
    test_word_list = []
    test_file = open(filename, "r")
    sentence_list = []
    sentence = []
    for line in test_file:
        word = line.strip()
        if not word == ".":
            sentence.append(word)
        else:
            sentence.append(word)
            sentence_list.append(sentence)
            sentence = []
        test_word_list.append(word)
    return sentence_list


def read_from_file(filename):
    """
    Load initial probabilities, transition probabilities, observation
    probabilities from training file

    :param filename: The name of the given file.
    :type filename: str

    """

    first_freq = {}  # counts how many times each tag is tag for the first word

    # tag_list = [] # unordered, tag list with no duplicates
    word_list = []
    tag_transitions = {} # transitions from one tag to another
    tag_word = {} # frequency of a tag with a particular word
    is_first_word = True
    num_sentences = 0 # total number of sentences
    prev_tag = None

    full_tag_list = ["AJ0", "AJC", "AJS", "AT0", "AV0", "AVP", "AVQ", "CJC", "CJS", "CJT", "CRD",
                     "DPS", "DT0", "DTQ", "EX0", "ITJ", "NN0", "NN1", "NN2", "NP0", "ORD", "PNI",
                     "PNP", "PNQ", "PNX", "POS", "PRF", "PRP", "PUL", "PUN", "PUQ", "PUR", "TO0",
                     "UNC", 'VBB', 'VBD', 'VBG', 'VBI', 'VBN', 'VBZ', 'VDB', 'VDD', 'VDG', 'VDI',
                     'VDN', 'VDZ', 'VHB', 'VHD', 'VHG', 'VHI', 'VHN', 'VHZ', 'VM0', 'VVB', 'VVD',
                     'VVG', 'VVI', 'VVN', 'VVZ', 'XX0', 'ZZ0', 'AJ0-AV0', 'AJ0-VVN', 'AJ0-VVD',
                     'AJ0-NN1', 'AJ0-VVG', 'AVP-PRP', 'AVQ-CJS', 'CJS-PRP', 'CJT-DT0', 'CRD-PNI', 'NN1-NP0', 'NN1-VVB',
                     'NN1-VVG', 'NN2-VVZ', 'VVD-VVN', 'AV0-AJ0', 'VVN-AJ0', 'VVD-AJ0', 'NN1-AJ0', 'VVG-AJ0', 'PRP-AVP',
                     'CJS-AVQ', 'PRP-CJS', 'DT0-CJT', 'PNI-CRD', 'NP0-NN1', 'VVB-NN1', 'VVG-NN1', 'VVZ-NN2', 'VVN-VVD']

    for file in filename:
        training_file = open(file, "r")
        for line in training_file:
            if line[0] == ":":
                text = line.split(":")
                line_text = ':'.join(text[:2]), ':'.join(text[2:])
            else:
                line_text = line.split(":")
            word = line_text[0].strip()
            tag = line_text[1].strip()
            # if tag not in tag_list:
            #     tag_list.append(tag)
            if word not in word_list:
                word_list.append(word)
            if prev_tag:
                # if (prev_tag, tag) not in tag_transitions:
                #     tag_transitions[(prev_tag, tag)] = 1
                # else:
                #     tag_transitions[(prev_tag, tag)] += 1
                if prev_tag not in tag_transitions:
                    tag_transitions[prev_tag] = {tag: 1}
                else:
                    if tag in tag_transitions[prev_tag]:
                        tag_transitions[prev_tag][tag] += 1
                    else:
                        tag_transitions[prev_tag][tag] = 1
            if tag not in tag_word:
                tag_word[tag] = {word: 1}
            else:
                if word in tag_word[tag]:
                    tag_word[tag][word] += 1
                else:
                    tag_word[tag][word] = 1

            if is_first_word: # check if word is the first word
                if tag not in first_freq:
                    first_freq[tag] = 1
                else:
                    first_freq[tag] += 1

            # checking if next word will be the first word
            if word == ".": # any word following a period is the first word
                num_sentences += 1
                is_first_word = True
            else:
                is_first_word = False

            prev_tag = tag

        training_file.close()


    # calculate initial probabilities
    initial_prob_dict = initial_probabilities(full_tag_list, first_freq, num_sentences)

    # calculate transition probabilities
    transitions_prob_dict = transition_probabilities(tag_transitions, full_tag_list)

    # calculate observation probabilities
    observation_prob_dict = observation_probabilities(tag_word, word_list, full_tag_list)

    return word_list, full_tag_list, initial_prob_dict, transitions_prob_dict, observation_prob_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trainingfiles",
        action="append",
        nargs="+",
        required=True,
        help="The training files."
    )
    parser.add_argument(
        "--testfile",
        type=str,
        required=True,
        help="One test file."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file."
    )
    args = parser.parse_args()

    training_list = args.trainingfiles[0]
    print("training files are {}".format(training_list))

    print("test file is {}".format(args.testfile))

    print("output file is {}".format(args.outputfile))

    print("Starting the tagging process.")

    file_info = read_from_file(training_list)
    word_list = file_info[0]
    tag_list = file_info[1]
    initial_prob = file_info[2]
    trans_prob = file_info[3]
    obsv_prob = file_info[4]

    test_words = read_test_file(args.testfile)
    final_predictions = []
    for sentence in test_words:
        pred = []
        pred.extend(tagging_words(word_list, tag_list, initial_prob, trans_prob, obsv_prob, sentence))
        final_predictions.extend(pred)

    original_stdout = sys.stdout
    with open(args.outputfile, "w") as output_tagger:
        sys.stdout = output_tagger
        for pred_tuple in final_predictions:
            print(pred_tuple[0] + " : " + pred_tuple[1])
        sys.stdout = original_stdout


# python3 tagger.py --trainingfiles training1.txt --testfile test1.txt --outputfile output.txt
