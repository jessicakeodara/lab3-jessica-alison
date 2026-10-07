##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
from typing import override
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab


#####################################################################
# BinaryLabels
#####################################################################

class BinaryLabels(PCLLabels):
    """docstring"""

    @override
    def _extract_label(self, example):
        """Semi-private function
           Extracts condescension atrributes
        Inputs: self
                example: XML file to extract
        Output: value of condescension for example"""
        return example.attrib["condescension"]


#####################################################################
# CategoryLabels
#####################################################################

class CategoryLabels(PCLLabels):
    """docstring"""

    @override
    def _extract_label(self, example):
        """Semi-private function
           Extracts condescension atrributes
        Inputs: self
                example: XML file to extract
                label: pull from this example category
        Output: value of label for example"""
        return example.attrib["category"]

#####################################################################
# MyFeatures
#####################################################################

class MyFeatures(PCLFeatures):
    """docstring"""

    def __init__(self, vocab):
        if vocab is None:
            super().__init__(PCLVocab("/courses/cs159/data/patronize/vocab.txt"))
        else:
            super().__init__(vocab)

    def vocab(self):
        print("MyFeatures' initial_vocab's Dict: " + str(self.initial_vocab._dict))
        print("MyFeatures' initial_vocab's Words: " + str(self.initial_vocab._words))

    @override
    def _extract_features(self, example):
        """Extracts features from example
        Inputs: self
                example: XML file to extract
        Output: list of words in the input vocabulary"""
        example_text =  self.extract_text(example)

        feature_list = []
        for word in example_text:
            print("Word in Example_text: " + str(word))
            if word in self.initial_vocab._words:
                print("Word in Initial_Vocab: " + str(word))
                feature_list.append(word)

        print("Feature List: " + str(feature_list))
        return feature_list

    @override
    def _get_feature_name(self, i):
        """ Returns a human-readable name for the ith feature in the DictVectorizer's internal vocabulary """
        return self.vectorizer.vocabulary_.get(i)

    @override        
    def _get_num_features(self):
        """ Return the total number of features """
        return len(self.vectorizer.vocabulary_)


def do_experiment(args): 
    raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=argparse.FileType('rb'), help="Data file containing labeled training instances")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="File containing vocabulary words")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=0 means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()
