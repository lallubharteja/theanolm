#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A module that implements the Vocabulary class.
"""

import numpy
import h5py

from theanolm.vocabulary.wordclass import WordClass
from theanolm.parsing import utterance_from_line
from theanolm.exceptions import IncompatibleStateError, InputError

class Vocabulary(object):
    """Word or Class Vocabulary

    Vocabulary class provides a mapping between the words and word IDs, as well
    as mapping from word IDs to class IDs. All shortlist words are mapped to a
    class. When classes are not used, the classes contain one word each. A
    vocabulary may also contain IDs for out-of-shortlist words. They are not
    mapped to a class and they are not predicted by the neural network.
    """

    def __init__(self, id_to_word, word_id_to_class_id, word_classes,
                 oos_words=None):
        """If the special tokens <s>, </s>, and <unk> don't exist in the word
        list, adds them and creates a separate class for each token. Then
        constructs a vocabulary based on given word-to-class mapping.

        The lists ``id_to_word`` and ``word_id_to_class_id`` have to be
        equal-sized. They are defined for every shortlist word. The list
        ``oos_words`` may contain out-of-shortlist words that will be added to
        the vocabulary if they don't exist in ``id_to_word`` already.

        :type id_to_word: list of strs
        :param id_to_word: mapping from word IDs to word names

        :type word_id_to_class_id: list of ints
        :param word_id_to_class_id: mapping from word IDs to indices in
                                    ``word_classes``

        :type word_classes: list of WordClass objects
        :param word_classes: list of all the word classes

        :type oos_words: list of strs
        :param oos_words: add words from this list to the vocabulary as
                          out-of-shortlist words, if they're not in
                          ``id_to_word``
        """

        if len(id_to_word) != len(word_id_to_class_id):
            raise ValueError("Vocabulary constructor requires equal-sized "
                             "id_to_word and word_id_to_class_id lists.")

        if '<s>' not in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('<s>')
            word_id_to_class_id.append(class_id)
            word_class = WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        if '</s>' not in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('</s>')
            word_id_to_class_id.append(class_id)
            word_class = WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        if '<unk>' not in id_to_word:
            word_id = len(id_to_word)
            assert word_id == len(word_id_to_class_id)
            class_id = len(word_classes)
            id_to_word.append('<unk>')
            word_id_to_class_id.append(class_id)
            word_class = WordClass(class_id, word_id, 1.0)
            word_classes.append(word_class)

        if oos_words is not None:
            for word in oos_words:
                if word not in id_to_word:
                    id_to_word.append(word)

        index = len(word_classes) - 1
        while index >= 0:
            word_class = word_classes[index]
            if len(word_class) == 1:
                word_id, _ = next(iter(word_class))
                if id_to_word[word_id].startswith('<'):
                    index -= 1
                    continue
            break
        self.num_normal_classes = index + 1

        for word_class in word_classes:
            word_class.normalize_probs()

        self.id_to_word = numpy.asarray(id_to_word, dtype=object)
        self.word_id_to_class_id = numpy.asarray(word_id_to_class_id)
        self._word_classes = numpy.asarray(word_classes)
        self.word_to_id = {word: word_id
                           for word_id, word in enumerate(self.id_to_word)}
        self._unigram_probs = None

    @classmethod
    def from_file(cls, input_file, input_format, oos_words=None):
        """Reads the shortlist words and possibly word classes from a vocabulary
        file.

        ``input_format`` is one of:

        * "words": ``input_file`` contains one word per line. Each word will be
                   assigned to its own class.
        * "classes": ``input_file`` contains a word followed by whitespace
                     followed by class ID on each line. Each word will be
                     assigned to the specified class. The class IDs can be
                     anything; they will be translated to consecutive numbers
                     after reading the file.
        * "srilm-classes": ``input_file`` contains a class name, membership
                           probability, and word, separated by whitespace, on
                           each line.

        The words read from the vocabulary file are put in the shortlist. If
        ``oos_words`` is given, those words are given an ID and added to the
        vocabulary as out-of-shortlist words if they don't exist in the
        vocabulary file.

        :type input_file: file object
        :param input_file: input vocabulary file

        :type input_format: str
        :param input_format: format of the input vocabulary file, "words",
                             "classes", or "srilm-classes"

        :type oos_words: list of strs
        :param oos_words: add words from this list to the vocabulary as
                          out-of-shortlist words, if they're not in the
                          vocabulary file
        """

        # We have also a set of the words just for faster checking if a word has
        # already been encountered.
        words = set()
        id_to_word = []
        word_id_to_class_id = []
        word_classes = []
        # Mapping from the IDs in the file to our internal class IDs.
        file_id_to_class_id = dict()

        for line in input_file:
            line = line.strip()
            fields = line.split()
            if not fields:
                continue
            if input_format == 'words' and len(fields) == 1:
                word = fields[0]
                file_id = None
                prob = 1.0
            elif input_format == 'classes' and len(fields) == 2:
                word = fields[0]
                file_id = int(fields[1])
                prob = 1.0
            elif input_format == 'srilm-classes' and len(fields) == 3:
                file_id = fields[0]
                prob = float(fields[1])
                word = fields[2]
            else:
                raise InputError("%d fields on one line of vocabulary file: %s"
                                 % (len(fields), line))

            if word in words:
                raise InputError("Word `%s' appears more than once in the "
                                 "vocabulary file." % word)
            words.add(word)
            word_id = len(id_to_word)
            id_to_word.append(word)

            if file_id in file_id_to_class_id:
                class_id = file_id_to_class_id[file_id]
                word_classes[class_id].add(word_id, prob)
            else:
                # No ID in the file or a new ID.
                class_id = len(word_classes)
                word_class = WordClass(class_id, word_id, prob)
                word_classes.append(word_class)
                if file_id is not None:
                    file_id_to_class_id[file_id] = class_id

            assert word_id == len(word_id_to_class_id)
            word_id_to_class_id.append(class_id)

        return cls(id_to_word, word_id_to_class_id, word_classes, oos_words)

    @classmethod
    def from_word_counts(cls, word_counts, num_classes=None):
        """Creates a vocabulary and classes from word counts. All the words will
        be in the shortlist.

        :type word_counts: dict
        :param word_counts: dictionary from words to the number of occurrences
                            in the corpus

        :type num_classes: int
        :param num_classes: number of classes to create in addition to the
                            special classes, or None for one class per word
        """

        # The special tokens are created automatically by the constructor. They
        # should not be included when creating the classes, so create a copy of
        # the word counts without the special tokens.
        word_counts = dict(word_counts)
        if '<s>' in word_counts:
            del word_counts['<s>']
        if '</s>' in word_counts:
            del word_counts['</s>']
        if '<unk>' in word_counts:
            del word_counts['<unk>']

        id_to_word = []
        word_id_to_class_id = []
        word_classes = []

        if num_classes is None:
            num_classes = len(word_counts)

        class_id = 0
        for word, _ in sorted(word_counts.items(),
                              key=lambda x: x[1]):
            word_id = len(id_to_word)
            id_to_word.append(word)

            if class_id < len(word_classes):
                word_classes[class_id].add(word_id, 1.0)
            else:
                assert class_id == len(word_classes)
                word_class = WordClass(class_id, word_id, 1.0)
                word_classes.append(word_class)

            assert word_id == len(word_id_to_class_id)
            word_id_to_class_id.append(class_id)
            class_id = (class_id + 1) % num_classes

        return cls(id_to_word, word_id_to_class_id, word_classes)

    @classmethod
    def from_state(cls, state):
        """Reads the vocabulary from a network state.

        :type state: hdf5.File
        :param state: HDF5 file that contains the architecture parameters
        """

        if 'vocabulary' not in state:
            raise IncompatibleStateError(
                "Vocabulary is missing from neural network state.")
        h5_vocabulary = state['vocabulary']

        if 'words' not in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'words' is missing from neural network "
                "state.")
        id_to_word = h5_vocabulary['words'].value

        if 'classes' not in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'classes' is missing from neural network "
                "state.")
        word_id_to_class_id = h5_vocabulary['classes'].value

        if 'probs' not in h5_vocabulary:
            raise IncompatibleStateError(
                "Vocabulary parameter 'probs' is missing from neural network "
                "state.")
        num_classes = word_id_to_class_id.max() + 1
        word_classes = [None] * num_classes
        h5_probs = h5_vocabulary['probs'].value
        for word_id, prob in enumerate(h5_probs):
            class_id = word_id_to_class_id[word_id]
            if word_classes[class_id] is None:
                word_class = WordClass(class_id, word_id, prob)
                word_classes[class_id] = word_class
            else:
                word_classes[class_id].add(word_id, prob)

        result = cls(id_to_word.tolist(),
                     word_id_to_class_id.tolist(),
                     word_classes)

        if 'unigram_probs' in h5_vocabulary:
            result._unigram_probs = h5_vocabulary['unigram_probs'].value

        return result

    def compute_probs(self, word_counts, update_class_probs=True):
        """Computes word unigram probabilities and possibly recomputes class
        membership probabilities from word counts. Class membership
        probabilities are updates only for classes whose words occur in
        ``word_counts``.

        Ensures that special tokens will always have nonzero probabilities.

        :type word_counts: dict
        :param words_counts: mapping from word strings to counts

        :type update_class_probs: bool
        :param input_files: input text files
        """

        counts = numpy.zeros(self.num_words(), dtype='int64')
        for word, count in word_counts.items():
            if word in self.word_to_id:
                word_id = self.word_to_id[word]
                if word_id < counts.size:
                    counts[word_id] = count
        self._unigram_probs = counts / counts.sum()

        if not update_class_probs:
            return

        sos_id = self.word_to_id['<s>']
        eos_id = self.word_to_id['</s>']
        unk_id = self.word_to_id['<unk>']
        counts[sos_id] = max(counts[sos_id], 1)
        counts[eos_id] = max(counts[eos_id], 1)
        counts[unk_id] = max(counts[unk_id], 1)

        for cls in self._word_classes:
            cls_counts = dict()
            for word_id, _ in cls:
                cls_counts[word_id] = counts[word_id]
            cls_total = sum(cls_counts.values())
            if cls_total > 0:
                for word_id, count in cls_counts.items():
                    cls.set_prob(word_id, float(count) / cls_total)
            else:
                prob = 1.0 / len(cls)
                for word_id, _ in cls:
                    cls.set_prob(word_id, prob)

    def get_state(self, state):
        """Saves the vocabulary in a network state file.

        If there already is a vocabulary in the state, it will be replaced, so
        it has to have the same number of words.

        :type state: h5py.File
        :param state: HDF5 file for storing the neural network parameters
        """

        h5_vocabulary = state.require_group('vocabulary')

        if 'words' in h5_vocabulary:
            state['words'][:] = self.id_to_word
        else:
            str_dtype = h5py.special_dtype(vlen=str)
            h5_vocabulary.create_dataset('words',
                                         data=self.id_to_word,
                                         dtype=str_dtype)

        if 'classes' in h5_vocabulary:
            state['classes'][:] = self.word_id_to_class_id
        else:
            h5_vocabulary.create_dataset('classes',
                                         data=self.word_id_to_class_id)

        probs = [self._word_classes[class_id].get_prob(word_id)
                 for word_id, class_id in enumerate(self.word_id_to_class_id)]
        if 'probs' in h5_vocabulary:
            state['probs'][:] = probs
        else:
            h5_vocabulary.create_dataset('probs', data=probs)

        if self.has_unigram_probs():
            if 'unigram_probs' in h5_vocabulary:
                state['unigram_probs'][:] = self._unigram_probs
            else:
                h5_vocabulary.create_dataset('unigram_probs',
                                             data=self._unigram_probs)

    def num_words(self):
        """Returns the number of words in the vocabulary. This includes
        shortlist and out-of-shortlist words.

        :rtype: int
        :returns: the number of words in the shirtlist
        """

        return self.id_to_word.size

    def num_shortlist_words(self):
        """Returns the number of words in the shortlist. Only the shortlist
        words are assigned to a class and are predicted by the neural network.

        :rtype: int
        :returns: the number of words in the shortlist
        """

        return self.word_id_to_class_id.size

    def num_classes(self):
        """Returns the number of word classes.

        :rtype: int
        :returns: the number of words classes
        """

        return self._word_classes.size

    def words_to_ids(self, words):
        """Translates words into word IDs.

        :type words: list of strs
        :param words: a list of words

        :rtype: ndarray
        :returns: the given words translated into word IDs
        """

        unk_id = self.word_to_id['<unk>']
        result = numpy.zeros(len(words), dtype='int64')
        for index, word in enumerate(words):
            if word in self.word_to_id:
                result[index] = self.word_to_id[word]
            else:
                result[index] = unk_id
        return result

    def class_ids_to_word_ids(self, class_ids):
        """Samples a word from the membership probability distribution of a
        class. (If classes are not used, returns the one word in the class.)

        :type class_ids: list of ints
        :param class_ids: list of class IDs

        :rtype: list of ints
        :returns: a word ID from each of the given classes
        """

        return [self._word_classes[class_id].sample()
                for class_id in class_ids]

    def get_word_prob(self, word_id):
        """Returns the class membership probability of a word.

        :type word_id: int
        :param word_id: ID of a word

        :rtype: float
        :returns: the probability of the word within its class
        """

        class_id = self.word_id_to_class_id[word_id]
        word_class = self._word_classes[class_id]
        return word_class.get_prob(word_id)

    def get_class_memberships(self, word_ids):
        """Finds the classes and class membership probabilities given a matrix
        of word IDs.

        Word IDs that are not in the shortlist are converted to ``<unk>``.

        :type word_ids: ndarray
        :param word_ids: a matrix containing word IDs

        :rtype: tuple of ndarrays
        :returns: two matrices, the first one containing class IDs and the
                  second one containing class membership probabilities
        """

        unk_id = self.word_to_id['<unk>']
        word_ids = numpy.copy(word_ids)
        word_ids[word_ids >= self.num_shortlist_words()] = unk_id
        class_ids = self.word_id_to_class_id[word_ids]
        word_classes = self._word_classes[class_ids]
        get_probs = numpy.vectorize(lambda wc, wid: wc.get_prob(wid))
        return class_ids, get_probs(word_classes, word_ids)

    def words(self):
        """A generator for iterating through the words in the vocabulary.

        :rtype: generator for str
        :returns: generates the next word in the vocabulary
        """

        for word in self.word_to_id.keys():
            yield word

    def in_shortlist(self, word_id):
        """Checks if the word with given ID is in the shortlist.

        :type word_id: int
        :param word_id: a word ID

        :rtype: bool
        :returns: ``True`` if the word is in shortlist, ``False`` otherwise
        """

        return word_id < self.word_id_to_class_id.size

    def has_unigram_probs(self):
        """Checks if the word unigram probabilities are computed and
        ``get_oos_logprobs()`` can be called.

        :rtype: bool
        :returns: ``True`` if the unigram probabilities are computed, ``False``
                  otherwise
        """

        return self._unigram_probs is not None

    def get_oos_logprobs(self):
        """Returns an array that can be indexed by word ID to obtain a log
        probability that should be added to the log probability predicted by the
        network.

        The returned probability is one (logprob is zero) for shortlist words,
        meaning that the shortlist probabilities are not affected. For other
        words, it's the unigram probability mass of the out-of-shortlist words
        divided according to their unigram frequencies.

        :rtype: ndarray
        :returns: an array that maps a word ID to the log probability that
                  should be added to the log probability predicted by the
                  network
        """

        shortlist_size = self.num_shortlist_words()
        oos_probs = numpy.copy(self._unigram_probs)
        oos_probs[:shortlist_size] = 1.0
        total_oos_prob = oos_probs[shortlist_size:].sum()
        oos_probs[shortlist_size:] /= total_oos_prob
        return numpy.log(oos_probs)

    def __contains__(self, word):
        """Tests if ``word`` is included in the vocabulary.

        :type word: str
        :param word: a word

        :rtype: bool
        :returns: True if ``word`` is in the vocabulary, False otherwise.
        """

        return word in self.word_to_id