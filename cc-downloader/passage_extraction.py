import re
import numpy as np
import nltk

class PassageExtractor:
    """ Extracts passages around keyword mentions.
    Parameters
    ----------
    text : str
        The text to extract passages from.
    keywords : list
        The keywords to extract passages around.
    return_paragraphs : bool
        Whether to return the entire paragraph containing a keyword mention.
    n_sent_backward : int
        The number of sentences to extract before the keyword mention. Does not apply if return_paragraphs
        is True.
    n_sent_forward : int
        The number of sentences to extract after the keyword mention. Does not apply if return_paragraphs
        is True.
    merge_passages : bool
        Whether to merge overlapping passages. Does not apply if return_paragraphs is True.
    char_limit : int
        The maximum number of characters to extract.
    Examples
    --------
    >>> text = 'This is a sentence This is second sentence! This is a third sentence mentioning a
    keyword. This is a fourth sentence. This is a fifth sentence. This is a sixth sentence.'
    >>> keywords = ['keyword']
    >>> extractor = PassageExtractor(text, keywords, n_sent_forward=2, n_sent_backward=1)
    >>> extractor.extract_relevant_passages()
    ['This is a second sentence! This is a third sentence mentioning a keyword. This is a fourth sentence.
    This is a fifth sentence.']
    """

    def __init__(self, text, keywords, return_paragraphs=False, n_sent_backward=2, n_sent_forward=4,
                 char_limit=3000, merge_passages=True):
        self.text = text
        self.keywords = keywords
        self.return_paragraphs = return_paragraphs
        self.n_sent_backward = n_sent_backward
        self.n_sent_forward = n_sent_forward
        self.merge_passages = merge_passages
        self.char_limit = char_limit
        if self.char_limit == None:
            self.char_limit = np.inf

    @staticmethod
    def mergeIntervals(arr):
        """ Merge overlapping intervals. """
        arr.sort(key=lambda x: x[0])
        index = 0
        for i in range(1, len(arr)):
            if (arr[index][1] >= arr[i][0]):
                arr[index][1] = max(arr[index][1], arr[i][1])
            else:
                index = index + 1
                arr[index] = arr[i]

        return arr[:index + 1]

    def extract_sentences_around_keyword_mention(self) -> list:
        sentence_boundary = '(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        # sentences = re.split(sentence_boundary, self.text)
        sentences = nltk.sent_tokenize(self.text)
        intervals = []
        for index, sentence in enumerate(sentences):
            for keyword in self.keywords:
                if keyword.casefold() in sentence.casefold():
                    keyword_mention_index = index
                    start_index = max(0, keyword_mention_index - self.n_sent_backward)
                    end_index = min(len(sentences), keyword_mention_index + self.n_sent_forward + 1)
                    intervals.append([start_index, end_index])
        if self.merge_passages:
            intervals = self.mergeIntervals(intervals)
        relevant_passages = [' '.join(sentences[start_index:end_index]) for start_index, end_index in
                             intervals]

        # enforce character limit
        relevant_passages = [passage for passage in relevant_passages if len(passage) < self.char_limit]

        return relevant_passages

    def extract_relevant_passages(self) -> list:
        relevant_passages = []

        if self.return_paragraphs == True:
            paragraphs = self.text.split('\n')
            relevant_passages += [paragraph for paragraph in paragraphs if any(keyword.casefold()
                                      in paragraph.casefold() for keyword in self.keywords)]
        else:
            relevant_passages += self.extract_sentences_around_keyword_mention()

        return list(set(relevant_passages))  # remove duplicates