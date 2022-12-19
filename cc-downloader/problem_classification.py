import re
from nltk.stem import WordNetLemmatizer

class ProblemClassifier:
    """
    A class to find keyword matches in a text.

    Parameters
    ----------
    topic_keywords : dict
        A dict with topic as key and list of keywords as value.
    lemmatize_keywords : bool
        Whether to add lemma of each keyword to keyword list.
    lemmatize_text : bool
        Whether to lemmatize the text before matching (computationally expensive).

    Attributes
    ----------
    topics : list
        The list of topics.
    nlp : spacy.lang
        The spacy model.
    topic_keywords : dict
        A dict with topic as key and list of keywords as value.
    topic_queries : dict
        A dict with topic as key and regex query as value.

    Methods
    -------
    lemmatize(text: str)
        Lemmatizes the text.
    form_queries()
        Forms regex queries for each topic.
    classify(text: str) -> dict
        Returns dict with topic as key and number of matches as value.

    Examples
    --------
    >>> topic_keywords = {'topic1': ['keyword1', 'keyword2'], 'topic2': ['keyword3', 'keyword4']}
    >>> classifier = ProblemClassifier(topic_keywords)
    >>> classifier.classify('This text contains keyword1, keyword2, and keyword3.')
    {'topic1': 2, 'topic2': 1}
    """
    def __init__(self, topic_keywords: dict, lemmatize_keywords: bool = True, lemmatize_text: bool = False):
        self.topic_keywords = topic_keywords
        self.topics = topic_keywords.keys()
        self.lemmatize_keywords = lemmatize_keywords
        self.lemmatize_text = lemmatize_text
        self.lemmatizer = WordNetLemmatizer()
        if lemmatize_keywords:
            self.topic_keywords = {topic: keywords + [self.lemmatize(keyword) for keyword in keywords] for topic, keywords in self.topic_keywords.items()}
        self.form_queries()

    # def lemmatize(self, text: str):
    #     doc = self.nlp(text)
    #
    #     return ' '.join([token.lemma_ for token in doc])

    def lemmatize(self, word: str):
        return self.lemmatizer.lemmatize(word)

    def form_queries(self):
        """ Forms regex query that matches on all keywords in the keyword list if they appear within word
        boundaries (so not within another word) """

        self.topic_queries = {}
        for topic, keywords in self.topic_keywords.items():
            self.topic_queries[topic] = '|'.join([r'\b' + keyword + r'\b' for keyword in keywords])

    def classify(self, text: str) -> dict:
        """ Returns dict with topic as key and number of matches as value """
        # if self.lemmatize_text:
        #     text = self.lemmatize(text)
        topic_matches = {}
        for topic, query in self.topic_queries.items():
            topic_matches[topic] = len(re.findall(query, text))

        return topic_matches

# problem_classifier = ProblemClassifier(topic_keywords)
# res = problem_classifier.classify('I have a problem with MAnufacturing my computer chips and traveling to meet my customers.')
#
# import pandas as pd
# df = pd.DataFrame({'text': ['I have a problem with MAnufacturing my computer chips.',
#                             'I have a problem traveling to meet my customers.'],})
# df = pd.concat([df, df['text'].apply(problem_classifier.classify).apply(pd.Series)], axis=1)

