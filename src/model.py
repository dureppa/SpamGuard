import math
from collections import defaultdict


class NaiveBayesClassifier:
    """
    Наивный байесовский классификатор для бинарной классификации текста (спам / не-спам).
    Реализован с нуля без использования sklearn.naive_bayes.
    """

    def __init__(self):
        self.vocab = None
        self.p_spam = 0.0
        self.p_ham = 0.0
        self.total_words_spam = 0
        self.total_words_ham = 0
        self.V = 0

    def fit(self, X, y):
        """
        Обучение модели на обучающей выборке.

        Параметры:
        ----------
        X : list of str
            Список очищенных текстов (после предобработки).
        y : list of str
            Список меток ('spam' или 'ham').
        """
        if len(X) != len(y):
            raise ValueError("Длины X и y должны совпадать.")

        total_samples = len(y)
        spam_count = y.count('spam')
        ham_count = y.count('ham')

        self.p_spam = spam_count / total_samples
        self.p_ham = ham_count / total_samples

        vocab = defaultdict(lambda: {'spam': 0, 'ham': 0})
        for text, label in zip(X, y):
            words = text.split()
            for word in words:
                vocab[word][label] += 1

        self.vocab = {
            word: counts
            for word, counts in vocab.items()
            if counts['spam'] + counts['ham'] >= 2
        }

        self.total_words_spam = sum(counts['spam']
                                    for counts in self.vocab.values())
        self.total_words_ham = sum(counts['ham']
                                   for counts in self.vocab.values())
        self.V = len(self.vocab)

    def _laplace_prob(self, count, total):
        """
        Вычисление условной вероятности с применением сглаживания Лапласа.

        P(word | class) = (count + 1) / (total_words_in_class + V)
        """
        return (count + 1) / (total + self.V)

    def predict_proba(self, text):
        """
        Вычисление логарифмов апостериорных вероятностей для каждого класса.

        Возвращает:
        ----------
        log_p_spam : float
            log(P(spam | text))
        log_p_ham : float
            log(P(ham | text))
        """
        words = text.split()
        log_p_spam = math.log(self.p_spam)
        log_p_ham = math.log(self.p_ham)

        for word in words:
            if word in self.vocab:
                cnt_spam = self.vocab[word]['spam']
                cnt_ham = self.vocab[word]['ham']

                p_w_given_spam = self._laplace_prob(
                    cnt_spam, self.total_words_spam)
                p_w_given_ham = self._laplace_prob(
                    cnt_ham, self.total_words_ham)

                log_p_spam += math.log(p_w_given_spam)
                log_p_ham += math.log(p_w_given_ham)

        return log_p_spam, log_p_ham

    def predict(self, text):
        """
        Предсказание класса для входного текста.

        Возвращает:
        ----------
        str : 'spam' или 'ham'
        """
        log_p_spam, log_p_ham = self.predict_proba(text)
        return 'spam' if log_p_spam > log_p_ham else 'ham'
