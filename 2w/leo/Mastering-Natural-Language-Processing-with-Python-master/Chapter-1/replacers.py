import re
from nltk.corpus import wordnet

replacement_patterns=[
(r'won\'t', 'will not'),
(r'cat\'t', 'cannot'),
(r'i\'m','i am'),
(r'ain\'t', 'is not'),
(r'(\w+)\'ll', '\g<l> will'),
(r'(\w+)n\'t', '\g<l> not'),
(r'(\w+)\'ve', '\g<l> have'),
(r'(\w+)\'s', '\g<l> is'),
(r'(\w+)\'re', '\g<l> are'),
(r'(\w+)\'d', '\g<l> would'),
]
class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in paterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.sbn(pattern, repl, s)
        return s

class RepeatReplacer(object):
    def __init__(self):
        self.repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'

    def replace(self, word):
        repl_word = self.repeat_regexp.sub(self.repl, word)
        if wordnet.synsets(word): return word
        if repl_word != word: return self.replace(repl_word)
        else : return repl_word

class WordReplacer(object):
    def __init__(self, word_map):
        self.word_map = word_map
    def replace(self, word):
        return self.word_map.get(word, word)
