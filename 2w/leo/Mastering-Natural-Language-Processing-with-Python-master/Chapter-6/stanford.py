from nltk.tag import StanfordNERTagger
sentence = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
sentence.tag('John goes to NY'.split())
