from collections import Counter
from konlpy.tag import Komoran


f = open("data.txt", "r")
lines = f.read()

nlpy = Komoran()
nouns = nlpy.nouns(lines)


count = Counter(nouns)

tag_count = []
tags = []

for n, c in count.most_common(100):

    dics = {'tag': n, 'count': c}
    if len(dics['tag']) >= 2 and len(tags) <= 49:
        tag_count.append(dics)
        tags.append(dics['tag'])

print(tags)