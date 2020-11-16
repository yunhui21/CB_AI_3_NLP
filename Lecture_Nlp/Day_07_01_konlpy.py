# Day_07_01_konlpy.py
import konlpy
import collections

print(konlpy.corpus.kolaw.fileids())    # ['constitution.txt']
print(konlpy.corpus.kobill.fileids())   # ['1809896.txt', '1809897.txt', ...]

# f = konlpy.corpus.kolaw.open('constitution.txt')
# print(f)
# print(f.read())
# f.close()

kolaw = konlpy.corpus.kolaw.open('constitution.txt').read()
tokens = kolaw.split()
print(tokens[:10])

tagger = konlpy.tag.Hannanum()
# tagger = konlpy.tag.Komoran()
# tagger = konlpy.tag.Kkma()
# tagger = konlpy.tag.Okt()
# tagger = konlpy.tag.Mecab()       # 추가 설치 필요

# pos: part of speech
pos = tagger.pos(kolaw)
print(pos[:10])
# [('대한민국헌법', 'N'), ('유구', 'N'), ('하', 'X'), ('ㄴ', 'E'), ('역사', 'N'), ('와', 'J'), ('전통', 'N'), ('에', 'J'), ('빛', 'N'), ('나는', 'J')]
# [('대한민국', 'NNP'), ('헌법', 'NNP'), ('유구', 'XR'), ('하', 'XSA'), ('ㄴ', 'ETM'), ('역사', 'NNG'), ('와', 'JC'), ('전통', 'NNG'), ('에', 'JKB'), ('빛나', 'VV')]
# [('대한민국', 'NNG'), ('헌법', 'NNG'), ('유구', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('역사', 'NNG'), ('와', 'JC'), ('전통', 'NNG'), ('에', 'JKM'), ('빛나', 'VV')]
# [('대한민국', 'Noun'), ('헌법', 'Noun'), ('\n\n', 'Foreign'), ('유구', 'Noun'), ('한', 'Josa'), ('역사', 'Noun'), ('와', 'Josa'), ('전통', 'Noun'), ('에', 'Josa'), ('빛나는', 'Verb')]

print(len(pos), len(set(pos)))          # 8549 1499
print(len(tokens), len(set(tokens)))    # 4178 2029

# 문제
# 대한민국헌법에 가장 많이 등장하는 명사 5개를 알려주세요(명사 = N)
# nouns = [(w, p) for w, p in pos if p == 'N']
nouns = ['{}/{}'.format(w, p) for w, p in pos if p == 'N']
print(nouns[:5])

freq = collections.Counter(nouns)
print(freq.most_common(5))
