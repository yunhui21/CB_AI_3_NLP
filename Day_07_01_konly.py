# Day_07_01_konly.py
import konlpy
import collections

print(konlpy.corpus.kolaw.fileids()) # ['constitution.txt'] 헌법
print(konlpy.corpus.kobill.fileids()) # ['1809890.txt', '1809891.txt', ...]

# f = konlpy.corpus.kolaw.open('constitution.txt')
# print(f) # <_io.TextIOWrapper name='C:\\Users\\yunhui\\AppData\\Local\\Programs\\Python\\Python37\\lib\\site-packages\\konlpy/data/corpus/kolaw/constitution.txt' mode='r' encoding='utf-8'>
# print(f.read()) # 제1조 이 헌법은 1988년 2월 25일부터 시행한다.
# f.close()

kolaw = konlpy.corpus.kolaw.open('constitution.txt').read()
tokens = kolaw.split()
print(tokens[:10])

tagger = konlpy.tag.Hannanum()
# tagger = konlpy.tag.Kkma()
# tagger = konlpy.tag.Komoran()
# tagger = konlpy.tag.Okt()
# tagger = konlpy.tag.Mecab() # 추가설치 필요

# pos: Part of speech
pos = tagger.pos(kolaw)
# print(pos[:10])
# [('대한민국헌법', 'N'), ('유구', 'N'), ('하', 'X'), ('ㄴ', 'E'),....]# hannanum
# [('대한민국', 'NNG'), ('헌법', 'NNG'), ('유구', 'NNG'), ('하', 'XSV'),# kkma
# [('대한민국', 'NNP'), ('헌법', 'NNP'), ('유구', 'XR'), ('하', 'XSA'),...] # Komoran
# [('대한민국', 'Noun'), ('헌법', 'Noun'), ('\n\n', 'Foreign'), ('유구', 'Noun'),# OKt

print(len(pos), len(set(pos)))          # 8549 1499 set을 통해서 중복된 데이터는 이게 더 작다.
print(len(tokens), len(set(tokens)))    # 4178 2029

# 문제
# 대한민국에 가장 많이 등장하는 명사 5개를 알려주세요.
# nouns = [(w,p) for w, p in pos if p == 'N' ] # [(('법률', 'N'), 115), (('수', 'N'), 91), (('①', 'N'), 78), (('때', 'N'), 55), (('국회', 'N'), 53)]
nouns = ['{}/{}'.format(w,p) for w, p in pos if p == 'N' ] # [('법률/N', 115), ('수/N', 91), ('①/N', 78), ('때/N', 55), ('국회/N', 53)]

print(nouns[:5])

freq = collections.Counter(nouns) # list를 풀어주어야 한다.
print(freq.most_common(5))


