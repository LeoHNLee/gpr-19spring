- [예제 코드](./Mastering-Natural-Language-Processing-with-Python-master)

## ch.1 문자열을 사용한 작업

### ss.1.1 토큰화

#### sss.1.1.1 텍스트를 문장으로 토큰화

- 주어진 텍스트를 개별 문장으로 토큰화
  - [nltk.tokenize.sent_tokenize()](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_1.py)
- 여러 문장을 토큰화 : PunktSentence Tokenizer
  - [nltk.data.load()](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_2.py)

#### sss.1.1.2 다양한 언어의 텍스트를 토큰화

- [프랑스어](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_3.py)

#### sss.1.1.3 문장을 단어로 토큰화

- [nltk.word_tokenize()](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_4.py)

#### sss.1.1.4 TreebankWordTokenizer를 사용한 토큰화

- 문장을 단어로 토큰화
- TreebankWordTokenizer는 Penn Treebank Corpus 기준을 따른다.
  - [분리 축약형으로 작동](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_7.py)
  - [사용법](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_6.py)
- PunkWordTokenizer
  - 분리된 문장부호로 작동
- [WordPunctTokenizer](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_8.py)
  - 문장부호를 새로운 토큰으로 분할하여 제공
- 토크나이저의 상속트리 : Tokenizer
  - PunkWordTokenizer
  - TreebankWordTokenizer
  - RegexpTokenizer
    - WordPunctTokenizer
    - WhitespaceTokenizer

#### sss.1.1.5 정규표현식을 사용한 토큰화

- RegexpTokenizer

### ss.1.2 정규화

- 문장 부호 제거, 소문자/대문자 변환, 숫자/단어 변환, 약어 전개, 텍스트 정규화

#### sss.1.2.1 [문장 부호 제거](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_22.py)

#### sss.1.2.2 [소문자/대문자 변환](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_23.py)

#### sss.1.2.3 불용어 처리

- stop words 위치
  - nltk_data/corpora/stopwords/
- [불용어 처리](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_24.py)

### ss.1.3 토큰의 대체 및 수정

- 약어 전개(don't to do not), 숫자를 단어로(1 to one)

#### sss.1.3.1 정규 표현식을 사용한 단어 대체

- [약어 대체](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/replacers.py)
  -
##### atc.1.3.1.1 [텍스트를 다른 텍스트로 대체하는 예제](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_27.py)

#### sss.1.3.2 [토큰화 전에 대체 수행](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch1_28.py)

#### sss.1.3.3 반복되는 문자 처리

- back-reference 방식으로 이전 문자를 참조하는 경우를 제거
  - [class code](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/replacers.py)
- [re-jump to python](https://wikidocs.net/4309)

##### atc.1.3.3.1 [반복 문자를 삭제하는 예제](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/ch_129.py)

- problem : happy to hapy
- [solv : wordnet 포함](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/replacers.py)

#### sss.1.3.4 단어를 동의어로 대체

- [WordReplacer 추가 : 단어와 동의어 매핑](Mastering-Natural-Language-Processing-with-Python-master/Chapter-1/replacers.py)
- 이건 수동이네

### ss.1.4 텍스트에 지프의 법칙 적용

- Zipf's law에 따르면, 토큰의 빈도는 정렬된 목록의 순위 혹은 위치에 정비례
- 양대수 그래프, log-log graph : x,y의 눈금이 log

### ss.1.5 유사 척도

- nltk.metrics
- 편집거리 / 리벤슈타인 편집 거리 : 두 문자열에 동일하게 하는 삽입, 대체, 삭제 될 수 있는 문자의 수를 계산하기 위해 사용
- 자카드 계수 / 타니모토 계수 : x,y 두 세트의 오러랩 측정
- 스미스 워터맨 거리 : dna 간의 광학 정력을 검출하기 위해 개발
- 이진 거리 : 문자열 유사도 메트릭, 두 개의 라벨이 동일한지
- 매시 거리 : 이진 거리에서 멀티 라벨

## ch.2 통계 언어 모델링

### ss.2.1 단어 빈도 이해

- 연어, collocations : 함께하는 경향이 있는 둘 이상의 토큰의 집합
- unigram : 하나의 토큰
  - [ngram 생성 코드](Mastering-Natural-Language-Processing-with-Python-master/Chapter-2/ch2_2.py)
- bigram : 토큰의 쌍
  - bigram 찾기 : 소문자 단어를 검색 => 소문자 단어 리스트 생성 => BigramCollocationFinder 생성
  - [nltk.metrics.BigramAssocMeasures](Mastering-Natural-Language-Processing-with-Python-master/Chapter-2/ch2_3.py)
  - [nltk.collocation.BigramCollocationFinder.from_words](Mastering-Natural-Language-Processing-with-Python-master/Chapter-2/ch2_5.py)

#### sss.2.1.1 텍스트의 MLE 개발

- MLE, multinomial logistic regression, conditionalexponential classifier

#### sss.2.1.2 Hidden Markov Model, HMM

### ss.2.2 MLE 모델의 스무딩 적용

- smoothing : 이전에 발생하지 않은 단어를 처리하는 데 사용

#### sss.2.2.1 에드온 스무딩

#### sss.2.2.2 Good Turing

#### sss.2.2.3 Kneser Ney

#### sss.2.2.4 Witten Bell

### ss.2.3 MLE의 백-오프 매커니즘 개발, Katz back-off

### ss.2.4 믹스 앤 매치를 얻기 위한 데이터 보간법 적용

### ss.2.5 혼잡도를 통한 언어 모델 평가

### ss.2.6 모델링 언어에서 메트로폴리스 헤이스팅스 적용

### ss.2.7 언어 처리에서 깁스 샘플링 적용

## ch.3 형태학 - 시작하기

### ss.3.1 형태학 소개

- 형태소의 도움으로 토큰 생산의 연구
- 형태소의 종류
  - 어간 : 자립 형태소
  - 접사 : 의존 형태소
- 언어의 종류 : 고립어, 교착어, 굴절어
  - 고립어 : 단어가 단순히 자립 형태소. 시제/수 정보를 전달하지 않는다.
  - 교착어 : 복합어 정보를 전달하기 위해 작은 단어로 함께 결함된 것 (eg 터키어)
  - 굴절어 : 단어를 단순한 단위로 세분화했지만, 모든 단순한 단위는 간단한 다른 의미를 나타낸다. (eg 라틴어)

### ss.3.2 스테머 이해

- stemming : 단어에서 접사를 제거하여 단어로부터 어간을 획득하는 과정
- Porter Stemming : 영어 단어에 존재하는 몇가지 잘 알려진 접미사를 대체하고 제거하도록 설계
- [nltk.stem.PorterStemmer](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_1.py)
- stemmer 1 interface 상속
  - PorterStemmer
  - LancasterStemmer
  - RegexpStemmer
  - SnowballStemmer
- Lancaster : 감정의 단어를 더 많이 사용한다.
  - [nltk.stem.LancasterStemmer](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_2.py)
- RegexpStemmer : 나만의 스테머
  - [nltk.stem.RegexpStemmer](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_3.py)
- SnowballStemmer : 영어 이외 13개의 언어에서 스테밍을 수행하는 데 사용
  - 먼저 해당 언어로 인스턴스를 생성한 후 스테밍 수행
  - [nltk.stem.SnowballStemmer](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_4.py)

### ss.3.3 원형복원 이해

- Lemmatization, 원형복원 : 다른 범주의 형태로 단어를 변환하는 과정
  - 원형을 찾을 수 없으면, 원래 집어넣은 단어를 내뱉는다.
  - [lemmatization eg](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_5.py)
- [lematization vs. stemming](Mastering-Natural-Language-Processing-with-Python-master/Chapter-3/ch3_6.py)

### ss.3.4 비영어 언어의 스테머 개발

- Polyglot : 토큰에서 형태소를 얻는 데 사용되는 morfessor 모델을 제공하는 데 사용되는 sw.
- python libarary가 있다.

### ss.3.5 형태소 분석기

- 형태소 기반 형태학
- 어휘 기반 형태학
- 단어 기반 형태학
- pyEnchant Dictionary 사용
  - 뭔 코드인지 모르겠는데?

### ss.3.6 형태소 생성기

- 숫자, 범주, 어간 등의 단어에 대한 설명이 주어지면, 원래 단어가 검색된다.
- 뭔가 다양한 언어를 위한 다양한 패키지가 있다.

### ss.3.7 검색 엔진

- PyStemmer는 snowball stemming 기반으로 정보 검색 수행

## ch.4 품사 태깅 - 단어 식별

### ss.4.1 품사 태깅 소개

- 품사, parts-of-speech, POS
- [nltk.pos_tag](Mastering-Natural-Language-Processing-with-Python-master/Chapter-4/ch4_1.py)
- [tag 목록](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
  - [tag 정보 확인 on python](Mastering-Natural-Language-Processing-with-Python-master/Chapter-4/ch4_2.py)
- 단어의 모호성 : 동사이면서 명사이고 그러면 뭘로 태깅되지??
  - [수동으로 특정 단어의 태깅 추가](Mastering-Natural-Language-Processing-with-Python-master/Chapter-4/ch4_5.py)

#### sss.4.1.1 기본태깅

- 모든 토큰에 동일한 품사를 태그

### ss.4.2 POS-tagged corpora 생성

- corpus : 문서의 집합
- corpora : 여러 corpus의 집합

### ss.4.3 기계 학습 알고리즘 선택

- POS 태깅 : 단어 범주 중의성, 문법 태깅
- rule-based
- stochastic/probabilistic

### ss.4.4 n-gram 접근법과 관련된 통계 모델링

- [nltk.tag.UnigramTagger](Mastering-Natural-Language-Processing-with-Python-master/Chapter-4/ch4_17.py)

### ss.4.5 pos-tagged data를 사용한 청커 개발

- chunking
  - 개체 검출을 수행하는 데 사용되는 과정
  - 문장에서 여러 개의 토큰 시퀀스를 세분화하고 라벨을 지정하는 데 사용

## ch.5 파싱 - 훈련 데이터 분석

- parsing, 구문 분석 : 자연어로 쓰여진 문자 순서가 형식 문법으로 정의된 규칙에 부합하는지 여부를 찾는 과정
  - 문장을 단어/구문 시퀀스로 분해하고, 특정 구성 요소 범주를 제공

### ss.5.1 파싱 소개

- 문법적으로는 정확하지만, 의미적으로는 부정확한 경우 : 파스 트리 생성에 이어 의미를 추가
- 파서 : 입력을 받아서 파스/구문 트리를 구성하는 sw
- 하향식 파싱
  - 시작 기호에서 시작
  - 개별 구성 요소에 도달할 때까지 계속
  - Recursive Descent, LL, Earley
- 상향식 파싱
  - 개별 구성 요소에서 시작
  - 시작 기호에 도달할 때까지 계속
  - Operator-precedence, simple precedence, simpole LR, LALR, Canonical LR, GLR, CYK, Recursive ascent, Shift-reduce
- nltk.parse.api.ParserI

### ss.5.2 트리뱅크 구성

- treebank corpus reader를 사용하는 penn treebank corpus를 접속하는 코드 :
  - [eg](Mastering-Natural-Language-Processing-with-Python-master/Chapter-5/ch5_2.py)
  - [이어지는 eg](Mastering-Natural-Language-Processing-with-Python-master/Chapter-5/ch5_3.py)

### ss.5.3 트리뱅크의 문맥 자유 문법 규칙 추출

- CFG, context free grammar, 자유 문법 규칙 : 촘스키가 자연어를 위해 정의한 문법
  - 구성요소
    - 비 단말 노드의 집합 N
    - 단말 노드의 집합 T
    - 시작 기호 S
    - 양식의 생성 규칙 집합 P
  - 구문 구조 규칙 : $`A\rightarrow a`$
  - 문장 구조 규칙
    - 선언적 구조 : 서술문
    - 명령형 구조 : 명령문, 제안문
    - 예/아니오 구조 : 질의 응답 문장을 처리
    - wh 질의구조 : 질의 응답 문장

### ss.5.4 CFG에서 확률적 문맥 자유 문법 생성

- PCFG, probabilistic Context-free Grammar
- 확률은 모든 생성 규칙에 추가된다.

### ss.5.5 CYK 차트 파싱 알고리즘

- 재귀가 아니라 DP로 구현, O(n3)

### ss.5.6 Earley 차트 파싱 알고리즘

## ch.6 의미 분석 - 본질 표현

- 문자 시퀀스, 단어 시퀀스의 의미를 결정하는 과정으로 정의
- 의미 판별 작업을 수행하기 위해 사용

### ss.6.1 의미 분석 소개

- 문장의 구문 구조가 설계되면, 이후에 문장의 의미 분석이 수행된다.
- 의미 해석은 의미를 문장에 매핑하는 것을 의미한다.
- 의미 분석의 기본 단위는 의미/감각으로 지칭된다.
- Gensim : 문서 인덱싱, 주제 모델링, 유사성 검색을 수행하는 데 사용할 수 있다.
- Polyglot : 다국어 앱
- MontyLingua : 영어 텍스트의 의미 해석을 수행

#### sss.6.1.1 NER 소개

- NER, named entity recognition : 고유 명사, 개체명이 문서에 위치하는 과정
  - NEP, person
  - NED, designation
  - NEO, organization
  - NEA, abbreviation
  - NEB, brand
  - NETP, title of person
  - NETO, title of object
  - NEL, location
  - NETI, time
  - NEN, number
  - NEM, measure
  - NETE, terms
- NER을 수행하기 위해 [nltk.tag.StanfordNERTagger](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/stanford.py)

#### sss.6.1.2 HMM을 사용한 NER 시스템

#### sss.6.1.3 기계 학습 툴킷을 사용한 NER 훈련

- 규칙 기반 / 수동
  - 리스트 룩업 접근
  - 언어 접근
- 기계 학습 기반 / 자동
  - HMM
  - 최대 엔트로피 마르코프 모델
  - 조건부 임의 필드
  - 서포트 벡터 머신
  - 의사결정 트리

#### sss.6.1.4 POS 태깅을 사용한 NER

- [사용할 수 있는 POS 태깅](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
- NNP로 태그된 토큰은 개체명

### ss.6.2 wordnet의 synset id 생성

- 상위어, 동의어, 반의어, 하의어 등 : 단어 간 개념의존성은 synsets을 사용해 찾을 수 있다.
- [nltk.corpus.wordnet.sunsets](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/ch6_13.py)

### ss.6.3 wordnet을 사용한 의미 판별

- 의미 판별 : 의미에 기초해, 둘 이상의 동일한 철자 혹은 동일한 소리 단어를 구별하는 작업
- WSD task의 구현
  - Lesk algorithms
    - original lesk
    - cosine lesk
    - simple lesk
    - apply/append lesk
    - reinforcement lesk
  - similarity
    - 정보 내용
    - 경로 유사성
  - supervised WSD
    - It Makes Sense
    - SVM WSD
  - vector 공간 모델
    - 토픽 모델, LDA
    - LSI/LSA
    - NMF
  - graph 기반 모델
    - Babelfly
    - UKB
  - baseline
    - Random sense
    - HIghest elmma counts
    - First NLTK sense
- NLTK에서 wordnet 감각 유사성이 포함하는 알고리즘
  - Resnik Score : 두 개의 토큰을 비교할 때 두 토큰의 유사성을 결정하는 점수가 반환
  - Wu-Palmer Similarity : 두 토큰의 유사성은 IS-분류법에서 계산된 최단 거리에 기초하여 결정
  - Leacock Chodorow Similarity : 유사성 점수는 분류에 존재하는 감각이 가장 짧은 경로 및 깊이에 기초하여 반환
  - Lin Similarity : 유사성 점수는 Least Common subsumer와 두 개의 입력 synsets의 정보 컨텐츠 정보에 기초하여 반환
  - Jiang-Conrath Similarity : 위와 비슷
- [path similarity](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/ch6_15.py)
- [Leacock Chodorow similarity](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/ch6_16.py)
- [Wu-Palmer similarity](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/ch6_17.py)
- [Resnik, Lin, Jiang-Conrath similarity](Mastering-Natural-Language-Processing-with-Python-master/Chapter-6/ch6_18.py)

## ch.7 감정 분석 - 나는 행복하다

### ss.7.1 감정 분석 소개

- Sentiment analysis : 긍정/부정/중립
- 이진분류 : 긍정/부정
- 멀티클래스 분류 : 긍정/부정/중립
- 토픽-감정 분석 : 감정 분석 + 토픽 마이닝
- 감정분석은 어휘목록(lexicon)을 사용해 수행할 수 있다.
- 단어 목록
  - ANEW, affective norms for english words : 1034단어, 학업 목적
    - DANEW : 네덜란드어
    - SSPANEW : 스페인어
  - AFINN : 2477단어, twitter text의 감정 분석을 수행할 목적
  - Balance Affective : 277 단어
    - BAWL : 독일어 2200
    - BAWL-R
  - Bilingual Finnish Affective Words : 210 영국 영어, 필란드 명사, 금기 단어 포함
  - Compass DeRose Guide to Emotino Words : 분류만 되었고 극성과 환기는 없음
  - DAL : 감정 분석에 사용할 수 있는 정서 단어
  - General Inquirer : 1915긍정, 2291부정
  - HL : 6800 단어
  - 아 귀찮아 기타 등등 있음
- [영화 리뷰](Mastering-Natural-Language-Processing-with-Python-master/Chapter-7/ch7_1.py)
- 정보 특징, informative features이 문서 내에 존재하는지 여부를 체크

#### sss.7.1.1 NER을 사용한 감정 분석

#### sss.7.1.2 기계 학습을 사용한 감정 분석

#### sss.7.1.3 NER 시스템의 평가

## ch.8 정보 검색 - 정보 접속

### ss.8.1 정보 검색 소개

- 사용자에 의해 수행되는 질의에 대한 응답으로 가장 적합한 정보를 검색하는 과정
- 색인 매커니즘 : 역 색인

### ss.8.2 벡터 공간 스코링 및 질의 연산자 상호 작용

### ss.8.3 잠재 의미 색인을 이용한 IR 시스템 개발

### ss.8.4 텍스트 요약

### ss.8.5 질의 응답 시스템

## ch.9 담화 분석 - 아는 것은 믿는 것이다

- 문맥 정보를 결정하는 과정

### ss.9.1 담화 분석 소개

- 텍스트 해석과 사회적 상호 작용을 인식하는 텍스트 혹은 언어 분석을 수행하는 과정
- 앞 문장에 기초하여 문장의 의미를 해석할 수 있다.
- ***GPR과 크게 연관 있다***
- DRT, discourse representation theory : AR을 수행하는 방법을 제공하기 위해 개발
- DRS, discourse representation structure : 담화 지시 대상 및 조건의 도움으로 담화의 의미를 제공하기 위해 개발
- FOPL, first order predicate logic : 명제 논리의 사상을 확장하기 위해 개발
   - 함수, 인자, 한정사의 사용 포함
- <code>nltk.sem.logic</code> : 1차 술어 논리 구현을 제공하는 데 사용
- <code>nltk.sem.drt</code> : 담화 표현 이론의 기초를 제공
- <code>nltk.corpus.reader.wordnet</code> : wordnet3.0에 대한 접속을 제공
- [DRS 구현](Mastering-Natural-Language-Processing-with-Python-master/Chapter-9/ch9_1.py)
- [DRS 연결 연산자를 사용](Mastering-Natural-Language-Processing-with-Python-master/Chapter-9/ch9_3.py)
- [한 DRS를 다른 것에 embed하는데 사용](Mastering-Natural-Language-Processing-with-Python-master/Chapter-9/ch9_4.py)
- [두 문장을 결합하는 데 사용](Mastering-Natural-Language-Processing-with-Python-master/Chapter-9/ch9_5.py)

#### sss.9.1.1 중심화 이론을 사용한 담화 분석

- centering theory를 사용한 담화 분석은 코퍼스 주석에 대한 첫 번째 단계
- 담화 참가자와 담화의 목적 혹은 의도 간의 상호 작용
- 참가자 태도
- 담화 구조

#### sss.9.1.2 대용어 복원, AR, anaphora resolution

- 대명사, 명사구가 해석되고 담화 지식에 기초해 특정한 개체를 언급하는 과정
- AR의 유형
  - 대명사 : 지시 대상은 대명사에 의해 참조된다.
  - 명확한 명사구 :
  - 수량사/서수 :
- 후방조응에서 지시 대상은 선행자 앞에 위치한다.
- 새로운 모듈은 기존 모듈인 <code>nltk.sem.logic, nltk.sem.drt</code> 상단에 개발되었다.
- <code>AbstractDRS.resolve()</code> : 특정 객체의 확인된 복사본으로 구성된 목록을 제공한다.
- 해결해야 하는 객체는 <code>readings()</code> 메소드를 오버라이드 해야한다.
- resolve 메소드는 traverse 함수를 사용하여 판독값을 생성하는 데 사용된다.
- traverse 함수는 작업 목록에서 정렬을 수행하는 데 사용된다.
- traverse 함수의 흐름도 :

## ch.10 NLP 시스템의 평가 - 성능 분석

### ss.10.1 NLP 시스템 평가의 필요성

### ss.10.2 IR 시스템의 평가

### ss.10.3 오류 식별 메트릭

### ss.10.4 어휘 매칭 기반 메트릭

### ss.10.5 구문 매칭 기반 메트릭

### ss.10.6 얕은 의미 매칭을 사용한 메트릭
