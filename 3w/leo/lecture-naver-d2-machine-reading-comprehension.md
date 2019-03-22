# machine reading comprehension

## links

- [youtube](https://www.youtube.com/watch?v=XBCkJck0cdY)
- [slide share](https://www.slideshare.net/NaverEngineering/ss-108892693)

## contents

- [1. encoder](#ch.1.-encoder) : 모델에 좋은 데이터를 집어넣기
- [2. attention](#ch.2.-attention) : 문맥 관계 파악 (질의와 문맥간의)
- [3. output](#ch.3.-output) : 단어 위치 추정
- [4. learning methods](#ch4.-learning-methods)

## ch.1. encoder

### ss.1.1 word embedding

- 단어를 기계가 알아들을 수 있는 숫자로 표현

#### sss.1.1.1 one hot encoding

- 최초의 연구

#### sss.1.1.2 word2vec

- 비슷한 단어가 가까운 벡터공간에 사영

##### atc.1.1.2.1 cbow

- 주변 단어로 타겟을 추정

##### atc.1.1.2.2 skip gram

- 타겟으로 주변단어를 추정

#### sss.1.1.3 advanced word embedding

##### atc.1.1.3.1 glove

- 문서 전체에서 (bi-gram 등)의 동시 등장 확률을 계산
- 2.2M 단어 학습한 모델 배포 중
- 단어 커버리지가 fast text보다 높다

##### atc.1.1.3.2 fast text

- 부분 단어를 사용해서 학습
- 노이즈(오타 등)에 강하다.
- 1M 단어 학습한 모델 배포 중
- 성능은 glove보다 좋은 듯

### ss.1.2 character embedding

- 미등록 단어 처리
- cnn, rnn 두 가지 방식
- 성능은 비슷하나, 속도과 rnn < cnn

### ss.1.3 contextual embedding

- 문맥을 사전학습한 모델을 가져다 사용하기

#### sss.1.2.1 cove

- mt-lstm을 통해 사전학습
- 기계번역의 디코더만 따와서 쓰는 방식 등

#### sss.1.2.2 elmo

- language model을 학습
  - 한국어로 치면 명사에 적당한 조사를 학습하는 등 일종의 문법 학습

### ss.1.4 other features

#### sss.1.4.1 종류

- linguatic feature : POS, named entity, dependency label
- term frequency
- co-occurrence : context와 question 모두에 나타나는가?
- soft alignment : glove 벡터 간의 내적 유사도

#### sss.1.4.2 reason

- 품사를 보고 정답 위치 추정 가능
- 구문 구조를 통해 정답 위치 추정 가능
- 질의 타입(의문사 종류)과 개체명을 대입

#### sss.1.4.3 make some error

- feature 간의 간섭
  - 서로 다른 피처가 상반되는 결론을 내놓을 경우
  - 복잡도가 상승
- feature selection : need experiments, resource waste
- 오류 전파
  - 애초에 품사 태깅이 잘못되었다던가

### ss.1.5 modeling to reflect context

- 정보의 중요도를 판단

#### sss.1.5.1 bi-rnn

#### sss.1.5.2 transformer

## ch.2. attention

### ss.2.1 attention mechanism

- 단어간의 관계 매칭

#### sss.2.1.1 bi-directional attention

- context to question attention
- question to context attention
- 두 정보를 취합해서 사용
- [Bi-Directional Attention Flow (Seo et al., 2017)]()

#### sss.2.1.2 fully-aware attention

- layer가 깊어질수록 고차원 추상화가 된다는 점을 활용
- input word를 추상화해서 저차원 추상화 정보와 고차원 추상화 정보 간의 관계를 이용
- [FusionNet (Huang et al., 2017)]()

#### sss.2.1.3 self attention

- 앞에는 co-attention이었는데 이거는 같은 문장 내에 다른 단어와의 관계
- [R-Net (Wang et al., 2017)]()
- 주의점 : 같은 문장을 비교하니까 같은 단어를 attention한다
  - sol) 동일한 단어의 위치를 무시하는 heuristic

## ch.3. output

- start-end의 확률 분포를 통해 계산
- pointer networks를 통해 계산

## ch4. learning methods

- negative log probability
- reinforce algorithm : f1 score를 reward로
  - [reinforce algorithm (Williams, 1992)]()
