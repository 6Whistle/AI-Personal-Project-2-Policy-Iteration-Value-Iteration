# 인공지능 Project 2

## 이름 : 이준휘

## 학번 : 2018202046

## 교수 : 박철수 교수님

## 분반 : 금 3, 4


## 1. Introduction

```
해당 과제는 Dynamic Programming(DP)를 이용하여 Policy evaluation, Policy improvement, 그리
고 Value iteration을 구현하는 것이다. 주어진 7 * 7 크기의 grid-world를 사용하며 Action은
상하좌우 4 가지 케이스로 이루어져 있다. 만약 grid-word 밖으로 나가는 action을 취할 경
우 제자리로 돌아오며 중간중간 reward = - 100 인 함정이 존재한다. Agent는 최단거리로 종
료점까지 가는 경로를 학습한다. Policy evaluation의 수렴값과, 업데이트 되는 과정, 그리고
각각의 state에서의 action을 매트릭스로 만들어 보고서에 작성한다.
```
## 2. Algorithm

### a. Dynamic Programming

```
일반적으로 상당수의 분할 정복 알고리즘(Divide-and-Conquer)은 Top-Down 방식으로 이
전에 해결한 문제를 다시 풀 수 있다는 단점을 가지고 있다. 이는 심각한 비효율성으
로 이어진다. 이를 해결하기 위한 Dynamic Programming은 기본적으로 큰 문제를 작은
문제로 나누어 작은 문제에서 구한 solution을 큰 문제에 사용하는 방식은 동일하나
이 과정에서 Memoization, 즉 이미 계산한 결과는 배열에 저장하는 방법을 통해 계산
을 최소화하는 알고리즘이다. 해당 알고리즘은 Top-Down 방식과 Bottom-Up 방식 모두
사용할 수 있으며 주로 Bottom-Up 방식을 활용한다.
```
### b. MDP & Bellman Equation

```
Markov Decision Process란 강화학습의 한 종류로 에이전트가 환경과 상호작용을 하며
보상을 최대화하는 방향으로 학습하는 것을 의미한다. 기본적으로 Markov Property에
의해 이전 시점에만 사건의 영향을 받는다. MDP의 목적은 보상, 즉 reward를 최대화하
는 것이며 이는 𝐺𝑡=𝑅𝑡+ 1 + 𝛾𝑅𝑡+ 2 +⋯ + 𝛾𝑇−𝑡𝑅𝑇로 나타낼 수 있다. 이 때 T가 매우
큰 경우 학습이 더딜 수 있기 때문에 discount factor를 이용하여 미래의 reward를 조정
한다.
Bellman Equation이란 value function을 정리한 것이다. Value function은 에이전트가 놓인
상태 또는 상태와 행동을 함수로 표현한 것이다. 𝑣𝜋(𝑠)=∑𝑎π(s|𝑎)𝑞𝜋(𝑠,𝑎)는 상태에
따른 가치를 나타내며, 𝑞𝜋(𝑠,𝑎)=∑𝑃(𝑠′,𝑎){𝑅+𝛾𝑣𝜋(𝑠′)}의 식으로 행동을 가치를 통해
나타낼 수 있다.
```
### c. Policy Evaluation

```
Policy evaluation이란 value가 어떻게 변할지를 예측(prediction)하는 것으로, 현재 주어진
policy에 대한 true value function을 구하는 것이다. 이 과정에서 Bellman equation을 사용
하며 현재 Policy를 가지고 이전 value를 이용하여 구하는 one step backup을 이용한다.
현재 상태의 value function을 update하는데 reward와 next state들의 value function을 사
용한다. 모든 state를 동시에 한 번씩 업데이트하는 시퀀스를 특정 횟수만큼 반복한다.
이를 무한대까지 계산하게 될 경우 현재 random policy에 대한 true value function을 구
할 수 있고 이를 policy evaluation이라 칭한다.
```
### d. Policy Improvement

```
Policy improvement란 위의 Policy evaluation을 통해 얻은 true value function을 통해 더 나
```

```
은 policy로 update하는 것을 의미한다. 반복의 횟수가 늘어날수록 optimal policy에 가
까워질 것이다. Improve의 방법으로 greedy improvement가 있다. 이는 다음 state 중에서
가장 높은 value fuction을 가진 state로 가는 것으로 max값을 취하는 것과 같다.
```
### e. Value Iteration

```
Value iteration은 Policy iteration과 다르게 Bellman Optimality Equation을 이용한다. 이를
통해 evaluation 과정에서 이동 가능한 state의 모든 value function을 더하여 value를 평
가하는 것과 다르게 이 중 max값을 통해 greedy한 방식으로 value fucntion을 구하여
improve를 진행하는 것이다. 즉 기존 식에서 action을 취할 확률을 곱하여 summation
하는 과정을 대신하여 max 값을 취하는 것으로 대체하면 된다. Policy improvement와의
차이점으로는 Policy Iteration의 경우 policy에 따라 value 값이 확률적으로 주어지지만
value iteration은 deterministic한 action으로 정해진다.
```
## 3. Result

### a. Policy Evaluation


해당 알고리즘은 google colab 환경에서 jupyter notebook을 통해 실행한 결과를 나타낸
다.
getState() 함수는 방향(a)에 따라 이동된 좌표를 반환하는 함수다. 만약 grid-world 밖으
로 나가는 경우에는 멈춘 상태를 유지한다.
GetRewardTable() 함수는 grid-world의 reward table을 만드는 함수다. 기본적으로 reward
는 - 1 로 설정되며 특정 부분에 함정에서는 - 100 으로 설정된다. 마지막으로 도착지의
reward는 0 으로 설정한다.
Policy_eval() 함수는 policy evaluation을 구현한 함수다. Post_value_t에는 grid_size *
grid_size 크기의 - 1 value 값으로 초기화된 table을 생성한다. 그 후 이전
getRewardTable()함수를 통해 reward table을 생성한다. 입력받은 iter의 횟수만큼 반복문
을 반복하게 된다. temp에는 임시로 저장할 다음 value의 값을 나타내는 table이며, 0 으
로 초기화한다. 그 후 각 state에 해대 다음 반복을 수행한다. 만약 현재 위치가 마지막


위치인 경우 temp의 해당 위치는 0 으로 설정하며, 아닐 경우 각 방향( 0 (up), 1(down),
2 (left), 3(right))에 대하여 현재 위치의 reward의 값과 이동한 위치의 value 값을 더하여
이를 0.25(방향이 4 개이기에 1/4)를 누적 시킨다. 위의 반복이 끝나면 temp에 저장했던
table을 post_value_t에 update하는 과정을 반복한다. 그 후 post_value_t를 반환한다.

위의 값은 policy evaluation에서 k=0일 때의 결과를 나타낸 것이다. 이는 기존에 초기화
된 - 1 table을 출력하는 정상적인 모습을 보인다.


#### K = 1로 반복을 하였을 때에는 기존에서 값이 변화된 모습을 볼 수 있다. 모두 이전의

value값을 통해 기대값을 예측한 것을 확인할 수 있으며 특히 함정의 위치의 경우
reward가 - 100 임으로 - 101 로 다른 위치에 비해 크게 나온 것을 알 수 있다. 그리고 결
승점


#### K = 2일 때 결과는 다음과 같다. (0,1)의 값을 분석해 보았을 때 3 * 0.25 * (- 1 - 2) + 0.25 * (-

#### 1 - 101 ) = 2.25 + 25.5 = 27.75로 직접 계산했을 때의 결과와 일치하는 것을 확인할 수 있

#### 다. 이를 통해 해당 결과가 정상적으로 나타나는 것을 알 수 있다.


해당 값은 k = 3일 때의 결과를 나타낸다. 이전보다 값이 증가된 것을 볼 수 있으며
goal에 가깝거나 함정에서 멀수록 value값이 작아지는 것을 확인할 수 있다.



#### 위의 결과는 값이 약 3000 이상으로 커지게 되었을 때 결과가 거의 변하지 않는다는

```
것을 보인다. 즉 이는 policy evaluation을 통해 구한 true value function임을 알 수 있다.
```
### b. Policy Improvement


다음 policy_improv() 함수는 policy_evaluation을 통해 구한 true function을 이용하여
policy_improvement를 구하는 함수다. 각 state에 대하여 도착지의 경우 policy을 0 으로
설정하며, 도착지가 아닌 경우 max값의 수를 확인하여 이를 확률로 표기하는 greedy한
방법을 확인한다.


다음은 0 에서의 value와 action을 나타낸 것이다. Action의 경우 모든 value가 동일하기에
policy는 1/4으로 설정된 것을 볼 수 있으며, 도착지의 경우 0, 0, 0, 0으로 이동하지 않는
것을 알 수 있다. 또한 policy를 통해 가능한 action을 나타낼 경우 다음과 같이 나타낼
수 있다.


다음은 k = 1일 때의 결과를 나타낸다. 이전과 다르게 (0, 1) 위치와 같은 경우 (0, 2) 위
치로 이동하는 것이 value가 작아지기에 해당 방향으로 이동하는 확률이 사라진 것을
확인할 수 있다. 이에 action을 확인할 경우 R이 X로 사라진 것을 확인할 수 있다. 이
외의 방향에서는 value가 같기에 모두 이동할 수 있도록 나온다.


다음 사진은 k = 2일 때를 나타낸다. 이 때 (0, 1) 위치를 보았을 때 왼쪽의 값이 가장
크기 때문에 greedy에서 왼쪽으로 이동하는 확률만 남았으며, 이는 action에도 반영된
것을 확인할 수 있다.


다음 결과는 k = 3일 때로 반복이 진행됨의 따라 policy가 optimize되고 있는 과정을 눈
으로 확인할 수 있다.


```
다음 결과는 true value function에서의 policy을 나타낸 것이다. 모든 state에서 하나의
state의 확률만 1 이며, start에서 출발하였을 때 종료지점까지 정확히 도달하는 모습을
볼 수 있다. 이를 통해 policy improvement 에서 true value function 으로 값이 수렴된
value값을 사용한 policy improvement는 optimal policy를 가진다는 것을 알 수 있다.
```
### c. Value Iteration


Value_iter() 함수는 value iteration을 구현한 함수다. 이전의 함수와 마찬가지로
post_value_t와 action_t, reward_t를 초기화한 후 i횟수만큼 반복을 수행한다. 각 state에
대하여 종료지점일 경우 temp와 action을 0 으로 설정하며, 이외의 경우에 대해서는
action_t에 다음 state의 기댓값을 저장한다. 이후 이 중 가장 큰 기대값만을 temp에 저
장한 후 temp와 같은 action_t 값을 1 로, 이외의 경우를 0 으로 설정한다.


다음은 k = 0일 때의 결과를 나타낸다. Value 값은 0 으로 초기화되어 있으며 action은 도
착지를 제외하곤 1 모든 방향에 대해 1 로 설정되어 있다.


다음 결과는 k = 1일 때의 결과다. (0, 1)을 보았을 때 이전 value 값은 모두 - 1 이었기 때
문에 value 값이 우측으로 - 101 이더라도 우측으로 아직 이동할 수 있는 상태를 보인다.
Value 값을 검증해보았을 때 각 이전 state는 모두 - 1 이기 때문에 - 1 + 현재 reward 값만
이 나올 수 밖에 없다. 이는 모든 방향이 동일하기 때문에 모든 방향기 이동 가능한
것이다.


다음은 k = 2일 때의 결과를 나타낸 것이다. 이 때 기존과 다르게 특정 이동방향이 막
힌 것을 확인할 수 있다. 또한 value의 경우 함정과 붙어있어도 - 3 으로 값이 이전과 다
르게 급격히 변하지 않은 모습이다. 이를 통해 주변 value 중 가장 max값을 취하며 이
동하고 있다는 것을 확인할 수 있다. 특히 도착 위치 주변은 value가 크기 때문에 - 1 로
값이 주변보다 작은 것을 확인할 수 있다.


K = 3일 때의 그림을 보았을 때 두드러지는 부분으로는 도착지점 주위로 갈수록 value
가 점점 작아지는 모습을 띈다는 것이다. 이를 통해 k를 특정 값 이상 늘리면 특정 값
으로 수렴할 것을 예측할 수 있다.


```
다음은 k=13으로 하였을 때의 결과로 value 값이 수렴한 case다. 해당 반복 이상으로
확인하더라도 같은 결과가 나온다. Value를 확인하였을 때 함정을 제외하고 도착지에
가까울수록 value가 작아지며 이에 따라 방향도 value가 작은 방향으로 결정되었다. 해
당 방식에서는 이전과 달리 방향이 2 개정도로 결정되는 경우도 많이 보이는 모습을
보였다. 또한 start에서 이동을 추적할 시 도착지로 가는 것을 확인하였다. 이를 통해
해당 알고리즘이 제대로 구현되었음을 확인하였다.
```
## 4. Consideration

```
해당 과제를 통해 Dynamic Programming을 사용하여 강화학습을 구현할 수 있다는 점을 알
수 있었으며, 이전의 value 1 step만을 저장하면 된다는 사실을 알았다. 특히 Markov
Decision Process와 Bellman Equation의 식에 대해 이해도를 높일 수 있었다. 그리고 Policy
```

Evaluation을 통해 직접 value가 어떻게 변하는지 눈으로 확인할 수 있었으며, Policy
improvement와 Value iteration을 직접 구현하여 비교함으로써, 두 모델의 세부적인 차이를

## 이해할 수 있었다. 마지막으로 이를 통해 시험 준비를 같이 할 수 있었던 과제였다.


