# Chapter 2 Questions

---

## Q. 왜 $P(x)$는 tractable하다고 하나?

확률론이나 머신러닝에서 어떤 확률 또는 분포가 tractable하다는 말은, 핵심적으로 그 값이 유한한 시간 안에 계산 가능한지를 뜻합니다.

보통 다음 세 가지 관점에서 판단합니다.

적분 또는 합의 계산 가능성입니다. 확률 밀도 함수에서 정규화 상수(normalizing constant)를 계산할 수 있으면 tractable하고, 상태 공간이 너무 크거나 적분이 닫힌 형식으로 떨어지지 않으면 intractable합니다.

주변화(marginalization)의 난이도입니다. 결합분포 $P(x, y)$에서 일부 변수를 제거해 원하는 분포를 구하는 계산이 쉬우면 tractable하고, 많은 변수들이 복잡하게 얽혀 있으면 intractable합니다.

사후확률(posterior)의 도출 가능성입니다. 베이즈 정리

$$
P(z \mid x) = \frac{P(x \mid z)P(z)}{P(x)}
$$

에서 분모 $P(x)$를 계산할 수 없으면 posterior를 직접 구할 수 없고, 이 경우 보통 intractable하다고 합니다.

요약하면 tractable은 직접 계산 가능한 경우이고, intractable은 정답을 바로 구하기 어려워서 MCMC나 Variational Inference 같은 근사 방법이 필요한 경우입니다.

---

## Q. $P(x)$에서 $x$를 무한대로 수집할 때도 tractable하다고 할 수 있나?

이론적으로는 보통 계산 복잡도 기준으로 polynomial time 안에 풀 수 있으면 tractable하다고 말합니다.

다만 실제 문맥에서는 "현실적으로 계산 가능하냐"라는 의미로 더 느슨하게 쓰이기도 합니다. 그래서 형식적으로는 tractable 분류에 들어가더라도, 실제로는 거의 불가능에 가까운 계산이라는 의미로 intractable에 가깝게 말하는 경우도 있습니다.

---

## Q. $P(x)$의 정확한 의미는?

MNIST를 예로 들면, $P(x)$는 $28 \times 28$ 픽셀 손글씨 숫자 이미지 $x$가 데이터 분포에서 나타날 확률을 의미합니다.

즉, $P(x)$는 개별 샘플 하나의 확률이라기보다 데이터셋 전체를 생성하는 underlying distribution의 관점에서 이해하는 것이 자연스럽습니다.

---

## Q. richer posterior class란 무엇인가?

posterior는 보통 $p(z \mid x)$ 같은 조건부분포를 뜻합니다. 그런데 실제 모델링에서는 이 posterior를 정확히 쓰기 어렵거나, 학습을 위해 특정한 형태로 제한해야 하는 경우가 많습니다.

이때 posterior class는 우리가 허용하는 posterior 후보들의 집합을 뜻합니다.

Variational Inference에서는

$$
q_\phi(\cdot \mid x) : \phi \in \Phi
$$

같은 형태의 근사 posterior family를 두고, 예를 들어 대각 가우시안, full-covariance Gaussian, normalizing flow 등이 posterior class가 됩니다.

diffusion 맥락에서 richer하다는 말은 더 복잡하고 다양한 조건분포를 표현할 수 있다는 뜻입니다.

많은 DDPM 계열 모델은 reverse step을 다음과 같이 단순한 가우시안으로 둡니다.

$$
p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t^2 I)
$$

즉 “가우시안(종종 분산은 고정/간단한 형태)”이라는 좁은 class로 제한합니다.

그런데 실제 $x_{t-1} \mid x_t$는 데이터와 시간에 따라 더 복잡할 수 있고, 특히 멀티모달 상황에서는 단일 가우시안이 표현력이 부족할 수 있습니다.

그래서 “richer posterior class”라고 하면 보통 예를 들어 다음과 같은 형태를 의미합니다.

- Mixture of Gaussians
- Full-covariance Gaussian
- Normalizing Flow로 만든 조건분포
- 더 일반적인 비가우시안 또는 implicit 조건분포

즉, richer posterior class는 posterior를 더 유연하고 강하게 표현할 수 있도록 분포 family의 제약을 푸는 것을 의미합니다.

---

## Q. Lemma 2.2.2가 성립하는 이유는? 조건부 순서를 바꾸면 결론도 달라지나?

결론부터 말하면 식의 전개 형태는 달라져도 값은 같습니다.

$P(A \mid B, C)$와 $P(A \mid C, B)$는 서로 다른 조건이 아니라 같은 결합사건 $B \cap C$를 조건으로 본 것이기 때문입니다.

정의로부터 쓰면

$$
P(A \mid B, C) = \frac{P(A, B, C)}{P(B, C)}
$$

$$
P(A \mid C, B) = \frac{P(A, C, B)}{P(C, B)}
$$

이고, 결합사건은 순서를 바꿔도 동일하므로 두 값은 같습니다.

겉으로 보기에는 베이즈 전개식이 다르게 보일 수 있습니다.

$$
P(A \mid B, C) = \frac{P(B \mid A, C) P(A \mid C)}{P(B \mid C)}
$$

$$
P(A \mid C, B) = \frac{P(C \mid A, B) P(A \mid B)}{P(C \mid B)}
$$

하지만 이는 같은 결합확률 $P(A, B, C)$를 서로 다른 순서로 인수분해한 것뿐입니다. 따라서 최종값은 동일합니다.

즉, Lemma 2.2.2에서 조건부의 나열 순서를 바꾼다고 해서 결론 자체가 바뀌지는 않습니다.

---

## Q. 디퓨전 모델에서 EMA는 어떤 역할을 하나?

참고: [Karras et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Karras_Analyzing_and_Improving_the_Training_Dynamics_of_Diffusion_Models_CVPR_2024_paper.pdf)

EMA는 보통 학습 중 파라미터의 변동을 평균화해서 더 안정적인 평가용 파라미터를 만드는 데 사용됩니다. 실질적으로는 overfitting 완화와 학습 안정화에 기여한다고 볼 수 있습니다.

그런데 diffusion에서 일반적으로 사용하는 best practice는 아닌듯 합니다.

---

## Q. DDPM $u$-prediction 학습이 불안정한 이유는?

DDPM loss식을 보면 weight가 앞에 있습니다.

epsilon-prediction의 경우 simple한 MSE이고, target 분포가 standard Gaussian으로 일정합니다.

그러나 $u$-prediction의 경우는 time마다 target scale이 달라서 학습이 불안정합니다. 실제로 DDPM 논문의 표를 보면 결과를 얻을 수 없었다고 나옵니다.

---

## Q. 35-36p Information-Theoretic View: ELBO as a Divergence Bound 이해가 어려움

논문 설명대로만 보면 ELBO를 최대화하는 게 true modeling error와는 무관한 inference error만 감소시키는 것처럼 보일 수 있습니다.

이 부분은 그냥 "total error bound, true modeling error, inference error의 관계식에서 ELBO가 어떤 항을 차지하는지 한번 알아봤다" 정도로 이해하고 넘어가면 될 것 같습니다.

그리고 실제로는 ELBO를 최대화하기 위해 업데이트되는 parameters는 true modeling error에도 영향을 줍니다.

---

## Q. 41p HVAE ELBO의 KL 항이 이해가 어려움

encoder의 $(i \mid i - 1)$과 매칭되는 decoder의 layer가 $(i - 1 \mid i)$가 아니고 $(i \mid i + 1)$인 이유는?

답변 없음.

---

## Open Question. DDPM decoder 관련 궁금증

`"변화가 작은 diffusion process는 Gaussian distribution을 따르고, 역방향은 정방향과 함수 형태가 동일하다"`는 (Feller, 1949) 정리에서 "변화가 작은 diffusion process는 Gaussian distribution을 따른다"는 것은 정확히는 continuous diffusion process를 표현하는 Brownian motion의 미세한 시간 동안의 변화량이 Gaussian을 따른다는 뜻으로 이해할 수 있습니다.

여기서 continuous diffusion process를 Brownian motion이 아닌 다른 방식으로 표현하는 것도 가능할지, 예를 들어 Lévy process 같은 것으로도 표현할 수 있을지는 열린 질문입니다.

---

## Q. 45-54p DDPM decoder의 학습 목표 관련 의문

생성모델은 MLE 혹은 ELBO만 잘 최대화하면 되고, 실제로 ELBO를 전개해보면 decoder가 근사해야 하는 건 $x_0$에 conditional한 reverse conditional transition kernels인데, 왜 DDPMs를 설명할 때는 "reverse transition kernels"를 근사해야 한다는 목표를 내세우는가?

두 가지 이유가 있을 것 같습니다.

DDPM 이전의 DPM 논문인 Sohl-Dickstein et al., 2015는 원래 non-equilibrium thermodynamics에서 출발했고, 이때는 오히려 ELBO는 관심 밖이었고 물리학에 기반한 diffusion process의 역방향 궤적을 찾는 것이 연구 목표였습니다.

그래서 "변화가 작은 diffusion process는 Gaussian distribution을 따르고, 역방향은 정방향과 함수 형태가 동일하다"는 reverse transition kernels에 관한 정리 (Feller, 1949)를 근거로 decoder를 Gaussian으로 가정한 것이 먼저였고, reverse conditional transition kernels은 이후에 학습을 하긴 해야 하니까 ELBO를 전개해봤을 때 도출되는 항일 뿐이었습니다. 즉, 애초에 목표가 reverse transition kernels를 근사하는 decoder를 ELBO로 학습시키자는 것이었습니다.

그러면 만약 decoder가 ELBO를 최대화해야 한다는 점만 가지고 얘기를 풀어나가면 어떻게 될까요. 실제로 DDPM 논문은 ELBO부터 전개를 해서 decoder가 reverse conditional transition kernels를 근사해야 한다는 사실부터 도출합니다. 그렇게 되면,

a) ELBO 전개
b) decoder가 reverse conditional transition kernels를 근사해야 한다는 KL 항이 나옴
c) 이미 forward process는 Gaussian으로 정의했기 때문에 reverse conditional transition kernels도 Gaussian임
d) reverse conditional transition kernels를 근사해야 하는 decoder도 Gaussian으로 가정할 수 있음

이렇게 위에서 언급한 정리 (Feller, 1949)를 사용하지 않고도 decoder를 Gaussian으로 가정할 수 있고, 따라서 reverse transition kernels를 근사하는 decoder라는 전제조건이 전혀 필요하지 않아 보입니다.

하지만 여기에는 논리적 비약이 있습니다. 애초에 reverse conditional transition kernels는 $x_0$를 알고 있고 decoder는 $x_0$를 모르기 때문에, 실제로 학습 시 decoder는 가능성 있는 모든 $x_0$에 대한 reverse conditional transition kernels의 가중 평균, 즉 mixture of Gaussians를 학습해야 한다는 설명이 가능합니다.

reverse conditional transition kernels가 단일 Gaussian이라고 해서 decoder도 단일 Gaussian으로 가정할 수는 없다는 것입니다.

결국 decoder가 Gaussian이 되기 위해서는,

c' ) decoder가 reverse conditional transition kernels를 근사하는 것은 reverse transition kernels를 근사하는 것과 같음을 증명
d' ) (Feller, 1949)에 따르면, 잘게 쪼개진 diffusion process의 reverse transition kernels는 forward process와 함수 형태가 동일함, 즉 Gaussian임
e) reverse transition kernels를 근사해야 하는 decoder도 Gaussian으로 가정할 수 있음

이렇게 (Feller, 1949) 정리를 사용해야 하고, decoder가 reverse transition kernels를 근사해야 한다는 해당 kernels와의 연관성이 꼭 언급되어야 합니다.

다음 문장도 이런 맥락에서 이해하는 것이 자연스럽습니다.

Leveraging the gradient-level equivalence as in Theorem 2.2.1 and the Gaussian form of the reverse conditional $p(x_{i-1}\mid x_i, x)$ as in Lemma 2.2.2, DDPM assumes that each reverse transition $p_\phi(x_{i-1}\mid x_i)$ is Gaussian

즉, 이 문장은 단순히 Lemma 2.2.2 때문에 바로 reverse transition이 Gaussian이라는 뜻이라기보다, reverse conditional과 reverse transition 사이의 연결을 전제로 읽어야 합니다.

---
