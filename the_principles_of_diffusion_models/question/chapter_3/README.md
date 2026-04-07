# Chapter 3 Questions

---

## Q. Higher Order Tweedie's Formula 에서 말하고자 하는 바가 무엇일까?

핵심은 classical Tweedie's formula를 posterior mean에서 끝내지 말고, posterior의 더 많은 통계량까지 확장해서 보자는 것입니다. 여기서 $\mathbf{x}$와 $\mathbf{\eta}$는 모두 다차원 벡터로 이해하면 됩니다.

classical Tweedie's formula는 noisy observation $\tilde{\mathbf{x}}$에 대해 score

$$
\nabla_{\tilde{\mathbf{x}}} \log p(\tilde{\mathbf{x}})
$$

를 이용해서 posterior mean

$$
\mathbb{E}[\mathbf{x}_0 \mid \tilde{\mathbf{x}}]
$$

를 표현할 수 있다고 말합니다.

Higher Order Tweedie's Formula는 여기서 한 걸음 더 나아가, posterior covariance와 그보다 더 높은 cumulants도 $\log p(\tilde{\mathbf{x}})$의 higher derivatives로 표현할 수 있다고 말합니다.

먼저 noisy model을 exponential family 형태로 씁니다.

$$
q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{\eta}) = \exp(\mathbf{\eta}^\top \tilde{\mathbf{x}} - \psi(\mathbf{\eta})) q_0(\tilde{\mathbf{x}})
$$

그리고

$$
\lambda(\tilde{\mathbf{x}}) := \log p_\sigma(\tilde{\mathbf{x}}) - \log q_0(\tilde{\mathbf{x}})
$$

를 정의하면, posterior $p(\mathbf{\eta} \mid \tilde{\mathbf{x}})$의 log-partition 역할을 하는 이 $\lambda(\tilde{\mathbf{x}})$의 미분들이 posterior cumulants를 만들어냅니다. 즉, $\mathbf{\eta} \in \mathbb{R}^D$인 자연 파라미터 벡터의 posterior 통계량을 $\tilde{\mathbf{x}}$에 대한 미분으로 읽어낼 수 있다는 뜻입니다.

즉,

$$
\nabla_{\tilde{\mathbf{x}}} \lambda(\tilde{\mathbf{x}}) = \mathbb{E}[\mathbf{\eta} \mid \tilde{\mathbf{x}}],
$$

$$
\nabla^2_{\tilde{\mathbf{x}}} \lambda(\tilde{\mathbf{x}}) = \operatorname{Cov}[\mathbf{\eta} \mid \tilde{\mathbf{x}}],
$$

그리고 더 높은 차수의 미분은 더 높은 차수의 conditional cumulants에 대응합니다.

Gaussian location model에서는 $\mathbf{\eta} = \mathbf{x} / \sigma^2$이므로 익숙한 형태로 다시 쓸 수 있습니다.

$$
\mathbb{E}[\mathbf{x} \mid \tilde{\mathbf{x}}] = \tilde{\mathbf{x}} + \sigma^2 \nabla_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}),
$$

$$
\operatorname{Cov}[\mathbf{x} \mid \tilde{\mathbf{x}}] = \sigma^2 I + \sigma^4 \nabla^2_{\tilde{\mathbf{x}}} \log p_\sigma(\tilde{\mathbf{x}}).
$$

여기서 posterior mean은 "$\tilde{\mathbf{x}}$가 주어졌을 때 가장 대표적인 복원 벡터가 어디냐"를 뜻합니다.

반면 posterior covariance는 "$\tilde{\mathbf{x}}$를 봤을 때 원래 $\mathbf{x}$가 그 평균 주변에서 얼마나 퍼져 있을 수 있느냐", 그리고 "각 좌표나 성분들이 서로 얼마나 함께 움직이느냐"를 뜻합니다.

즉, covariance는 단순한 평균 복원값 하나가 아니라 복원의 불확실성 구조를 나타냅니다. 어떤 방향은 거의 확실해서 분산이 작고, 어떤 방향은 애매해서 분산이 클 수 있으며, 두 성분이 함께 커지거나 작아지는 경향이 있으면 공분산이 생깁니다.

그보다 더 높은 cumulants는 평균과 공분산만으로는 설명되지 않는 분포의 모양을 뜻합니다.

- 1차 cumulant는 평균입니다.
- 2차 cumulant는 분산 또는 공분산입니다.
- 3차 cumulant는 비대칭성(skewness)과 관련 있습니다.
- 4차 cumulant는 뾰족함이나 꼬리(kurtosis)와 관련 있습니다.
- 더 높은 cumulants는 더 복잡한 비가우시안 구조를 나타냅니다.

즉, higher cumulants를 본다는 것은 posterior가 단순히 "평균 주변에 타원형으로 퍼진 가우시안"이 아니라, 치우쳐 있거나, 더 뾰족하거나, 더 두꺼운 꼬리를 가지는지까지 본다는 뜻입니다.

따라서 이 부분이 말하고자 하는 핵심은, score는 단순히 denoising mean만 알려주는 것이 아니라, Hessian은 uncertainty를, 더 높은 derivatives는 posterior shape의 더 미세한 정보까지 담고 있다는 것입니다.

그래서 이 섹션의 의미는 다음처럼 이해하면 됩니다.

- score matching이 단지 평균 복원과만 연결되는 것이 아니라 posterior statistics 전체와 연결된다는 점을 보여줍니다.
- 1차 score는 mean, 2차 score는 covariance, 고차 score는 higher cumulants를 준다는 통일된 관점을 제공합니다.
- diffusion이나 denoising 문제에서 $\log p_\sigma(\tilde{\mathbf{x}})$의 미분을 잘 추정하면, 단순 복원뿐 아니라 uncertainty와 posterior geometry까지 읽어낼 수 있다는 메시지를 줍니다.

한 줄로 요약하면, Higher Order Tweedie's Formula는 "score는 posterior mean의 도구일 뿐 아니라 posterior 전체 구조를 담고 있는 통계량"이라는 점을 보여주려는 것입니다.

---

## Q. EBM에서 $\tilde{p}_\phi(x)$는 계산 가능한데 왜 $p_\phi(x)$는 intractable할까? 또 비율만 알면 확률을 복원할 수 있는 것 아닌가?

맞습니다. 여기서 구분해야 하는 핵심은 "각 점에서의 unnormalized density는 계산 가능하다"와 "그걸 전체 확률분포로 정규화할 수 있다"는 서로 다른 문제라는 점입니다.

EBM에서는 보통

$$
\tilde{p}_\phi(x) = \exp(-E_\phi(x))
$$

로 unnormalized density를 정의합니다.

이 값 자체는 tractable합니다. 왜냐하면 주어진 $x$에 대해 neural network로 $E_\phi(x)$를 계산하고, 거기에 $\exp$를 씌우면 되기 때문입니다. 즉, 특정한 입력 $x$ 하나에 대해서는 $\tilde{p}_\phi(x)$를 바로 계산할 수 있습니다.

하지만 실제 확률분포는

$$
p_\phi(x) = \frac{\tilde{p}_\phi(x)}{Z_\phi}, \qquad Z_\phi = \int \tilde{p}_\phi(x) \, dx
$$

처럼 partition function $Z_\phi$로 나누어야 합니다.

문제는 바로 이

$$
Z_\phi = \int \exp(-E_\phi(x)) \, dx
$$

를 계산하는 것입니다.

이산 유한 상태라면 이 문제가 없습니다. 예를 들어 $x \in \{1,2,3\}$이고 비율이

$$
\tilde{p}(1) : \tilde{p}(2) : \tilde{p}(3) = 1:2:3
$$

이면

$$
Z = 1 + 2 + 3 = 6
$$

이라서

$$
p(1)=\frac{1}{6}, \qquad p(2)=\frac{2}{6}, \qquad p(3)=\frac{3}{6}
$$

처럼 바로 정규화할 수 있습니다. 이 경우에는 "비율만 알면 확률을 복원할 수 있다"는 말이 정확히 맞습니다. 상태가 3개뿐이어서 분모를 전부 더할 수 있기 때문입니다.

하지만 EBM의 $x$는 보통 연속 고차원 벡터입니다. 이미지라면 $x \in \mathbb{R}^D$이고 $D$가 매우 큽니다. 이때는

$$
Z_\phi = \int_{\mathbb{R}^D} \exp(-E_\phi(x)) \, dx
$$

를 계산해야 하는데, 이것은 "모든 가능한 $x$에 대해 더한다"는 뜻입니다. 연속 공간에서는 점이 무한히 많고, 고차원에서는 격자로 근사하려 해도 필요한 점의 수가 차원에 따라 폭발적으로 증가합니다. 즉, 비율을 아는 것 자체는 충분하지 않고, 그 비율을 전체 공간에서 합친 정규화 상수를 알아야 실제 확률을 만들 수 있는데, 그 단계가 intractable한 것입니다.

그렇다면 "샘플을 몇 개만 뽑아서 분모를 근사하면 되지 않나?"라는 생각이 자연스럽게 나옵니다. 이를 쓰면

$$
Z_\phi = \int \exp(-E_\phi(x)) \, dx \approx \frac{1}{N}\sum_{i=1}^N \exp(-E_\phi(x_i))
$$

처럼 Monte Carlo 적분을 하고 싶어집니다.

방향 자체는 맞지만, 여기서 핵심 문제는 $x_i$를 어디서 샘플링하느냐입니다.

uniform distribution에서 뽑으면, 고차원 공간에서는 대부분의 샘플이 실제 데이터 manifold와 전혀 상관없는 곳에 떨어집니다. 그런 점들에서는 보통

$$
\exp(-E_\phi(x)) \approx 0
$$

가 되어 추정량이 매우 불안정해지고, variance가 커져서 쓸 만한 추정이 되기 어렵습니다.

그럼 좋은 샘플을 얻기 위해 $p_\phi$ 자체에서 뽑으면 될 것 같지만,

$$
p_\phi(x) = \frac{\exp(-E_\phi(x))}{Z_\phi}
$$

이므로 $p_\phi$에서 샘플링하려면 이미 $Z_\phi$를 알아야 합니다. 순환 논리가 생깁니다.

또 학습 데이터, 즉 $p_{\mathrm{data}}$에서 샘플링하는 것도 직접적인 해결책은 아닙니다. 일반적으로 샘플 분포가 $q(x)$일 때는

$$
\int f(x)\,dx = \mathbb{E}_{q(x)}\left[\frac{f(x)}{q(x)}\right]
$$

로 써야 하므로, $q = p_{\mathrm{data}}$라면

$$
\frac{\exp(-E_\phi(x))}{p_{\mathrm{data}}(x)}
$$

를 평균내야 합니다. 그런데 $p_{\mathrm{data}}(x)$ 자체를 모르기 때문에 이 방법도 곧바로 쓸 수 없습니다.

정리하면 다음과 같습니다.

- $\tilde{p}_\phi(x)=\exp(-E_\phi(x))$는 각 $x$에 대해 계산 가능하므로 tractable합니다.
- 그러나 실제 확률 $p_\phi(x)$를 얻으려면 전체 공간에 대한 정규화 상수 $Z_\phi$가 필요합니다.
- 유한한 이산 상태공간에서는 $Z$를 전부 더할 수 있으므로 비율만으로 확률을 복원할 수 있습니다.
- 연속 고차원 공간에서는 그 "전부 더하기"가 사실상 불가능하므로 $p_\phi(x)$의 exact computation이 intractable합니다.

한 줄로 말하면, 네 직관은 유한 이산 공간에서는 완전히 맞고, EBM에서 어려운 이유는 개념이 달라서가 아니라 상태공간이 너무 커서 정규화 상수를 계산할 수 없기 때문입니다.

---

## Q. Page 68에서 trace를 기댓값으로 바꿨을 뿐인데, 왜 계산량이 줄어들까?

trace 자체를 더하는 연산이 어려운 것은 아닙니다. 진짜 비싼 것은

$$
\operatorname{Tr}(\nabla_x s_\phi(x))=\sum_{i=1}^D \frac{\partial s_{\phi,i}(x)}{\partial x_i}
$$

에서 각 대각 원소

$$
\frac{\partial s_{\phi,i}(x)}{\partial x_i}
$$

를 전부 구하는 과정입니다. 여기서 $s_\phi(x)\in\mathbb{R}^D$ 이므로 $\nabla_x s_\phi(x)$는 $D\times D$ Jacobian입니다.

중요한 점은 보통 Jacobian 전체를 미리 만들어 두지 않는다는 것입니다. 대신 필요한 방향벡터 $v$가 들어올 때마다 자동미분으로

$$
Jv
$$

또는

$$
v^\top J
$$

를 계산합니다.

exact trace를 구하려면 basis vector $e_1,\dots,e_D$를 써서

$$
\operatorname{Tr}(J)=\sum_{i=1}^D e_i^\top J e_i
$$

를 계산해야 하므로, 결국 각 $i$마다 $J e_i$를 구해서 대각 원소 $J_{ii}$를 뽑는 작업을 반복하게 됩니다. 즉, 방향별 미분 연산이 $D$번 필요합니다.

Hutchinson estimator는 이 trace를

$$
\operatorname{Tr}(J)=\mathbb{E}_u[u^\top J u]
$$

$$
\mathbb{E}[uu^\top]=I
$$

로 바꿉니다. 그러면 basis vector를 전부 쓰는 대신 랜덤 벡터 $u_1,\dots,u_K$만 뽑아서

$$
\operatorname{Tr}(J)\approx \frac{1}{K}\sum_{k=1}^K u_k^\top J u_k
$$

로 근사할 수 있습니다. 이때도 미분은 여전히 필요하지만, full Jacobian을 다루는 것이 아니라 랜덤 방향에 대한 Jacobian-vector product만 계산하면 됩니다.

따라서 차이는 다음과 같습니다.

- exact 방식: basis 방향 $e_i$를 모두 써야 해서 미분 관련 계산이 $D$번 필요합니다.
- Hutchinson 방식: 랜덤 방향 $u_k$ 몇 개만 써서 미분 관련 계산이 $K$번 필요합니다.
- 보통 $K \ll D$ 이므로 계산량이 크게 줄어듭니다.

한 줄로 요약하면, 둘 다 score의 미분은 하지만 exact는 모든 좌표축 방향을 다 봐야 하고 Hutchinson은 랜덤 방향 몇 개만 보면 되기 때문에, full $D\times D$ Jacobian을 만들지 않고도 훨씬 싸게 trace를 추정할 수 있습니다.

---

## Q. Langevin Dynamics와 Tweedie's Formula는 어떤 관계가 있을까? score-based model에서는 두 식이 sampling에 쓰이는지, training에 쓰이는지, 그리고 어떤 역할을 하는지?

둘은 서로 경쟁하는 개념이 아니라, **같은 score를 서로 다른 방식으로 사용하는 두 관점**이라고 보면 됩니다.

이 장의 score-based model에서 핵심 객체는 score

$$
s_\theta(x,\sigma) \approx \nabla_x \log p_\sigma(x)
$$

입니다. 여기서 $p_\sigma$는 데이터에 Gaussian noise가 섞인 noisy distribution입니다.

Langevin Dynamics는 이 score를 **샘플링에 직접 사용하는 알고리즘**입니다. 기본 형태는

$$
x_{k+1} = x_k + \frac{\epsilon}{2} s_\theta(x_k,\sigma) + \sqrt{\epsilon}\, z_k
$$

$$
z_k \sim \mathcal{N}(0,I)
$$

처럼 쓰입니다. 여기서 score는 $\log p_\sigma(x)$의 gradient이므로, 샘플을 확률밀도가 높은 방향으로 밀어 주는 drift 역할을 합니다. 즉, Langevin Dynamics는 학습된 score field를 따라가며 점차 데이터 같은 샘플을 만들어내는 **sampling procedure**입니다.

반면 Tweedie's Formula는 score의 **통계적 의미를 해석해 주는 공식**입니다. Gaussian noise 모델에서

$$
\tilde{x} = x_0 + \sigma \varepsilon
$$

$$
\varepsilon \sim \mathcal{N}(0,I)
$$

이면

$$
\mathbb{E}[x_0 \mid \tilde{x}]=\tilde{x} + \sigma^2 \nabla_{\tilde{x}} \log p_\sigma(\tilde{x})
$$

가 성립합니다. 즉, noisy sample $\tilde{x}$에서 clean data $x_0$의 posterior mean을 복원하려면 noisy density의 score만 알면 된다는 뜻입니다.

그래서 Tweedie's Formula는 "score가 단순한 gradient가 아니라, 사실상 denoising 정보도 담고 있다"는 것을 보여줍니다. 즉, 같은 score가

- 한편으로는 Langevin Dynamics에서 샘플을 움직이는 힘으로 쓰이고,
- 다른 한편으로는 Tweedie's Formula에서 clean sample의 추정값을 주는 denoiser로 해석됩니다.

둘의 관계를 한 문장으로 말하면 이렇습니다.

> Langevin Dynamics는 score를 사용해 샘플을 생성하고, Tweedie's Formula는 그 score가 왜 denoising에 해당하는지를 설명한다.

그럼 training과 sampling 중 어디에 쓰이느냐를 나누어 보면 다음과 같습니다.

- Langevin Dynamics: 주로 **sampling**에 쓰입니다.
- Tweedie's Formula: 주로 **해석과 연결고리** 역할을 합니다.

조금 더 정확히 말하면, score-based model의 training에서는 보통 denoising score matching을 통해

$$
s_\theta(x,\sigma) \approx \nabla_x \log p_\sigma(x)
$$

를 학습합니다. 이때 Tweedie's Formula는 "왜 noisy score를 배우는 것이 denoising과 연결되는가?"를 이해하게 해 줍니다. 즉, training objective 자체가 Tweedie's Formula를 직접 최적화하는 것은 아니지만, 학습된 score가 clean data 복원 정보를 담는다는 해석을 제공합니다.

반대로 sampling 단계에서는 학습된 score를 실제로 이용해 Langevin Dynamics나 Annealed Langevin Dynamics를 돌립니다. 즉, 모델이 배운 score field를 따라 노이즈에서 데이터 쪽으로 샘플을 이동시키는 데 사용됩니다.

정리하면 역할 분담은 다음과 같습니다.

- training: denoising score matching으로 noisy distribution의 score를 학습합니다.
- Tweedie's Formula: 그 score가 posterior mean denoiser와 연결된다는 이론적 해석을 줍니다.
- sampling: 학습된 score를 이용해 Langevin Dynamics로 샘플을 생성합니다.

한 줄 요약하면, Tweedie's Formula는 "배운 score가 무엇을 의미하는가"를 설명하고, Langevin Dynamics는 "배운 score를 가지고 실제로 어떻게 샘플을 만들 것인가"를 담당합니다.

---
