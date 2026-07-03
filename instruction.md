`Task Introduction` 

TTT는 attention을 대체하는 방법론으로, linear time complexity을 가지고 있어. 

TTT는 quadratic time complexity인 attention을 대체하는 방법으로 주목을 받았지.

하지만, Attention에는 camera positional embedding을 통해 camera 정보를 주입하는 방법(e.g. CaPE, GTA, PRoPE, RayRoPE)이 잘 연구된 반면

TTT는 camera 정보을 어떻게 주입하는지에 대한 연구는 이루어 지지 않았어. 

`Related Work` Prope (논문명: Cameras as Relative Positional Encoding)

Prope (Projection Rotary Positional Embedding)는 attention에서 camera 정보를 주입하는 가장 최신의 연구야. 

구체적으로 Prope는 attention 이전 q, k, v에 embedding matrix를 곱해서 camera 정보를 입혀. (Prope는 매번 attention 할때 마다 적용이 돼.)

이때 attention 과정에서 query key에 곱해진 embedding matrix가 relative하게 표현돼.

또한 value 곱해진 embedding matrix도 relative하게 표현되지.

이때 embedding matrix의 구성을 projection matrix (extrinsic과 intrinsic 모두 포함함)로 만들어서 

따라서 camera 정보를 relative하게 볼 수 있다는 장점있어. 

하지만, Prope는 attention 메커지즘에만 적용되고, TTT로는 확장이 되지 못해. 

`Related Work` RayRoPE

RayRope도 attention에서 camera 정보를 주입하는 가장 최신의 연구야. 

Proejction matrix를 사용했던 RoPE와 달리 RayRoPE는 depth 예측을 통해 3D point를 기반으로 conditioning을 했어.

`Related Work` TTT (논문명: Learning to (learn at test time): Rnns with expressive hidden states)

TTT layer는 test time training layer의 약자로 attention 과정을 모방하는 layer야 연구야.

q, k, v가 주어졌을 때, TTT layer는 key와 value를 통해 backpropagation을 통해 한번 update돼. 

구체적으로 key를 input으로 넣고, TTT layer (mlp) 를 통해 output을 예측하면 이것을 value와 loss를 통해 update하지.

이때 기존의 TTT와 다르게 SwiGLU로 큰 mlp를 만들어서 learning space를 크게했어. 

이를 통해 TTT layer는 key → value 메핑을 학습할 수 있어. 

이후 query를 TTT layer에 넣으면 query에 해당하는 value값이 나와.

Attention의 motivation은 query와 key의 유사성을 바탕으로 value를 Interpolation하는 것인데

TTT도 이 motivation을 그대로 공유하는 거지.

`Related Work` LaCT (논문명 : Test-Time Training Done Right)

Large Chunk TTT (LaCT)은 TTT layer의 일종으로, GPU 활용도 크게 높인 연구야. 

구체적으로, LaCT은 한 번에 많은 token들 (q, k, v 쌍)을 바탕으로 TTT layer를 업데이트해서, 병렬성(GPU 활용도)를 높였어. 

LaCT은 

이를 통해 TTT가 Vision task같은 large token이 사용되는 분야에 확장될 수 있음을 보였지.

특히, LaCT은 novel view synthesis 모델(LVSM)에 적용했을 때 Full Attention(requiring quadratic cost)과 거의 유사한 성능을 보이면서 몇배나 빠른 속도를 달성했어.

다만 TTT layer는 linear 하게 표현되지 않기 때문에 ProPE같은 embedding 방법을 그대로 적용할 수 없어. 

`Task Goal` LaCT에 camera 정보를 주입하는 방법 개발

1) TTT layer를 통해 camera pose를 relative 하게 표현할 수 있는 embedding이 있는가?

2) 혹은 relative 하게 표현할 수 없더라고 camera 정보를 embedding할 수 있는 방법이 무엇인가? 

이 질문에 대해 깊게 생각해줘. 수학적 intuition 이 있으면 좋아. 

이때 매 TTT layer에 적용할 수 있으면 좋을듯해.

꼭 intuition이 수학적으로 표현되지 않아도 돼. 예를 들어, ProPE의 그냥 intrinsic 정보는 relative하게 표현되지 않지만, intrinsic matrix를 넣어주기 위한 방법인거지. 

최대한 여러 가설을 세워봐줘. 한 10가지 아이디어가 있으면 좋을 것 같아. 이떄 아이디어를 subagent한테 생각해달라하고 추합하는 방법도 좋을 것같아. 최대한 다양한 방향으로 아이디어를 생각해줘. 

이후 코딩을 통해서  가설이 맞는지 실험해줘. GPU는 총 B200 4대라 한 번에 최대 4개 실험을 동시에 실험 할 수 있겠지?

이때 baseline (LaCT LVSM)에 비해 얼마나 성능이 오르는지 각각 비교해줘. 

`Code Structure` 

paper : 관련 연구들에 대한 pdf와 latex file이 있어. pdf만 읽으면 수식을 보기 어려울 까봐 latex file도 넣어뒀어. 

prope : Prope에 관한 코드야. 

RayRoPE : rayrope에 관한 코드야. 

lact_nvs : LaCT으로 구현한 LVSM 코드야. 이 코드를 기반으로 여러 파생 아이디어들을 실험해줘.