# coding: utf-8

# 인공지능개론 #Homework 1
# 간단한 XOR Table을 학습하는 NN 을 구성하는 문제입니다.
#
#  1-Layer, 2-Layer model을 각각 구성하여 XOR 결과를 비교합니다.
#  1-Layer, 2-Layer의 model을 Back propagation을 이용하여 학습시킵니다.
#  주어진 양식을 활용해 주시며, scale, 차원의 순서, hyper parameter등은 결과가 잘 나오는 방향으로 Tuning하셔도 무방합니다.
#  Layer의 Activation 함수 Sigmoid는 54줄의 함수를 사용하시면 됩니다.
#  결과 재현을 위해 Weight, bias 값을 저장하여 함께 첨부해 주시기 바랍니다.
#  각 모델에서 loss 그래프와 testing step을 첨부하여 간단하게 자유 양식 결과 보고서(2~3장 내외)로 작성해 주세요.
#
#
# * python으로 코드를 작성하는 경우, 양식에서 활용하는 라이브러리 외에 추가로 import 하여 사용하실 수 없습니다.

## 이 외에 추가 라이브러리 사용 금지
import numpy as np
import random
import matplotlib.pyplot as plt

# Hyper parameters
# 학습의 횟수와 Gradient update에 쓰이는 learning rate입니다.
# 다른 값을 사용하여도 무방합니다.
epochs = 20000
learning_rate = 0.1

# Input data setting
# XOR data
# 입력 데이터들, XOR Table 에 맞게 정의되어 있습니다.
train_inp = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
train_out = np.array([0, 1, 1, 0])

# Weight Setting
# 학습에 사용되는 Weigth 들의 초기값을 선언해 줍니다. 다른 값을 사용하여도 무방합니다.
W = np.random.randn(2,1) / 10
b = np.random.randn(1,1) / 10

##-----------------------------------##
##------- Activation Function -------##
##-----------------------------------##
def sigmoid(x):
    return 1 / (np.exp(-x)+1)

# ----------------------------------- #
# --------- Training Step ----------- #
# ----------------------------------- #
# 학습이 시작됩니다.
# epoch 사이즈만큼 for 문을 통해 학습됩니다.
# 빈 칸을 채워 Weight과 bias를 학습하는 신경망을 설계하세요.
# 양식의 모든 내용을 무조건 따를 필요는 없습니다. 각자에게 편하게 수정하셔도 좋습니다.


errors = []

for epoch in range(epochs):
    for batch in range(4):
        idx = random.randint(0, 3)

        # 입력 데이터 x1, x2와 해당하는 정답 ans 불러오기
        x1 = train_inp[idx][0]
        x2 = train_inp[idx][1]
        ans = train_out[idx]

        # Layer에 맞는 Forward Network 구성
        # H1 : hidden layer
        # z : output layer
        # y : output
        z = x1 * W[0][0] + x2 * W[1][0] + b
        y = sigmoid(z)

        # Binary Corss Entropy(BCE)로 loss 계산
        # loss 미분함수 diff_loss, delta 계산
        loss = -(ans * np.log10(y) + (1 - ans) * np.log10(1 - y))
        diff_loss = -ans / (np.log(10) * y) + (ans - 1) / (np.log(10) * (y - 1))
        delta = diff_loss * y * (1 - y)

        # Weight 초기값을 설정
        delta_W = np.zeros((2, 1))
        delta_b1 = np.zeros((1, 1))

        # Back propagation을 통한 Weight의 Gradient update step
        delta_W[0][0] = delta * x1
        delta_W[1][0] = delta * x2
        delta_b = delta

        W[0][0] = W[0][0] - learning_rate * delta_W[0][0]
        W[1][0] = W[1][0] - learning_rate * delta_W[1][0]
        b = b - learning_rate * delta_b



    if epoch % 10 == 0:
        errors.append(float(loss))
    if epoch % 400 == 0:
        print("epoch[{}/{}] loss: {:.4f}".format(epoch, epochs, float(loss)))

plt.plot(errors)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()


for idx in range(4):
    x1 = train_inp[idx][0]
    x2 = train_inp[idx][1]
    ans = train_out[idx]

    z = x1 * W[0][0] + x2 * W[1][0] + b
    y = sigmoid(z)

    print("input: ", x1, x2, ", answer: ", ans, ", pred: {:.4f}".format(float(y)))


#-----------------------------------#
#--------- Weight Saving -----------#
#-----------------------------------#

# weight, bias를 저장하는 부분입니다.
# 학번에 자신의 학번으로 대체해 주세요.

    #layer 1개인 경우
np.savetxt("학번_layer1_weight",(W, b),fmt="%s")

    #layer 2개인 경우
#np.savetxt("학번_layer2_weight",(W1, W2, b1, b2),fmt="%s")
