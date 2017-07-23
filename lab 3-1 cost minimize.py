import tensorflow as tf
import matplotlib.pyplot as plt

# X와 Y 데이터
X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# 가설 : WX (y = ax)
hypothesis = X * W

# cost = (hypothesis - y)^2 / m
cost = tf.reduce_mean(tf.square(hypothesis - Y))

sess = tf.Session()
# tf가 사용할 변수들을 초기화
sess.run(tf.global_variables_initializer())

# 그래프를 실행시켜 W 값과 cost 값을 저장시킬 리스트 생성
W_val = []
cost_val = []

for i in range(-30, 50) :
    # W를 -3 부터 5까지의 값을 0.1간격으로 제공
    feed_W = i * 0.1
    # W 값을 넘기면서 cost와 W의 값을 위 리스트에 저장
    curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
    W_val.append(curr_W)
    cost_val.append(curr_cost)

# x축이 W고 y축이 cost인 그래프를 그림
plt.plot(W_val, cost_val)
plt.show()