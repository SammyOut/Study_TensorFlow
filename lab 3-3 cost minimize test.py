import tensorflow as tf

# X데이터와 Y데이터 제공
X = [1, 2, 3]
Y = [1, 2, 3]

# W값을 5.0으로 지정
# W = tf.Variable(5.0)

# W값을 -3.0으로 지정
W = tf.Variable(-3.0)

#  가설 : WX (y = ax)
hypothesis = X * W

# cost = (hypothesis - y)^2 / m
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Gradient Descent algorithm을 이용하여 cost 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

# Session을 열어줌
sess = tf.Session()
# tf가 사용할 변수들을 초기화
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)