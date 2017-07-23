import tensorflow as tf

#x, y의 데이터 값
x_train = [1, 2, 3]
y_train = [1, 2, 3]
#tf가 사용할 변수
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 가설 : Wx+b (y = ax+b)
hyposis = x_train * W + b
# cost = (hypothesis - y)^2 /m
cost = tf.reduce_mean(tf.square(hyposis - y_train))

# cost를 최소화 하는방향으로 train
optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# session을 만듦
sess = tf.Session()

# tf가 사용할 변수들을 초기화
sess.run(tf.global_variables_initializer())

for step in range(2001) :
    # train을 실행
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))