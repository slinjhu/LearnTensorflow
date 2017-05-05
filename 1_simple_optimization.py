import tensorflow as tf

# Define a linear model = k * x + b, where k and b are TBD
k = tf.Variable(2.0, tf.float32)
b = tf.Variable(1.0, tf.float32)
x = tf.placeholder(tf.float32)
model = k * x + b

# Define loss function
y = tf.placeholder(tf.float32)
loss = tf.reduce_sum(tf.square(model - y))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


data = {x: [1, 2, 3], y: [1.1, 2.2, 3.3]}

# Define and start the session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(200):
    sess.run([train], data)
    print(sess.run([loss], data))