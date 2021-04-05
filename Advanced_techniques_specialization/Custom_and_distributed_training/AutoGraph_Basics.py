import tensorflow as tf
tf.get_logger().setLevel('ERROR')


@tf.function
def add(a, b):
    return a + b

a = tf.Variable([[1.,2.],[3.,4.]])
b = tf.Variable([[4.,0.],[1.,5.]])
print(tf.add(a, b))

# simple function that returns the square if the input is greater than zero
@tf.function
def f(x):
    if x>0:
        x = x * x
    return x

@tf.function
def fizzbuzz(max_num):
    counter = 0
    for num in range(max_num):
        if num % 3 == 0 and num % 5 == 0:
            print('FizzBuzz')
        elif num % 3 == 0:
            print('Fizz')
        elif num % 5 == 0:
            print('Buzz')
        else:
            print(num)
        counter += 1
    return counter

if __name__ == '__main__':
    # See what the generated code looks like
    print(tf.autograph.to_code(add.python_function))
    # print(tf.autograph.to_code(f.python_function))
    # print(tf.autograph.to_code(fizzbuzz.python_function))
