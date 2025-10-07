''' test load data buffer and moidify it'''
import tensorflow as tf

from tools.buffer import Buffer

# save_dir = 'saved_data/kuka-pushT-dim2_dimo52_1106_episode0-10/'
save_dir = 'saved_data/robosuite_square/Implicit_CLIC_save_dataset/'
buffer_test = Buffer(min_size=1, max_size=100000)
buffer_test.load_from_file(save_dir + 'buffer_data.pkl')
print("load data")
print("length of buffer: ", buffer_test.length())

modified_buffer = []
for s, a, h, s_next in buffer_test.buffer:
    print("s: ", s)
    print("a: ", a)
    print("h: ", h)
    # h = h + a 
    # a = tf.zeros_like(a)  # a set to zero, the same shape of original a
    modified_buffer.append([s, a + h])
buffer_test.buffer = modified_buffer
buffer_test.save_to_file(save_dir + 'buffer_data_bc.pkl')