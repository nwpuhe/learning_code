import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import skimage.io as io
img_width = 224
img_height = 224


def get_file(file_dir):
    images = []
    temp = []
    for root, sub_folders, files in os.walk(file_dir):
        for name in files:
            images.append(os.path.join(root, name))
        for name in sub_folders:
            temp.append(os.path.join(root, name))
    labels = []
    for one_folder in temp:
        n_img = len(os.listdir(one_folder))
        letter = one_folder.split('\\')[-1]
        if letter == 'cat':
            labels = np.append(labels, n_img * [0])
        else:
            labels = np.append(labels, n_img * [1])
    # shuffle
    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]

    return image_list, label_list

image_list, label_list = get_file("C:\cat_and_dog_r")

def get_batch(image_list, label_list,img_width,img_height,batch_size,capacity):
    image = tf.cast(image_list,tf.string)
    label = tf.cast(label_list,tf.int32)

    input_queue = tf.train.slice_input_producer([image,label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)

    image = tf.image.resize_image_with_crop_or_pad(image,img_width,img_height)
    image = tf.image.per_image_standardization(image) #将图片标准化
    #image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=batch_size * 5,capacity=batch_size * 7)
    image_batch, label_batch = tf.train.shuffle_batch([image,label],batch_size=batch_size,capacity=12 * batch_size,min_after_dequeue=10 * batch_size,num_threads=2)
    label_batch = tf.reshape(label_batch,[batch_size])

    return image_batch,label_batch


def plot_images(images, labels):
    '''plot one batch size
    '''
    for i in np.arange(0, 25):
        plt.subplot(5, 5, i + 1)
        plt.axis('off')
        plt.title("lover", fontsize = 14)

        plt.imshow(images[i])
    plt.show()



import matplotlib.pyplot as plt

BATCH_SIZE = 200
CAPACITY = 2000
IMG_W = 224
IMG_H = 224

#使用示例
if __name__ == "__main__":
    image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    with tf.Session() as sess:
       #i = 0
       coord = tf.train.Coordinator()
       threads = tf.train.start_queue_runners(coord=coord)

       try:
           #while not coord.should_stop() and i<1:
           for _ in range(2):
               image, label = sess.run([image_batch, label_batch])
               plot_images(image, label)
               #i += 1  # 如果把这个注销了，就会一直运行下去

       except tf.errors.OutOfRangeError:
           print('done!')
       finally:
           coord.request_stop()
       coord.join(threads)