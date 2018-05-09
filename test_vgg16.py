import numpy as np
import tensorflow as tf
import os

import vgg16
import utils


def get_1365_vec(vgg, input_tensor, sess, folder_path):
    files = os.listdir(folder_path)
    imgs = []
    for file in files:
        if not os.path.isdir(file) and file.endswith('.jpg'):
            print(file)
            image = utils.load_image(file)
            print(image.shape)
            imgs.append(image)
    imgs = np.stack(imgs)
    feed_dict = {input_tensor: imgs}
    probs = sess.run(vgg.prob, feed_dict=feed_dict)
    print(probs)
    utils.print_prob(probs[0], './synset.txt')
    utils.print_prob(probs[1], './synset.txt')

    return probs


def main():
    folder_path = 'images'
    with tf.Session() as sess:
        input_tensor = tf.placeholder("float", [2, 224, 224, 3])
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(input_tensor)
        get_1365_vec(vgg, input_tensor, sess, folder_path)


if __name__ == '__main__':
    main()
