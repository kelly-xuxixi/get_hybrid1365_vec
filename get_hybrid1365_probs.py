import vgg16
import utils
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


def get_1365_vec(vgg, input_tensor, sess, folder_path):
    files = os.listdir(folder_path)
    imgs = []
    num = 0
    for file in files:
        if not os.path.isdir(file) and (file.endswith('.jpg') or file.endswith('jpeg')):
            print(file)
            image = utils.load_image(os.path.join(folder_path, file))
            print(image.shape)
            imgs.append(image)
            num += 1
    while num % 64 != 0:
        imgs.append(np.zeros([224, 224, 3]))
        num += 1
    imgs = np.stack(imgs)
    print(imgs.shape)
    all_probs = []
    for i in range(int(len(imgs) / 64)):
        feed_dict = {input_tensor: imgs[i * 64: (i + 1) * 64]}
        probs = sess.run(vgg.prob, feed_dict=feed_dict)
        all_probs.append(probs)
    all_probs = np.vstack(all_probs)
    all_probs = all_probs[:len(files)]
    print(all_probs.shape)
    utils.print_prob(all_probs[0], './synset.txt')
    utils.print_prob(all_probs[1], './synset.txt')
    utils.print_prob(all_probs[2], './synset.txt')

    return all_probs


def main():
    with tf.Session() as sess:
        # add a loop for all folders
        folder_path = 'images'
        input_tensor = tf.placeholder("float", [64, 224, 224, 3])
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(input_tensor)
        probs_for_folder = get_1365_vec(vgg, input_tensor, sess, folder_path)
        np.savetxt(os.path.join(folder_path, 'probs.txt'), probs_for_folder)


if __name__ == '__main__':
    main()
