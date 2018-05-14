import vgg16
import utils
import numpy as np
import tensorflow as tf
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


root_folder = '/m/data/med/frame'
probs_folder = '/m/data/med/probs'


def get_1365_vec(vgg, input_tensor, sess, folder_path):
    print(folder_path)
    try:
        files = os.listdir(folder_path)
        imgs = []
        num = 0
        for file in files:
            if not os.path.isdir(file) and file.endswith('.jpg'):
                image = utils.load_image(os.path.join(folder_path, file))
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
    except Exception as e:
        print(e)


def main():
    with tf.Session() as sess:
        input_tensor = tf.placeholder("float", [64, 224, 224, 3])
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(input_tensor)
        # a loop for all folders
        folders = os.listdir(root_folder)
        folders.sort()
        try:
            start = int(sys.argv[1])
            end = int(sys.argv[2])
        except:
            start = 0
            end = len(folders)
        for folder in folders[start:end]:
            folder_path = os.path.join(root_folder, folder)
            prob_path = os.path.join(probs_folder, folder + '.txt')
            if os.path.exists(prob_path):
                continue
            probs_for_folder = get_1365_vec(vgg, input_tensor, sess, folder_path)
            np.savetxt(prob_path, probs_for_folder)


if __name__ == '__main__':
    main()
