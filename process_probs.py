import numpy as np
import os


def process_probs(folder_path):
    try:
        probs = np.loadtxt(os.path.join(folder_path, 'probs.txt'))
    except IOError:
        print(folder_path + ' contains no prob.txt')
        return []
    print('processing ' + folder_path)
    processed_probs = np.mean(probs, axis=0)
    return processed_probs


def main():
    probs = []
    # loop over all folders
    root_folder = '.'
    folders = os.listdir(root_folder)
    all_probs = []
    for folder in folders:
        if os.path.isdir(folder):
            folder_path = os.path.join(root_folder, folder)
            processed_probs = process_probs(folder_path)
            if len(processed_probs) > 0:
                all_probs.append(processed_probs)
    all_probs = np.vstack(all_probs)
    print(all_probs.shape)
    np.savetxt(os.path.join(root_folder, 'all_probs.txt'), all_probs)


if __name__ == '__main__':
    main()