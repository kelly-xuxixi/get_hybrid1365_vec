import numpy as np
import os


def process_probs(file_path):
    probs = np.loadtxt(file_path)
    print('processing ' + file_path + str(probs.shape))
    try:
        tmp = probs[1]
    except IndexError:
        probs.shape = (1, len(probs))
    processed_probs = np.mean(probs, axis=0)
    print(processed_probs.shape)
    return processed_probs


def main():
    folder = '/m/data/med/probs_bg'
    prob_files = os.listdir(folder)
    prob_files.sort()
    all_probs = []
    for file in prob_files:
        file_path = os.path.join(folder, file)
        processed_probs = process_probs(file_path)
        if processed_probs.shape[0] == 1365:
            all_probs.append(processed_probs)
        else:
            print(file_path, processed_probs.shape)
    all_probs = np.vstack(all_probs)
    print(all_probs.shape)
    np.savetxt(os.path.join('/m/data/med', 'mean_probs_bg.txt'), all_probs)


if __name__ == '__main__':
    main()