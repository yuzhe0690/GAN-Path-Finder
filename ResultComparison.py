import cv2
import numpy as np
import csv
import os


from matplotlib import pyplot as plt


def preprocess_gan_prediction(pred, ideal_map, size):
    img = np.copy(pred)

    for i in range(size):
        for j in range(size):
            if ideal_map[i, j] == 128:
                img[i, j] = 128

            if ideal_map[i, j] == 0:
                img[i, j] = 0

    return img


if __name__ == '__main__':
    counter = 1
    mc = 1.0

    gan_log = cv2.imread(f'prediction_{counter - 1}.png', cv2.IMREAD_GRAYSCALE)
    ideal_map = cv2.imread(f'{counter}_ideal_img.png', cv2.IMREAD_GRAYSCALE)
    ideal_log = cv2.imread(f'{counter}_ideal_log.png', cv2.IMREAD_GRAYSCALE)
    astar_log = cv2.imread(
        f'{counter}_astar_ideal_{mc}.png', cv2.IMREAD_GRAYSCALE)
    pgan_log = preprocess_gan_prediction(gan_log, ideal_map, 64)

    id_vs_as = np.subtract(astar_log, ideal_log)
    id_vs_gan = np.subtract(pgan_log, ideal_log)
    a_diff = np.bitwise_xor(ideal_log, id_vs_as)
    g_diff = np.bitwise_xor(ideal_log, id_vs_gan)

    plt.figure()
    plt.subplot(141)
    plt.imshow(ideal_log, cmap='gray')
    plt.subplot(142)
    plt.imshow(astar_log, cmap='gray')
    plt.subplot(143)
    plt.imshow(id_vs_as, cmap='gray')
    plt.subplot(144)
    plt.imshow(a_diff, cmap='gray')

    plt.figure()
    plt.subplot(141)
    plt.imshow(ideal_log, cmap='gray')
    plt.subplot(142)
    plt.imshow(pgan_log, cmap='gray')
    plt.subplot(143)
    plt.imshow(id_vs_gan, cmap='gray')
    plt.subplot(144)
    plt.imshow(g_diff, cmap='gray')

    actual = []
    for i in range(64):
        for j in range(64):
            if ideal_log[i, j] == 0:
                actual.append((i, j))

    a_path = []
    for i in range(64):
        for j in range(64):
            if a_diff[i, j] == 0:
                a_path.append((i, j))

    g_path = []
    for i in range(64):
        for j in range(64):
            if g_diff[i, j] == 0:
                g_path.append((i, j))

    print(f'actual path length: {len(actual)}')
    print(f'astar diff path length: {len(a_path)}')
    print(f'gan diff path length: {len(g_path)}')
    similarity_id_vs_as = np.round((((len(a_path)) / len(actual)) * 100), 2)
    print(similarity_id_vs_as)
    similarity_id_vs_gan = np.round((((len(g_path)) / len(actual)) * 100), 2)
    print(similarity_id_vs_gan)

    blocker = cv2.hconcat([np.full((64, 2), 255, dtype=np.uint8), np.full(
        (64, 1), 200, dtype=np.uint8), np.full((64, 2), 255, dtype=np.uint8)])
    sample = cv2.hconcat([ideal_log, blocker, astar_log, blocker, a_diff])
    cv2.imwrite(f'{counter}_ideal_astar.png', sample)
    sample = cv2.hconcat([ideal_log, blocker, pgan_log, blocker, g_diff])
    cv2.imwrite(f'{counter}_ideal_gan.png', sample)
    benchmark_path = os.path.join('./', f'{counter}_ideal.csv')
    with open(benchmark_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write headers
        writer.writerow(['', 'Ideal', 'GAN-Path-Finder', 'AStar'])
        # Write data
        writer.writerow(['Path len', len(actual), len(g_path), len(a_path)])
        writer.writerow(
            ['Similarity (%)', 100, similarity_id_vs_gan, similarity_id_vs_as])
