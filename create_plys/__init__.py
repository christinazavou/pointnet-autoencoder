import os
import random


with open("../data/shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt", "r") as fin:
    lines = fin.readlines()
    categories = {l.rstrip().split('\t')[1]: l.rstrip().split('\t')[0] for l in lines}


os.makedirs("../data/ply_2048_with_labels", exist_ok=True)
skip_cnt = 0
for cat in categories:
    for root, dirs, files in os.walk("../data/shapenetcore_partanno_segmentation_benchmark_v0/{}/points".format(cat)):
        for file in files:
            with open(os.path.join(root, file), "r") as fin:
                points = fin.readlines()
                if len(points) < 2048:
                    skip_cnt += 1
                    continue
                points = random.sample(points, 2048)

                with open("../data/ply_2048_with_labels/{}".format(file.replace(".pts", ".ply")), "w") as fout:
                    fout.write('ply\n'
                                'format ascii 1.0\n'
                                'element vertex 2048\n'
                                'property float x\n'
                                'property float y\n'
                                'property float z\n'
                                'property uchar red\n'
                                'property uchar green\n'
                                'property uchar blue\n'
                                'property uchar alpha\n'
                                'property float label\n'
                                'end_header\n')
                    for point in points:
                        X, Y, Z = point.strip().split()
                        fout.write('%s %s %s %s %d %d %d %s\n' % (X, Y, Z, 255, 255, 255, 255, cat))
print("skipped: {}".format(skip_cnt))
