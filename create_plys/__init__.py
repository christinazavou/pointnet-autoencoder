import os
import random

#
# with open("../data/shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt", "r") as fin:
#     lines = fin.readlines()
#     categories = {l.rstrip().split('\t')[0]: l.rstrip().split('\t')[1] for l in lines}
#
#
# os.makedirs("../data/ply_2048_with_labels", exist_ok=True)
# skip_cnt = 0
# for cat_idx, cat in enumerate(categories):
#     for root, dirs, files in os.walk("../data/shapenetcore_partanno_segmentation_benchmark_v0/{}/points".format(categories[cat])):
#         for file in files:
#             with open(os.path.join(root, file), "r") as fin:
#                 points = fin.readlines()
#                 if len(points) < 2048:
#                     skip_cnt += 1
#                     continue
#                 points = random.sample(points, 2048)
#
#                 with open("../data/ply_2048_with_labels/{}".format(file.replace(".pts", ".ply")), "w") as fout:
#                     fout.write('ply\n'
#                                 'format ascii 1.0\n'
#                                 'element vertex 2048\n'
#                                 'property float x\n'
#                                 'property float y\n'
#                                 'property float z\n'
#                                 'property uchar red\n'
#                                 'property uchar green\n'
#                                 'property uchar blue\n'
#                                 'property uchar alpha\n'
#                                 'property float label\n'
#                                 'end_header\n')
#                     for point in points:
#                         X, Y, Z = point.strip().split()
#                         fout.write('%s %s %s %s %d %d %d %d\n' % (X, Y, Z, 255, 255, 255, 255, cat_idx))
# print("skipped: {}".format(skip_cnt))
# print(categories)
# {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}

files = os.listdir("/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/ply_2048_with_labels")
random.shuffle(files)
train_size = int(len(files) * 0.7)
val_size = int(train_size * 0.3)
val = files[:val_size]
train = files[val_size:train_size]
test = files[train_size:]

os.makedirs("/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/train_val_test_split", exist_ok=True)

with open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/train_val_test_split/train.txt", "w") as f_train,\
    open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/train_val_test_split/val.txt", "w") as f_val,\
    open("/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/train_val_test_split/test.txt", "w") as f_test:
    f_train.writelines(["/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/ply_2048_with_labels/{}\n".format(l) for l in train])
    f_val.writelines(["/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/ply_2048_with_labels/{}\n".format(l) for l in val])
    f_test.writelines(["/media/graphicslab/BigData/zavou/ANNFASS_CODE/pointnet-autoencoder/data/ply_2048_with_labels/{}\n".format(l) for l in test])
