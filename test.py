import tensorflow as tf
import numpy as np
import argparse
import os
import part_dataset
from models import model as MODEL
from plyfile import PlyData, PlyElement

BASE_DIR = ""

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log.buildnet', help='Log dir [default: log]')
parser.add_argument('--eval_dir', default='evaluation', help='Eval dir [default: evaluation]')
parser.add_argument('--model_path', default='model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--data_path', required=True)
parser.add_argument('--num_group', type=int, default=1,
                    help='Number of groups of generated points -- used for hierarchical FC decoder. [default: 1]')
FLAGS = parser.parse_args()

MODEL_PATH = os.path.join(FLAGS.log_dir, FLAGS.model_path)
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
LOG_DIR = FLAGS.log_dir
DATA_PATH = FLAGS.data_path
EVAL_DIR = FLAGS.eval_dir
# DATA_PATH = os.path.join(BASE_DIR, 'data/shapenetcore_partanno_segmentation_benchmark_v0')
# TEST_DATASET = part_dataset.PartDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False,
#                                         class_choice=FLAGS.category, split='test', normalize=True)
TEST_DATASET = part_dataset.BuildnetPartDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normalize=True)
print(len(TEST_DATASET))
ACTUAL_DIR = "actual"
PREDICTIONS_DIR = "predictions"
EMBEDDINGS_DIR = "embeddings"


def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            saver = tf.train.Saver()
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'embedding': end_points['embedding']}
        return sess, ops


def inference(sess, ops, pc, batch_size):
    ''' pc: BxNx3 array, return BxN pred '''
    assert pc.shape[0] % batch_size == 0
    num_batches = pc.shape[0] // batch_size
    logits = np.zeros((pc.shape[0], pc.shape[1], 3))
    embeddings = []
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
                     ops['is_training_pl']: False}
        batch_logits, batch_embeddings = sess.run([ops['pred'], ops['embedding']], feed_dict=feed_dict)
        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        embeddings.append(batch_embeddings)
    return logits, np.vstack(embeddings)


def save_point_cloud(points_3d, filename, binary=True, with_label=False, verbose=True):
    """Save an RGB point cloud as a PLY file.

  Args:
    points_3d: Nx6 matrix where points_3d[:, :3] are the XYZ coordinates and points_3d[:, 4:] are
        the RGB values. If Nx3 matrix, save all points with [128, 128, 128] (gray) color.
  """
    assert points_3d.ndim == 2
    if points_3d.shape[1] == 3:
        gray_concat = np.tile(np.array([128], dtype=np.uint8), (points_3d.shape[0], 3))
        points_3d = np.hstack((points_3d, gray_concat))
    assert points_3d.shape[1] == 6
    python_types = (float, float, float, int, int, int)
    npy_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                 ('blue', 'u1')]
    if binary is True:
        # Format into NumPy structured array
        vertices = []
        for row_idx in range(points_3d.shape[0]):
            cur_point = points_3d[row_idx]
            vertices.append(tuple(dtype(point) for dtype, point in zip(python_types, cur_point)))
        vertices_array = np.array(vertices, dtype=npy_types)
        el = PlyElement.describe(vertices_array, 'vertex')

        # Write
        PlyData([el]).write(filename)
    else:
        # PlyData([el], text=True).write(filename)
        with open(filename, 'w') as f:
            f.write('ply\n'
                    'format ascii 1.0\n'
                    'element vertex %d\n'
                    'property float x\n'
                    'property float y\n'
                    'property float z\n'
                    'property uchar red\n'
                    'property uchar green\n'
                    'property uchar blue\n'
                    'property uchar alpha\n'
                    'end_header\n' % points_3d.shape[0])
            for row_idx in range(points_3d.shape[0]):
                X, Y, Z, R, G, B = points_3d[row_idx]
                f.write('%f %f %f %d %d %d 0\n' % (X, Y, Z, R, G, B))
    if verbose is True:
        print('Saved point cloud to: %s' % filename)


if __name__ == '__main__':

    os.makedirs(os.path.join(LOG_DIR, EVAL_DIR, ACTUAL_DIR), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, EVAL_DIR, PREDICTIONS_DIR), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, EVAL_DIR, EMBEDDINGS_DIR), exist_ok=True)
    sess, ops = get_model(batch_size=1, num_point=NUM_POINT)
    indices = np.arange(len(TEST_DATASET))
    np.random.shuffle(indices)
    for i in range(len(TEST_DATASET)):
        ps, seg = TEST_DATASET[indices[i]]
        pred, emb = inference(sess, ops, np.expand_dims(ps, 0), batch_size=1)
        np.save(os.path.join(LOG_DIR, EVAL_DIR, EMBEDDINGS_DIR, "{}.npy".format(i)), emb)
        save_point_cloud(ps, os.path.join(LOG_DIR, EVAL_DIR, ACTUAL_DIR, "{}.ply".format(i)))
        save_point_cloud(pred[0], os.path.join(LOG_DIR, EVAL_DIR, PREDICTIONS_DIR, "{}.ply".format(i)))
