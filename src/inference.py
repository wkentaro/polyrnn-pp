import glob
import json
import os

import matplotlib
matplotlib.use('agg')  # NOQA

import numpy as np
import skimage.io
import tensorflow as tf
import tqdm

from EvalNet import EvalNet
from GGNNPolyModel import GGNNPolygonModel
from PolygonModel import PolygonModel
import utils


_BATCH_SIZE = 1
_FIRST_TOP_K = 5

_INPUT_FOLDER = 'imgs/'
_OUTPUT_FOLDER = 'output/'


class Inferencer(object):

    PolyRNN_metagraph = 'models/poly/polygonplusplus.ckpt.meta'
    PolyRNN_checkpoint = 'models/poly/polygonplusplus.ckpt'
    EvalNet_checkpoint = 'models/evalnet/evalnet.ckpt'
    GGNN_checkpoint = 'models/ggnn/ggnn.ckpt'
    GGNN_metagraph = 'models/ggnn/ggnn.ckpt.meta'
    Use_ggnn = True

    def __init__(self):
        # Creating the graphs
        evalGraph = tf.Graph()
        polyGraph = tf.Graph()

        # Evaluator Network
        tf.logging.info("Building EvalNet...")
        with evalGraph.as_default():
            with tf.variable_scope("discriminator_network"):
                evaluator = EvalNet(_BATCH_SIZE)
                evaluator.build_graph()
            saver = tf.train.Saver()

            # Start session
            evalSess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True
            ), graph=evalGraph)
            saver.restore(evalSess, self.EvalNet_checkpoint)

        # PolygonRNN++
        tf.logging.info("Building PolygonRNN++ ...")
        model = PolygonModel(self.PolyRNN_metagraph, polyGraph)

        model.register_eval_fn(
            lambda input_: evaluator.do_test(evalSess, input_)
        )

        polySess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True
        ), graph=polyGraph)

        model.saver.restore(polySess, self.PolyRNN_checkpoint)
        self.model = model
        self.polySess = polySess

        if self.Use_ggnn:
            ggnnGraph = tf.Graph()
            tf.logging.info("Building GGNN ...")
            ggnnModel = GGNNPolygonModel(self.GGNN_metagraph, ggnnGraph)
            ggnnSess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True
            ), graph=ggnnGraph)

            ggnnModel.saver.restore(ggnnSess, self.GGNN_checkpoint)
            self.ggnnModel = ggnnModel
            self.ggnnSess = ggnnSess

    def __call__(self, image_np):
        image_np = np.expand_dims(image_np, axis=0)
        preds = [
            self.model.do_test(self.polySess, image_np, top_k)
            for top_k in range(_FIRST_TOP_K)
        ]

        # sort predictions based on the eval score and pick the best
        preds = sorted(preds, key=lambda x: x['scores'][0], reverse=True)[0]

        if self.Use_ggnn:
            polys = np.copy(preds['polys'][0])
            feature_indexs, poly, mask = utils.preprocess_ggnn_input(polys)
            preds_gnn = self.ggnnModel.do_test(
                self.ggnnSess, image_np, feature_indexs, poly, mask,
            )
            output = {
                'polys': preds['polys'],
                'polys_ggnn': preds_gnn['polys_ggnn'],
            }
        else:
            output = {'polys': preds['polys']}

        return output


# -----------------------------------------------------------------------------


def save_to_json(crop_name, predictions_dict):
    output_dict = {
        'img_source': crop_name,
        'polys': predictions_dict['polys'][0].tolist(),
    }
    if 'polys_ggnn' in predictions_dict:
        output_dict['polys_ggnn'] = predictions_dict['polys_ggnn'][0].tolist()

    fname = os.path.basename(crop_name).split('.')[0] + '.json'

    fname = os.path.join(_OUTPUT_FOLDER, fname)

    json.dump(output_dict, open(fname, 'w'), indent=4)


def main():
    inferencer = Inferencer()

    tf.logging.info("Testing...")
    if not os.path.isdir(_OUTPUT_FOLDER):
        tf.gfile.MakeDirs(_OUTPUT_FOLDER)
    crops_path = glob.glob(os.path.join(_INPUT_FOLDER, '*.png'))

    for crop_path in tqdm.tqdm(crops_path):
        image_np = skimage.io.imread(crop_path)

        output = inferencer(image_np)

        # dumping to json files
        save_to_json(crop_path, output)


if __name__ == '__main__':
    main()
