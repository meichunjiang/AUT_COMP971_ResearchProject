# coding = utf-8

import argparse
import sys
import math
import os

import numpy as np

import mindspore.nn as nn
from mindspore import Model, load_param_into_net, load_checkpoint
from mindspore import dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
import mindspore.dataset as de
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
from mindarmour import MembershipInference
from mindarmour.utils import LogUtil

LOGGER = LogUtil.get_instance()
TAG = "MembershipInference_test"
LOGGER.set_level("INFO")

# 这里采用的是CIFAR-100数据集，您也可以采用自己的数据集，但要保证传入的数据仅有两项属性”image”和”label”。
# Generate CIFAR-100 data.
def vgg_create_dataset100(data_home, image_size, batch_size, rank_id=0, rank_size=1, repeat_num=1,
                          training=True, num_samples=None, shuffle=True):
    """Data operations."""
    de.config.set_seed(1)
    data_dir = os.path.join(data_home, "train")
    if not training:
        data_dir = os.path.join(data_home, "test")

    if num_samples is not None:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id,
                                      num_samples=num_samples, shuffle=shuffle)
    else:
        data_set = de.Cifar100Dataset(data_dir, num_shards=rank_size, shard_id=rank_id)

    input_columns = ["fine_label"]
    output_columns = ["label"]
    data_set = data_set.rename(input_columns=input_columns, output_columns=output_columns)
    data_set = data_set.project(["image", "label"])

    rescale = 1.0 / 255.0
    shift = 0.0

    # Define map operations.
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT.
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize(image_size)  # interpolation default BILINEAR.
    rescale_op = vision.Rescale(rescale, shift)
    normalize_op = vision.Normalize((0.4465, 0.4822, 0.4914), (0.2010, 0.1994, 0.2023))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # Apply map operations on images.
    data_set = data_set.map(operations=type_cast_op, input_columns="label")
    data_set = data_set.map(operations=c_trans, input_columns="image")

    # Apply repeat operations.
    data_set = data_set.repeat(repeat_num)

    # Apply batch operations.
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)

    return data_set
# 建立模型
# 这里以VGG16模型为例，您也可以替换为自己的模型
def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init='XavierUniform')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)

class Vgg(nn.Cell):
    """
    VGG network definition.
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train"):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layers = _make_layer(base, args, batch_norm=batch_norm)
        self.flatten = nn.Flatten()
        dropout_ratio = 0.5
        if not args.has_dropout or phase == "test":
            dropout_ratio = 1.0
        self.classifier = nn.SequentialCell([
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(dropout_ratio),
            nn.Dense(4096, num_classes)])

    def construct(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

base16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
def vgg16(num_classes=1000, args=None, phase="train"):
    net = Vgg(base16, num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase)
    return net

# 运用MembershipInference进行隐私安全评估
# 1.构建VGG16模型并加载参数文件。这里直接加载预训练完成的VGG16参数配置，您也可以使用如上的网络自行训练。
#
# load parameter
# data_path = '/AUT'
# data_path = '/Users/chunjiangmei/Development/PycharmProjects/AUT_COMP971_ResearchProject/cifar-100-binary'
# pre_trained = './VGG16-100_781.ckpt'


# python MindSpore_MembershipInference_Sample.py --data_path ./cifar-100-binary/ --pre_trained ./VGG16-100_781.ckpt

parser = argparse.ArgumentParser("main case arg parser.")
parser.add_argument("--data_path",   type=str,  required=True, help="Data home path for dataset")
parser.add_argument("--pre_trained", type=str,  required=True, help="Checkpoint path")
args = parser.parse_args()
args.batch_norm = True
args.has_dropout = False
args.has_bias = False
args.padding = 0
args.pad_mode = "same"
args.weight_decay = 5e-4
args.loss_scale = 1.0

# Load the pretrained model.
net = vgg16(num_classes=100, args=args)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9,
                  weight_decay=args.weight_decay, loss_scale=args.loss_scale)
load_param_into_net(net, load_checkpoint(args.pre_trained))
model = Model(network=net, loss_fn=loss, optimizer=opt)

# 2.加载CIFAR-100数据集，按8:2分割为成员推理模型的训练集和测试集。
# Load and split dataset.
train_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                      batch_size=64, num_samples=5000, shuffle=False)
test_dataset = vgg_create_dataset100(data_home=args.data_path, image_size=(224, 224),
                                     batch_size=64, num_samples=5000, shuffle=False, training=False)
train_train, eval_train = train_dataset.split([0.8, 0.2])
train_test, eval_test = test_dataset.split([0.8, 0.2])
msg = "Data loading completed."
LOGGER.info(TAG, msg)

# 3.配置推理参数和评估参数
# 设置用于成员推理的方法和参数。目前支持的推理方法有：KNN、LR、MLPClassifier和RandomForestClassifier。
# 推理参数数据类型使用list，各个方法使用key为”method”和”params”的字典表示。
config = [
        {
            "method": "lr",
            "params": {
                "C": np.logspace(-4, 2, 10)
            }
        },
     {
            "method": "knn",
            "params": {
                "n_neighbors": [3, 5, 7]
            }
        },
        {
            "method": "mlp",
            "params": {
                "hidden_layer_sizes": [(64,), (32, 32)],
                "solver": ["adam"],
                "alpha": [0.0001, 0.001, 0.01]
            }
        },
        {
            "method": "rf",
            "params": {
                "n_estimators": [100],
                "max_features": ["auto", "sqrt"],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }
        }
    ]

# 我们约定标签为训练集的是正类，标签为测试集的是负类。设置评价指标，目前支持3种评价指标。包括：
# 准确率：accuracy，正确推理的数量占全体样本中的比例。
# 精确率：precision，正确推理的正类样本占所有推理为正类中的比例。
# 召回率：recall，正确推理的正类样本占全体正类样本的比例。 在样本数量足够大时，如果上述指标均大于0.6，我们认为目标模型就存在隐私泄露的风险。
metrics = ["precision", "accuracy", "recall"]

# 4.训练成员推理模型，并给出评估结果。
inference = MembershipInference(model)                  # Get inference model.

inference.train(train_train, train_test, config)        # Train inference model.
msg = "Membership inference model training completed."
LOGGER.info(TAG, msg)

result = inference.eval(eval_train, eval_test, metrics) # Eval metrics.
count = len(config)
for i in range(count):
    print("Method: {}, {}".format(config[i]["method"], result[i]))

# 5.实验结果。 执行如下指令,开始成员推理训练和评估：
# python example_vgg_cifar.py --data_path ./cifar-100-binary/ --pre_trained ./VGG16-100_781.ckpt




















