# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, \
            `Kazutaka Morita <https://github.com/kazum>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to official installation guide.
https://mxnet.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import tvm
import tvm.relay as relay
import numpy as np

import time

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

#model = "resnet18_v1"
model = "squeezenet1.0"
block = get_model(model, pretrained=True)
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
img_path = download_testdata(img_url, "cat.png", module="data")
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())
image = Image.open(img_path).resize((224, 224))
plt.imshow(image)
plt.show()


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)
print("x", x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
shape_dict = {"data": x.shape}

build_start = time.time()

mod, params = relay.frontend.from_mxnet(block, shape_dict)
## we want a probability so add a softmax operator
func = mod["main"]
func = relay.Function(func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs)

######################################################################
# now compile the graph
# target = "cuda"
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(func, target, params=params)

    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))
