from collections import namedtuple
import logging
import os

import tvm
from tvm import te
from tvm import autotvm
from tvm import topi
import vta
import vta.testing

env = vta.get_env()

Workload = namedtuple(
    "Conv2DWorkload",
    [
        "batch",
        "height",
        "width",
        "in_filter",
        "out_filter",
        "hkernel",
        "wkernel",
        "hpad",
        "wpad",
        "hstride",
        "wstride",
    ],
)

resnet_wkls = [
    # Workloads of resnet18 on imagenet
    # ('resnet-18.C1',  Workload(env.BATCH, 224, 224, 3,   64,  7, 7, 3, 3, 2, 2)),
    ("resnet-18.C2", Workload(env.BATCH, 56, 56, 64, 64, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C3", Workload(env.BATCH, 56, 56, 64, 128, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C4", Workload(env.BATCH, 56, 56, 64, 128, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C5", Workload(env.BATCH, 28, 28, 128, 128, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C6", Workload(env.BATCH, 28, 28, 128, 256, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C7", Workload(env.BATCH, 28, 28, 128, 256, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C8", Workload(env.BATCH, 14, 14, 256, 256, 3, 3, 1, 1, 1, 1)),
    ("resnet-18.C9", Workload(env.BATCH, 14, 14, 256, 512, 3, 3, 1, 1, 2, 2)),
    ("resnet-18.C10", Workload(env.BATCH, 14, 14, 256, 512, 1, 1, 0, 0, 2, 2)),
    ("resnet-18.C11", Workload(env.BATCH, 7, 7, 512, 512, 3, 3, 1, 1, 1, 1)),
]

from tvm.autotvm.task import TaskExtractEnv

@tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
def my_clip(x, a_min, a_max):
    """Unlike topi's current clip, put min and max into two stages."""
    const_min = tvm.tir.const(a_min, x.dtype)
    const_max = tvm.tir.const(a_max, x.dtype)
    x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
    x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
    return x

TaskExtractEnv()

@autotvm.template("test/conv2d_packed.vta")
def conv2d(N, CI, H, W, CO, KH, KW, strides, padding, dilation):
    data_shape = (N // env.BATCH, CI // env.BLOCK_IN, H, W, env.BATCH, env.BLOCK_IN)
    kernel_shape = (CO // env.BLOCK_OUT, CI // env.BLOCK_IN, KH, KW, env.BLOCK_OUT, env.BLOCK_IN)
    bias_shape = (N // env.BATCH, CO // env.BLOCK_OUT, 1, 1, env.BATCH, env.BLOCK_OUT)

    data = te.placeholder(data_shape, name="data", dtype=env.inp_dtype)
    kernel = te.placeholder(kernel_shape, name="kernel", dtype=env.wgt_dtype)
    bias = te.placeholder(bias_shape, name="bias", dtype=env.acc_dtype)

    with tvm.target.vta():
        # res = topi.nn.conv2d(
        #     input=data,
        #     filter=kernel,
        #     padding=padding,
        #     strides=strides,
        #     dilation=dilation,
        #     layout="NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN),
        #     out_dtype=env.acc_dtype,
        # )
        res = vta.top.conv2d_packed(
                data, kernel, padding, strides, dilation, "NCHW%dn%dc" % (env.BATCH, env.BLOCK_IN), env.acc_dtype)
        res = topi.right_shift(res, env.WGT_WIDTH)
        res = topi.add(res, bias)
        res = my_clip(res, 0, (1 << env.OUT_WIDTH - 1) - 1)
        res = topi.cast(res, env.out_dtype)

    # if tvm.target.Target.current().device_name == "vta":
    #     s = topi.generic.schedule_conv2d_nchw([res]) ## 好像直接create_schedule就行了吧
    # else:
    #     s = te.create_schedule([res.op])
    s = vta.top.schedule_conv2d_packed

    return s, [data, kernel, bias, res]

if __name__ == "__main__":

    # Logging config (for printing tuning log to the screen)
    logging.basicConfig()
    # logging.getLogger('autotvm').setLevel(logging.DEBUG)

    # Tuning log files
    log_file = "%s.conv2d.log" % (env.TARGET)
    # create tmp log file
    tmp_log_file = log_file + ".tmp"
    if os.path.exists(log_file):
        os.remove(log_file)

    # Get tracker info from env
    tracker_host = os.environ.get("TVM_TRACKER_HOST", "0.0.0.0")
    tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
    if not tracker_host or not tracker_port:
        print("Set your AutoTVM tracker node host and port variables to run the autotuner")
        exit()

    for idx, (wl_name, wl) in enumerate(resnet_wkls):
        prefix = "[Task %2d/%2d] " % (idx, len(resnet_wkls))

        # Read in workload parameters
        N = wl.batch
        CI = wl.in_filter
        H = wl.height
        W = wl.width
        CO = wl.out_filter
        KH = wl.hkernel
        KW = wl.wkernel
        strides = (wl.hstride, wl.wstride)
        padding = (wl.hpad, wl.wpad)
        dilation = (1, 1)

        # Create task
        task = autotvm.task.create(
            "test/conv2d_packed.vta",
            args=(N, CI, H, W, CO, KH, KW, strides, padding, dilation),
            target=tvm.target.vta(),
            target_host=env.target_host
        )
        print(task.config_space)

        # Tune
        measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                env.TARGET,
                host=tracker_host,
                port=int(tracker_port),
                number=5,
                timeout=60,
                check_correctness=True,
            ),
        )

        # Run Tuner
        tuner = autotvm.tuner.RandomTuner(task)
        tuner.tune(
            n_trial=len(task.config_space),
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(len(task.config_space), prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # Pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_file)
    os.remove(tmp_log_file)
