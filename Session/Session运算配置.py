# tf.configProto用于在创建Session的时候配置Session的运算方式，即使用GPU运算或CPU运算；
import tensorflow as tf
import os

# 1. tf.ConfigProto()中的基本参数：
session_config = tf.ConfigProto(
    log_device_placement=True,  # 设置为True时，会打印出Tensorflow 使用了哪种操作
    allow_soft_placement=True,  # 当运行设备不满足要求时，会自动分配GPU或CPU
    inter_op_parallelism_threads=0, # 一个操作内部并行运算的线程数
    intra_op_parallelism_threads=0  # 多个操作并行运算的线程数
)
sess = tf.Session(config=session_config)


# 2.tf.ConfigProto配置GPU

# 2.1 判断Tensorflow是否能够使用GPU运算
result = tf.test.is_built_with_cuda()
print(result)

# 2.2 两种方式配置使用具体哪块GPU
# 方式一：在pytohn程序中设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1' # GPU号

# 方式二：在执行py文件时，指定具体的GPU块
CUDA_VISIBLE_DEVICES=0,1 python test.py

# 2.3 动态申请GPU显存
session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True # 动态申请
session = tf.Session(config=session_config)

# 2.4 限制GPU的使用率
session_config  = tf.ConfigProto()
session_config.gpu_options.per_process_gpu_memory_fraction = 0.4 # 占用40%的显存
session = tf.Session(config=session_config)

