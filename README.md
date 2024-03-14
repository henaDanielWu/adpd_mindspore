# ADPD:Adaptive Decoupled Pose Knowledge Distillation
 This code is based on [MindSpore](https://gitee.com/mindspore/mindspore).

## Install
a. Install the [MindSpore(CPU)](https://www.mindspore.cn/install): 

If your machine is Linux-x86_64 and you have Python 3.7, you can run the following command:

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.11/MindSpore/unified/x86_64/mindspore-2.2.11-cp37-cp37m-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

b. Install the dependent libraries as follows: 

Install the dependent python libraries: 

```shell
pip install -r requirements.txt
```

c. Run the following command to compile the c++ extension file:

```shell
cd lib
make
```

## Testing

* Test:

Run the following command：
```shell
python tools/test.py
```
## Reference
If you use ADPD in your research, please cite our work by using the following BibTeX entry:
```shell
@inproceedings{xu2023adaptive,
  title={Adaptive Decoupled Pose Knowledge Distillation},
  author={Xu, Jie and Zhang, Shanshan and Yang, Jian},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4401--4409},
  year={2023}
}
```
