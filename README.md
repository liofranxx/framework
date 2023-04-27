# framework
一个结合wandb和pytorch-lighting的项目框架
框架目录：
```
.
├── config                 # 项目配置目录
│   └── test.cfg           # 项目配置文件
├── data                   # 项目数据存放目录
│   ├── category
│   │   ├── eval.jsonl
│   │   └── train.jsonl
│   └── telecom
│       ├── eval.jsonl
│       └── train.jsonl
├── logs                   # 项目日志目录
│   └── monitor
├── model                  # 项目模型目录
│   ├── Bert.py            # 项目具体模型文件，如BERT、T5或自己具体实现的model
│   ├── model.py           # 项目模型文件，利用pytorch-lighting构建pl模型，包含模型结构、训练/评估过程、优化器和评价方法
│   └── __pycache__
│       ├── Bert.cpython-38.pyc
│       └── model.cpython-38.pyc
├── peft_test.py
├── readme.md
├── requirements.txt
├── test.ipynb
├── test.py
├── test.sh
├── utils                  # 项目工具目录，包含cofigue类、process类和一些工具metric方法
│   ├── configue.py        
│   ├── process.py
│   ├── __pycache__
│   │   ├── configue.cpython-310.pyc
│   │   ├── configue.cpython-38.pyc
│   │   ├── process.cpython-310.pyc
│   │   ├── process.cpython-38.pyc
│   │   ├── training_arguments.cpython-310.pyc
│   │   ├── training_arguments.cpython-38.pyc
│   │   ├── utils.cpython-310.pyc
│   │   └── utils.cpython-38.pyc
│   ├── task.py
│   ├── training_arguments.py
│   └── utils.py
└── zero-shot.py
```
### 加注释文件和目录为框架必要文件目录
