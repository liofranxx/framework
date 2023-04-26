'''
Descripttion: 
Author: Lyc
Date: 2022-11-08 09:28:06
LastEditors: Lyc
LastEditTime: 2022-11-10 18:31:01
'''
import os
import torch
import random
import logging
import numpy as np
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report


class Logger(logging.Logger):
    """
    logger.debug("This is an debug.")
    logger.info("This is an info.")
    logger.warning("This is an warning.")
    logger.error("This is an error.")
    logger.critical("This is an critical.")
    """

    def __init__(self, name='root', level=logging.DEBUG, debug_name="debug.log"):
        super().__init__(name=name, level=level)
        logs_dir = './logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)

        # 用于在标准输出上打印日志
        # stdout_debug_handler = logging.StreamHandler(sys.stdout)
        # stdout_debug_handler.setFormatter(logging.Formatter(
        #     fmt='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
        #     datefmt='%Y-%m-%d %H:%M:%S'
        # ))

        # 在debug.log中记录全量日志
        debug_log_path = os.path.join(logs_dir, debug_name)
        debug_handler = logging.FileHandler(
            debug_log_path, mode='a', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # 在err.log中记录Error, Critical级别的日志
        err_log_path = os.path.join(logs_dir, 'err10.12.log')
        err_handler = logging.FileHandler(
            err_log_path, mode='a', encoding='utf-8')
        err_handler.setLevel(logging.ERROR)
        err_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))

        # self.addHandler(stdout_debug_handler)
        self.addHandler(debug_handler)
        self.addHandler(err_handler)

    def debug(self, *infos):
        info = ' '.join(str(info) for info in infos)
        self._log(level=logging.DEBUG, msg=info, args=None)

    def info(self, *infos):
        info = ' '.join(str(info) for info in infos)
        self._log(level=logging.INFO, msg=info, args=None)

    def warning(self, *infos):
        info = ' '.join(str(info) for info in infos)
        self._log(level=logging.WARNING, msg=info, args=None)

    def error(self, *infos):
        info = ' '.join(str(info) for info in infos)
        self._log(level=logging.ERROR, msg=info, args=None)

    def critical(self, *infos):
        info = ' '.join(str(info) for info in infos)
        self._log(level=logging.CRITICAL, msg=info, args=None)

    def class_info(self, infos):
        # infos = "\n" + infos
        self._log(level=logging.INFO, msg=infos, args=None)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次运行结果一样

# 精度计算


def skl_metric(labels_flat, pred_flat):
    # pred_flat = np.argmax(preds, axis=1).flatten()
    # labels_flat = labels.flatten()
    accuracy = accuracy_score(labels_flat, pred_flat)  # 使用sklearn。metric模块进行计算
    recall = recall_score(labels_flat, pred_flat, average="micro")
    precision = precision_score(labels_flat, pred_flat, average="micro")
    f1 = f1_score(labels_flat, pred_flat, average="micro")

    return accuracy, recall, precision, f1


class MetricForClassification(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.class_list = [type_ for _, type_ in self.id2label.items()]
        self.digits = 3
        self.reset()

    def reset(self):
        self.origins = []
        self.preds = []
        self.rights = []

    def compute(self, origins, preds, rights):
        recall = 0 if origins == 0 else (rights / origins)
        precision = 0 if preds == 0 else (rights / preds)
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = (2*precision*recall) / (precision+recall)
        return precision, recall, f1

    def update(self, labels, targets):
        '''
        Example: [batch_size, 1].flatten()
            >>> labels = [1,2,2,3,1,4]
            >>> targets = [1,1,2,3,1,4]
        '''
        self.origins.extend(labels)
        self.preds.extend(targets)
        # assert len(self.origins) == len(self.preds), "数据格式出错"
        length = len(labels)
        for i in range(0, length):
            if targets[i] == labels[i]:
                self.rights.append(targets[i])

    def result(self):
        headers = ["precision", "recall", "f1-score", "support"]
        width = max(len(cn) for cn in self.class_list)
        head_fmt = '{:>{width}s} ' + ' {:>9}' * len(headers)
        report = head_fmt.format('', *headers, width=width)
        report += '\n\n'
        row_fmt = '{:>{width}s} ' + ' {:>9.{digits}f}' * 3 + ' {:>9}\n'

        macro_pre = 0
        macro_rec = 0
        macro_f1 = 0
        # 计算各类f1值
        origin_Counter = Counter(self.origins)
        pred_Counter = Counter(self.preds)
        right_Counter = Counter(self.rights)
        for type_, count in origin_Counter.items():
            origin = count
            pred = pred_Counter.get(type_)
            right = right_Counter.get(type_)
            precision, recall, f1 = self.compute(origin, pred, right)
            report += row_fmt.format(self.id2label[type_], precision, recall,
                                     f1, count, width=width, digits=self.digits)
            macro_pre += precision
            macro_rec += recall
            macro_f1 += f1
            # class_info[self.id2label[type_]] = {"precision": round(precision, 3), 'recall': round(
            #     recall, 3), 'f1': round(f1, 3)}  # 每一个类别的信息
        report += '\n'
        # 计算macro F1
        macro_pre = macro_pre / len(self.class_list)
        macro_rec = macro_rec / len(self.class_list)
        macro_f1 = macro_f1 / len(self.class_list)
        report += row_fmt.format("macro avg", macro_pre, macro_rec,
                                 macro_f1, len(self.origins), width=width, digits=self.digits)
        # 计算micro F1
        origin = len(self.origins)
        pred = len(self.preds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(
            origin, pred, right)  # micro整体类别的信息
        report += row_fmt.format("micro avg", precision, recall,
                                 f1, len(self.origins), width=width, digits=self.digits)
        return macro_f1, report

    def Skelearn_result(self, target_names):
        # accuracy, recall, precision, f1 = skl_metric(self.origins, self.preds)
        return classification_report(self.origins, self.preds, target_names=target_names)
