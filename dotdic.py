import numpy as np
import copy

# 重写了几个特殊方法以提供更方便的属性访问和复制功能
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	# 重写了 copy.deepcopy 的行为，使其在复制一个 DotDic 实例时保留其属性的特殊行为。
	# 这允许复制一个 DotDic 实例时保持其属性访问和设置方式。
	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))