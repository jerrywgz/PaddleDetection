# -*- coding: utf-8 -*-


def __init__():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    _global_dict[key] = value


def get_value(key, defValue=0.):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
