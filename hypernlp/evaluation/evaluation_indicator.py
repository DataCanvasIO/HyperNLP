import numpy as np


def mse_indicator(pred_res):
    data = pred_res
    residual = np.zeros(len(data[0][0]))
    for d in data:
        pred, true = d[0], d[1]
        assert len(d[0]) == len(d[1])
        residual += (pred - true) ** 2 / data.shape[0]
    return sum(residual), "mse"


def abs_indicator(pred_res):
    data = pred_res
    residual = np.zeros(len(data[0][0]))
    for d in data:
        pred, true = d[0], d[1]
        assert len(d[0]) == len(d[1])
        residual += np.abs(pred - true) / data.shape[0]
    return sum(residual), "abs"


def acc_indicator(pred_res):
    data = pred_res
    hits = 0
    for d in data:
        pred, true = d[0], d[1]
        if pred.shape[0] == 1:
            index = 1 if pred[0] >= 0.5 else 0
        else:
            index = np.argmax(pred)
        if index == true:
            hits += 1
    return float(hits) / len(data), "acc"


def precision_indicator(pred_res):
    data = pred_res
    hits = {}
    preds = {}
    trues = {}
    res = {}
    for d in data:
        pred, true = d[0], d[1]
        if pred.shape[0] == 1:
            index = 1 if pred[0] >= 0.5 else 0
        else:
            index = np.argmax(pred)
        if index in preds.keys():
            preds[index] += 1
        else:
            preds[index] = 1
        if true in trues.keys():
            trues[true] += 1
        else:
            trues[true] = 1
        if true not in hits.keys():
            hits[true] = 0
        if index == true:
            hits[true] += 1
    data = {"predicts": preds, "hits": hits, "gt": trues}
    for cls in hits.keys():
        if cls in preds.keys():
            res[cls] = float(hits[cls]) / preds[cls]
    data["results"] = res
    return data, "precision"


def recall_indicator(pred_res):
    data = pred_res
    hits = {}
    preds = {}
    trues = {}
    res = {}
    for d in data:
        pred, true = d[0], d[1]
        if pred.shape[0] == 1:
            index = 1 if pred[0] >= 0.5 else 0
        else:
            index = np.argmax(pred)
        if index in preds.keys():
            preds[index] += 1
        else:
            preds[index] = 1
        if true in trues.keys():
            trues[true] += 1
        else:
            trues[true] = 1
        if true not in hits.keys():
            hits[true] = 0
        if index == true:
            hits[true] += 1
    data = {"predicts": preds, "hits": hits, "gt": trues}
    for cls in hits.keys():
        if cls in preds.keys():
            res[cls] = float(hits[cls]) / trues[cls]
    data["results"] = res
    return data, "recall"


def f1_score_indicator(pred_res):
    precision, _ = precision_indicator(pred_res)
    recall, _ = recall_indicator(pred_res)
    indicator = 0
    for cls in precision["results"].keys():
        indicator += (2 * precision["results"][cls] * recall["results"][cls]) / (
                    precision["results"][cls] + recall["results"][cls])
    return indicator / len(precision.keys()), "f1_score"
