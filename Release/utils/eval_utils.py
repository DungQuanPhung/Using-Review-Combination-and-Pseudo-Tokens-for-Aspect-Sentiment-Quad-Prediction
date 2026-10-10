from utils.parse_utils import parse_aste
import json
import os


def parse_and_score(pred, target, data_args):
    all_labels, all_preds = [], []
    for i in range(len(pred)):
        pred_list = parse_aste(pred[i])
        all_preds.append(pred_list)
    for i in range(len(target)):
        gold_list = parse_aste(target[i])
        all_labels.append(gold_list)
    # if target_format == "AO":
    #     all_preds = distance_aware_merge_preds(all_preds)
    #     all_labels = all_labels[: int(len(all_labels) / 2)]
    min_votes = getattr(data_args, "min_votes", -1)
    all_preds = distance_aware_merge_preds(all_preds, data_args.topk, min_votes)
    all_labels = distance_aware_merge_preds(all_labels, data_args.topk, min_votes)
    raw_score = score(all_preds, all_labels, data_args)
    return raw_score


# def distance_aware_merge_preds(preds):
#     pred_nums = len(preds) // 2
#     merged_preds = []
#     for i in range(pred_nums):
#         ao_pred, oa_pred = preds[i], preds[i + pred_nums]
#         if ao_pred == oa_pred:
#             pred = ao_pred
#         else:
#             ao_pred = list([x for x in ao_pred])
#             oa_pred = list([x for x in oa_pred])
#             pred = []
#             ao_pred_dup, oa_pred_dup = ao_pred.copy(), oa_pred.copy()
#             # add the common triplet into ans (intersection set)
#             for cur in ao_pred:
#                 if cur in oa_pred_dup:
#                     ao_pred_dup.remove(cur)
#                     oa_pred_dup.remove(cur)
#                     pred.append(cur)
#         merged_preds.append(pred)
#     return merged_preds

def distance_aware_merge_preds(preds, top_K, min_votes=-1):
    # original release hard-coded `count > 14` (unanimous vote for topk=15)
    if min_votes is None or min_votes <= 0:
        min_votes = top_K
    min_votes = min(min_votes, top_K)
    group_size = top_K
    grouped_preds = [preds[i:i + group_size] for i in range(0, len(preds), group_size)]
    merged_preds = []
    for grouped_pred in grouped_preds:
        # # 初始化交集为第一个子列表的集合
        # intersection = set(grouped_pred[0])
        #
        # # 循环遍历剩余的子列表，计算交集
        # for sublist in grouped_pred[1:]:
        #     intersection &= set(sublist)

        # intersection = set(grouped_pred[0])
        # # 循环遍历剩余的子列表，计算并集
        # for sublist in grouped_pred[1:]:
        #     intersection = intersection.union(sublist)

        # 不按照交集 也不按照并集 count表示元组存在的个数
        tuple_counts = {}
        intersection = []
        for sublist in grouped_pred:
            # 使用集合存储唯一的元组
            unique_tuples = set(sublist)
            # 更新字典中每个元组的出现次数
            for tuple_item in unique_tuples:
                tuple_counts[tuple_item] = tuple_counts.get(tuple_item, 0) + 1
        for tuple_item, count in tuple_counts.items():
            if count >= min_votes:
                intersection.append(tuple_item)
        merged_preds.append(intersection)
        # merged_preds.append(list(intersection))
    return merged_preds


# def score(all_preds, all_labels):
#     assert len(all_preds) == len(all_labels)
#     n_preds, n_labels, n_common = 0, 0, 0
#     for pred, label in zip(all_preds, all_labels):
#         n_preds += len(pred)
#         n_labels += len(label)
#         label_dup = label.copy()
#         for p in pred:
#             if p in label_dup:
#                 n_common += 1
#                 label_dup.remove(p)
#     if n_preds == 0:
#         precision = 0
#     else:
#         precision = n_common / n_preds
#     recall = n_common / n_labels
#     f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
#     return {'precision': precision, "recall": recall, "f1_score": f1_score,
#             "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}


def _prf(n_common, n_preds, n_labels):
    precision = n_common / n_preds if n_preds > 0 else 0.0
    recall = n_common / n_labels if n_labels > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _quad_type(quad):
    # quad = (aspect, category, sentiment, opinion); 'null' marks implicit
    if len(quad) < 4:
        return "other"
    a_imp = quad[0] == "null"
    o_imp = quad[3] == "null"
    if not a_imp and not o_imp:
        return "EA_EO"
    if a_imp and not o_imp:
        return "IA_EO"
    if not a_imp and o_imp:
        return "EA_IO"
    return "IA_IO"


def score(all_preds, all_labels, data_args):
    assert len(all_preds) == len(all_labels)
    n_preds, n_labels, n_common = 0, 0, 0
    types = ["EA_EO", "IA_EO", "EA_IO", "IA_IO"]
    type_counts = {t: {"n_preds": 0, "n_labels": 0, "n_common": 0} for t in types}
    json_data = {"examples": [], "result": {}}

    for pred, label in zip(all_preds, all_labels):
        is_exist = 0
        n_preds += len(pred)
        n_labels += len(label)
        for q in pred:
            t = _quad_type(q)
            if t in type_counts:
                type_counts[t]["n_preds"] += 1
        for q in label:
            t = _quad_type(q)
            if t in type_counts:
                type_counts[t]["n_labels"] += 1
        label_dup = label.copy()
        for p in pred:
            if p in label_dup:
                n_common += 1
                is_exist = 1
                label_dup.remove(p)
                t = _quad_type(p)
                if t in type_counts:
                    type_counts[t]["n_common"] += 1
        info_dict = {
            "labels": label,
            "preds": pred,
            "isExist": is_exist
        }
        json_data['examples'].append(info_dict)

    precision, recall, f1_score = _prf(n_common, n_preds, n_labels)
    result = {'precision': precision, "recall": recall, "f1_score": f1_score,
              "n_preds": n_preds, "n_labels": n_labels, "n_common": n_common}
    for t in types:
        c = type_counts[t]
        p_t, r_t, f_t = _prf(c["n_common"], c["n_preds"], c["n_labels"])
        result[f"{t}_f1"] = f_t
        result[f"{t}_n_labels"] = c["n_labels"]
    # implicit-aspect / implicit-opinion aggregates
    for name, members in (("IA", ["IA_EO", "IA_IO"]), ("IO", ["EA_IO", "IA_IO"])):
        nc = sum(type_counts[m]["n_common"] for m in members)
        npd = sum(type_counts[m]["n_preds"] for m in members)
        nl = sum(type_counts[m]["n_labels"] for m in members)
        result[f"{name}_f1"] = _prf(nc, npd, nl)[2]

    json_data['result'] = result
    output_json_path = getattr(data_args, "output_json_path", None)
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
    return result
