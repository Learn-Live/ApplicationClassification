"""
https://github.com/pangwong/pytorch-multi-label-classifier/blob/master/test/celeba/label.txt
"""

import collections


def load_label(label_file):
    rid2name = list()  # rid: real id, same as the id in label.txt
    id2rid = list()  # id: number from 0 to len(rids)-1 corresponding to the order of rids
    rid2id = list()
    with open(label_file) as l:
        rid2name_dict = collections.defaultdict(str)
        id2rid_dict = collections.defaultdict(str)
        rid2id_dict = collections.defaultdict(str)
        new_id = 0
        for line in l.readlines():
            line = line.strip('\n\r').split(';')
            if len(line) == 3:  # attr description
                if len(rid2name_dict) != 0:
                    rid2name.append(rid2name_dict)
                    id2rid.append(id2rid_dict)
                    rid2id.append(rid2id_dict)
                    rid2name_dict = collections.defaultdict(str)
                    id2rid_dict = collections.defaultdict(str)
                    rid2id_dict = collections.defaultdict(str)
                    new_id = 0
                rid2name_dict["__name__"] = line[2]
                rid2name_dict["__attr_id__"] = line[1]
            elif len(line) == 2:  # attr value description
                rid2name_dict[line[0]] = line[1]
                id2rid_dict[new_id] = line[0]
                rid2id_dict[line[0]] = new_id
                new_id += 1
        if len(rid2name_dict) != 0:
            rid2name.append(rid2name_dict)
            id2rid.append(id2rid_dict)
            rid2id.append(rid2id_dict)
    return rid2name, id2rid, rid2id


if __name__ == '__main__':
    label_file = '../data/label.txt'
    rid2name, id2rid, rid2id = load_label(label_file)

    print(rid2name)
    print(id2rid)
    print(rid2id)

    num_classes = [len(item) - 2 for item in rid2name]
    print(num_classes)
