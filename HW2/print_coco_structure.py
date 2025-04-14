import json
from collections import defaultdict


def print_dict_tree(d, indent=0):
    prefix = " " * indent + "├── "
    if isinstance(d, dict):
        for k, v in d.items():
            print(f"{prefix}{k}", end="")
            if isinstance(v, (dict, list)):
                print()
                print_dict_tree(
                    v[0] if isinstance(v, list) and len(v) > 0 else v, indent + 4
                )
            else:
                print(f": {type(v).__name__}")
    elif isinstance(d, list):
        print(f"{prefix}[{len(d)} items]")
        if len(d) > 0:
            print_dict_tree(d[0], indent + 4)


# 載入 coco json
with open("nycu-hw2-data/train.json", "r") as f:
    coco = json.load(f)

print("COCO JSON 結構：")
print_dict_tree(coco)
