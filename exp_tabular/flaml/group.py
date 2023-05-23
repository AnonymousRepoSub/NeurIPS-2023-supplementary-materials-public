import json
import os


def set_test_group(group_value=None, group_num=-1, train_len=0):
    dictionary = {
        'group_num': group_num,
        'group_value':group_value,
    }
    folder = 'group_file'
    if not os.path.exists(folder):
        os.mkdir(folder)
    name = f"{folder}/group_{train_len}.txt"

    if os.path.exists(name):
        print(f"{folder}/group_{train_len}.txt already exists.", flush=True)
        return
    print(f"Write to {folder}/group_{train_len}.txt", flush=True)
    with open(name, 'w') as f:
        json.dump(dictionary, f)


def get_test_group(train_len):
    try:
        # print(f"opening group_{train_len}.txt")
        with open(f"./group_file/group_{train_len}.txt", 'r') as f:
            return json.load(f)
    except Exception as e:
        result ={
        'group_num': -1,
        'group_value': None,
        }
        return result
