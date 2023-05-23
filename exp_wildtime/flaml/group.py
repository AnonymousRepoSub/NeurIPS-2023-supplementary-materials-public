import json
def set_test_group(group_value=None, group_num=-1, train_len=0):
    dictionary = {
        'group_num': group_num,
        'group_value':group_value,
        'train_len': train_len,
    }
    with open('group.txt', 'w') as f:
        json.dump(dictionary, f)


def get_test_group():
    try:
        with open('group.txt', 'r') as f:
            return json.load(f)
    except json.decoder.JSONDecodeError:
        result ={
        'group_num': -1,
        'group_value': None,
        'train_len': 0,
        }
        return result
