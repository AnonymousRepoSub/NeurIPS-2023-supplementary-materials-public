import pickle
from collections import defaultdict

final_result = defaultdict(list)

method = 'pre'

for seed in range(1,6):
    path = "{method}_seed_{seed}.pckl".format(method = method, seed = seed)
    savepath = "processed_{method}_seed_{seed}.pckl".format(method = method, seed = seed)
    f = open(path,"rb")
    result = pickle.load(f)
    f.close()

    keys = result.keys()

    start_time = 0
    for key in keys:
        start_time += result[key]['time_total_s']
        final_result["wall_clock_time"].append(start_time)
        final_result["error_rate"].append(result[key]['error_rate'])
        final_result["flops"].append(result[key]['flops'])

    f = open(savepath, "wb")
    pickle.dump(final_result, f)
    f.close()

