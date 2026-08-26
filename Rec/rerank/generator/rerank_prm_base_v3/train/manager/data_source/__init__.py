import json
import sys, os

def load_json(path):
    import json
    lines = []
    with open(path) as f:
        for row in f.readlines():
            if row.strip().startswith("//"):
                continue
            lines.append(row)
    return json.loads("\n".join(lines))

def process(filename):
    try:
        file = "/manager/data_source/{}".format(filename)
        file = os.path.abspath(sys.path[0]) + file
        dul_crf_conf = load_json(file)
        return dul_crf_conf
    except IOError:
        file = "/../manager/data_source/{}".format(filename)
        file = os.path.abspath(sys.path[0]) + file
        dul_crf_conf = load_json(file)
        return dul_crf_conf

dul_crf_conf = process("dynamic_json_config_dul_crf.json")

hot_rerank_conf = process("dynamic_json_config_old_hot_rerank.json")


