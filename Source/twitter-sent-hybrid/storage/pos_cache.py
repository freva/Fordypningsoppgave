from subprocess import Popen, PIPE, STDOUT
import io
import re
import os
import random
import cache
pos_cache = cache.load_pickle("pos_cache", False)
pos_cache = pos_cache if pos_cache else {}
pos_tags_RE = re.compile(ur'_([A-Z]+)\s')


def get_pos_tags(tweets):
    raw_tweets = [tweet for tweet in tweets if tweet not in pos_cache]
    gate_tagger_path = "../data/gate_pos_tagger/"

    if raw_tweets:
        temp_file = "temp_" + ('%030x' % random.randrange(16**30)) + ".temp" 
        f = io.open(gate_tagger_path + temp_file, "w", encoding="utf-8")
        f.write("\n".join(raw_tweets))
        f.close()

        old_dir = os.getcwd()
        os.chdir(gate_tagger_path)
        p = Popen(['java', '-Xmx1024m', '-jar', 'twitie_tag.jar', 'models/gate-EN-twitter.model', temp_file],
                  stdout=PIPE, stderr=STDOUT)
        os.chdir(old_dir)

        raw_pos = [pos_tags_RE.findall(line) for line in p.stdout if "_" in line]
        pos_cache.update(dict(zip(raw_tweets, raw_pos)))
        cache.save_pickle("pos_cache", pos_cache, False)
        os.remove(gate_tagger_path + temp_file)
    return [pos_cache[tweet] for tweet in tweets]
