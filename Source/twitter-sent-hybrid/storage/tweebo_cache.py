import cache

class TweeboCacher(object):
    pos_counts = cache.load_json("pos_counts_cache")
    tokens = cache.load_json("tokens_cache")
    pos_tokens = cache.load_json("pos_tokens_cache")
    dependencies = cache.load_json("dependency_cache")

    @staticmethod
    def get_cached_pos_counts():
        return TweeboCacher.pos_counts

    @staticmethod
    def get_cached_pos_tokens():
        return TweeboCacher.pos_tokens

    @staticmethod
    def get_cached_dependency():
        return TweeboCacher.dependencies

    @staticmethod
    def get_cached_tokens():
        return TweeboCacher.tokens
