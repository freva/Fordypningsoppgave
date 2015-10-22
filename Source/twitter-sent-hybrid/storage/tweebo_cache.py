import cache

class TweeboCacher(object):
    pos_counts = cache.get("pos_counts_cache", False)
    tokens = cache.get("token_cache", False)
    pos_tokens = cache.get("pos_tokens_cache", False)
    dependencies = cache.get("dependency_cache", False)

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
