from sklearn.feature_extraction.text import TfidfVectorizer
from storage import cache
from storage import data
import scipy


class TfidfNegTransformer(TfidfVectorizer):
    def __init__(self, negate=True, **kwargs):
        super(TfidfNegTransformer, self).__init__(self, **kwargs)
        self.negate = negate

    def transform(self, tweets, **kwargs):
        resname = ''.join(tweets) + str(**kwargs)
        cached = cache.get(resname)
        if not isinstance(cached, scipy.sparse.csr.csr_matrix):
            print "cached not found"
            print type(cached)
            cached = super(TfidfNegTransformer, self).transform(tweets, **kwargs)
            cache.save(resname, cached)
        else:
            self.X = cached
        return cached

