import filters as f
import preprocess as p


def no_prep(text):
    return text


def no_usernames(text):
    return f.no_username(text)


def html_decode(text):
    return p.html_decode(text)


def remove_noise(text):
    text = p.html_decode(text)
    text = f.no_url(text)
    text = f.no_username(text)
    text = f.hash_as_normal(text)
    text = f.no_rt_tag(text)
    text = f.reduce_letter_duplicates(text)
    text = f.quote_placeholder(text)
    text = p.naive_negation_attachment(text)
    return text.strip()


def remove_for_negation(text):
    text = p.html_decode(text)
    text = f.no_url(text)
    text = f.no_username(text)
    text = f.hash_as_normal(text)
    text = f.no_rt_tag(text)
    text = f.reduce_letter_duplicates(text)
    text = f.quote_placeholder(text)
    return text.strip()


def remove_all(text):
    text = p.html_decode(text)
    text = f.no_url(text)
    text = f.no_username(text)
    text = f.no_hash(text)
    text = f.no_emoticons(text)
    text = f.no_rt_tag(text)
    text = p.naive_negation_attachment(text)
    return text


def placeholders(text):
    text = f.url_placeholder(text)
    text = f.username_placeholder(text)
    text = f.hash_placeholder(text)
    return text


def reduced_attached(text):
    text = f.reduce_letter_duplicates(text)
    text = p.naive_negation_attachment(text)
    return text


def no_url_username(text):
    text = p.html_decode(text)
    text = f.no_url(text)
    text = f.no_username(text)
    return text.strip()


def no_url_username_reduced_attached(text):
    text = f.no_url(text)
    text = f.no_username(text)
    text = f.reduce_letter_duplicates(text)
    text = p.naive_negation_attachment(text)
    return text


def all(text):
    text = html_decode(text)
    text = text.lower()
    text = f.no_url(text)
    text = f.no_username(text)
    text = f.no_emoticons(text)
    text = f.no_hash(text)
    text = f.no_rt_tag(text)
    text = f.reduce_letter_duplicates(text)
    text = p.naive_negation_attachment(text)
    return text
