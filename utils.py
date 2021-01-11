def get_or_default(d, key, default_value):
    if key in d:
        return d[key]
    else:
        return default_value
