BOUNDING_BOX_TEMPLATE = "bounding_box{}.csv"
LANDMARKS_TEMPLATE = "landmarks{}.csv"
EMBEDDINGS_TEMPLATE = "embeddings{}.csv"

def file_name(template, suffix):
    if suffix:
        suffix = "_" + suffix
    return template.format(suffix)

def bounding_box_file_name(suffix):
    return file_name(BOUNDING_BOX_TEMPLATE, suffix)

def landmarks_file_name(suffix):
    return file_name(LANDMARKS_TEMPLATE, suffix)

def embeddings_file_name(suffix):
    return file_name(EMBEDDINGS_TEMPLATE, suffix)