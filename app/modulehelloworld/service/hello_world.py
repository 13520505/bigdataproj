from nar_module.nar.utils import deserialize


def hello_world(name):
    if name is None or len(name) == 0:
        return "Hello World!!!"
    else:
        return  name + " Hello "

def load_acr_module_resources(path):
    (acr_label_encoders, articles_metadata_df, content_article_embeddings) = \
        deserialize(path)
    return acr_label_encoders, articles_metadata_df, content_article_embeddings