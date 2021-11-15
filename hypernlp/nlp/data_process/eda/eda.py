'''
EDA model: https://arxiv.org/abs/1901.11196
'''


def eda_model(model_type, num_aug=9):

    if model_type == 'chinese':
        from hypernlp.nlp.data_process.eda.eda_chinese import EdaChinese
        return EdaChinese(num_aug)
    elif model_type == 'cased' or model_type == 'uncased':
        from hypernlp.nlp.data_process.eda.eda_english import EdaEnglish
        return EdaEnglish(num_aug)
    else:
        raise ValueError('Unknown EDA type for {}!'.format(model_type))

