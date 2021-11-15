import os


def generate_model_name(name, framework, model_type):
    if framework == 'tensorflow':
        framework_token = 'tf'
    elif framework == 'pytorch':
        framework_token = 'pt'
    else:
        return None
    return "".join([name, "_", framework_token, "_", model_type])


def home_path():
    return os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".") + '/../'


if __name__ == '__main__':

    print(generate_model_name('bert', 'tensorflow', 'chinese'))
    print(generate_model_name('bert', 'keras', 'chinese'))
