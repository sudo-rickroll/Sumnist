import configparser

def parse_config(path):
    config_dict={}
    parser = configparser.ConfigParser()
    parser.read(path)
    for section in parser.sections():
        config_dict.setdefault(section,{})
        for key, value in parser.items(section):
            config_dict[section].setdefault(key, value)
    return config_dict



