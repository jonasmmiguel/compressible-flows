import dill
from time import strftime


def save_session(user_file_name):
    time_now = strftime("%Y%m%d-%H%M%S")
    filename = (user_file_name + time_now + '.pkl')
    dill.dump_session(filename)


def load_session(file_name_pattern):
    dill.load_session(filename)