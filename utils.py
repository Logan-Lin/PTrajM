import os
import string
import random
from datetime import datetime


def get_datetime_key():
    """ Get a string key based on current datetime. """
    return 'D' + datetime.now().strftime("%Y_%m_%dT%H_%M_%S_") + get_random_string(4)


def get_random_string(length):
    letters = string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def create_if_noexists(path):
    if not os.path.exists(path):
        os.makedirs(path)
