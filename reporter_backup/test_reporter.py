#!/usr/bin/env python
# coding=utf-8

import os
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime, subprocess
from reminders.get_document import *
from email_api.gmailapi import *
from drive_api.drive_api import *
from reporting_functions.calling_functions import *


def test_extract_reminders_document():
    document = ("Testing List$$ Ffoutfdrdr7isyidd$$ May "
               "09, 2020 at 11:41PM$$ None$$ //-//")
    result = extract_reminders_document(document)
    testing_list =[['Testing List',
                    ' Ffoutfdrdr7isyidd',
                    ' May 09, 2020 at 11:41PM',
                    ' None']]
    assert result == testing_list


def test_log_tasks():
    tasks_for_logging_test = [
    ['Testing List', ' Ffoutfd', ' May 09, 2020 at 11:41PM', ' None'],
     ['Testing List', ' Jgkccccccui', ' May 09, 2020 at 11:41PM', ' None'],
     ['Testing List', ' Hgkcgjcutcu', ' May 09, 2020 at 11:41PM', ' Medium']
    ]
    log = log_tasks("unit_test", tasks_for_logging_test )
    expected = " Ffoutfd,  Jgkccccccui,  Hgkcgjcutcu, "
    assert log == expected


def test_check_for_high_priority():
    reminder = [
        ['Testing List', ' Hgkcgjcut', ' May 09, 2020 at 11:41PM', ' Medium'],
        ['Testing List', ' Ffoutf', ' May 09, 2020 at 11:41PM', ' None']
    ]
    from_function = check_for_high_priority(reminder)
    expected = [" Hgkcgjcut"]
    assert expected == from_function


def test_get_tasks_in_time_range():











