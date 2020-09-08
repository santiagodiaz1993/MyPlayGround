# TODO(santiago): organize and documnet
# TODO(santiago): create unit testing for all of the functions
import os
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime, subprocess
from reminders.get_document import *
from email_api.gmailapi import *
from drive_api.drive_api import *


def log_tasks(file_name, tasks_to_log):
    current_date = str(dt.today().date())
    path_and_name = "task_logging/" + current_date + "/" + file_name + ".txt"
    path = "task_logging/" + current_date
    subprocess.call("mkdir " + path, shell=True)
    subprocess.call("touch " + path_and_name, shell=True)
    log = ""
    for tasks in tasks_to_log:
        log = log + tasks[1] + ", "
    with open(path_and_name, "w") as task_completed:
        task_completed.write(log)
    return log


def extract_reminders_document(reminder_doc):
    reminder_doc = reminder_doc.split(" //-//")
    reminder_doc = [reminder.strip() for reminder in reminder_doc]
    reminder_doc = [reminder.split("$$") for reminder in reminder_doc]
    reminder_doc = [reminder[:-1] for reminder in reminder_doc]
    reminder_doc = reminder_doc[:-1]
    return reminder_doc


def check_for_high_priority(tasks_document):
    for reminder in tasks_document:
        if reminder[3] != " None":
            tasks_with_priority = []
            tasks_with_priority.append(reminder[1])
    print(tasks_with_priority)
    return tasks_with_priority


def get_name_of_reminders(reminders):
    reminders_names = ""
    for reminder in reminders:
        reminders_names.append(reminders + " ")
    return reminders_names


def get_tasks_in_time_range(reminders, number_of_days):
    last_week_reminders = []
    for reminder in reminders:
        date = reminder[2]
        date = dt.strptime(date, " %b %d, %Y at %I:%M%p")
        today = dt.today()
        start_date_range = today - datetime.timedelta(days=number_of_days)
        if date >= start_date_range:
            last_week_reminders.append(reminder)
        else:
            print("There where no new reminders found")
    return last_week_reminders


def classify_by_date_range(reminders, number_of_days):
    range_assignmnet_dates = {}
    starting_limit_date = dt(2020, 5, 1).date()
    date_range = []
    date = dt.today().date()
    while starting_limit_date < date:
        date_range.append(date)
        date = date - datetime.timedelta(days=number_of_days)
    for index in range(1, len(date_range)):
        range_assignmnet_dates["range" + str(index)] = 0
    for index in range(len(date_range) - 1):
        for index2, reminder in enumerate(reminders):
            date_format = " %b %d, %Y at %I:%M%p"
            date_reminder = dt.strptime(reminder[2], date_format)
            date_reminder = date_reminder.date()
            if date_range[index] >= date_reminder >= date_range[index + 1]:
                range_assignmnet_dates["range" + str(index + 1)] += 1
    return range_assignmnet_dates


def categorize_tasks(list_of_tasks):
    categories = {
        "Work": 0,
        "Personal Errands": 0,
        "Machine Learning Project":0,
        "Artificial Inteligence Podcast": 0,
        "Movies to do": 0,
        "VIM Learning": 0,
        "Testing List": 0
    }
    for reminder in list_of_tasks:
        for category in categories:
            if reminder[0] == category:
                categories[category] += 1
    return categories


def load_template(template, reminders_completed, reminders_initiated):
    template_path = "templates/" + str(template) + ".txt"
    with open(template_path) as chosen_template:
        read_file = chosen_template.read()
        read_file = read_file % (len(reminders_completed),
                                 len(reminders_initiated),
                                 check_for_high_priority(reminders_completed),
                                 check_for_high_priority(reminders_initiated))
        print(read_file)
        return read_file


def get_tasks_name(list_of_tasks):
    name_of_tasks = ""
    for task in list_of_tasks:
        tasks_name = tasks[1]
        name_of_tasks.append(tasks_name + ",")
    return name_of_tasks

