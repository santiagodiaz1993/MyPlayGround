# TODO(santiago): organize and documnet
# TODO(santiago): convert all code into functions and put it in another file
import os
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime, subprocess
from reminders.get_document import *
from email_api.gmailapi import *
from drive_api.drive_api import *
from reporting_functions.calling_functions import *


# TODO(santiagodiaz): Rre)ove unecessary scopes
SCOPES = ["https://www.googleapis.com/auth/drive.activity.readonly",
          "https://www.googleapis.com/auth/documents",
          "https://mail.google.com/",
          "https://www.googleapis.com/auth/drive.activity.readonly"]

reminders_completed = get_text(
    "19GFEhbZ0KlWknhEw6Js0mCEeIwNQgVeif-3Bw_yFpVs")
reminders_started = get_text(
    "1HuonqcF3SwcfwTebKL2hCNW3LjZ87jrC4byV_Zh7PQ0")

reminders_completed_list = extract_reminders_document(reminders_completed)
reminders_initiated_list = extract_reminders_document(reminders_started)


log_tasks("tasks_completed", reminders_completed_list)


current_date = str(dt.today().date())
range_of_reminders_completed = classify_by_date_range(reminders_completed_list, 3)
plt.bar(range_of_reminders_completed.keys(), range_of_reminders_completed.values())
plt.savefig("productivity_chart_" + current_date + ".png")
plt.show()

number_of_tasks_completed = len(reminders_completed_list)
number_of_tasks_initiated = len(reminders_initiated_list)

categorized_tasks = categorize_tasks(reminders_completed_list)
floats = [int(categorized_tasks[v]) for v in categorized_tasks]
keys = [keys for keys in categorized_tasks]
plt.pie(floats,labels = keys)
plt.savefig("productivity_distribution_" + str(current_date) + ".png")
plt.show()

request_message = CreateMessageWithAttachment(
    "santiagobmxdiaz@gmail.com",
    "santiagobmxdiaz@gmail.com",
    "weekly Productivity Report",
    load_template(
        "template1",
        reminders_completed_list,
        reminders_completed_list),
    "",
    ["productivity_distribution_" + current_date + ".png",
     "productivity_chart_" + current_date + ".png"])

SendMessage(authorization(), "santiagobmxdiaz@gmail.com", request_message)

