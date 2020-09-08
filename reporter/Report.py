# TODO(santiago): create unit testing for all of the functions
import subprocess
import matplotlib.pyplot as plt
from datetime import datetime as dt
import datetime
from reminders.get_document import *
from email_api.gmailapi import *


class ReportGenerator:
    """contains a set of functions that maniulates a list of tasks for the
    creation of a report"""

    reminders_formatted = []

    def __init__(self, document):
        self.document = document
        self.tasks_with_priority_set = []
        self.number_of_tasks_completed = int()
        self.number_of_tasks_initiated = int()
        self.tasks_distribution = {}
        self.has_tasks_with_priority = False

    # TODO(santiago): change this function so it gets the reminders from
    # the documents by requesting the json format of the file
    def get_reminders_from_document(self):
        """takes document as a string and return list of tasks and their
        attributes"""
        reminders_list = self.split(" //-//")
        reminders_formatted = []
        for reminder in reminders_list:
            reminder = reminder[1:]
            reminders = reminder.split("$$")
            reminders = reminders[:-1]
            reminders_formatted.append(reminders)
        reminders_formatted = reminders_formatted[:-1]
        return reminders_formatted

    def get_tasks_with_priority_set(self):
        """takes a list of tasks and their attributes and returns a list
        with the name of tasks that have a priority set"""
        tasks_with_priority_set = []
        for reminder in self:
            if (
                reminder[3] == " Low"
                or reminder[3] == " Medium"
                or reminder[3] == " High"
            ):
                tasks_with_priority_set.append(reminder[1])
        return tasks_with_priority_set

    def get_tasks_name(self):
        """takes a list of tasks and their attributes and returns the name of
        the tasks in string format"""
        reminders_names = ""
        for reminder in self:
            reminders_names = reminders_names + reminder[1] + ", "
        return reminders_names

    def get_tasks_in_time_range(self, number_of_days):
        """takes a list of tasks and returns the tasks that have a
        date in  between numbers_of_days and the present"""
        last_week_reminders = []
        for reminder in self:
            date = reminder[2]
            date = dt.strptime(date, " %b %d, %Y at %I:%M%p")
            today = dt.today()
            start_date_range = today - datetime.timedelta(days=number_of_days)
            if date >= start_date_range:
                last_week_reminders.append(reminder)
            else:
                print("There where no new reminders found")
        return last_week_reminders

    def create_date_ranges(self):
        "This comment is done using vim"
        """creates a list with date ranges"""
        range_assignment_dates = {}
        starting_limit_date = dt(2020, 5, 1).date()
        date_range = []
        date = dt.today().date()
        while starting_limit_date <= date:
            date_range.insert(0, date)
            date = date - datetime.timedelta(days=self)
        for index in range(len(date_range) - 1):
            date_key_name = (
                str(date_range[index]) + " to " + str(date_range[index + 1])
            )
            range_assignment_dates[date_key_name] = 0
        return range_assignment_dates, date_range

    def classify_tasks_in_date_range(self, number_of_days):
        """counts tasks that fall in each date range"""
        (
            range_assignment_dates,
            date_range,
        ) = ReportGenerator.create_date_ranges(number_of_days)
        for index1 in range(len(date_range) - 1):
            for reminder in self:
                date_format = " %b %d, %Y at %I:%M%p"
                date_reminder = dt.strptime(reminder[2], date_format).date()
                if (
                    date_range[index1]
                    <= date_reminder
                    < date_range[index1 + 1]
                ):
                    date_key_name = (
                        str(date_range[index1])
                        + " to "
                        + str(date_range[index1 + 1])
                    )
                    range_assignment_dates[date_key_name] += 1
        return range_assignment_dates

    def categorize_tasks(self):
        """counts tasks for each task category"""
        categories_count = {
            "Work": 0,
            "Personal Errands": 0,
            "Machine Learning Project": 0,
            "Artificial Inteligence Podcast": 0,
            "Movies to do": 0,
            "VIM Learning": 0,
            "Testing List": 0,
        }
        for reminder in self:
            for category in categories_count:
                if reminder[0] == category:
                    categories_count[category] += 1
        return categories_count

    def get_number_of_reminders(self):
        """retusn number of reminders given"""
        return len(self)


class ReportGraphing:
    def build_bar_chart(self, number_of_days):
        """creates a bar chart of for showing progression of initated and
        closed tasks over time"""
        plt.bar(
            ReportGenerator.classify_tasks_in_date_range(
                self, number_of_days
            ).keys(),
            ReportGenerator.classify_tasks_in_date_range(
                self, number_of_days
            ).values(),
        )

    def build_pie_chart(self):
        """creates pie chart for the number of tasks closed and initated in
        each working category"""
        tasks_to_graph = ReportGenerator.categorize_tasks(self)
        keys = [key for key in tasks_to_graph]
        values = [int(tasks_to_graph[value]) for value in tasks_to_graph]
        plt.pie(values, labels=keys)


class TaskLogging:
    def load_template(
        self, tasks_organized_completed, tasks_organized_initiated
    ):
        """loads templaes that contain templates for emial reporting"""
        template_path = "templates/" + str(self) + ".txt"
        with open(template_path) as chosen_template:
            read_file = chosen_template.read()
            read_file = read_file % (
                ReportGenerator.get_number_of_reminders(
                    tasks_organized_completed
                ),
                ReportGenerator.get_number_of_reminders(
                    tasks_organized_initiated
                ),
                ReportGenerator.get_tasks_with_priority_set(
                    tasks_organized_completed
                ),
                ReportGenerator.get_tasks_with_priority_set(
                    tasks_organized_initiated
                ),
            )
        return read_file

    def log_tasks(self, tasks_to_log):
        """creates a directory and saves tasks as a document in string
        format"""
        current_date = str(dt.today().date())
        path_and_name = "task_logging/" + current_date + "/" + self + ".txt"
        path = "task_logging/" + current_date
        subprocess.call("mkdir " + path, shell=True)
        subprocess.call("touch " + path_and_name, shell=True)
        log = ""
        for tasks in tasks_to_log:
            log = log + tasks[1] + ", "
        with open(path_and_name, "w") as task_completed:
            task_completed.write(log)
        return log


# TODO(santiago): Remove unnecessary scopes
SCOPES = [
    "https://www.googleapis.com/auth/drive.activity.readonly",
    "https://www.googleapis.com/auth/documents",
    "https://mail.google.com/",
]


reminders_completed = get_text("19GFEhbZ0KlWknhEw6Js0mCEeIwNQgVeif-3Bw_yFpVs")
reminders_started = get_text("1HuonqcF3SwcfwTebKL2hCNW3LjZ87jrC4byV_Zh7PQ0")

tasks_organized = ReportGenerator.get_reminders_from_document(
    reminders_completed
)
# TaskLogging.log_tasks("test", tasks_organized)
# ReportGraphing.build_bar_chart(tasks_organized, 3)
# ReportGraphing.build_pie_chart(tasks_organized)
# ReportGenerator.get_tasks_with_priority_set(tasks_organized)
# ReportGenerator.get_number_of_reminders(tasks_organized)
# ReportGenerator.get_tasks_name(tasks_organized)
# ReportGenerator.get_tasks_in_time_range(tasks_organized, 22)
# ReportGenerator.create_date_ranges(3)
# ReportGenerator.classify_tasks_in_date_range(tasks_organized, 4)
# ReportGenerator.categorize_tasks(tasks_organized)

current_date = str(dt.today().date())
request_message = EmailInteraction.CreateMessageWithAttachment(
    "santiagobmxdiaz@gmail.com",
    "santiagobmxdiaz@gmail.com",
    "weekly Productivity Report",
    TaskLogging.load_template("template1", tasks_organized, tasks_organized),
    "",
    [
        "productivity_distribution_" + current_date + ".png",
        "productivity_chart_" + current_date + ".png",
    ],
)

EmailInteraction.SendMessage(
    EmailInteraction.authorization(),
    "santiagobmxdiaz@gmail.com",
    request_message,
)
