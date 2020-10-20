#!/usr/bin/env python
# coding=utf-8


def print_messag(message):
    print(message)

    def print_message_second_time(message2):
        print(message2)

    return print_message_second_time


print_messag("hellow WOrld")
