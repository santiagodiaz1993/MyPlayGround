"""This script tests the ability of a user to upload their resume after filling up the 
application page.

Fature: The job application page should allow users to submit their resume through a i
upload pop up box

Scenario:Resume should uploaded when the user fills up all requiered fields a


 Given: I am on Qualitest's home page 
 When I click on "Careers" on the 'Home' page
   And I click on "Open positions" on the "Careers" page
   And I click on "Austin" on the "Open positions" page
   And I click on "6036- Bilingual Engineer - Spanish" link under "Open Positions" page 
   And I click on "Apply for this job" on the "6036- Bilingual Engineer - Spanish" page
   And I enter "Santiago" in the "First Name" field
   And I enter "Diaz" in the "Last name" field
   And I enter "santi@gmail.com" in the "email" field
   And I enter "hello" in the "headline" field
   And I enter "1590 Lexington Ave" in the "Address" field
   And I enter "I am a Qualitester" in the "summary" field
   And I click on "Upload my resume" in the "6036- Bilingual Engineer - Spanish" page
 Then the file selection box should become visibile for resume selection

"""

import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
print("Script Started...")


browser = webdriver.Firefox()
browser.get("http://qualitestgroup.com/")

careers = browser.find_element_by_id("mega-menu-item-699")
careers.click()

open_positions = browser.find_element_by_id("menu-item-1158")
open_positions.click()

# Scoll until all cities becomes visibile + wait 1 second
browser.execute_script("window.scrollTo(0, 888)")
time.sleep(2)

austin = browser.find_elements_by_class_name("cities_item")[3]
austin.click()

# Scroll until the open positions are visibile + wait 1 second
browser.execute_script("window.scrollTo(0, 888)")
time.sleep(2)

position_name = browser.find_element_by_link_text("6036- Bilingual Engineer - Spanish")
position_name.click()
time.sleep(2)

browser.switch_to.window(browser.window_handles[1])

browser.execute_script("window.scrollTo(0, 99999)")
time.sleep(2)

apply_job_button = browser.find_element_by_link_text("Apply for this job")
apply_job_button.click()

first_name_field = browser.find_element_by_id("firstname")
first_name_field.send_keys("Santiago")

last_name_field = browser.find_element_by_id("lastname")
last_name_field.send_keys("Diaz")

email_field = browser.find_element_by_id("email")
email_field.send_keys("santi@gmail.com")

headline_field = browser.find_element_by_id("headline")
headline_field.send_keys("hello")

address_field = browser.find_element_by_id("address")
address_field.send_keys("1590 lexington ave")

browser.execute_script("window.scrollTo(0, 1200)")
time.sleep(2)

summary_field = browser.find_element_by_id("summary")
summary_field.send_keys("I am a Qualitester")

upload_resume = browser.find_element_by_xpath("//*[@class='_-_-shared-ui-atoms-button-base-___button__button _-_-shared-ui-atoms-button-base-___button__normal _-_-shared-ui-atoms-button-tertiary-___tertiary__default ']")
upload_resume.click()

upload_resume.send_keys(os.getcwd() + "/resume_recommendation.pdf")

print("Script finished successfully")
