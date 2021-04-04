import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import smtplib
import configparser
import logging
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import warnings
from email.mime.text import MIMEText
from io import StringIO 
warnings.filterwarnings("ignore")

home_dir = 'H:/FYP-django/iSegmentor/'

class Alerter:

    def __init__(self):
        self.email_config = configparser.ConfigParser()

    def send_emails(self, html):
        #print("hi")
        self.email_config.read(
            '{}/config_files/email_config.ini'.format(home_dir))
        email_list = self.email_config.get('LIST', 'emails').strip().split(',')
        #print(email_list)
        COMMASPACE = ', '

        msg = MIMEMultipart()
        msg['Subject'] = 'iSegmentor Alert'
        msg['From'] = "isegmentor.info@gmail.com"
        msg['To'] = COMMASPACE.join(email_list)
        #print(msg)

        message = MIMEText(html, 'plain')
        msg.attach(message)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login("isegmentor.info@gmail.com", "iSegmentor97")
        server.send_message(msg)
        server.quit()

    # def main(self):
    #     print("Sending E-Mails... ")
    #     print('')

    #     string = "Test Email" 
    #     self.send_emails(string)
