from alert_service import Alerter
import logging
import traceback
import time
from threading import Thread

def manager():
    alerter = Alerter()
    alerter.send_emails("Traing Started")


if __name__ == "__main__":
    t1 = Thread(target=manager)
    t1.start()