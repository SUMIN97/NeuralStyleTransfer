import sys, os
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib.request
from urllib.request import urlopen
import requests
import random
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import errno

###initial set
driver = webdriver.Chrome('/Users/sumin/Downloads/chromedriver')
driver.get('https://bam-dataset.org/')
driver.implicitly_wait(5)

MEDIA_NUM = 7
CONTENT_NUM = 9
IMG_NUM = 48
n_epoch = 10
# baseDir = "/Users/sumin/PycharmProjects/NeuralStyleTransfer/code/data"
baseDir = os.path.abspath("./")

MEDIA_LIST = ['3DGraphics', 'Comic', 'Pencil', 'Oil', 'Pen', 'VectorArt', 'Watercolor']
CONTENT_LIST = ['Bicycle', 'Bird', 'Building', 'Cars', 'Cat', 'Dog', 'Flower', 'People', 'Tree']

try:
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".Media"))
    )
finally:
    for epoch in range(1, n_epoch):
        print(epoch)
        for m in range(1, MEDIA_NUM+1):
            driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[1]/ul[1]/button[%d]' % m).click()
            driver.implicitly_wait(10)
            for c in range(1, CONTENT_NUM+1):
                driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[1]/ul[2]/button[%d]' % c).click()
                driver.implicitly_wait(10)

                for idx in range(IMG_NUM):
                    img = driver.find_element_by_xpath('/html/body/div[1]/div[2]/div[2]/div[1]/div/img[{}]'.format(idx + 2))
                    src = img.get_attribute('src')
                    # print(src)
                    if src[-2] == 'n':
                        continue

                    folder_name = MEDIA_LIST[m-1] + '_' + CONTENT_LIST[c-1]
                    folder_path = os.path.join(baseDir, 'data', folder_name)
                    if os.path.exists(folder_path) == False:
                        os.mkdir(folder_path)
                    
                    img_idx = epoch * 50 + idx
                    path = os.path.join(folder_path,(str(img_idx) + '.jpg' ))
                    try:
                        urllib.request.urlretrieve(src, path)
                    except:
                        print("not save")
                        pass

# /Users/sumin/PycharmProjects/NeuralStyleTransfer/code/data











