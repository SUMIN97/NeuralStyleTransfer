import sys, os
from bs4 import BeautifulSoup
from selenium import webdriver
import urllib, urllib.request
import requests
import random
import time
from selenium.webdriver.common.keys import Keys
import errno
###initial set


folder = "./"
url = "https://www.google.com/search"

searchItem = "판화"
size = 100

params ={
   "q":searchItem
   ,"tbm":"isch"
   ,"sa":"1"
   ,"source":"lnms&tbm=isch"
}

url = url+"?"+urllib.parse.urlencode(params)
browser = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver")
time.sleep(0.5)
browser.get(url)
html = browser.page_source
time.sleep(0.5)

### get number of image for a page

soup_temp = BeautifulSoup(html, 'html.parser')
img4page = len(soup_temp.findAll("img"))

### page down

elem = browser.find_element_by_tag_name("body")
imgCnt = 0

while imgCnt < size * 10:
   elem.send_keys(Keys.PAGE_DOWN)
   rnd = random.random()
   time.sleep(rnd)
   imgCnt += img4page

html = browser.page_source
soup = BeautifulSoup(html,'html.parser')
img = soup.findAll("img")

browser.find_elements_by_tag_name('img')

fileNum=0
srcURL=[]

for line in img:
   if str(line).find('data-src') != -1 and str(line).find('http')<100:
      print(fileNum, " : ", line['data-src'])
      srcURL.append(line['data-src'])
      fileNum+=1


### make folder and save picture in that directory

saveDir = folder+searchItem

try:
    if not(os.path.isdir(saveDir)):
        os.makedirs(os.path.join(saveDir))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise

for i,src in zip(range(fileNum),srcURL):
    urllib.request.urlretrieve(src, saveDir+"/"+str(i)+".jpg")
    print(i,"saved")

fileNum=0
srcURL=[]

for line in img:
   if str(line).find('data-src') != -1 and str(line).find('http')<100:
      print(fileNum, " : ", line['data-src'])
      srcURL.append(line['data-src'])
      fileNum+=1

