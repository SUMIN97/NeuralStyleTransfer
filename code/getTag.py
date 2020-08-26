from selenium import webdriver
import json
from collections import OrderedDict

import os

###initial set
driver = webdriver.Chrome('/Users/sumin/Downloads/chromedriver')
driver.implicitly_wait(3)


NUM_WORKS = 1598931
base_url = 'https://grafolio.naver.com/works/'
#Tag
TAGS = []
DATA = OrderedDict()

for work in range(520, NUM_WORKS):
    url = base_url + str(work)
    driver.get(url)

    try:
        driver.implicitly_wait(5)
        a_tags = driver.find_elements_by_css_selector('.tag_area > a')

        for tag in a_tags:
            txt = str(tag.text)
            if txt not in TAGS:
                TAGS.append(txt)

        driver.implicitly_wait(2)
    except:
        pass

DATA['tags'] = TAGS

with open('./tag.json', 'w', encoding='utf-8') as make_file:
    json.dump(DATA, make_file, ensure_ascii=False, indent="\t")



driver.quit()


