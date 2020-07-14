from bs4 import BeautifulSoup
import requests

maximum = 3
page = 1

## BeautifulSoup으로 html소스를 python객체로 변환하기
## 첫 인자는 html소스코드, 두 번째 인자는 어떤 parser를 이용할지 명시.
## 이 글에서는 Python 내장 html.parser를 이용했다.
#https://grafolio.naver.com/searchList.grfl?query=%EC%86%8C%EB%AC%98&order=wRecommend&type=works&haveProductYn=&page=1&categoryNo=&storyCategoryNo=&termType=entire&wallpaperYn=N#middleTab


#whole_source 는 크롤링할 모든 페이지의 HTML 소스를 전부 저장할 변수
whole_source = ""
for page_number in range(1, maximum+1):
    URL = 'https://www.singulart.com/ko/%EC%86%8C%EB%AC%98/%EB%A8%B9?page=' + str(page_number)
    response = requests.get(URL)
    whole_source = whole_source + response.text
soup = BeautifulSoup(whole_source, 'html.parser')
find_title = soup.select("picture > img")

print(soup)

for title in find_title:
	print(title.text)
# for i in enumerate(img_data[1:]):
#     # 딕셔너리를 순서대로 넣어줌
#     t = urlopen(i[1].attrs['src']).read()
#     filename = "byeongwoo_" + str(i[0] + 1) + '.jpg'
#     with open(filename, "wb") as f:
#         f.write(t)
#     print("Img Save Success")