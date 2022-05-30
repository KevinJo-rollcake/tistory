import pandas as pd
from selenium import webdriver
import time

chrome_driver_path = 'E:\crawling\chromedriver.exe'
url = 'https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx'

driver = webdriver.Chrome(chrome_driver_path)
driver.get(url)

# column 추출
result = driver.find_element_by_xpath('//*[@id="cphContents_cphContents_cphContents_udpContent"]/div[3]/table/thead/tr')
col_list = result.text.split(' ')
print(col_list)

id_list = []
row_list = []


def getData():
    # playerID 추출
    a_list = driver.find_elements_by_css_selector('#cphContents_cphContents_cphContents_udpContent > div.record_result > table > tbody > tr > td > a')
    for a in a_list:
        id_list.append(a.get_attribute('href')[-5:])
    print(id_list)

    # row 추출
    player_list = driver.find_elements_by_css_selector('#cphContents_cphContents_cphContents_udpContent > div.record_result > table > tbody > tr')
    for player in player_list:
        data_list = player.text.split(' ')
        row_list.append(data_list)
    print(row_list)


for i in range(1, 4):
    driver.find_element_by_xpath('//*[@id="cphContents_cphContents_cphContents_ucPager_btnNo' + str(i) + '"]').click()
    time.sleep(2)
    getData()

df = pd.DataFrame(row_list, columns=col_list, index=id_list)
print(df)