import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import pandas as pd

# 创建一个webdriver实例，这里以Chrome为例
driver = webdriver.Safari()

data = pd.read_excel('Problem_C_Data_Wordle.xlsx')

# read data
dict = {}

# remove the keys that exists in the dict
data = data[~data['Word'].isin(dict.keys())]

try:
    for word in data['Word']:
        driver.get(
            "https://books.google.com/ngrams/graph?content=" + word + "&year_start=2021&year_end=2022&corpus=en&smoothing=3")
        element = driver.find_element('id', 'main-content')
        actions = ActionChains(driver)

        # 添加鼠标按下的动作，移动到元素的起始位置
        actions.move_to_element(element)

        # 执行水平移动，'xoffset'是水平方向移动的距离，'yoffset'是垂直方向移动的距离
        actions.click_and_hold().move_by_offset(xoffset=260, yoffset=0).release()

        # 执行这个动作链
        actions.perform()

        # 找tt_right_colum class
        element = driver.find_element('class name', 'tt_right_column')
        # 找里面tspan的内容
        element = element.find_elements('tag name', 'tspan')

        # 保存数据
        dict[word] = [e.text for e in element]
        time.sleep(0.5)
finally:
    driver.quit()

    # 保存数据
    pd.DataFrame(dict).to_excel('词频.xlsx', index=False)
