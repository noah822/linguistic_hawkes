from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.webdriver import WebDriver

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.support.wait import WebDriverWait

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

from typing import (
    List, Tuple, Dict, Union
)

from lxml import etree
import re

from collections import OrderedDict
import time


# thin wrapper of webdriver project
class _BaseProxy:
    def __init__(self, driver: WebDriver):
        self.driver = driver
        self._default_wait_time = 10
    
    def get(self, url: str):
        self.driver.get(url)
        return self

    def __call__(self,
                 selector: Tuple[By, str],
                 explicit_wait: float=None,
                 return_multiple=False,
                 until_clickable=False) -> WebElement:
        return self._wait_barrier(
            selector,
            return_multiple,
            until_clickable,
            explicit_wait
        )
    
    # explict wait until element appears
    def _wait_barrier(self,
                      selector: Tuple[By, str],
                      wait_multiple: bool=False,
                      util_clickable: bool=False,
                      explicit_wait: float=None) -> Union[WebElement, List[WebElement]]:
        if wait_multiple:
            condition = EC.presence_of_all_elements_located
        else:
            condition = EC.presence_of_element_located

        if util_clickable:
            element = WebDriverWait(self.driver, self._default_wait_time).until(
                EC.element_to_be_clickable(selector)
            )
        else:
            if explicit_wait is not None:
                time.sleep(explicit_wait)
            element = WebDriverWait(self.driver, self._default_wait_time).until(
                condition(selector)
            )
        return element

# thin wrapper of htlm string using lxml etree
class HtmlElement:
    def __init__(self,
                 html_str: str,
                 dummy_root: bool=True):
        if dummy_root:
            self.html_str = f'<li>{html_str}</li>'
        else:
            self.html_str = html_str
        self.element = etree.fromstring(self.html_str)
    
    def find_text_by_xpath(self, pattern: str) -> str:
        return self.element.find(pattern).text
        
def _setup_driver(driver='chrome') -> WebDriver:
    if driver == 'chrome':
        default_version = '114.0.5735.90'
        default_caps = DesiredCapabilities().CHROME
        default_caps['pageLoadStrategy']='none'

        driver = webdriver.Chrome(
            service=Service(
                ChromeDriverManager(version=default_version).install()
            ),
            desired_capabilities=default_caps
        )
    else:
        assert NotImplementedError
    return _BaseProxy(driver=driver)
    
        

def fetch_works(driver='chrome') -> List[Tuple[str, str]]:
    homepage = 'https://www.opensourceshakespeare.org/views/plays/plays_alpha.php'

    proxy = _setup_driver(driver).get(homepage)
    # get a list of play title element
    play_title_elements: List[WebElement] = \
        proxy((By.XPATH, '//li[@class="playlist"]'),
              return_multiple=True)

    play_and_date: Dict[str, str] = {}

    for element in play_title_elements:
        wrapped_tag = HtmlElement(element.get_attribute('innerHTML'))
        play_name = wrapped_tag.find_text_by_xpath('.//strong')
        written_date = wrapped_tag.find_text_by_xpath('.//em')
        
        play_and_date[play_name] = re.search(r'\((.*?)\)', written_date).group(1)
    
    ordered_by_written_date = sorted([(v, k) for k, v in play_and_date.items()])
    return ordered_by_written_date

import os
import requests
def download_plays(plays: Union[List[str], str],
                   verbose=False,
                   save_path='./Shakespeare',
                   driver='chrome'):
    
    homepage = 'https://www.folger.edu/explore/shakespeares-works/download'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    proxy = _setup_driver(driver).get(homepage)

    if isinstance(plays, str):
        plays = [plays]

    for name in plays:

        response = requests.get(_get_download_link(proxy, name))
        with open(os.path.join(save_path, '{}.txt'.format(name.replace(' ', '-'))), 'w') as f:
            f.write(response.text)
        if verbose:
            print(f'{name} sucessfully downloaded!')


# input driver is assumed to have loaded the download page
# reduce reload overhead
def _get_download_link(driver: _BaseProxy,
                       playname: str,
                       format: str='TXT') -> str:
    # take care of single quote
    if "'" in playname:
        quoted_playname = f'\"{playname}\"'
    else:
        quoted_playname = f"'{playname}'"
    xpath_pattern = (
'''//button[.//span[text()={}]]/parent::div/following-sibling::\
*/div/ul/li[.//span[text()='{}']]/a'''
        ).format(quoted_playname, format)
    selector = (By.XPATH, xpath_pattern)

    download_href = driver(selector).get_attribute('href')
    return download_href






    




    

