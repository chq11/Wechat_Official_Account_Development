# -*- coding: utf-8 -*-

from urllib import request
import urllib
import time
import json

class Basic:
    def __init__(self):
        self.__accessToken = ''
        self.__leftTime = 0
        self.__startTime = time.time()

    def __real_get_access_token(self):
        appId = "wx8d38d6b618bb7e7c"
        appSecret = "4fa432869571d4eb6f17733c4f3b01a6"
        postUrl = ("https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=%s&secret=%s" % (appId, appSecret))
        urlResp = request.urlopen(postUrl)
        # urlResp = urllib.urlopen(postUrl)
        urlResp = json.loads(urlResp.read().decode('utf-8'))
        # print (urlResp)
        self.__accessToken = urlResp['access_token']
        # print(self.__accessToken)
        self.__leftTime = urlResp['expires_in']
        # fw = open('access_token', 'w')
        # fw.write(self.__accessToken)
        # fw.close()
        return self.__accessToken


    def get_access_token(self):
        currenTime = time.time()
        if currenTime - self.__startTime > 600:
            self.__startTime = currenTime
            self.__accessToken = self.__real_get_access_token()
            return self.__accessToken
        else:
            return self.__accessToken

    def first_get_access_token(self):
        self.__accessToken = self.__real_get_access_token()
        return self.__accessToken

    def run(self):
        while(True):
            if self.__leftTime > 10:
                time.sleep(2)
                self.__leftTime -= 2
            else:
                self.__real_get_access_token()

# if __name__ == '__main__':
#     accesstoken = Basic()
    # accesstoken.run()

