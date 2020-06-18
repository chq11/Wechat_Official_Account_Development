import urllib
import time
import json

def get_accetoken():
        appId = "wx8d38d6b618bb7e7c"
        appSecret = "4fa432869571d4eb6f17733c4f3b01a6"
        postUrl = ("https://api.weixin.qq.com/cgi-bin/token?grant_type=client_credential&appid=%s&secret=%s" % (appId, appSecret))
        #urlResp = urllib.request.urlopen(postUrl)
        urlResp = urllib.urlopen(postUrl)
        urlResp = json.loads(urlResp.read().decode('utf-8'))
        print (urlResp)
        access_Token = urlResp['access_token']
        acct = urlResp['access_token']
        #leftTime = urlResp['expires_in']

        fw = open('access_token', 'w')
        fw.write(access_Token)
        fw.close()
        return acct
