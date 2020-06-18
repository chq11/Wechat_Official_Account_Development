# -*- coding: utf-8 -*-
# filename: media.py
#import urllib.request
import urllib
import requests
from urllib import request
import poster3.encode
from poster3.streaminghttp import register_openers
import json
from imagetype import *
# from basic import Basic

class Media(object):
    def get(self, accessToken, mediaId):
        postUrl = "https://api.weixin.qq.com/cgi-bin/media/get?access_token=%s&media_id=%s" % (accessToken, mediaId)
        urlResp = request.urlopen(postUrl)
        # urlResp = urllib.urlopen(postUrl)
        #print(urlResp.info())
        #headers = urlResp.info().__dict__['headers']
        #if ('Content-Type: application/json\r\n' in headers) or ('Content-Type: text/plain\r\n' in headers):
            #jsonDict = json.loads(urlResp.read())
            #print (jsonDict)
        #else:
        buffer = urlResp.read()
        # print(buffer)
        image_type = binfiletype(buffer[:20])
        mediaFile = open("./DRN/received_image/{}.{}".format(mediaId, image_type), "wb")
        mediaFile.write(buffer)
        print("get successful")
        return image_type

    def upload(self, accessToken, mediaId, mediaType):
        img_url = "https://api.weixin.qq.com/cgi-bin/media/upload?access_token=%s&type=%s" % (accessToken, mediaType)
        files = {'p_w_picpath': open('./DRN/received_image/{}1.jpg'.format(mediaId), 'rb')}
        r = requests.post(img_url, files=files)
        re = json.loads(r.text)['media_id']
        return re

# if __name__ == '__main__':
    # myMedia = Media()
    # accessToken = '9_8uylO_OK25D6WXNTXUSIvZqUahE1YThFV43sK8OCQKoC18fah-6OAh787DWvzv1yHgWz_inm1VD9gkZZX3dPquBEP5CY7JtnrqGQ0AbotG1AAT-yqSLnO-HSkR-aLUl9jWKssSlCyslgpe_6QKHgAIATNK'
    #accessToken = Basic().get_access_token()
    # mediaId = "MkjpTk3e1NsjWLTxjEiTtfc6hE8YULHBTyFDBdBmfQXDGX8cFSuXxu7t_ThaIKRl"
    # myMedia.get(accessToken, mediaId)
