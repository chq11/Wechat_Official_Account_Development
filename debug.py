from urllib import request
import requests
import json
import poster3
from GT_chat import *
from imagetype import *
import basic
from basic import Basic
from hashmap import *

# accesstoken = Basic()
# accessToken = '34_wVbsM6YP0ibzPmbFRP0oS62FzkW9XpHUgQZ1lckqr4m_li7W8keQZ664dx2unsjm1JJjI1iqdklGjzOjGCATxdtEYcWER2G39nQ_BlIf8UPn-IhNxHBseDD4oke8lhReBUJ7NabM4u0uOtcwFEBbAJAVUR'
# # # mediaId = 'RAYUH050IEYi_q5Ng9qeoEI6W_P962kaLCjsLiKADsLvR_KSUppMBynYRfeOY9IY'
# # # postUrl = "https://api.weixin.qq.com/cgi-bin/media/get?access_token=%s&media_id=%s" % (accessToken, mediaId)
# img_url = "https://api.weixin.qq.com/cgi-bin/media/upload?access_token=%s&type=image" % (accessToken)
# filePath = './DRN/received_image/babyx4.png'
# # mediaType = "image"
# files = {'p_w_picpath': open(filePath, 'rb')}
# # # openFile = open(filePath, "rb")
# r = requests.post(img_url, files=files)
# re = json.loads(r.text)
# print(re)

# param = {'media': openFile}
# postData, postHeaders = poster3.encode.multipart_encode(param)
#
# postUrl = "https://api.weixin.qq.com/cgi-bin/media/upload?access_token=%s&type=%s" % (accessToken, mediaType)
# request1 = request.Request(postUrl, postData, postHeaders)
# urlResp = request.urlopen(request1)
# print(urlResp.read())

# class a():
#     dd = 1
#     ww = True
#
# ss = a()
#
# print(ss.dd,ss.ww)
#
# def b():
#     qq = None
#     return qq
#
# ww = b()
# if ww is None:
#     print('aa')

# print('读文件二进制码中……')
# binfile = open('./DRN/received_image/babyx4.png', 'rb')  # 必需二制字读取
# # print('提取关键码……')
# aa = binfiletype(binfile)
# print(aa)

# accessToken = accesstoken.get_access_token()
# print(accessToken)

# import string

m = HashMap(maplen=4)
# s = string.ascii_lowercase

# for k, v in enumerate(s):
m.add('qwe123', '123')
m.add('qwe123', '456')
m.add('as', '897')
# m.add('as1','352')
m.add('qwe123','8917')
m.add('qwz','86917')
# try:
# 	if m.get('as1') == 'qwesdssss':
# 		print('kkkkkk')
# except:
# 	m.add('as1','qwesdssss')
print(m.get('qwe123'))
get_k = m.get('qwe123')
if '456' in get_k:
	print('iiiin')
print(m.num)
# print(m)
# print(m.key())
print(m.print_map())
# for k in range(len(s)):
# 	print(k, m.get(k))

# a = []
# a.append(('wqwa', '222'))
# a.append(('w', '123'))
# a.append(('w1', '2222'))
# a.append(('w2', '222'))
# lena = len(a)
# i = 0
# while i < lena:
# 	print(lena)
# 	if int(a[i][1]) == 222:
# 		a.pop(i)
# 		lena -= 1
# 		continue
# 	i += 1
# 	# print(a[i][1])
# print(a)
# print(len(a))

# try:
# 	if recMsg.CreateTime in m.get(recMsg.FromUserName):
# 		return reply.Msg().send()
# except:
# 	m.add(recMsg.FromUserName, recMsg.CreateTime)