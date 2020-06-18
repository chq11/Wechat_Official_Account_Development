# -*- coding: utf-8 -*-
# filename: handle.py
import hashlib
# import tensorflow as tf
import reply
import receive
import web
from media import Media
from test_model import *
from DRN.my_function import *
from GT_chat import *
from action import check_action
from hashmap import *
m = HashMap(maplen=4)


@check_action
def format_output(output_str, raw_input):
    return output_str


class Handle(object):
    def GET(self):
        try:
            data = web.input()
            if len(data) == 0:
                return "hello, this is handle view"
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = "haiquan123"
            # print(token)
            list = [token, timestamp, nonce]
            list.sort()
            sha1 = hashlib.sha1()
            # map(sha1.update, list)
            sha1.update(''.join(list).encode('utf-8'))  # 将py3中的字符串编码为bytes类型
            hashcode = sha1.hexdigest()
            print ("handle/GET func: hashcode, signature: ", hashcode, signature)
            if hashcode == signature:
                return echostr
            else:
                return ""
        except Exception:
            return
    def POST(self):
        try:
            webData = web.data()
            # print "Handle Post webdata is ", webData   #后台打日志
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg):
                toUser = recMsg.FromUserName #粉丝号
                fromUser = recMsg.ToUserName #公众号
                # print(recMsg.CreateTime, recMsg.FromUserName)
                if recMsg.MsgType == 'text':
                    print(recMsg)
                    #content = "test"
                    content = recMsg.Content.decode('utf-8')
                    # print is_alphabet(content.decode('utf-8'))
                    print(content)
                    if is_alphabet(content):
                        re_content1 = pred_f(Sess=sess,encod_inputs=encoder_inputs, decod_inputs=decoder_inputs,map1=map1,pred=pred,recive_c=content).encode('utf-8').decode('utf-8')
                        re_content2 = translation_sys_en_to_zn(content).encode('utf-8').decode('utf-8')
                        replyMsg = reply.TextMsg(toUser, fromUser, 'little beauty：' + re_content1 + '\ntranslation：' + re_content2)

                    elif is_chinese(content):
                        re_content_weather = format_output(content, content).encode('utf-8').decode('utf-8')
                        if re_content_weather != content:
                            replyMsg = reply.TextMsg(toUser, fromUser, re_content_weather)
                        else:
                            re_content2 = translation_sys_zn_to_en(content).encode('utf-8').decode('utf-8')
                            # re_content2 = translation_sys_zn_to_en(content)
                            chat_input = str(re_content2).strip('.')
                            # print(chat_input)
                            re_content1 = pred_f(Sess=sess, encod_inputs=encoder_inputs, decod_inputs=decoder_inputs, map1=map1,
                                   pred=pred, recive_c=chat_input).encode('utf-8').decode('utf-8')
                            re_content1 = translation_sys_en_to_zn(re_content1).encode('utf-8').decode('utf-8')
                            re_content_add = '小权权：' + re_content1 + '\n翻译：' + re_content2
                            replyMsg = reply.TextMsg(toUser, fromUser, re_content_add)

                        # replyMsg = format_output(replyMsg, content).encode('utf-8').decode('utf-8')
                        # replyMsg = reply.TextMsg(toUser, fromUser, re_content2)

                    else:
                        replyMsg = reply.TextMsg(toUser, fromUser, 'Please enter content that conforms to the specification.\n 请输入符合规范的内容。')

                    return replyMsg.send()
                elif recMsg.MsgType == 'image':
                    if recMsg.CreateTime in m.get(recMsg.FromUserName):
                        return_url = 'http://www.dldebug.top/image/' + recMsg.MediaId + '1'
                        replyMsg = reply.TextMsg(toUser, fromUser, return_url)
                        return replyMsg.send()
                    else:
                        m.add(recMsg.FromUserName, recMsg.CreateTime)
                        mediaId = recMsg.MediaId
                        image_url = recMsg.PicUrl
                        myMedia = Media()
                        image_type = myMedia.get_url(image_url, mediaId)

                        receive_info = super_resolution(mediaId, image_type)
                        if receive_info is None:
                            return_url = 'http://www.dldebug.top/image/' + recMsg.MediaId + '1'
                            replyMsg = reply.TextMsg(toUser, fromUser, return_url)
                        else:
                            replyMsg = reply.TextMsg(toUser, fromUser, receive_info)
                        return replyMsg.send()

                else:
                    return reply.Msg().send()
            else:
                print("暂且不处理")
                return reply.Msg().send()
                #return "success"
        except Exception:
            return
