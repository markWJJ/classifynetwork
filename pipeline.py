from flask import Flask, render_template,json
import requests, time
import numpy as np

# index_dict={0:'买入股票',1:'卖出股票',2:'个股详情',3:'个股诊断',4:'other',5:'other',6:'查看大盘',7:'增减持',8:'查看研报'}
def api_call(url, data):
    r = requests.post(url, data = json.dumps(data) )
    result = r.json()
    return result

import time
start = time.time()

data_list=["stock停牌了吗",
           "多买stock",
           "怎么才能做空stock",
           "我想卖stock",
           "怎么看stock的行情",
           "今天星期几？",
           "提供一下最新的研报",
           "最新研报",
"客户看中了一只香港的股票，为什么港股通账户无法买入呢？",
"我想吃东西",
"如果某一只股票同为沪港通和深港通的标的，那么客户可以通过沪港通账户买入再通过深港通账户卖出吗？",
"机构客户申请参与挂牌公司股票公开转让需要具备什么条件？",
"个人投资者在第二家期权经营机构开立新的期权账户时，投资者适当性评估要如何管理？"
                                                ]

s = api_call("http://192.168.3.131:8086/classifynet", {"question":data_list})

print(s)

# print(time.time()-start)
#
# print(s[0])
# print(s[1])
#
# for e,sent in zip(s[1]['esim'],data_list):
#     if int(e) in index_dict:
#         print(sent,'\t\t',index_dict[int(e)])
#     else:
#         print(sent,'\t\t','other')