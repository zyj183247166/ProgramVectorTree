# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 22:11:40 2018

@author: a
"""
import logging
class xiaojie_log_class():
    def log(self,message):
        #logger = logging.getLogger(__name__)
        logger = logging.getLogger(__name__)
        #这里进行判断，如果logger.handlers列表为空，则添加，否则，直接去写日志
        if not logger.handlers:
            logger.setLevel(level = logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            #文本中输出一份
            handler = logging.FileHandler("./3RvNNoutData/曾杰训练RvNN全记录.txt")
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            
            #控制台页输出一份
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            
            logger.addHandler(handler)
            logger.addHandler(console)
        try:
            logger.info(message)
        except Exception as e:
            pass
if __name__ == "__main__":
    logs = xiaojie_log_class()
    logs.log("Start print log")
