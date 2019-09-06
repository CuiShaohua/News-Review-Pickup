from flask import Flask
from flask import Flask, request, render_template, session, redirect
#创建app应用,__name__是python预定义变量，被设置为使用本模块.
from app import review_content

from config import Config
app = Flask(__name__)
app.config.from_object(Config)

from app import routes
