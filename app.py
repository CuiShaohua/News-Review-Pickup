# coding=utf-8
# Pro1
import pickup
# Pro2 
import main_pro2
import sents2vec
from preprocess_autosummary import get_word_frequency,sentence_and_doc 

from flask_wtf import FlaskForm

from flask import Flask, render_template, request  # 引入类

class NameForm(FlaskForm):
    text = ''

app = Flask(__name__)  # Flask实例化一个对象，与我们当前运行的模块名当做参数
app.config["SECRET_KEY"] = "12345678"

@app.route('/')
def hh():
    return render_template('index.html')

content1 = None

@app.route('/news-extraction', methods=["POST","GET"])
def deal_project1():
    form = NameForm()
    if request.form:
        global content1
        content1 = request.form.get("desc")
        #print(content)
        user_input = content1
        form.text = pickup.main(user_input)
        return render_template('news-extraction.html',text=form.text, form=form)

    else:
        return render_template('news-extraction.html',form=form)  # 没提交的时候显示页面

content2 = None

@app.route("/autoSummary", methods=["POST", "GET"])
def deal_project2():
    form = NameForm()
    if request.form:
        global content2
        content2 = request.form.get("desc")
        #print(content)
        user_input = content2
        form.text = main_pro2.autoSummary(user_input)
        return render_template('autoSummary.html',text=form.text, form=form)

    else:
        return render_template('autoSummary.html',form=form)  # 没提交的时候显示页面

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
