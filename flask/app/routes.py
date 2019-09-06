#从app模块中即从__init__.py中导入创建的app应用
from app import app
from flask_bootstrap import Bootstrap
from flask import  render_template, redirect
from app.forms import NameForm
#from app.numpy import 
#建立路由，通过路由可以执行其覆盖的方法，可以多个路由指向同一个方法。
import pandas
from app import review_content

@app.route('/')
@app.route('/index')
def index():
     return render_template("index.html")

@app.route('/Review_Extraction',methods=['GET','POST'])
def Review_Extraction ():
    name = None
    form = NameForm()
    if form.validate_on_submit():

        name = form.name.data

        
        old_width = pandas.get_option('display.max_colwidth')
        pandas.set_option('display.max_colwidth', -1)

        df = review_content.review_fc(name)
        form.name.data = ''
        columns = ['Person','SBV','speech_content','sentence']
        return render_template('review_extraction.html', title='Zh_Cola',form=form,tables=[df.to_html(classes='data',escape=True,index=True,sparsify=False,border=1,index_names=False,header=True,columns=columns)], titles=df.columns.values)
        
        pandas.set_option('display.max_colwidth', old_width)
    else:
        return render_template('review_extraction.html', title='Zh_Cola',form=form)


@app.route('/News_push')
def News_push():
    return "更新中~~~"

bootstrap = Bootstrap(app)
