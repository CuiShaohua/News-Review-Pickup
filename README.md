# Project-1  
# 0 整体目标  
## 0.1 初级目标：新闻人物言论自动提取 即 得到说话的人和说话的内容  
> ### 思路：  
>> * 1) 加载语料库，语料库来源于2部分：  
>>>>（1）wiki百科，使用wiki_extractor进行抽取，中文繁体转换为中文简体，具体操作请看[]  
>>>> (2) 新闻语料库，使用18年新华社新闻库[]  

>> * 2) 加载模型（ltp分词、词性标注、依存句法分析）（这些在哈工大的ltp语言模型中都有的，只要安装好就可以用）  
>>>>(1)cws.model对应分词模型  
>>>>(2)parser.model对应依存句法分析模型  
>>>>(3)pos.model对应词性标注模型  
>>>>(4)pisrl.model对应语义角色分析模型  
>> * 3) 根据上述模型和语料库（按行处理）得到依存句法关系parserlist。  
>>>> * 这部分处理主要是为了找出所有的SBV主谓结构的句子，为后来的验证模型做准备。  
>> * 4) 使用gensim工具加载预训练好的词向量模型word2vec.model  
>>> * gensim.Word2Vec  
>> * 4) 使用gensim工具加载预训练好的词向量模型word2vec.model  
## 0.2 中级目标：由新闻人物言论的自动提取，找到相似文章，并根据文章内容进行新闻推送  
> ### 思路：  

## 0.3 高级目标：对多个说话人针对同一件事情的不同观点进行整理，输出人物的正负面评价分析  
> ### 思路：  

# 1 实现方法  
## 1.1 准备环境  
> * OS环境：华为云主机CentOS7.6
> * 软件环境：  
>> * 1) Python3.5.1  
>> * 2) jieba、pyltp3.4.0、gensim、  
>> * 3) flask、Bootstrap、html

## 1.2 各部分代码详解  
* Flask部分  
```Python  
#从app模块中即从__init__.py中导入创建的app应用
from app import app
from flask_bootstrap import Bootstrap
from flask import  render_template, redirect
from app.forms import NameForm
#from app.numpy import
#建立路由，通过路由可以执行其覆盖的方法，可以多个路由指向同一个方法。
import pandas
from app import review_content
from app import news_extract
@app.route('/')  # 指向/
@app.route('/index') # 指向/index.html
def index():
     return render_template("index.html")

@app.route('/Review_Extraction',methods=['GET','POST'])
def Review_Extraction ():
    name = None
    form = NameForm()
    if form.validate_on_submit(): # 当提交的内容不为空时
        name = form.name.data
        old_width = pandas.get_option('display.max_colwidth')
        pandas.set_option('display.max_colwidth', -1)

        df = review_content.review_fc(name)
        df = news_extract.main(name)
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
```

## 1.3 运行方法  
> * 1 flask文件夹下，指定app：  
```Shell  
export FLASK_APP=myproject.py  
```
> * 2 flask文件夹下，运行：  
```Shell  
nohup firefox &  
```  
> * 3 flask文件夹下，运行：  
```Shell  
flask run  
```
> * 4 firefox浏览器运行<127.0.0.1:5000>  

# 2 目录结构  
.
├── app  
│   ├── forms.py  
│   ├── forms.py.bak  
│   ├── __init__.py  
│   ├── review_content.py  
│   ├── routes.py  
│   ├── static  
│   │   └── images  
│   │       ├── \345\257\271\350\257\235\346\203\205\347\273\252\350\257\206\345\210\253.PNG  
│   │       ├── \346\203\205\346\204\237\345\200\276\345\220\221\345\210\206\346\236\220.PNG  
│   │       ├── \346\226\207\347\253\240\345\210\206\347\261\273.PNG  
│   │       ├── \346\226\207\347\253\240\346\240\207\347\255\276\347\256\241\347\220\206.PNG  
│   │       ├── \346\226\260\351\227\273\346\221\230\350\246\201.PNG  
│   │       ├── \350\207\252\344\270\273\347\272\240\351\224\231.PNG  
│   │       └── \350\257\204\350\256\272\350\247\202\347\202\271\346\217\220\345\217\226.PNG  
│   └── templates  
│       ├── base.html  
│       ├── form.html  
│       ├── index.html  
│       └── review_extraction.html  
├── config.py  
├── cws.model  
├── md5.txt  
├── merge_corpus  
├── merge_corpus.model  
├── merge_corpus.model.trainables.syn1neg.npy  
├── merge_corpus.model.wv.vectors.npy  
├── myproject.py  
├── ner.model  
├── parser.model  
├── pisrl.model  
├── pos.model  
├── review_content.py  
└── stopwords.txt  

# 3 效果展示  
**Home Page**  
![HomePage](https://github.com/CuiShaohua/project1/blob/master/Home.PNG)  
**review content extraction**  
![review content extraction](https://github.com/CuiShaohua/project1/blob/master/review_content.PNG)  
# 4 update  
（1）指代消解的算法还需要再写入初级目标；  
（2）Tfidf在判断文档相似性上还需要细心考虑一些权重。  
（3）正负面评价分析目前需要参考硕博论文再定思路。
