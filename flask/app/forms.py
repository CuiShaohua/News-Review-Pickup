from flask_wtf import FlaskForm
from wtforms import StringField,PasswordField,BooleanField,SubmitField
from wtforms.validators import DataRequired

class NameForm(FlaskForm):
    #DataRequired，当你在当前表格没有输入而直接到下一个表格时会提示你输入
    name = StringField('Input News in Chinese',validators=[DataRequired(message='Input NEWs')])
    submit = SubmitField('Submit')
