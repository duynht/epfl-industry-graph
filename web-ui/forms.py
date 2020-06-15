from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired

class ExplorerForm(FlaskForm):
    node_str = StringField('Node String', validators=[DataRequired()])
    src_type = SelectField("Source Node's  Type", choices=[('company','company'),('field','field')], validators=[DataRequired()])
    dst_type = SelectField("Destined Nodes' Type", choices=[('company','company'),('field','field')], validators=[DataRequired()])

    submit = SubmitField('Submit')