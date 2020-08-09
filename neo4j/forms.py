from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField
from wtforms.validators import DataRequired, Optional
from evaluate_neo4j import QueryType

class RequiredIf(object):
    """Validates field conditionally.
    Usage::
        login_method = StringField('', [AnyOf(['email', 'facebook'])])
        email = StringField('', [RequiredIf(login_method='email')])
        password = StringField('', [RequiredIf(login_method='email')])
        facebook_token = StringField('', [RequiredIf(login_method='facebook')])
    """
    def __init__(self, *args, **kwargs):
        self.conditions = kwargs

    def __call__(self, form, field):
        for name, data in self.conditions.items():
            if name not in form._fields:
                Optional(form, field)
            else:
                condition_field = form._fields.get(name)
                if condition_field.data == data and not field.data:
                    DataRequired()(form, field)
        Optional()(form, field)

class ExplorerForm(FlaskForm):
    # eval_list = {}
    # def __init__(self, evaluator):
    #     super().__init__()
    #     self.eval_company2company = SelectField("Node", choices=evaluator.get_eval_company2company())
    #     self.eval_company2field =  SelectField("Node", choices=evaluator.get_eval_company2field())
    #     self.eval_field2company =  SelectField("Node", choices=evaluator.get_eval_field2company())

    query_type = SelectField('Query Type', coerce=QueryType, choices=[(type_enum, type_enum.name) for type_enum in QueryType], validators=[DataRequired()], default=QueryType['company2company'])

    node = SelectField("Node", validators=[DataRequired()])

    # src_type = SelectField("Source Node's  Type", choices=[('company','company'),('field','field')], validators=[DataRequired()])
    # dst_type = SelectField("Destined Nodes' Type", choices=[('company','company'),('field','field')], validators=[DataRequired()])

    # zefix_uid = StringField('ZEFIX UID', validators=[RequiredIf(src_type='company')], filters = [lambda x: x or None])
    # node_str = StringField('Node String', validators=[DataRequired()])

    submit = SubmitField('Submit')