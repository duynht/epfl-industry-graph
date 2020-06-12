from flask import Flask, render_template, url_for
from flask_ngrok import run_with_ngrok
from naive_db import NaiveDB
from forms import ExplorerForm

app = Flask(__name__)
app.config.update(dict(
    SECRET_KEY="powerful secretkey",
    WTF_CSRF_SECRET_KEY="a csrf secret key"
))
run_with_ngrok(app)  # Start ngrok when app is run
db = NaiveDB()

@app.route("/", methods=['GET','POST'])
@app.route("/explorer", methods=['GET','POST'])
def explorer():
    form = ExplorerForm()
    data = []
    if form.validate_on_submit():
        data = db.get_related_nodes(form.node_str.data, form.src_type.data, form.dest_type.data)
    return render_template('explorer_form.html', title='Neighbor Explorer', form=form, data=data)


if __name__ == '__main__':
    app.run()