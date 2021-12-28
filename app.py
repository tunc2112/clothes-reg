import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import main

UPLOAD_FOLDER = 'static/upload'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECENT_UPLOAD'] = ""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            app.config['RECENT_UPLOAD'] = app.config['UPLOAD_FOLDER'] + "/" + secure_filename(filename)
            file.save(app.config['RECENT_UPLOAD'])
            return redirect(url_for('show_result'))

    return render_template('upload.html', site_title="Clothes")


@app.route('/result')
def show_result():
    try:
        if app.config["RECENT_UPLOAD"]:
            return render_template('result.html', site_title="Clothes", src_filename=app.config["RECENT_UPLOAD"], tags=main.predict(app.config["RECENT_UPLOAD"]))
    except:
        pass

    return redirect(url_for('index'))


@app.route('/')
def index():
    return render_template('upload.html', site_title="Clothes")


if __name__ == '__main__':
    app.run(host="localhost", port=3000, debug=True)
