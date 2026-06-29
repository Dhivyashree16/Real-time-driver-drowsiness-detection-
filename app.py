from flask import Flask, render_template, Response, redirect, url_for
from detection import generate_frames

app = Flask(__name__)

def is_detection_on():
    try:
        with open("detection_active.txt") as f:
            return f.read().strip() == "1"
    except:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    open("detection_active.txt", "w").write("1")
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    open("detection_active.txt", "w").write("0")
    return redirect(url_for('index'))

@app.route('/video')
def video():
    if is_detection_on():
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return ""

if __name__ == "__main__":
    app.run(debug=True)
