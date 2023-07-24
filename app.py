from flask import Flask, render_template, request, redirect, url_for, session
import matplotlib.pyplot as plt
import implementation

app = Flask("Flask-Heart-Rate-Counter")
app.secret_key = 'BAD_SECRET_KEY'



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    # Get video file from the request
    video = request.files['video']

    # Save video file to a temporary location
    video_path = 'recorded_video.webm'
    video.save(video_path)


    rPPG_filtered, rPPG_peaks = implementation.find_rPPG(video_path) 
    if rPPG_filtered is None:
        session['jumlah_peaks'] = 0
        return {
            'status': 'Tak De Mukee',
            'code': 404,    
            }

    # Plot the rPPG signal and peaks within the app context
    with app.app_context():
        plt.switch_backend('Agg')  # Use Agg backend for saving figures
        fig, ax = plt.subplots(figsize=(20, 5))
        ax.plot(rPPG_filtered, label='rPPG Signal')
        if len(rPPG_peaks) > 0:
            ax.plot(rPPG_peaks, rPPG_filtered[rPPG_peaks], "x", label='Peaks')
        fig.savefig('static/gambar.png')
        plt.close(fig)

    heart_rate = len(rPPG_peaks) * 6

    # Pass the results to the template
    session['jumlah_peaks'] = heart_rate
    return {
            'status': 'success',
            'code': 200,
            'jumlah_peaks': heart_rate,
            'image': url_for('static', filename='gambar.png'), 
            }


if __name__ == '__main__':
    app.run()
