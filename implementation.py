import cv2
import numpy as np
import skin_detector
import dlib
from scipy.signal import find_peaks
import scipy.signal as signal
# import matplotlib.pyplot as plt


def find_rPPG(video_path):
    vid = cv2.VideoCapture(video_path)

    # Get the frames per second
    fps = vid.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second using video.get(cv2.CAP_PROP_FPS) : {fps}")


    detector = dlib.get_frontal_face_detector()
    left_expand_ratio = 0.25
    top_expand_ratio = 0.25
    # Get the total numer of frames in the video.
    frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"Total number of frames using video.get(cv2.CAP_PROP_FRAME_COUNT) : {frame_count}")

    f_cnt = 0
    i_cnt = 0
    mean_rgb = None
    while True:
        ret, frame = vid.read()
        if ret:
            h, w, _ = frame.shape

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if f_cnt == 0:
                rect = detector(gray_frame, 0)
                print(rect)
                if len(rect) == 0:
                    print("No face detected")
                    break
                rect = rect[0]
                left, right, top, bottom = rect.left(), rect.right(), rect.top(), rect.bottom()

                width = abs(right - left)
                height = abs(bottom - top)
                face_left = int(left - (left_expand_ratio/2 * width))
                face_top = int(top - (top_expand_ratio/2 * height))
                face_right = right
                face_bottom = bottom

            face = frame[face_top:face_bottom, face_left:face_right]
            mask = skin_detector.process(face)
            n_skinpixels = np.sum(mask)
            if n_skinpixels == 0:
                print(f"No skin pixels detected. {n_skinpixels}, frame {f_cnt}")

            # cv2.imshow('skin', s_det)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            masked_face = cv2.bitwise_and(face, face, mask=mask)

            mean_r = np.sum(masked_face[:,:,2]) / n_skinpixels
            mean_g = np.sum(masked_face[:,:,1]) / n_skinpixels
            mean_b = np.sum(masked_face[:,:,0]) / n_skinpixels

            if f_cnt == 0:
                mean_rgb = np.array([mean_r, mean_g, mean_b])
            else:
                mean_rgb = np.vstack((mean_rgb, np.array([mean_r, mean_g, mean_b])))

            f_cnt += 1
            i_cnt += 1
        else:
            break

    vid.release()
    cv2.destroyAllWindows()

    print("DONE")
    l = int(fps * 1.6)
    if mean_rgb is None:
        print("No face detected")
        return None, None
    rPPG_signals = np.zeros(mean_rgb.shape[0])

    for t in range(0, mean_rgb.shape[0] - l):
        C = mean_rgb[t:t+l-1,:].T
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, C)
        projection_matrix = np.array([[0, 1, -1], [-2, 1, 1]])

        S = np.matmul(projection_matrix, Cn)

        std = np.array([1, np.std(S[0,:]) / np.std(S[1,:])])

        P = np.matmul(std, S)

        rPPG_signals[t:t+l-1] = rPPG_signals[t:t+l-1] + (P-np.mean(P)) / np.std(P)


    lowcut = 0.8
    highcut = 2

    b, a = signal.butter(2, [lowcut, highcut], btype='bandpass', fs=fps)
    rPPG_filtered = signal.filtfilt(b, a, rPPG_signals)

    #  z Normalize
    # rPPG_filtered = (rPPG_filtered - np.mean(rPPG_filtered)) / np.std(rPPG_filtered)

    # minmax normalize
    rPPG_filtered = (rPPG_filtered - np.min(rPPG_filtered)) / (np.max(rPPG_filtered) - np.min(rPPG_filtered))

    rPPG_peaks, _ = find_peaks(rPPG_filtered, height=0.4, prominence=0.08)
    ## Check the result
    # plt.figure(figsize=(20,5))
    # plt.plot(rPPG_filtered)
    # plt.title(f"rPPG Signal of {video_path} after filtering")
    # plt.show()
    # print(f"rPPG Peaks = {len(rPPG_peaks)}")

    return rPPG_filtered, rPPG_peaks

