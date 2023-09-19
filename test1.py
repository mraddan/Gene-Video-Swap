import cv2
import os
import insightface
import time
from functools import partial
from insightface.app import FaceAnalysis
from moviepy.editor import VideoFileClip, ImageSequenceClip
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

input_path = "D:\_New Fiture\Video Face Swap/asset\meme15.mp4"
output_frame_path = 'frames/'
input_image = cv2.imread('D:\_New Fiture\Video Face Swap/asset/bowo1.jpeg')
output_name = 'meme15'
swapper = insightface.model_zoo.get_model('inswapper_128.onnx')

total_succesfull = 0
total_failed = 0
batch_size = 8
total_processed = 0
resolution = (1280, 720)

def extract_frames(input_path, output_path):
    videocapture = cv2.VideoCapture(input_path)
    success, image = videocapture.read()
    count = 0
    total_frames = int(videocapture.get(cv2.CAP_PROP_FRAME_COUNT)) 
    vid_frame = int(videocapture.get(cv2.CAP_PROP_FPS))
    frames_per_update = total_frames / 100  
    start_time = time.time()  

    while success:
        cv2.imwrite(os.path.join(output_path, f"{count}.jpg"), image)
        success, image = videocapture.read()
        count += 1
        
        if count % frames_per_update == 0:
            percentage = (count / total_frames) * 100
            elapsed_time = time.time() - start_time
            print(f"Progress ekstraksi frame: {percentage:.2f}% ({count}/{total_frames} frame) - Waktu berlalu: {elapsed_time:.2f} detik")

    end_time = time.time()  
    elapsed_time = end_time - start_time  
    print('Frames extraction has ended')
    return vid_frame

def swap_face(inp, img):
    faces = app.get(img)
    faces_inp = app.get(inp)
    source_faces = faces_inp[0]

    res = img.copy()
    for i in faces:
        res = swapper.get(res, i, source_faces, paste_back=True)
    return res

def count_files_in_folder(folder_path):
    file_list = os.listdir(folder_path)
    file_count = len(file_list)
    return file_count

def downscale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def upscale_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

def process_image(image_file):
    image_path = os.path.join(output_frame_path, image_file)

    try:
        img = cv2.imread(image_path)
        downscale_percent = 50
        img = downscale_image(img, downscale_percent)
        swaped_img = swap_face(input_image, img)

        upscale_percent = 200
        swaped_img = upscale_image(swaped_img, upscale_percent)

        processed_image_path = os.path.join('results/', image_file)
        cv2.imwrite(processed_image_path, swaped_img)
        return 1

    except Exception as e:
        print(f"Terjadi kesalahan saat memproses {image_file}: {str(e)}")
        return 0

def process_batch_images(image_files, start_time):
    results = []
    global total_processed

    for i, image_file in enumerate(image_files):
        result = process_image(image_file)
        results.append(result)
        total_processed += 1
        show_progress(total_processed, len(image_files), start_time)
    return results

def rebuilding_video(file_count, fps, output, resolution):
    frames = []
    for i in range(file_count):
        image = cv2.imread(os.path.join('results/', f"{i}.jpg"))
        if image is not None:
            height, width, layers = image.shape
            frames.append(image)
        else:
            print(f"Gambar {i}.jpg tidak dapat dibaca.")
    output_video_path = f'{output}.avi'
    
    video = ImageSequenceClip(frames, fps=fps)
    video = video.resize(resolution)
    video.write_videofile(output_video_path, codec='rawvideo', fps=fps)
    print("Video output telah dibuat.")

def inject_audio(input_path, output):
    output_video_path = f'{output}.avi'

    input_video = VideoFileClip(input_path)
    audio_clip = input_video.audio

    output_video = VideoFileClip(output_video_path)
    video_with_audio = output_video.set_audio(audio_clip)

    output_video_with_audio_path = f'{output}.mp4'
    video_with_audio.write_videofile(output_video_with_audio_path, codec="libx264")
    print("Video output dengan audio telah disimpan.")
    return video_with_audio

def del_file():
    result_files = os.listdir('results')
    for file in result_files:
        file_path = os.path.join('results', file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Terjadi kesalahan saat menghapus {file_path}: {str(e)}")

    frame_files = os.listdir('frames')
    for file in frame_files:
        file_path = os.path.join('frames', file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Terjadi kesalahan saat menghapus {file_path}: {str(e)}")
    print("Semua file dalam folder 'results' dan 'frames' telah dihapus.")

def show_progress(current, total, start_time):
    elapsed_time = time.time() - start_time
    percentage = (current / total) * 100
    print(f"Progress: {percentage:.2f}% ({current}/{total} gambar) - Waktu berlalu: {elapsed_time:.2f} detik")

def delete_video_file(output_name):
    try:
        avi_file = f"{output_name}.avi"
        if os.path.exists(avi_file):
            os.remove(avi_file)
        else:
            print(f"File {avi_file} tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan saat menghapus file .avi: {str(e)}")

def execute(batch_size):
    fps = extract_frames(input_path, output_frame_path)
    file_list = os.listdir(output_frame_path)
    total_files = len(file_list)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(partial(process_batch_images, start_time=start_time), [file_list[i:i+batch_size] for i in range(0, total_files, batch_size)]))
    file_count = count_files_in_folder(output_frame_path)
    rebuilding_video(file_count, fps, output_name, resolution)
    inject_audio(input_path, output_name)
    del_file()
    delete_video_file(output_name)

    global total_succesfull
    global total_failed
    
    jumlah_berhasil = sum([sum(batch) for batch in results])
    jumlah_gagal = len(file_list) - jumlah_berhasil

    print(f"Jumlah gambar berhasil diproses dan disimpan: {jumlah_berhasil}")
    print(f"Jumlah gambar gagal diproses: {jumlah_gagal}")

start_time = time.time()
execute(batch_size)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total waktu yang dibutuhkan: {elapsed_time:.2f} detik")
