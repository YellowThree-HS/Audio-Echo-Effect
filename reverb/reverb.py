import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import threading
import pygame
import tempfile
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.io import wavfile
from scipy import signal
import librosa



# 设置默认值
DEFAULT_FILE_PATH = "漠河舞厅.wav"
DEFAULT_DELAY_MS = "500"
DEFAULT_DECAY = "0.5"
DEFAULT_REPETITIONS = "3"

# 函数：添加回声效果
def add_echo(input_file, output_file, delay_ms, decay, repetitions):
    # 加载音频文件
    audio = AudioSegment.from_file(input_file)

    # 将音频数据转换为NumPy数组
    samples = np.array(audio.get_array_of_samples())

    # 计算延迟样本数（将毫秒转换为样本数）
    delay_samples = int(delay_ms * audio.frame_rate / 1000)

    # 初始化回声音频数组为原始音频数组
    echoed_samples = samples.copy()

    # 循环应用回声效果多次
    for i in range(1, repetitions + 1):
        # 计算当前回声的位置
        position = i * delay_samples

        # 应用音频衰减
        decayed_samples = (echoed_samples[:len(samples) - position] * decay).astype(np.int16)

        # 将衰减后的样本叠加到回声音频上
        echoed_samples[position:] += decayed_samples

    # 创建带有回声效果的新音频对象
    echoed_audio = AudioSegment(
        echoed_samples.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=samples.dtype.itemsize,
        channels=audio.channels
    )

    # 导出生成的音频到输出文件（格式为WAV）
    echoed_audio.export(output_file, format="wav")

# 去除回声
# 创建卡尔曼滤波器函数
# 初始化卡尔曼滤波器参数
# 这些参数需要根据具体问题进行调整
A = np.array([[1]])  # 状态转移矩阵
H = np.array([[1]])  # 观测矩阵
Q = np.array([[0.01]])  # 状态噪声协方差
R = np.array([[1]])  # 观测噪声协方差
x = np.array([[0]])  # 初始状态估计
P = np.array([[1]])  # 初始状态协方差估计
def kalman_filter(z):
    global x, P  # 将x和P声明为全局变量
    # 预测步骤
    x_hat = np.dot(A, x)
    P_hat = np.dot(np.dot(A, P), A.T) + Q

    # 更新步骤
    K = np.dot(np.dot(P_hat, H.T), np.linalg.inv(np.dot(np.dot(H, P_hat), H.T) + R))
    x = x_hat + np.dot(K, (z - np.dot(H, x_hat)))
    P = P_hat - np.dot(np.dot(K, H), P_hat)

    return x
def remove_echo(input_file, output_file):
    # 加载音频文件
    audio = AudioSegment.from_file(input_file, format="wav")

    # 将音频数据转换为numpy数组
    audio_array = np.array(audio.get_array_of_samples())

    filtered_signal = []

    for z in audio_array:
        # 使用卡尔曼滤波器进行回声消除
        x = kalman_filter(z)
        filtered_signal.append(x[0])
        print(str(x[0]))

    # 将回声消除后的numpy数组转换回AudioSegment对象
    clean_audio = AudioSegment(
        samples=np.array(filtered_signal, dtype=np.int16),
        frame_rate=input_file.frame_rate,
        sample_width=input_file.sample_width,
        channels=input_file.channels
    )

    clean_audio.export(output_file, format="wav")

    print("Clean audio saved to {}".format(output_file))
    play_output_audio(output_file)
    return clean_audio

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)


def plot_amplitude_spectrum(audio, title):
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    time = np.arange(0, len(samples)) / sample_rate
    plt.plot(time, samples)

    plt.show()

# 绘制音频频谱
def plot_frequency_spectrum(audio, title):
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    frequencies, amplitudes = compute_frequency_spectrum(samples, sample_rate)
    plt.plot(frequencies, amplitudes)

    # 设置 x 轴刻度范围，将最大值限制为采样率的一半
    plt.xlim(0, sample_rate / 2)

    plt.show()

# 计算音频频谱
def compute_frequency_spectrum(samples, sample_rate):
    n = len(samples)
    dft = np.fft.fft(samples) / n
    frequencies = np.fft.fftfreq(n, 1 / sample_rate)

    return frequencies, np.abs(dft)

def play_audio():
    input_file = file_entry.get()
    delay_ms = int(delay_entry.get())
    decay = float(decay_entry.get())
    repetitions = int(repetitions_entry.get())

    output_file = "output_audio.wav"
    add_echo(input_file, output_file, delay_ms, decay, repetitions)

    threading.Thread(target=play_output_audio, args=(output_file,)).start()

    # # 绘制原始音频幅度谱
    # original_audio = AudioSegment.from_file(input_file)
    # plot_amplitude_spectrum(original_audio, "Original Audio")
    #
    # # 绘制生成音频幅度谱
    # generated_audio = AudioSegment.from_file(output_file)
    # plot_amplitude_spectrum(generated_audio, "Generated Audio")

def play_output_audio(output_file):
    pygame.init()
    pygame.mixer.init()

    temp_dir = tempfile.gettempdir()
    temp_audio_file = os.path.join(temp_dir, "temp.wav")

    audio = AudioSegment.from_file(output_file)
    audio.export(temp_audio_file, format="wav")

    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)

    pygame.mixer.quit()
    pygame.quit()

# 函数：应用高通滤波器
def apply_highpass_filter(input_file, output_file, cutoff_frequency=400):
    # 读取音频文件
    audio = AudioSegment.from_file(input_file, format="wav")

    # 应用高通滤波器
    filtered_audio = audio.high_pass_filter(cutoff_frequency)

    # 保存滤波后的音频
    filtered_audio.export(output_file, format="wav")
    # 打印信息
    print("Filtered audio saved to {}".format(output_file))
    play_output_audio(output_file)

# 低通滤波器
def apply_lowpass_filter(input_file, output_file, cutoff_frequency=400):
    # 读取音频文件
    audio = AudioSegment.from_file(input_file, format="wav")

    # 应用低通滤波器
    filtered_audio = audio.low_pass_filter(cutoff_frequency)

    # 保存滤波后的音频
    filtered_audio.export(output_file, format="wav")
    # 打印信息
    print("Filtered audio saved to {}".format(output_file))
    play_output_audio(output_file)

# 去除噪声
def remove_noise(input_file, output_file):
    # 加载音频文件
    y, sr = librosa.load(input_file)


# 创建主窗口
root = tk.Tk()
root.title("回声大师")
root.geometry("500x500+200+200")

# 创建文件选择框
file_label = tk.Label(root, text="选择音频文件:")
file_label.place(x=10, y=50)
file_entry = tk.Entry(root)
file_entry.insert(0, DEFAULT_FILE_PATH)  # 设置默认文件路径
file_entry.place(x=100, y=50)
browse_button = tk.Button(root, text="浏览", command=browse_file)
browse_button.place(x=250, y=45)


# 创建回声参数输入框
delay_label = tk.Label(root, text="延迟时间（毫秒）:")
delay_label.place(x=10, y=100)
delay_entry = tk.Entry(root)
delay_entry.insert(0, DEFAULT_DELAY_MS)  # 设置默认延迟时间
delay_entry.place(x=150, y=100)

decay_label = tk.Label(root, text="衰减幅度:")
decay_label.place(x=10, y=150)
decay_entry = tk.Entry(root)
decay_entry.insert(0, DEFAULT_DECAY)  # 设置默认衰减幅度
decay_entry.place(x=150, y=150)

repetitions_label = tk.Label(root, text="回声次数:")
repetitions_label.place(x=10, y=200)
repetitions_entry = tk.Entry(root)
repetitions_entry.insert(0, DEFAULT_REPETITIONS)  # 设置默认回声次数
repetitions_entry.place(x=150, y=200)

# 创建播放按钮
play_original_button = tk.Button(root, text="播放原始音频", command=lambda: threading.Thread(target=play_output_audio, args=(file_entry.get(),)).start())
play_original_button.place(x=10, y=250)
play_button = tk.Button(root, text="播放回声效果", command=play_audio)
play_button.place(x=150, y=250)
clean_button = tk.Button(root, text="去除回声效果", command=lambda: threading.Thread(target=remove_echo, args=("output_audio.wav", "output_audio_clean.wav")).start())
clean_button.place(x=290, y=250)
# 创建截止频率输入框
cutoff_frequency_label1 = tk.Label(root, text="截止频率:")
cutoff_frequency_label1.place(x=80, y=305)
cutoff_frequency_entry1 = tk.Entry(root)
cutoff_frequency_entry1.insert(0, "400")  # 设置默认截止频率
cutoff_frequency_entry1.place(x=150, y=305)
cutoff_hz_label1 = tk.Label(root, text="Hz")
cutoff_hz_label1.place(x=300, y=305)
# 创建高通滤波按钮
highpass_button = tk.Button(root, text="高通滤波", command=lambda: threading.Thread(target=apply_highpass_filter, args=(file_entry.get(), "output_audio_highpass.wav", int(cutoff_frequency_entry1.get()))).start())
highpass_button.place(x=10, y=300)

# 创建截止频率输入框
cutoff_frequency_label2 = tk.Label(root, text="截止频率:")
cutoff_frequency_label2.place(x=80, y=355)
cutoff_frequency_entry2 = tk.Entry(root)
cutoff_frequency_entry2.insert(0, "400")  # 设置默认截止频率
cutoff_frequency_entry2.place(x=150, y=355)
cutoff_hz_label2 = tk.Label(root, text="Hz")
cutoff_hz_label2.place(x=300, y=355)
# 创建低通滤波按钮
lowpass_button = tk.Button(root, text="低通滤波", command=lambda: threading.Thread(target=apply_lowpass_filter, args=(file_entry.get(), "output_audio_lowpass.wav", int(cutoff_frequency_entry2.get()))).start())
lowpass_button.place(x=10, y=350)


# # 创建去除噪声按钮
# remove_noise_button = tk.Button(root, text="去除噪声", command=lambda: threading.Thread(target=remove_noise, args=(file_entry.get(), "output_audio_remove_noise.wav")).start())
# remove_noise_button.place(x=10, y=350)
# 运行主循环
root.mainloop()

