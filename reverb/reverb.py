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

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)


def plot_audio_waveform(audio, title):
    samples = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    time = np.arange(0, len(samples)) / sample_rate
    plt.plot(time, samples)

    return plt


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

def play_audio():
    input_file = file_entry.get()
    delay_ms = int(delay_entry.get())
    decay = float(decay_entry.get())
    repetitions = int(repetitions_entry.get())

    output_file = "output_audio.wav"
    add_echo(input_file, output_file, delay_ms, decay, repetitions)

    threading.Thread(target=play_output_audio, args=(output_file,)).start()

    # 绘制原始音频幅度谱
    original_audio = AudioSegment.from_file(input_file)
    plot_amplitude_spectrum(original_audio, "Original Audio")

    # 绘制生成音频幅度谱
    generated_audio = AudioSegment.from_file(output_file)
    plot_amplitude_spectrum(generated_audio, "Generated Audio")

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

# 创建主窗口
root = tk.Tk()
root.title("回声大师")
root.geometry("300x350+200+200")

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


# 运行主循环
root.mainloop()

