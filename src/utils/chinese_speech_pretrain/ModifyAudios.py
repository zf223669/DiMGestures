import os
import librosa
import soundfile as sf

# 输入音频目录和输出音频目录
# 设置输入和输出目录
input_dir = '/data/HOME/zhangfan/Chinese-speech-pretrain/wav'
output_dir = '/data/HOME/zhangfan/Chinese-speech-pretrain/wav/Mono'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入音频目录中的所有文件
for filename in os.listdir(input_dir):
    print(f'Processing {filename}')
    input_path = os.path.join(input_dir, filename)

    # 检查文件是否为音频文件
    if input_path.endswith('.wav') or input_path.endswith('.mp3'):
        # 读取音频文件
        audio, sr = librosa.load(input_path, sr=16000, mono=True)  # 保持声道分离

        # # 合并左右声道
        # if audio.shape[0] == 2:
        #     audio_merged = librosa.to_mono(audio)  # 合并为单声道
        # else:
        #     audio_merged = audio[0]  # 如果已经是单声道则保持不变
        #
        # # 修改采样率
        target_sr = 16000  # 目标采样率
        # audio_resampled = librosa.resample(audio_merged, orig_sr=sr, target_sr=target_sr)
        # # 构建输出文件路径
        output_filename = f"{filename}"
        output_path = os.path.join(output_dir, output_filename)
        # librosa.output.write_wav(output_path, audio, sr, norm=False)
        # 保存处理后的音频文件
        sf.write(output_path, audio, target_sr)
        print(f"Processed {input_path} and saved to {output_path}")

