import onnxruntime
from waveAnalyzer import MelAnalysis, F0Analyzer, resample_align_curve
import dataclasses
import soundfile as sf
import numpy as np
import torch
import argparse  # 新增参数解析模块

@dataclasses.dataclass
class Config:
    sampling_rate: int = 44100
    win_size: int = 2048
    hop_size: int = 512
    n_mels: int = 128
    n_fft: int = 2048
    mel_fmin: float = 20.0
    mel_fmax: float = 16000.0
    f0_extractor: str = 'parselmouth'
    f0_min: float = 20.0
    f0_max: float = 1600
    vocoder_path: str = r"path/to/your/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"

def parse_args():  # 新增参数解析函数[6,7](@ref)
    parser = argparse.ArgumentParser(description='Audio processing with controllable parameters')
    parser.add_argument('--mel_keyshift', type=float, default=0, 
                       help='Mel spectrogram key shift (semitones)')
    parser.add_argument('--speed', type=float, default=1.0,
                       help='Speed ratio (1.0=original speed)')
    parser.add_argument('--vocoder_keyshift', type=float, default=0,
                       help='Vocoder key shift (semitones)')
    parser.add_argument('--input', type=str, required=True,
                       help='Input audio file path')
    return parser.parse_args()

def wave_to_mel(wave, mel_keyshift, speed):
    '''
    wave shape=(n_samples,)
    mel_keyshift: float
    speed: float,                        不变速为1.0
    '''
    wave = torch.from_numpy(wave).float()
    melAnalysis = MelAnalysis(
        sampling_rate=Config.sampling_rate, 
        win_size=Config.win_size, 
        hop_size=Config.hop_size, 
        n_mels=Config.n_mels, 
        n_fft=Config.n_fft, 
        mel_fmin=Config.mel_fmin, 
        mel_fmax=Config.mel_fmax
        )
    mel = melAnalysis(
        wave,
        mel_keyshift,
        speed,
        diffsinger = True  #是否使用diffsinger风格的padding，必须为True
        )
    
    f0Analyzer = F0Analyzer(
        sampling_rate = Config.sampling_rate,
        f0_extractor = Config.f0_extractor,
        hop_size = Config.hop_size,
        f0_min = Config.f0_min,
        f0_max = Config.f0_max
        )
    
    f0, _ = f0Analyzer(wave, n_frames=mel.shape[1],speed=speed)
    
    #根据mel_keyshift调整f0
    f0 = f0*(2 ** (mel_keyshift / 12))

    # mel shape: (n_mels, n_frames*speed)
    # f0 shape:  (n_frames*speed, )

    return mel, f0

def mel_to_wave(mel, f0, vocoder_keyshift, speed):
    '''
    mel shape = (n_mels, n_frames*speed)
    f0 shape = (n_frames*speed,)
    vocoder_keyshift shape = (n_frames,)
    '''
    timestep=Config.hop_size/Config.sampling_rate
    vocoder_keyshift = resample_align_curve(
        vocoder_keyshift, 
        timestep, 
        timestep*speed, 
        mel.shape[1]
    ) #调整vocoder_keyshift到mel的长度
    f0 = f0*(2 ** (vocoder_keyshift / 12))

    # 加载vocoder模型
    ort_session = onnxruntime.InferenceSession(Config.vocoder_path)

    # 准备输入
    mel = mel.numpy()
    f0 = f0.astype(np.float32)
    mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
    f0 = np.expand_dims(f0, axis=0)
    input_data = {
        'mel': mel,
        'f0': f0,
    }

    output = ort_session.run(['waveform'], input_data)[0]

    wave = output[0]
    return wave

def main(wave, mel_keyshift, speed, vocoder_keyshift):
    # 音频变速变调
    mel, f0 = wave_to_mel(wave, mel_keyshift, speed)
    wave = mel_to_wave(mel, f0, vocoder_keyshift, speed)
    return wave

if __name__ == '__main__':
    args = parse_args()
    
    try:
        wave, _ = sf.read(args.input)
        if wave.ndim > 1:  # 处理多声道音频
            wave = wave.mean(axis=1)
            
        # 生成vocoder参数数组
        vocoder_shift_array = np.zeros(len(wave)//Config.hop_size) + args.vocoder_keyshift
        
        # 执行处理流程
        processed_wave = main(wave, args.mel_keyshift, args.speed, vocoder_shift_array)
        
        # 生成输出路径
        output_path = args.input.replace('.wav', 
            f'_mel{args.mel_keyshift:.1f}_speed{args.speed:.1f}_voc{args.vocoder_keyshift:.1f}.wav')
        
        # 保存结果
        sf.write(output_path, processed_wave, Config.sampling_rate)
        print(f"Successfully saved to: {output_path}")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)