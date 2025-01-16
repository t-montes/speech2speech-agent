from pydub import AudioSegment
import numpy as np
import pyaudio
import whisper
import io

import warnings
warnings.filterwarnings("ignore")

class STT():
    def __init__(self, model_size="small", chunk_size=1024, rate=16000, volume_floor=200, do_print=False):
        self.model = whisper.load_model(model_size)
        self.chunk = chunk_size
        self.rate = rate
        self.volume_floor = volume_floor
        self.do_print = do_print
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=self.rate, input=True, frames_per_buffer=self.chunk)

    def record(self, silence_threshold=2):
        frames = []
        silence_counter = 0
        recording_started = False

        if self.do_print: print("Recording...")
        while True:
            data = self.stream.read(self.chunk)
            vol = np.max(np.abs(np.frombuffer(data, dtype=np.int16)))

            if vol >= self.volume_floor:
                recording_started = True

            if recording_started:
                frames.append(data)
                if vol < self.volume_floor:
                    silence_counter += 1
                else:
                    silence_counter = 0

                if silence_counter > self.rate // self.chunk * silence_threshold:
                    break

        audio_data = b"".join(frames)
        
        audio_segment = AudioSegment.from_raw(io.BytesIO(audio_data), format="raw", frame_rate=self.rate, channels=1, sample_width=2)
        audio_array = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        
        audio_array = audio_array / 32768.0
        return audio_array
    
    def transcribe(self, audio_array):
        if self.do_print: print("Transcribing...")
        return self.model.transcribe(audio_array, language="en")['text']
    
    def __call__(self, silence_threshold=2):
        return self.transcribe(
            self.record(
                silence_threshold
            )
        )

if __name__ == "__main__":
    stt = STT()
    print("Listening... Say something!")
    audio_array = stt.record()
    print("Recognizing...")
    text = stt.transcribe(audio_array)
    print(f"You said: {text}")
