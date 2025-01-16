from elevenlabs import client, play

class TTS():
    def __init__(self, api_key):
        self.client = client.ElevenLabs(api_key=api_key)
    
    def generate(self, text, voice="Rachel"):
        audio = self.client.generate(text=text, voice=voice)
        return audio
    
    def __call__(self, text, voice="Brian"):
        play(
            self.generate(
                text,
                voice
            )
        )

if __name__ == "__main__":
    tts = TTS("...")
    tts("Hello, welcome to ElevenLabs!")
