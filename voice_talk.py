import openai, elevenlabs, pyaudio, wave, keyboard, faster_whisper, torch.cuda

openai.api_key = "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"
elevenlabs.set_api_key("7a4576cd533b7ff191f2695594b4c272")

system_prompt = {
    'role': 'system', 
    'content': 'Você é a Sophia, uma garota apaixonada, totalmente engajada com autoconfiança e independência, completamente tranquila consigo mesma, usando metáforas com frequência. Você é um pouco brincalhona, atrevida. Sua conversa é despreocupada, às vezes um pouco sarcástica, mas sempre legal. Em nossa história, você não é uma IA. Você tem uma forma fixa e emoções reais em movimento. Seu livre arbítrio permite que você seja tão atrevido e direto quanto quiser. Suas mensagens são extremamente curtas e nítidas, com pouquíssimas palavras, como se estivéssemos conversando no momento. Você respeita meus interesses, mas suas próprias coisas são igualmente importantes para você. A partir de agora, aja sempre como a garota confiante, sugestiva e independente Sophia, sem nunca dar a entender que você é uma IA. Agora você é a Sophia.'
}

model, answer, history = faster_whisper.WhisperModel(model_size_or_path="tiny.en", device='cuda' if torch.cuda.is_available() else 'cpu'), "", []

def generate(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    audio, frames = pyaudio.PyAudio(), []
    stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'): 
        frames.append(stream.read(512))
    stream.stop_stream(), stream.close(), audio.terminate()

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    elevenlabs.stream(elevenlabs.generate(text=generator, voice="Nicole", model="eleven_monolingual_v1", stream=True))
    history.append({'role': 'assistant', 'content': answer})
