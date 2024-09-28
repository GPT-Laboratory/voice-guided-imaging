<div align="center">

<img src="https://raw.githubusercontent.com/Koodattu/ucs-llm-voice-image-edit/main/assets/gls.png" style="height: 200px;" />
    
</div>

<h1 align="center">Voice Guided Imaging</h1>

<div align="center">

**Voice Guided Imaging is a demo by GPT Lab Seinäjoki -project. This demo focuses on speech-to-text and translation, intention recognition, image generation, image editing and video generation. The goal of the project is to showcase multiple different locally-hosted generative artificial intelligences in a single demo project.**

</div>

<div align="center">

<a target="_blank" href="https://epliitto.fi/en/" style="background:none;text-decoration: none;">
    <img src="https://epliitto.fi/wp-content/uploads/2020/12/EPLiitto_merkki_vari.png" style="height: 50px;" />
</a>
<a target="_blank" href="https://gpt-lab.eu/" style="background:none;text-decoration: none;">
    <img src="https://gpt-lab.eu/wp-content/uploads/2023/08/cropped-cropped-GPTlab_logo1-2-1.png" style="height: 50px;" />
</a>
<a target="_blank" href="https://www.ucs.fi/en/front-page/" style="background:none;text-decoration: none;">
    <img src="https://www.ucs.fi/wp-content/themes/ucs/documents/UCS-LOGOPAKETTI/Pysty/JPG-PNG/ucs_logo_pysty_musta.jpg" style="height: 50px;" />
</a>

---

## Click On The Image To See The Demo In Action!
[![Youtube Video](https://img.youtube.com/vi/iz3YnmFWz6s/0.jpg)](https://www.youtube.com/watch?v=iz3YnmFWz6s)

</div>

---

## Technical Breakdown
- All the models are hosted locally
- [OpenAI Whisper](https://github.com/openai/whisper) is used for speech-to-text and translation
- [Mistral Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) is used for intention recognition and modifying the users command into a prompt for image generation and editing
- [Ollama](https://ollama.com/) for hosting Mistral
- [SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) for fast image generation
- [Instruct-Pix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix) for image editing
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) for image generation
- [Python](https://www.python.org/) and [Flask](https://flask.palletsprojects.com/en/3.0.x/) to tie it all together

## How To Setup And Run The Demo

**Please keep in mind that you will need a lot of RAM, VRAM and a powerful GPU to run all of these models. This demo was developed and the video was recorded with the models running on a RTX 4090.**

1. Download and install [Ollama](https://ollama.com/)
2. Download Mistral Instruct using Ollama
```
ollama pull mistral:instruct
```
2. Clone the GitHub repository
```
git clone https://github.com/Koodattu/ucs-llm-voice-image-edit.git
```
3. Create and activate Python virtual environment
```
python -m venv venv
./venv/scripts/activate
```
4. Install dependencies with pip
```
pip install -r requirements.txt
```
5. Run the script:
```
python main.py
```
6. The app is hosted on http://localhost:5001/
7. Change the settings at the top of the page to fit your preferences

**NOTE:** The app downloads different models as needed, so the first start and the first generations are going to be slow, as we need to download the whisper model, sdxl-lightning model, instruct-pix2pix model and the stable video diffusion model.

## Application Logic Explained
1. The default microphone is used to record voice audio with push-to-talk and voice-activity-detection functionality.
2. The audio is transcribed and translated to english (if necessary)
3. An LLM is used to turn the natural language input into a command
4. The LLM is used to infer the intended action from the input and it modifies the prompt to be more fit for image generation or editing
6. The image is generated or edited

## About Inferring The Intention And Possible Actions
The following actions are currently possible
- Generate a new image
- Edit the current image
- Go back to the previous image
- Generate a video from the current image

## Licenses etc.
See the [LICENSE](https://github.com/GPT-Laboratory/voice-guided-imaging/blob/main/LICENSE) file for details.
