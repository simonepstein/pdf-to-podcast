import concurrent.futures as cf
import glob
import io
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Literal, ClassVar

import gradio as gr
import sentry_sdk
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger
from openai import OpenAI
from promptic import llm
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader
from tenacity import retry, retry_if_exception_type
import requests
from bs4 import BeautifulSoup

sentry_sdk.init(os.getenv("SENTRY_DSN"))

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


class DialogueItem(BaseModel):
    text: str
    speaker: Literal["female-1", "male-1", "female-2"]

    @property
    def voice(self):
        return {
            "female-1": "alloy",
            "male-1": "onyx",
            "female-2": "shimmer",
        }[self.speaker]


class Dialogue(BaseModel):
    scratchpad: str
    dialogue: List[DialogueItem]

class DialogueOptions(BaseModel):
    target_languages: ClassVar[List[str]] = ["English", "French", "German", "Spanish", "Chinese"]
    title: str
    organisation: str
    participants: List[str]
    language: str = "English"
    with_metadata: bool

    def __init__(self, title: str, organisation: str, participants:List[str], language: str, with_metadata: bool):
        super().__init__(
            title=title,
            organisation=organisation,
            participants=participants,
            language=language,
            with_metadata=with_metadata
        )
        if language not in DialogueOptions.target_languages:
            raise ValueError(f"Language {language} not supported.")
    
    def title_prompt(self):
        if self.title.strip():
            return "The podcast should be called " + self.title + "."
        else:
            return ""
    def organisation_prompt(self):
        if self.organisation.strip():
            return "The podcast is being produced by " + self.organisation + ". Be sure to mention this in the podcast/"
        else:
            return ""

    def participants_prompt(self):
        parts = [p for p in self.participants if p.strip()]
        if len(parts) == 2:
            return "The participants in the podcast are " + " and ".join(parts) + ".  If present, use their roles to frame how they contribute to the discussion."
        elif len(parts) == 1:
            return "One of the participants in the podcast is " + parts[0] + ".  If present, use their role to frame how they contribute to the discussion."
        else:
            return ""

    def metadata_prompt(self):
        if self.with_metadata:
            return """
            Start by introducing the subject of the podcast, citing the title and any main authors you have identified.
            You can talk about the authors but they do not participate in the podcast.
            """
        else:
            return ""

@retry(retry=retry_if_exception_type(ValidationError))
@llm(model="gpt-4o", max_tokens=4096)
def generate_dialogue(text: str, title_prompt: str, organisation_prompt:str, participants_prompt:str, metadata_prompt:str, language: str) -> Dialogue:
    """
    Your task is to take the input text provided and turn it into an engaging, informative podcast dialogue in {language}. The input text may be messy or unstructured, as it could come from a variety of sources like PDFs or web pages. Don't worry about the formatting issues or any irrelevant information; your goal is to extract the key points and interesting facts that could be discussed in a podcast.

    Here is the input text you will be working with:

    <input_text>
    {text}
    </input_text>

    {title_prompt}

    {organisation_prompt}

    {participants_prompt}

    First, carefully read through the input text and identify the main topics, key points, and any interesting facts or anecdotes. Think about how you could present this information in a fun, engaging way that would be suitable for an audio podcast.

    <scratchpad>
    Brainstorm creative ways to discuss the main topics and key points you identified in the input text. Consider using analogies, storytelling techniques, or hypothetical scenarios to make the content more relatable and engaging for listeners.

    Keep in mind that your podcast should be accessible to a general audience, so avoid using too much jargon or assuming prior knowledge of the topic. If necessary, think of ways to briefly explain any complex concepts in simple terms.

    Use your imagination to fill in any gaps in the input text or to come up with thought-provoking questions that could be explored in the podcast. The goal is to create an informative and entertaining dialogue, so feel free to be creative in your approach.

    Write your brainstorming ideas and a rough outline for the podcast dialogue here. Be sure to note the key insights and takeaways you want to reiterate at the end.
    </scratchpad>

    Now that you have brainstormed ideas and created a rough outline, it's time to write the actual podcast dialogue. Aim for a natural, conversational flow between the host and any guest speakers. Incorporate the best ideas from your brainstorming session and make sure to explain any complex topics in an easy-to-understand way.

    <podcast_dialogue>
    Write your engaging, informative podcast dialogue here, based on the key points and creative ideas you came up with during the brainstorming session. Use a conversational tone and include any necessary context or explanations to make the content accessible to a general audience. Use made-up names for the hosts and guests to create a more engaging and immersive experience for listeners. Do not include any bracketed placeholders like [Host] or [Guest]. Design your output to be read aloud -- it will be directly converted into audio.
    {metadata_prompt}
    Make the dialogue as long and detailed as possible, while still staying on topic and maintaining an engaging flow. Aim to use your full output capacity to create the longest podcast episode you can, while still communicating the key information from the input text in an entertaining way.

    At the end of the dialogue, have the host and guest speakers naturally summarize the main insights and takeaways from their discussion. This should flow organically from the conversation, reiterating the key points in a casual, conversational manner. Avoid making it sound like an obvious recap - the goal is to reinforce the central ideas one last time before signing off.
    </podcast_dialogue>
    """

def get_mp3(text: str, voice: str, api_key: str = None) -> bytes:
    client = OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
    )

    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice=voice,
        input=text,
    ) as response:
        with io.BytesIO() as file:
            for chunk in response.iter_bytes():
                file.write(chunk)
            return file.getvalue()


def generate_audio_from_dialogue(llm_output: Dialogue, openai_api_key: str = None) -> bytes:
    logger.info("===== Generating dialog =====")
    logger.info(f"Generated: {llm_output}")
    
    audio = b""
    transcript = ""

    characters = 0

    with cf.ThreadPoolExecutor() as executor:
        futures = []
        for line in llm_output.dialogue:
            transcript_line = f"{line.speaker}: {line.text}"
            future = executor.submit(get_mp3, line.text, line.voice, openai_api_key)
            futures.append((future, transcript_line))
            characters += len(line.text)

        for future, transcript_line in futures:
            audio_chunk = future.result()
            audio += audio_chunk
            transcript += transcript_line + "\n\n"

    logger.info(f"Generated {characters} characters of audio")

    temporary_directory = "./gradio_cached_examples/tmp/"
    os.makedirs(temporary_directory, exist_ok=True)

    # we use a temporary file because Gradio's audio component doesn't work with raw bytes in Safari
    temporary_file = NamedTemporaryFile(
        dir=temporary_directory,
        delete=False,
        suffix=".mp3",
    )
    temporary_file.write(audio)
    temporary_file.close()

    # Delete any files in the temp directory that end with .mp3 and are over a day old
    for file in glob.glob(f"{temporary_directory}*.mp3"):
        if os.path.isfile(file) and time.time() - os.path.getmtime(file) > 24 * 60 * 60:
            os.remove(file)

    return temporary_file.name, transcript

def generate_audio(text: str, options: DialogueOptions, openai_api_key: str = None) -> bytes:
    logger.info("===== Generating audio =====")
    logger.info(f"Options: {options} Input: {text}")

    if len(text.strip()) == 0:
        raise gr.Error("Failed to extract text from input")

    llm_output = generate_dialogue(
        text,
        options.title_prompt(),
        options.organisation_prompt(),
        options.participants_prompt(),
        options.metadata_prompt(),
        options.language
    )
    return generate_audio_from_dialogue(llm_output, openai_api_key)

def handle_file_upload(file:str, title:str, organisation:str , participant1:str, participant2: str, language: str, with_metadata: bool, openai_api_key: str = None) -> bytes:
    if not os.getenv("OPENAI_API_KEY", openai_api_key):
        raise gr.Error("OpenAI API key is required")
    logger.info("===== Processing file =====")
    logger.info(file)
    with Path(file).open("rb") as f:
        if file.endswith(".pdf"):
            logger.info("===== Processing PDF =====")
            reader = PdfReader(f)
            text = "\n\n".join([page.extract_text() for page in reader.pages])
        else:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        return generate_audio(text, DialogueOptions(title, organisation, [participant1, participant2], language, with_metadata), openai_api_key)

def handle_url(url:str, title:str, organisation:str,  participant1:str, participant2: str, language: str, with_metadata: bool, openai_api_key: str = None) -> bytes:
    if not os.getenv("OPENAI_API_KEY", openai_api_key):
        raise gr.Error("OpenAI API key is required")
    logger.info("===== Fetching URL =====")
    logger.info(url)
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return generate_audio(text, DialogueOptions(title, organisation, [participant1, participant2], language, with_metadata), openai_api_key)
        else:
            raise gr.Error("Failed to retrieve the webpage")
        
    except Exception as e:
        raise gr.Error("Failed to retrieve content from URL: " + str(e))

def handle_transcript(transcript:str, openai_api_key: str = None) -> bytes:
    if not os.getenv("OPENAI_API_KEY", openai_api_key):
        raise gr.Error("OpenAI API key is required")
    logger.info("===== Parising transcript =====")
    dialog_items = [DialogueItem(text=l.split(':', 1)[1], speaker=l.split(':', 1)[0]) for l in transcript.split("\n\n") if l.strip()]
    dialog = Dialogue(scratchpad="", dialogue=dialog_items)
    return generate_audio_from_dialogue(dialog, openai_api_key)[0]

file_interface = gr.Interface(
    description=Path("description_file_upload.md").read_text(),
    fn=handle_file_upload,
    examples=[[str(p), "", "", "Simon [host]", "Alex [co-host]", "English", True] for p in Path("examples").glob("*.pdf")],
    inputs=[
        gr.File(
            label="File (html or pdf)",
        ),
        gr.Textbox(
            label="Podcast title",
        ),
        gr.Textbox(
            label="Organisation name",
        ),
        gr.Textbox(
            label="Participant 1",
            value="Simon [host]",
            info="A participants name and optional role"
        ),
        gr.Textbox(
            label="Participant 2",
            value="Alex [co-host]",
            info="A participants name and optional role"
        ),
        gr.Dropdown(
            DialogueOptions.target_languages, label="Language", info="Desired podcast language", value="English",
        ),
        gr.Checkbox(
            label="Include title and author of orignal content in podcast",
            value=False,
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Textbox(label="Transcript", show_copy_button=True),
    ],
    allow_flagging="never",
    clear_btn=None,
    cache_examples="lazy",
    api_name=False,
)

url_interface = gr.Interface(
    description=Path("description_url.md").read_text(),
    fn=handle_url,
    examples=[
        ["https://www.theguardian.com/lifeandstyle/article/2024/may/28/where-the-wild-things-are-the-untapped-potential-of-our-gardens-parks-and-balconies", "Nature's Banter", "The Guardian", "Simon [host]", "Alex [Phd researcher]", "English", True],
        ["https://www.oneusefulthing.org/p/what-apples-ai-tells-us-experimental", "Mind Bytes", "67 Bricks", "Simon [co-host]", "Alex [host]", "Spanish", False],
        ["https://blog.67bricks.com/?p=739", "Testing times", "The Guardian", "Simon [reluctant guest]", "Alex [host]", "English", True],
    ],
    inputs=[
        gr.Textbox(
            label="URL"
        ),
        gr.Textbox(
            label="Podcast title",
        ),
        gr.Textbox(
            label="Organisation name",
        ),
        gr.Textbox(
            label="Participant 1",
            value="Simon [host]",
            info="A participants name and optional role"
        ),
        gr.Textbox(
            label="Participant 2",
            value="Alex [co-host]",
            info="A participants name and optional role"
        ),
        gr.Dropdown(
            DialogueOptions.target_languages, label="Language", info="Desired podcast language", value="English",
        ),
        gr.Checkbox(
            label="Include title and author of orignal content in podcast",
            value=False,
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
        gr.Textbox(label="Transcript", show_copy_button=True),
    ],
    allow_flagging="never",
    clear_btn=None,
    cache_examples="lazy",
    api_name=False,
)

transcript_interface = gr.Interface(
    description=Path("description_transcript.md").read_text(),
    fn=handle_transcript,
    inputs = [
        gr.Textbox(
            label="Transcript",
            lines=10,
        ),
    ],
    outputs=[
        gr.Audio(label="Audio", format="mp3"),
    ],
    allow_flagging="never",
    clear_btn=None,
    cache_examples="lazy",
    api_name=False,
)

demo = gr.TabbedInterface(
    title="PDF or Web page to Podcast",
    head=os.getenv("HEAD", "") + Path("head.html").read_text(), 
    interface_list=[file_interface, url_interface, transcript_interface],
    tab_names=["Convert file", "Convert web page", "Process transcript (advanced)"],
)
demo = demo.queue(
    max_size=20,
    default_concurrency_limit=20,
)

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    demo.launch(show_api=False)
