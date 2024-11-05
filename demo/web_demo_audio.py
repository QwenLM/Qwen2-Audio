import gradio as gr
import modelscope_studio as mgr
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from argparse import ArgumentParser

DEFAULT_CKPT_PATH = 'Qwen/Qwen2-Audio-7B-Instruct'


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8001,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def add_text(chatbot, task_history, input):
    text_content = input.text
    content = []
    if len(input.files) > 0:
        for i in input.files:
            content.append({'type': 'audio', 'audio_url': i.path})
    if text_content:
        content.append({'type': 'text', 'text': text_content})
    task_history.append({"role": "user", "content": content})

    chatbot.append([{
        "text": input.text,
        "files": input.files,
    }, None])
    return chatbot, task_history, None


def add_file(chatbot, task_history, audio_file):
    """Add audio file to the chat history."""
    task_history.append({"role": "user", "content": [{"audio": audio_file.name}]})
    chatbot.append((f"[Audio file: {audio_file.name}]", None))
    return chatbot, task_history


def reset_user_input():
    """Reset the user input field."""
    return gr.Textbox.update(value='')


def reset_state(task_history):
    """Reset the chat history."""
    return [], []


def regenerate(chatbot, task_history):
    """Regenerate the last bot response."""
    if task_history and task_history[-1]['role'] == 'assistant':
        task_history.pop()
        chatbot.pop()
    if task_history:
        chatbot, task_history = predict(chatbot, task_history)
    return chatbot, task_history


def predict(chatbot, task_history):
    """Generate a response from the model."""
    print(f"{task_history=}")
    print(f"{chatbot=}")
    text = processor.apply_chat_template(task_history, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in task_history:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(ele['audio_url'], sr=processor.feature_extractor.sampling_rate)[0]
                    )

    if len(audios)==0:
        audios=None
    print(f"{text=}")
    print(f"{audios=}")
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    if not _get_args().cpu_only:
        inputs["input_ids"] = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"{response=}")
    task_history.append({'role': 'assistant',
                         'content': response})
    chatbot.append((None, response))  # Add the response to chatbot
    return chatbot, task_history


def _launch_demo(args):
    with gr.Blocks() as demo:
        gr.Markdown(
            """<p align="center"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/assets/blog/qwenaudio/qwen2audio_logo.png" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen2-Audio-Instruct Bot</center>""")
        gr.Markdown(
            """\
    <center><font size=3>This WebUI is based on Qwen2-Audio-Instruct, developed by Alibaba Cloud. \
    (本WebUI基于Qwen2-Audio-Instruct打造，实现聊天机器人功能。)</center>""")
        gr.Markdown("""\
    <center><font size=4>Qwen2-Audio <a href="https://modelscope.cn/models/qwen/Qwen2-Audio-7B">🤖 </a> 
    | <a href="https://huggingface.co/Qwen/Qwen2-Audio-7B">🤗</a>&nbsp ｜ 
    Qwen2-Audio-Instruct <a href="https://modelscope.cn/models/qwen/Qwen2-Audio-7B-Instruct">🤖 </a> | 
    <a href="https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct">🤗</a>&nbsp ｜ 
    &nbsp<a href="https://github.com/QwenLM/Qwen2-Audio">Github</a></center>""")
        chatbot = mgr.Chatbot(label='Qwen2-Audio-7B-Instruct', elem_classes="control-height", height=750)

        user_input = mgr.MultimodalInput(
            interactive=True,
            sources=['microphone', 'upload'],
            submit_button_props=dict(value="🚀 Submit (发送)"),
            upload_button_props=dict(value="📁 Upload (上传文件)", show_progress=True),
        )
        task_history = gr.State([])

        with gr.Row():
            empty_bin = gr.Button("🧹 Clear History (清除历史)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")

        user_input.submit(fn=add_text,
                          inputs=[chatbot, task_history, user_input],
                          outputs=[chatbot, task_history, user_input]).then(
            predict, [chatbot, task_history], [chatbot, task_history], show_progress=True
        )
        empty_bin.click(reset_state, outputs=[chatbot, task_history], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot, task_history], show_progress=True)

    demo.queue().launch(
        share=False,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


if __name__ == "__main__":
    args = _get_args()
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.checkpoint_path,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model.generation_config.max_new_tokens = 2048  # For chat.
    print("generation_config", model.generation_config)
    processor = AutoProcessor.from_pretrained(args.checkpoint_path, resume_download=True)
    _launch_demo(args)
