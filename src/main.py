import argparse
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def translate(save_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MINE"))
    assistant = client.beta.assistants.create(
        instructions="You are a professor specializing in machine learning. Summarize the given paper.",
        tools=[{"type": "file_search"}],
        model="gpt-4o-mini",
    )
    vector_store = client.beta.vector_stores.create(name="PDFstore")
    file_streams = [open(save_path, "rb")]

    _ = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={
            "file_search": {"vector_store_ids": [vector_store.id]}
        },
        temperature=0,
    )
    message_file = client.files.create(
        file=open(save_path, "rb"), purpose="assistants"
    )

    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": "この論文を日本語で要約してください。ただし、以下の注意に従ってください。\n\n* 私は機械学習を専攻する博士課程の学生です。論文の背景や関連研究についてはよく知っているので、それ以外の部分の詳細を優先してまとめてください。\n* 論文の内容を過不足なく忠実にまとめてください。必要に応じて数式を用いて構わないので、正確に記述してください。\n* 論文中の1つの節が箇条書きの1つの項目に対応するように、箇条書きでまとめてください。要約が長くなっても構いません。",
                "attachments": [
                    {
                        "file_id": message_file.id,
                        "tools": [{"type": "file_search"}],
                    }
                ],
            }
        ]
    )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(
        client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
    )
    response = messages[0].content[0].text.value
    return response


def main(pdf_path, output_dir):
    text_ja = translate(pdf_path)
    print(text_ja)

    with open(f"{output_dir}/output_ja.md", "w") as f:
        f.write(text_ja)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdf_path", type=str)
    parser.add_argument("-o", "--output_dir", type=str, default="output")
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args.pdf_path, args.output_dir)
