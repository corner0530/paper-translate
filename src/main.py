import argparse
import os
import shutil
from openai import OpenAI
from pix2text import Pix2Text


def extract_text(pdf_path, output_dir):
    p2t = Pix2Text.from_config()
    doc = p2t.recognize_pdf(pdf_path)
    doc.to_markdown(output_dir)


def translate(text):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_MINE"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "あなたは翻訳ツールです。入力されたMarkdownの文章を日本語で全て翻訳してください。翻訳結果以外の文章は出力しないでください。なお、インライン数式は$, $で、ブロック数式は$$, $$で囲われていることに注意してください。また、専門用語は無理に訳さず英語のまま表記してください。",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.0,
    )
    text_ja = response.choices[0].message.content
    return text_ja


def main(pdf_path, output_dir):
    shutil.rmtree(output_dir, ignore_errors=True)

    extract_text(pdf_path, output_dir)

    with open(f"{output_dir}/output.md", "r") as f:
        text = f.read()
    text_ja = translate(text)

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
