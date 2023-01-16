import json
import os
import re
from typing import List, Optional

from pydantic import BaseModel

FILES_DIRS = ['/home/lg/PycharmProjects/work/ppp/gambling_text_generation/texts/texts/',
              '/home/lg/PycharmProjects/work/ppp/gambling_text_generation/texts/texts2/']


class TextInfo(BaseModel):
    text: Optional[str]
    word_count: Optional[int]
    char_count: Optional[int]


class Example(BaseModel):
    title: Optional[str]
    description: Optional[str]
    keywords: List[str] = []

    text: Optional[TextInfo]



patterns = {
    "title": r'title: <(.*?)>',
    "description": r'description: <(.*?)>',
    "keywords": r'keywords: <(.*?)>',

    "divider": r'-----',

    "main_text_pattern_start": r'^<(.*?)',
}

unique = ['title', 'description', 'keywords']


def parse_files(generated_path: str = './out.json') -> str:
    with open(generated_path, 'a') as out:
        exmps: List[dict] = []

        for source_files_dir in FILES_DIRS:


            for filepath in os.listdir(source_files_dir):

                with open(source_files_dir + filepath, 'r') as file:
                    lines = file.readlines()

                    exmp = Example()
                    params = unique.copy()

                    for line in lines:

                        if re.search(patterns.get('main_text_pattern_start'), line):
                            index = lines.index(line)
                            text = ''.join(lines[index:])

                            exmp.text = TextInfo(text=text, word_count=len(text.split()), char_count=len(text))

                        if re.search(patterns.get('divider'), line):
                            continue

                        else:

                            for p in params:

                                match = re.search(patterns.get(p), line)
                                if match:
                                    value = match.groups()[0]

                                    if p == 'keywords':
                                        value = value.split(',')

                                    exmp.__setattr__(p, value)

                                    params.pop(params.index(p))

                                    continue

                    exmps.append(exmp.dict())

        out.write(json.dumps(exmps, ensure_ascii=False))

    return generated_path


parse_files()
