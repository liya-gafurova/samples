import json
from typing import Optional

from pydantic import BaseModel

from text_generation.source_parser import Example

FILEPATH = '/home/lg/PycharmProjects/my/samples/text_generation/out_all.json'


class DataExample(BaseModel):
    source: Optional[str]
    target: Optional[str]


def upload_data(filepath: str) -> list[Example]:
    datas = []

    with open(filepath, 'r') as file:
        raw = file.read()
        data = json.loads(raw)

        exmps = []
        for obj in data:
            exmp: Example = Example.parse_obj(obj)
            d = DataExample(
                source=f'<title>: {exmp.title}, <description>: {exmp.description}, <keywords>: {", ".join(exmp.keywords)}',
                target=f'{exmp.text.text}'
            )
            datas.append(d.dict())

    with open('./dataset.json', 'w') as out:
        out.write(json.dumps(datas))

    return exmps


def count_data_len(filepath: str, field: str):
    all_lengths: int = 0
    objs_count: int = 0

    with open(filepath, 'r') as file:
        raw = file.read()
        data: dict = json.loads(raw)

        for d in data:
            field_len = len(d[field])
            all_lengths += field_len
            objs_count += 1

    print(f'Avg length of field {field}: {all_lengths / objs_count}')


def enrich_dataset(filepath: str, new_dataset_filepath: str):
    with open(filepath) as file:
        raw = file.read()
        data = json.loads(raw)

    objs: list[DataExample] = [DataExample(**d) for d in data]
    new_objs = []
    for obj in objs:
        sentences = obj.target.split('\n')
        sentences = list(filter(lambda s: s not in ['', ' ', '  ', '\t'], sentences))

        for i, sentence in enumerate(sentences):
            new_data = DataExample()
            if i == 0:
                new_data.source = obj.source
                new_data.target = sentence
            else:
                new_data.source = obj.source + f', <prompt>: {sentences[i - 1]}'
                new_data.target = sentence

            new_objs.append(new_data.dict())

    with open(new_dataset_filepath, 'w') as out:
        out.write(json.dumps(new_objs))

enrich_dataset('dataset.json', './dataset_enriched.json')