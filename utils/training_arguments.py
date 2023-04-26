from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

@dataclass
class MyTrainingArguments():
    cfg: str = field(
        default=None, metadata={"help": "the path of the configuration"}
    )

if __name__ == "__main__":
    @dataclass()
    class A():
        a: str = field()

    @dataclass()
    class B():
        b: str = field(
            default="111",
        )
    
    parser = HfArgumentParser((A, B))
    a, b = parser.parse_args_into_dataclasses()

    print(a.a)
    print(b.b)
