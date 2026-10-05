"""Streaming readers for local FineWeb2 and Tatoeba files."""

import argparse
import json
from contextlib import closing
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Document:
    source: str
    source_id: str
    text: str
    source_file: str
    row_index: int
    metadata: dict

    @property
    def doc_id(self) -> str:
        return f"{self.source}:{self.source_id}"


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def iter_fineweb(
    path: Path, batch_size: int = 1024
) -> Iterator[Document]:
    path = Path(path)
    if batch_size < 1:
        raise ValueError("batch_size 必须大于 0。")

    with pq.ParquetFile(path) as parquet:
        names = parquet.schema_arrow.names
        if not {"text", "id"}.issubset(names):
            raise ValueError(f"{path.name}: 缺少 text 或 id 字段。")

        row_index = 0
        for batch in parquet.iter_batches(batch_size=batch_size):
            for row in batch.to_pylist():
                source_id = row["id"]
                text = row["text"]
                if not isinstance(source_id, str) or not source_id:
                    raise ValueError(
                        f"{path.name}, row {row_index}: id 无效。"
                    )
                if not isinstance(text, str):
                    raise ValueError(
                        f"{path.name}, row {row_index}: text 无效。"
                    )

                yield Document(
                    source="fineweb",
                    source_id=source_id,
                    text=text,
                    source_file=relative_path(path),
                    row_index=row_index,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"text", "id"}
                    },
                )
                row_index += 1


def iter_tatoeba(path: Path) -> Iterator[Document]:
    path = Path(path)

    with path.open(encoding="utf-8-sig") as stream:
        for row_index, line in enumerate(stream):
            # Preserve literal quotes in the sentence text.
            parts = line.rstrip("\r\n").split("\t", 2)
            if len(parts) != 3:
                raise ValueError(
                    f"{path.name}, row {row_index}: TSV 格式错误。"
                )

            source_id, language, text = parts
            if not source_id:
                raise ValueError(
                    f"{path.name}, row {row_index}: id 为空。"
                )
            if language != "jpn":
                raise ValueError(
                    f"{path.name}, row {row_index}: "
                    f"预期 jpn，实际为 {language!r}。"
                )

            yield Document(
                source="tatoeba",
                source_id=source_id,
                text=text,
                source_file=relative_path(path),
                row_index=row_index,
                metadata={"language": language},
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit 必须大于 0")

    fineweb_files = sorted(
        (ROOT / "datasets/fineweb-2-edu-japanese").glob("*.parquet")
    )
    if not fineweb_files:
        raise FileNotFoundError("没有找到 FineWeb2 parquet。")

    inputs = [
        (path, iter_fineweb) for path in fineweb_files
    ]
    inputs.append((
        ROOT / "datasets/Tatoeba/jpn_sentences.tsv",
        iter_tatoeba,
    ))

    preview_documents = 0
    for path, reader in inputs:
        with closing(reader(path)) as documents:
            for document in islice(documents, args.limit):
                print(json.dumps({
                    "doc_id": document.doc_id,
                    "source": document.source,
                    "source_file": document.source_file,
                    "row_index": document.row_index,
                    "characters": len(document.text),
                    "preview": document.text[:100],
                }, ensure_ascii=False))
                preview_documents += 1

    print(json.dumps({
        "input_files": len(inputs),
        "preview_documents": preview_documents,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()