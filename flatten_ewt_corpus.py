#!/usr/bin/env python3
"""
flatten_ewt_corpus.py

Flatten all UD-style CoNLL-U treebanks inside a folder named EWT-corpus.

Default expected structure:

    your_project_folder/
    ├── flatten_ewt_corpus.py
    └── EWT-corpus/
        ├── en_ewt-ud-train.conllu
        ├── en_ewt-ud-dev.conllu
        └── en_ewt-ud-test.conllu

The script recursively finds all .conllu files inside EWT-corpus and creates
one flattened output file per scheme per input file.

Default output folder:

    EWT-flattened/

Supported output schemes:

1. surface
   Raw token forms only.

2. upos_deprel
   UPOS and dependency relation for each token.

3. local
   UPOS, dependency relation, and head direction for each token.

4. distance
   UPOS, dependency relation, head direction, and bucketed head distance.

5. arc
   One dependency arc statement per token, using token positions but not explicit signpost tokens.

6. signpost
   Explicit token IDs/signposts and head IDs.

7. shuffled_deprel_control
   Preserves sentence/token structure but shuffles dependency labels within each sentence.

8. shuffled_head_control
   Preserves tokens and labels but shuffles heads within each sentence.

Basic usage:

    python flatten_ewt_corpus.py

Run only selected schemes:

    python flatten_ewt_corpus.py --schemes local distance shuffled_deprel_control

Test on first 10000 sentences per file:

    python flatten_ewt_corpus.py --max_sentences 10000

Use a different input/output folder:

    python flatten_ewt_corpus.py --input_dir EWT-corpus --output_dir EWT-flattened

No external libraries required.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_INPUT_DIR = Path("EWT-corpus")
DEFAULT_OUTPUT_DIR = Path("EWT-flattened")

SUPPORTED_SCHEMES = {
    "surface",
    "upos_deprel",
    "local",
    "distance",
    "arc",
    "signpost",
    "shuffled_deprel_control",
    "shuffled_head_control",
}

SPECIAL_SCHEME_ALL = "all"


@dataclass
class UDToken:
    """
    A regular syntactic token from a CoNLL-U sentence.

    We skip:
    - multiword token lines like 1-2
    - empty nodes like 3.1

    Fields:
    - idx: integer token index in sentence
    - form: surface form
    - lemma: lemma
    - upos: universal POS tag
    - xpos: language-specific POS tag
    - feats: morphological features
    - head: integer head index, where 0 means ROOT
    - deprel: dependency relation
    - deps: enhanced dependencies
    - misc: miscellaneous field
    """
    idx: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str
    deps: str
    misc: str


@dataclass
class UDSentence:
    """
    One UD sentence with optional metadata and regular syntactic tokens.
    """
    metadata: Dict[str, str]
    tokens: List[UDToken]


def normalize_token(text: str, lowercase: bool = False) -> str:
    """
    Normalize a token so flattened outputs remain whitespace-safe.
    """
    if text is None or text == "":
        text = "_"

    text = text.strip()

    if lowercase:
        text = text.lower()

    text = re.sub(r"\s+", "_", text)
    return text


def normalize_label(text: str) -> str:
    """
    Normalize annotation labels for pretraining text.
    """
    if text is None or text == "":
        return "_"

    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    return text


def parse_conllu(path: Path) -> Iterable[UDSentence]:
    """
    Parse a CoNLL-U file.

    This parser intentionally handles standard UD files without external dependencies.

    It skips:
    - comments except for storing metadata
    - multiword tokens with IDs like 1-2
    - empty nodes with IDs like 3.1
    - malformed token lines
    """
    metadata: Dict[str, str] = {}
    tokens: List[UDToken] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if not line.strip():
                if tokens:
                    yield UDSentence(metadata=metadata, tokens=tokens)
                metadata = {}
                tokens = []
                continue

            if line.startswith("#"):
                meta = line[1:].strip()
                if "=" in meta:
                    key, value = meta.split("=", 1)
                    metadata[key.strip()] = value.strip()
                continue

            fields = line.split("\t")

            if len(fields) != 10:
                continue

            token_id = fields[0]

            # Skip multiword token ranges, e.g. 1-2
            if "-" in token_id:
                continue

            # Skip empty nodes, e.g. 3.1
            if "." in token_id:
                continue

            try:
                idx = int(token_id)
            except ValueError:
                continue

            try:
                head = int(fields[6])
            except ValueError:
                continue

            token = UDToken(
                idx=idx,
                form=fields[1],
                lemma=fields[2],
                upos=fields[3],
                xpos=fields[4],
                feats=fields[5],
                head=head,
                deprel=fields[7],
                deps=fields[8],
                misc=fields[9],
            )

            tokens.append(token)

    if tokens:
        yield UDSentence(metadata=metadata, tokens=tokens)


def head_direction(token: UDToken) -> str:
    """
    Return dependency direction relative to the token.

    ROOT means head = 0.
    SELF is rare/invalid in UD but handled safely.
    """
    if token.head == 0:
        return "ROOT"

    if token.head < token.idx:
        return "HEAD_LEFT"

    if token.head > token.idx:
        return "HEAD_RIGHT"

    return "HEAD_SELF"


def distance_bucket(token: UDToken) -> str:
    """
    Return a coarse bucket for absolute head distance.

    Exact head indices are likely less C-RASP-friendly.
    Bucketing keeps distance information without arbitrary retrieval.
    """
    if token.head == 0:
        return "DIST_ROOT"

    dist = abs(token.head - token.idx)

    if dist == 0:
        return "DIST_SELF"

    if dist == 1:
        return "DIST_1"

    if dist == 2:
        return "DIST_2"

    if 3 <= dist <= 4:
        return "DIST_3_4"

    if 5 <= dist <= 8:
        return "DIST_5_8"

    if 9 <= dist <= 16:
        return "DIST_9_16"

    return "DIST_17_PLUS"


def signpost(
    token_idx: int,
    sentence_id: Optional[int] = None,
    include_sentence_id: bool = False,
) -> str:
    """
    Create a token signpost.

    If include_sentence_id is False:
        ID_3

    If include_sentence_id is True:
        S_12_ID_3
    """
    if include_sentence_id:
        if sentence_id is None:
            raise ValueError("sentence_id is required when include_sentence_id=True")
        return f"S_{sentence_id}_ID_{token_idx}"

    return f"ID_{token_idx}"


def flatten_surface(sentence: UDSentence, lowercase_forms: bool = False) -> str:
    """
    Surface word sequence.
    """
    return " ".join(
        normalize_token(tok.form, lowercase=lowercase_forms)
        for tok in sentence.tokens
    )


def flatten_upos_deprel(sentence: UDSentence) -> str:
    """
    UPOS + dependency label per token.

    Example:
        DET__det NOUN__nsubj VERB__root
    """
    pieces = []

    for tok in sentence.tokens:
        upos = normalize_label(tok.upos)
        deprel = normalize_label(tok.deprel)
        pieces.append(f"{upos}__{deprel}")

    return " ".join(pieces)


def flatten_local(sentence: UDSentence) -> str:
    """
    UPOS + dependency label + head direction.

    This is the most conservative C-RASP-ish flattening.

    Example:
        DET__det__HEAD_RIGHT NOUN__nsubj__HEAD_RIGHT VERB__root__ROOT
    """
    pieces = []

    for tok in sentence.tokens:
        upos = normalize_label(tok.upos)
        deprel = normalize_label(tok.deprel)
        direction = head_direction(tok)
        pieces.append(f"{upos}__{deprel}__{direction}")

    return " ".join(pieces)


def flatten_distance(sentence: UDSentence) -> str:
    """
    UPOS + dependency label + head direction + bucketed head distance.

    Example:
        DET__det__HEAD_RIGHT__DIST_1
        NOUN__nsubj__HEAD_RIGHT__DIST_1
        VERB__root__ROOT__DIST_ROOT
    """
    pieces = []

    for tok in sentence.tokens:
        upos = normalize_label(tok.upos)
        deprel = normalize_label(tok.deprel)
        direction = head_direction(tok)
        dist = distance_bucket(tok)
        pieces.append(f"{upos}__{deprel}__{direction}__{dist}")

    return " ".join(pieces)


def flatten_arc(sentence: UDSentence, lowercase_forms: bool = False) -> str:
    """
    Arc statements without explicit signpost tokens.

    This contains exact structural information but introduces exact head positions.
    """
    pieces = []

    for tok in sentence.tokens:
        form = normalize_token(tok.form, lowercase=lowercase_forms)
        upos = normalize_label(tok.upos)
        deprel = normalize_label(tok.deprel)

        pieces.append(
            f"TOK_{tok.idx}_FORM_{form}_UPOS_{upos}_HEAD_{tok.head}_REL_{deprel}"
        )

    return " ".join(pieces)


def flatten_signpost(
    sentence: UDSentence,
    sentence_id: Optional[int] = None,
    include_sentence_id: bool = False,
    lowercase_forms: bool = False,
) -> str:
    """
    Explicit signpost/head-signpost format.

    Example:
        ID_1 FORM_The UPOS_DET HEAD_ID_2 REL_det ;
        ID_2 FORM_dog UPOS_NOUN HEAD_ID_3 REL_nsubj ;
        ID_3 FORM_barked UPOS_VERB HEAD_ROOT REL_root ;
    """
    pieces = []

    for tok in sentence.tokens:
        tok_id = signpost(
            tok.idx,
            sentence_id=sentence_id,
            include_sentence_id=include_sentence_id,
        )

        if tok.head == 0:
            head_id = "HEAD_ROOT"
        else:
            head_id = "HEAD_" + signpost(
                tok.head,
                sentence_id=sentence_id,
                include_sentence_id=include_sentence_id,
            )

        form = normalize_token(tok.form, lowercase=lowercase_forms)
        upos = normalize_label(tok.upos)
        deprel = normalize_label(tok.deprel)

        pieces.extend(
            [
                tok_id,
                f"FORM_{form}",
                f"UPOS_{upos}",
                head_id,
                f"REL_{deprel}",
                ";",
            ]
        )

    return " ".join(pieces)


def flatten_shuffled_deprel_control(sentence: UDSentence, rng: random.Random) -> str:
    """
    Control condition:
    preserve UPOS and head direction, but shuffle dependency relation labels
    within the sentence.
    """
    if not sentence.tokens:
        return ""

    shuffled_labels = [tok.deprel for tok in sentence.tokens]
    rng.shuffle(shuffled_labels)

    pieces = []

    for tok, shuffled_deprel in zip(sentence.tokens, shuffled_labels):
        upos = normalize_label(tok.upos)
        deprel = normalize_label(shuffled_deprel)
        direction = head_direction(tok)
        pieces.append(f"{upos}__{deprel}__{direction}")

    return " ".join(pieces)


def valid_shuffled_heads(
    tokens: List[UDToken],
    rng: random.Random,
    max_attempts: int = 100,
) -> List[int]:
    """
    Create a shuffled list of heads for a sentence.

    We preserve the multiset of head values but try to avoid:
    - token heading itself
    - losing all roots

    This does NOT guarantee a valid dependency tree.
    That is okay because it is a corruption/control condition.
    """
    original_heads = [tok.head for tok in tokens]

    if len(tokens) <= 1:
        return original_heads[:]

    token_indices = [tok.idx for tok in tokens]

    for _ in range(max_attempts):
        candidate_heads = original_heads[:]
        rng.shuffle(candidate_heads)

        has_root = any(h == 0 for h in candidate_heads)
        no_self_heads = all(
            h != idx for h, idx in zip(candidate_heads, token_indices)
        )

        if has_root and no_self_heads:
            return candidate_heads

    # Fallback: shuffle and repair obvious self-heads when possible.
    candidate_heads = original_heads[:]
    rng.shuffle(candidate_heads)

    for i, tok in enumerate(tokens):
        if candidate_heads[i] == tok.idx:
            for j, other_tok in enumerate(tokens):
                if i == j:
                    continue

                if candidate_heads[j] != tok.idx and candidate_heads[i] != other_tok.idx:
                    candidate_heads[i], candidate_heads[j] = (
                        candidate_heads[j],
                        candidate_heads[i],
                    )
                    break

    return candidate_heads


def flatten_shuffled_head_control(sentence: UDSentence, rng: random.Random) -> str:
    """
    Control condition:
    preserve UPOS and dependency labels, but shuffle head assignments.

    Output format mirrors the distance scheme:
        UPOS__deprel__direction__distance
    """
    if not sentence.tokens:
        return ""

    shuffled_heads = valid_shuffled_heads(sentence.tokens, rng)

    pieces = []

    for tok, new_head in zip(sentence.tokens, shuffled_heads):
        corrupted = UDToken(
            idx=tok.idx,
            form=tok.form,
            lemma=tok.lemma,
            upos=tok.upos,
            xpos=tok.xpos,
            feats=tok.feats,
            head=new_head,
            deprel=tok.deprel,
            deps=tok.deps,
            misc=tok.misc,
        )

        upos = normalize_label(corrupted.upos)
        deprel = normalize_label(corrupted.deprel)
        direction = head_direction(corrupted)
        dist = distance_bucket(corrupted)

        pieces.append(f"{upos}__{deprel}__{direction}__{dist}")

    return " ".join(pieces)


def flatten_sentence(
    sentence: UDSentence,
    scheme: str,
    rng: random.Random,
    sentence_id: Optional[int] = None,
    include_sentence_id: bool = False,
    lowercase_forms: bool = False,
) -> str:
    """
    Dispatch flattening by scheme name.
    """
    if scheme == "surface":
        return flatten_surface(sentence, lowercase_forms=lowercase_forms)

    if scheme == "upos_deprel":
        return flatten_upos_deprel(sentence)

    if scheme == "local":
        return flatten_local(sentence)

    if scheme == "distance":
        return flatten_distance(sentence)

    if scheme == "arc":
        return flatten_arc(sentence, lowercase_forms=lowercase_forms)

    if scheme == "signpost":
        return flatten_signpost(
            sentence,
            sentence_id=sentence_id,
            include_sentence_id=include_sentence_id,
            lowercase_forms=lowercase_forms,
        )

    if scheme == "shuffled_deprel_control":
        return flatten_shuffled_deprel_control(sentence, rng=rng)

    if scheme == "shuffled_head_control":
        return flatten_shuffled_head_control(sentence, rng=rng)

    raise ValueError(f"Unsupported scheme: {scheme}")


def resolve_schemes(raw_schemes: Sequence[str]) -> List[str]:
    """
    Resolve --schemes argument.

    Allows:
        --schemes all
        --schemes surface local signpost
    """
    if not raw_schemes:
        raise ValueError("At least one scheme must be provided.")

    if SPECIAL_SCHEME_ALL in raw_schemes:
        if len(raw_schemes) > 1:
            raise ValueError("Use either --schemes all or list individual schemes, not both.")

        return sorted(SUPPORTED_SCHEMES)

    unknown = [scheme for scheme in raw_schemes if scheme not in SUPPORTED_SCHEMES]

    if unknown:
        raise ValueError(
            "Unsupported scheme(s): "
            + ", ".join(unknown)
            + "\nSupported schemes: "
            + ", ".join(sorted(SUPPORTED_SCHEMES))
        )

    return list(raw_schemes)


def find_conllu_files(input_dir: Path) -> List[Path]:
    """
    Recursively find all .conllu files inside input_dir.
    """
    files = sorted(input_dir.rglob("*.conllu"))
    return files


def output_path_for_file(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    scheme: str,
) -> Path:
    """
    Create an output path for one input file and one scheme.

    If input file is:

        EWT-corpus/en_ewt-ud-train.conllu

    output becomes:

        EWT-flattened/en_ewt-ud-train.local.txt

    If the input file is inside a subfolder, the subfolder structure is preserved.
    """
    relative = input_file.relative_to(input_dir)

    if relative.name.endswith(".conllu"):
        stem = relative.name[:-len(".conllu")]
    else:
        stem = relative.stem

    output_relative = relative.with_name(f"{stem}.{scheme}.txt")
    return output_dir / output_relative


def write_flattened_for_one_file(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    schemes: Sequence[str],
    seed: int = 42,
    max_sentences: Optional[int] = None,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
    include_sentence_id: bool = False,
    lowercase_forms: bool = False,
) -> Dict[str, int]:
    """
    Flatten one .conllu file into one output file per scheme.
    """
    rng = random.Random(seed)

    output_paths = {
        scheme: output_path_for_file(
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            scheme=scheme,
        )
        for scheme in schemes
    }

    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    writers = {
        scheme: output_paths[scheme].open("w", encoding="utf-8")
        for scheme in schemes
    }

    n_seen = 0
    n_written = 0
    n_skipped_short = 0
    n_skipped_long = 0
    n_empty_outputs = 0

    try:
        for sent_id, sentence in enumerate(parse_conllu(input_file), start=1):
            n_seen += 1
            n_tokens = len(sentence.tokens)

            if n_tokens < min_tokens:
                n_skipped_short += 1
                continue

            if max_tokens is not None and n_tokens > max_tokens:
                n_skipped_long += 1
                continue

            if max_sentences is not None and n_written >= max_sentences:
                break

            any_written_for_sentence = False

            for scheme in schemes:
                flattened = flatten_sentence(
                    sentence=sentence,
                    scheme=scheme,
                    rng=rng,
                    sentence_id=sent_id,
                    include_sentence_id=include_sentence_id,
                    lowercase_forms=lowercase_forms,
                ).strip()

                if flattened:
                    writers[scheme].write(flattened + "\n")
                    any_written_for_sentence = True

            if any_written_for_sentence:
                n_written += 1
            else:
                n_empty_outputs += 1

    finally:
        for writer in writers.values():
            writer.close()

    print(f"\nFinished: {input_file}")
    print(f"  Sentences seen: {n_seen}")
    print(f"  Sentences written: {n_written}")
    print(f"  Skipped too short: {n_skipped_short}")
    print(f"  Skipped too long: {n_skipped_long}")
    print(f"  Empty outputs: {n_empty_outputs}")
    print("  Output files:")

    for scheme in schemes:
        print(f"    {scheme}: {output_paths[scheme]}")

    return {
        "seen": n_seen,
        "written": n_written,
        "skipped_short": n_skipped_short,
        "skipped_long": n_skipped_long,
        "empty_outputs": n_empty_outputs,
    }


def write_flattened_corpus(
    input_dir: Path,
    output_dir: Path,
    schemes: Sequence[str],
    seed: int = 42,
    max_sentences: Optional[int] = None,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
    include_sentence_id: bool = False,
    lowercase_forms: bool = False,
) -> None:
    """
    Flatten all .conllu files inside input_dir.
    """
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Make sure your treebanks are in a folder named EWT-corpus, "
            "or pass a different folder with --input_dir."
        )

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    conllu_files = find_conllu_files(input_dir)

    if not conllu_files:
        raise FileNotFoundError(
            f"No .conllu files found inside: {input_dir}\n"
            "Check that the EWT-corpus folder contains files ending in .conllu."
        )

    print("Starting UD flattening.")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Schemes: {', '.join(schemes)}")
    print(f"Found {len(conllu_files)} .conllu file(s):")

    for path in conllu_files:
        print(f"  - {path}")

    total_seen = 0
    total_written = 0
    total_skipped_short = 0
    total_skipped_long = 0
    total_empty_outputs = 0

    for file_index, input_file in enumerate(conllu_files, start=1):
        # Offset the seed by file index so controls are deterministic but not identical
        # across files with the same sentence structures.
        file_seed = seed + file_index

        stats = write_flattened_for_one_file(
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            schemes=schemes,
            seed=file_seed,
            max_sentences=max_sentences,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            include_sentence_id=include_sentence_id,
            lowercase_forms=lowercase_forms,
        )

        total_seen += stats["seen"]
        total_written += stats["written"]
        total_skipped_short += stats["skipped_short"]
        total_skipped_long += stats["skipped_long"]
        total_empty_outputs += stats["empty_outputs"]

    print("\nAll done.")
    print(f"Total files processed: {len(conllu_files)}")
    print(f"Total sentences seen: {total_seen}")
    print(f"Total sentences written: {total_written}")
    print(f"Total skipped too short: {total_skipped_short}")
    print(f"Total skipped too long: {total_skipped_long}")
    print(f"Total empty outputs: {total_empty_outputs}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten all UD-style .conllu treebanks inside EWT-corpus "
            "into multiple pre-pretraining formats."
        )
    )

    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing UD .conllu files. Default: EWT-corpus",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where flattened .txt files will be written. Default: EWT-flattened",
    )

    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["all"],
        help=(
            "Flattening schemes to produce. Use 'all' for all schemes. "
            f"Supported: {', '.join(sorted(SUPPORTED_SCHEMES))}. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for control conditions. Default: 42",
    )

    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help=(
            "Optional maximum number of sentences to write per .conllu file. "
            "Useful for quick testing."
        ),
    )

    parser.add_argument(
        "--min_tokens",
        type=int,
        default=1,
        help="Minimum number of syntactic tokens per sentence. Default: 1",
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional maximum number of syntactic tokens per sentence.",
    )

    parser.add_argument(
        "--include_sentence_id",
        action="store_true",
        help=(
            "Include sentence IDs in signpost tokens, e.g. S_12_ID_3 instead of ID_3. "
            "Usually not needed if each sentence is one line."
        ),
    )

    parser.add_argument(
        "--lowercase_forms",
        action="store_true",
        help="Lowercase surface forms in schemes that include word forms.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    schemes = resolve_schemes(args.schemes)

    write_flattened_corpus(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        schemes=schemes,
        seed=args.seed,
        max_sentences=args.max_sentences,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        include_sentence_id=args.include_sentence_id,
        lowercase_forms=args.lowercase_forms,
    )


if __name__ == "__main__":
    main()