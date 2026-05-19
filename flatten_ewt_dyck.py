#!/usr/bin/env python3
"""
flatten_ewt_dyck.py

Create annotation-only, Dyck-like flattenings from UD-style CoNLL-U treebanks.

Default input folder:

    EWT-corpus/

Default output folder:

    EWT-dyck-flattened/

This script uses NO WORD FORMS and NO LEMMAS.
It only uses structural labels such as:

    UPOS
    DEPREL
    HEAD direction
    dependency tree structure

Supported schemes:

1. deprel_sequence
   Surface-order dependency relation sequence.

   Example:
       det nsubj root obj punct

2. upos_deprel_sequence
   Surface-order UPOS + dependency relation sequence.

   Example:
       DET__det NOUN__nsubj VERB__root NOUN__obj PUNCT__punct

3. dep_dyck_unlabeled
   Tree shape only, using generic brackets.

   Example:
       OPEN OPEN CLOSE OPEN CLOSE CLOSE

4. dep_dyck_labeled
   Nested dependency tree, with DEPREL as bracket type.

   Example:
       OPEN_root OPEN_nsubj OPEN_det CLOSE_det CLOSE_nsubj OPEN_obj CLOSE_obj CLOSE_root

5. dep_dyck_oriented
   Nested dependency tree, with head direction + DEPREL as bracket type.

   Direction is from dependent to head:
       L means the dependent's head is to its left.
       R means the dependent's head is to its right.
       ROOT means the token is attached to root.

   Example:
       OPEN_ROOT_root OPEN_R_nsubj OPEN_R_det CLOSE_R_det CLOSE_R_nsubj CLOSE_ROOT_root

6. dep_dyck_upos
   Nested dependency tree, with DEPREL + UPOS as bracket type.

   Example:
       OPEN_root_VERB OPEN_nsubj_NOUN OPEN_det_DET CLOSE_det_DET CLOSE_nsubj_NOUN CLOSE_root_VERB

7. dep_dyck_oriented_upos
   Nested dependency tree, with direction + DEPREL + UPOS as bracket type.

   Example:
       OPEN_ROOT_root_VERB OPEN_R_nsubj_NOUN CLOSE_R_nsubj_NOUN CLOSE_ROOT_root_VERB

8. dep_arc_shuffle
   Linear arc-event encoding. For each dependency arc, emit an OPEN event at
   the earlier endpoint and a CLOSE event at the later endpoint. Crossing arcs
   therefore produce interleaved, Shuffle-Dyck-like patterns.

   Example:
       OPEN_root OPEN_nsubj OPEN_det CLOSE_det CLOSE_nsubj CLOSE_root

   This is not a tree traversal. It is an arc-span event language over the
   surface order.

9. dep_arc_shuffle_oriented
   Same as dep_arc_shuffle, but arc labels include direction.

10. shuffled_deprel_dyck_control
    Same tree shape as dep_dyck_labeled, but DEPREL labels are shuffled within
    each sentence.

11. shuffled_deprel_arc_control
    Same arc-event structure as dep_arc_shuffle, but DEPREL labels are shuffled
    within each sentence.

Basic usage:

    python flatten_ewt_dyck.py

Only selected schemes:

    python flatten_ewt_dyck.py --schemes dep_dyck_labeled dep_arc_shuffle shuffled_deprel_dyck_control

Quick test:

    python flatten_ewt_dyck.py --max_sentences 1000

No external libraries required.
"""

from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = Path("EWT-corpus")
DEFAULT_OUTPUT_DIR = Path("EWT-dyck-flattened")

SUPPORTED_SCHEMES = {
    "deprel_sequence",
    "upos_deprel_sequence",
    "dep_dyck_unlabeled",
    "dep_dyck_labeled",
    "dep_dyck_oriented",
    "dep_dyck_upos",
    "dep_dyck_oriented_upos",
    "dep_arc_shuffle",
    "dep_arc_shuffle_oriented",
    "shuffled_deprel_dyck_control",
    "shuffled_deprel_arc_control",
}

SPECIAL_SCHEME_ALL = "all"


@dataclass
class UDToken:
    idx: int
    upos: str
    head: int
    deprel: str


@dataclass
class UDSentence:
    metadata: Dict[str, str]
    tokens: List[UDToken]


def normalize_label(text: str) -> str:
    """
    Make labels whitespace-safe and tokenizer-friendly.

    Examples:
        nsubj:pass -> nsubj_PASS
        obl:tmod   -> obl_TMOD
    """
    if text is None or text == "":
        return "_"

    text = text.strip()
    text = re.sub(r"\s+", "_", text)
    text = text.replace(":", "_")
    text = text.replace("-", "_")
    text = text.replace("/", "_")
    text = text.replace("\\", "_")

    if text == "":
        return "_"

    return text


def parse_conllu(path: Path) -> Iterable[UDSentence]:
    """
    Parse a CoNLL-U file.

    Uses only:
        ID
        UPOS
        HEAD
        DEPREL

    Skips:
        multiword token lines, e.g. 1-2
        empty nodes, e.g. 3.1
        malformed lines
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

            if "-" in token_id:
                continue

            if "." in token_id:
                continue

            try:
                idx = int(token_id)
            except ValueError:
                continue

            upos = normalize_label(fields[3])

            try:
                head = int(fields[6])
            except ValueError:
                continue

            deprel = normalize_label(fields[7])

            tokens.append(
                UDToken(
                    idx=idx,
                    upos=upos,
                    head=head,
                    deprel=deprel,
                )
            )

    if tokens:
        yield UDSentence(metadata=metadata, tokens=tokens)


def head_direction(token: UDToken) -> str:
    """
    Direction from dependent to head.

    If head is left of dependent:
        L

    If head is right of dependent:
        R

    If attached to root:
        ROOT
    """
    if token.head == 0:
        return "ROOT"

    if token.head < token.idx:
        return "L"

    if token.head > token.idx:
        return "R"

    return "SELF"


def token_map(sentence: UDSentence) -> Dict[int, UDToken]:
    return {tok.idx: tok for tok in sentence.tokens}


def build_children(sentence: UDSentence) -> Dict[int, List[UDToken]]:
    """
    Build head -> children mapping.

    Artificial root is 0.

    If a token points to a missing head, attach it to artificial root 0.
    This makes the script robust to malformed or partial treebanks.
    """
    tok_by_id = token_map(sentence)
    children: Dict[int, List[UDToken]] = {0: []}

    for tok in sentence.tokens:
        children.setdefault(tok.idx, [])

    for tok in sentence.tokens:
        if tok.head == 0:
            parent = 0
        elif tok.head in tok_by_id:
            parent = tok.head
        else:
            parent = 0

        children.setdefault(parent, []).append(tok)

    for parent in children:
        children[parent].sort(key=lambda t: t.idx)

    return children


def find_root_tokens(sentence: UDSentence) -> List[UDToken]:
    """
    Return tokens whose HEAD is 0.

    If no explicit root exists, fall back to first token.
    """
    roots = [tok for tok in sentence.tokens if tok.head == 0]

    if roots:
        return sorted(roots, key=lambda t: t.idx)

    if sentence.tokens:
        return [sentence.tokens[0]]

    return []


def bracket_label(token: UDToken, mode: str) -> str:
    """
    Create bracket label for a token.

    mode options:
        unlabeled
        labeled
        oriented
        upos
        oriented_upos
    """
    deprel = normalize_label(token.deprel)
    upos = normalize_label(token.upos)
    direction = head_direction(token)

    if mode == "unlabeled":
        return ""

    if mode == "labeled":
        return deprel

    if mode == "oriented":
        return f"{direction}_{deprel}"

    if mode == "upos":
        return f"{deprel}_{upos}"

    if mode == "oriented_upos":
        return f"{direction}_{deprel}_{upos}"

    raise ValueError(f"Unknown bracket label mode: {mode}")


def open_token(label: str) -> str:
    if label == "":
        return "OPEN"
    return f"OPEN_{label}"


def close_token(label: str) -> str:
    if label == "":
        return "CLOSE"
    return f"CLOSE_{label}"


def traverse_dependency_tree(
    token: UDToken,
    children: Dict[int, List[UDToken]],
    mode: str,
    output: List[str],
    visited: Optional[set[int]] = None,
) -> None:
    """
    Preorder open, recursively emit children, then close.

    This produces a Dyck-like tree traversal.

    Cycle protection is included in case of malformed input.
    """
    if visited is None:
        visited = set()

    if token.idx in visited:
        return

    visited.add(token.idx)

    label = bracket_label(token, mode)
    output.append(open_token(label))

    for child in children.get(token.idx, []):
        traverse_dependency_tree(
            token=child,
            children=children,
            mode=mode,
            output=output,
            visited=visited,
        )

    output.append(close_token(label))


def flatten_dyck(sentence: UDSentence, mode: str) -> str:
    """
    Create a nested dependency-tree bracket sequence.

    Important:
        This is a traversal of the dependency tree, not surface-order arc events.
        It works best as a Dyck-like representation of tree structure.
    """
    if not sentence.tokens:
        return ""

    children = build_children(sentence)
    roots = find_root_tokens(sentence)

    output: List[str] = []
    visited: set[int] = set()

    for root in roots:
        traverse_dependency_tree(
            token=root,
            children=children,
            mode=mode,
            output=output,
            visited=visited,
        )

    # If malformed input caused disconnected components, include them too.
    for tok in sorted(sentence.tokens, key=lambda t: t.idx):
        if tok.idx not in visited:
            traverse_dependency_tree(
                token=tok,
                children=children,
                mode=mode,
                output=output,
                visited=visited,
            )

    return " ".join(output)


def flatten_deprel_sequence(sentence: UDSentence) -> str:
    return " ".join(normalize_label(tok.deprel) for tok in sentence.tokens)


def flatten_upos_deprel_sequence(sentence: UDSentence) -> str:
    return " ".join(
        f"{normalize_label(tok.upos)}__{normalize_label(tok.deprel)}"
        for tok in sentence.tokens
    )


def make_arc_events(
    sentence: UDSentence,
    oriented: bool = False,
    label_override: Optional[Dict[int, str]] = None,
) -> List[Tuple[int, int, str]]:
    """
    Create linear arc events.

    For each dependency arc:
        earlier endpoint gets OPEN_label
        later endpoint gets CLOSE_label

    This makes projective arcs look nested and crossing arcs look interleaved.

    Returns tuples:
        (position, priority, event)

    priority ensures stable ordering:
        opens before closes at the same position for a slightly more permissive
        Shuffle-Dyck-like encoding.
    """
    events: List[Tuple[int, int, str]] = []

    n = len(sentence.tokens)
    if n == 0:
        return events

    for tok in sentence.tokens:
        if label_override is not None and tok.idx in label_override:
            base_label = normalize_label(label_override[tok.idx])
        else:
            base_label = normalize_label(tok.deprel)

        if oriented:
            label = f"{head_direction(tok)}_{base_label}"
        else:
            label = base_label

        if tok.head == 0:
            left = 1
            right = n
            root_label = label

            events.append((left, 0, open_token(root_label)))
            events.append((right, 1, close_token(root_label)))
            continue

        left = min(tok.idx, tok.head)
        right = max(tok.idx, tok.head)

        events.append((left, 0, open_token(label)))
        events.append((right, 1, close_token(label)))

    events.sort(key=lambda item: (item[0], item[1], item[2]))
    return events


def flatten_arc_shuffle(
    sentence: UDSentence,
    oriented: bool = False,
    label_override: Optional[Dict[int, str]] = None,
) -> str:
    """
    Linear arc-event representation.

    This is the closest thing here to a dependency-derived Shuffle-Dyck language.

    It is intentionally not a clean tree traversal. It linearizes arc spans over
    the sentence. Crossing dependencies produce crossing/interleaved events.
    """
    events = make_arc_events(
        sentence=sentence,
        oriented=oriented,
        label_override=label_override,
    )

    return " ".join(event for _, _, event in events)


def make_shuffled_deprel_map(sentence: UDSentence, rng: random.Random) -> Dict[int, str]:
    """
    Shuffle dependency relation labels within a sentence.

    Preserves:
        number of tokens
        tree shape
        UPOS sequence
        multiset of dependency labels

    Destroys:
        correct mapping between tokens and dependency labels
    """
    labels = [tok.deprel for tok in sentence.tokens]
    rng.shuffle(labels)

    return {
        tok.idx: normalize_label(label)
        for tok, label in zip(sentence.tokens, labels)
    }


def clone_sentence_with_deprel_override(
    sentence: UDSentence,
    deprel_map: Dict[int, str],
) -> UDSentence:
    """
    Return a sentence with the same tree but replaced dependency labels.
    """
    new_tokens: List[UDToken] = []

    for tok in sentence.tokens:
        new_tokens.append(
            UDToken(
                idx=tok.idx,
                upos=tok.upos,
                head=tok.head,
                deprel=deprel_map.get(tok.idx, tok.deprel),
            )
        )

    return UDSentence(metadata=dict(sentence.metadata), tokens=new_tokens)


def flatten_shuffled_deprel_dyck_control(
    sentence: UDSentence,
    rng: random.Random,
) -> str:
    deprel_map = make_shuffled_deprel_map(sentence, rng)
    corrupted = clone_sentence_with_deprel_override(sentence, deprel_map)
    return flatten_dyck(corrupted, mode="labeled")


def flatten_shuffled_deprel_arc_control(
    sentence: UDSentence,
    rng: random.Random,
) -> str:
    deprel_map = make_shuffled_deprel_map(sentence, rng)
    return flatten_arc_shuffle(
        sentence=sentence,
        oriented=False,
        label_override=deprel_map,
    )


def flatten_sentence(
    sentence: UDSentence,
    scheme: str,
    rng: random.Random,
) -> str:
    if scheme == "deprel_sequence":
        return flatten_deprel_sequence(sentence)

    if scheme == "upos_deprel_sequence":
        return flatten_upos_deprel_sequence(sentence)

    if scheme == "dep_dyck_unlabeled":
        return flatten_dyck(sentence, mode="unlabeled")

    if scheme == "dep_dyck_labeled":
        return flatten_dyck(sentence, mode="labeled")

    if scheme == "dep_dyck_oriented":
        return flatten_dyck(sentence, mode="oriented")

    if scheme == "dep_dyck_upos":
        return flatten_dyck(sentence, mode="upos")

    if scheme == "dep_dyck_oriented_upos":
        return flatten_dyck(sentence, mode="oriented_upos")

    if scheme == "dep_arc_shuffle":
        return flatten_arc_shuffle(sentence, oriented=False)

    if scheme == "dep_arc_shuffle_oriented":
        return flatten_arc_shuffle(sentence, oriented=True)

    if scheme == "shuffled_deprel_dyck_control":
        return flatten_shuffled_deprel_dyck_control(sentence, rng)

    if scheme == "shuffled_deprel_arc_control":
        return flatten_shuffled_deprel_arc_control(sentence, rng)

    raise ValueError(f"Unsupported scheme: {scheme}")


def resolve_schemes(raw_schemes: Sequence[str]) -> List[str]:
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
    return sorted(input_dir.rglob("*.conllu"))


def output_path_for_file(
    input_file: Path,
    input_dir: Path,
    output_dir: Path,
    scheme: str,
) -> Path:
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
    seed: int,
    max_sentences: Optional[int],
    min_tokens: int,
    max_tokens: Optional[int],
) -> Dict[str, int]:
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
        for sentence in parse_conllu(input_file):
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

            wrote_any = False

            for scheme in schemes:
                flattened = flatten_sentence(
                    sentence=sentence,
                    scheme=scheme,
                    rng=rng,
                ).strip()

                if flattened:
                    writers[scheme].write(flattened + "\n")
                    wrote_any = True

            if wrote_any:
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
    seed: int,
    max_sentences: Optional[int],
    min_tokens: int,
    max_tokens: Optional[int],
) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}\n"
            "Expected a folder named EWT-corpus, or pass another folder with --input_dir."
        )

    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    conllu_files = find_conllu_files(input_dir)

    if not conllu_files:
        raise FileNotFoundError(
            f"No .conllu files found inside: {input_dir}\n"
            "Check that EWT-corpus contains files ending in .conllu."
        )

    print("Starting annotation-only Dyck-style UD flattening.")
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
        description="Create annotation-only, Dyck-like flattenings from UD CoNLL-U treebanks."
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
        help="Directory where flattened .txt files will be written. Default: EWT-dyck-flattened",
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
        help="Random seed for control schemes. Default: 42",
    )

    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Optional maximum number of sentences to write per .conllu file.",
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
    )


if __name__ == "__main__":
    main()