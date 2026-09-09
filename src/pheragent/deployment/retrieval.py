from __future__ import annotations

import math
import posixpath
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import PurePosixPath

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_COMMAND = re.compile(
    r"(?im)^\s*(?:helm(?:file)?\b|kubectl\s+(?:apply|create|wait|rollout)\b|"
    r"docker\s+compose\b|terraform\s+(?:apply|plan)\b|ansible-playbook\b|"
    r"(?:\./|\.\./)[^\s;&|]+\.(?:sh|bash)\b).*$"
)
_INSTALLER_NAMES = frozenset({"install.sh", "deploy.sh", "setup.sh"})
_INSTALLER_NAME = re.compile(r"^(?:install|deploy|setup)(?:[-_.][a-z0-9_.-]+)?\.(?:sh|bash)$")
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_YAML_SUFFIXES = {".yaml", ".yml"}
_SHELL_SUFFIXES = {".sh", ".bash", ".zsh"}
_MARKDOWN_HEADING = re.compile(r"^\s*#{1,6}\s+\S")
_SHELL_FUNCTION = re.compile(
    r"^\s*(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*\s*(?:\(\s*\))?\s*\{\s*$"
)
_MAX_PASSAGE_LINES = 80
_PASSAGE_OVERLAP_LINES = 8


def normalize_terms(value: str) -> tuple[str, ...]:
    """Normalize prose and identifiers without relying on a project vocabulary."""
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return tuple(token.casefold() for token in _TOKEN.findall(separated))


def is_installer_path(path: str) -> bool:
    """Recognize conventional installer scripts without requiring one exact filename."""
    return bool(_INSTALLER_NAME.fullmatch(PurePosixPath(path).name.casefold()))


@dataclass(frozen=True, slots=True)
class RetrievalDocument:
    source_id: str
    source_kind: str
    path: str
    text: str
    roles: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    start_line: int = 1
    end_line: int = 1
    kind: str = "file"

    @property
    def key(self) -> str:
        return f"{self.file_key}:{self.start_line}:{self.end_line}:{self.kind}"

    @property
    def file_key(self) -> str:
        return f"{self.source_id}:{self.path}"


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    terms: tuple[str, ...]
    source_kind: str | None = None
    path_prefix: str | None = None
    seed_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    document: RetrievalDocument
    score: float
    line: int
    reasons: tuple[str, ...]


@dataclass(slots=True)
class _IndexedDocument:
    document: RetrievalDocument
    fields: dict[str, Counter[str]]
    lengths: dict[str, int]


class DeploymentRetrievalEngine:
    """A bounded lexical and structural retriever for deployment evidence."""

    _FIELD_WEIGHTS = {
        "filename": 5.0,
        "commands": 4.0,
        "roles": 3.0,
        "facts": 3.0,
        "path": 2.5,
        "body": 1.0,
    }

    def __init__(
        self,
        documents: list[RetrievalDocument],
        *,
        relations: tuple[tuple[str, str], ...] = (),
    ):
        self._documents = {document.key: document for document in documents}
        self._indexed = [self._index(document) for document in documents]
        self._document_frequency = self._document_frequencies(self._indexed)
        self._average_lengths = self._field_averages(self._indexed)
        self._neighbors = self._build_neighbors(documents, relations)

    def search(self, query: RetrievalQuery, *, limit: int = 3) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        query_tokens = self._query_tokens(query)
        eligible = [item for item in self._indexed if self._eligible(item.document, query)]
        lexical = sorted(
            ((self._lexical_score(item, query_tokens), item.document.key) for item in eligible),
            reverse=True,
        )
        lexical = [(score, key) for score, key in lexical if score > 0]
        structural = self._structural_ranking(query, lexical[: max(limit, 3)])
        fused: dict[str, float] = defaultdict(float)
        reasons: dict[str, set[str]] = defaultdict(set)
        for label, ranking in (("lexical", lexical), ("structural", structural)):
            for rank, (_score, key) in enumerate(ranking, start=1):
                fused[key] += (0.5 if label == "structural" else 1.0) / (40 + rank)
                reasons[key].add(label)
        ranked = self._select_diverse(
            sorted(fused, key=lambda key: (-fused[key], key)),
            query=query,
            limit=limit,
        )
        return [
            RetrievalHit(
                document=self._documents[key],
                score=fused[key],
                line=self._best_line(self._documents[key], query_tokens),
                reasons=tuple(sorted(reasons[key])),
            )
            for key in ranked
        ]

    @classmethod
    def passages(
        cls,
        *,
        source_id: str,
        source_kind: str,
        path: str,
        text: str,
        roles: tuple[str, ...] = (),
        facts: tuple[str, ...] = (),
    ) -> list[RetrievalDocument]:
        """Split source text at format-aware boundaries while retaining provenance."""
        lines = text.splitlines()
        if not lines:
            return []
        kind = _passage_kind(path)
        return [
            RetrievalDocument(
                source_id=source_id,
                source_kind=source_kind,
                path=path,
                text="\n".join(lines[start:end]),
                roles=roles,
                facts=facts,
                start_line=start + 1,
                end_line=end,
                kind=kind,
            )
            for start, end in _passage_ranges(path, lines)
            if any(line.strip() for line in lines[start:end])
        ]

    def _index(self, document: RetrievalDocument) -> _IndexedDocument:
        path = PurePosixPath(document.path)
        commands = "\n".join(match.group(0) for match in _COMMAND.finditer(document.text))
        fields = {
            "filename": Counter(normalize_terms(path.name)),
            "path": Counter(normalize_terms(document.path)),
            "roles": Counter(normalize_terms(" ".join(document.roles))),
            "facts": Counter(normalize_terms(" ".join(document.facts))),
            "commands": Counter(normalize_terms(commands)),
            "body": Counter(normalize_terms(document.text)),
        }
        return _IndexedDocument(
            document=document,
            fields=fields,
            lengths={name: sum(tokens.values()) for name, tokens in fields.items()},
        )

    def _lexical_score(self, item: _IndexedDocument, query_tokens: set[str]) -> float:
        score = 0.0
        document_count = max(1, len(self._indexed))
        for field_name, weight in self._FIELD_WEIGHTS.items():
            tokens = item.fields[field_name]
            length = item.lengths[field_name]
            average = self._average_lengths.get(field_name, 1.0) or 1.0
            for token in query_tokens:
                frequency = tokens.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[field_name].get(token, 0)
                inverse_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (0.25 + 0.75 * length / average)
                score += weight * inverse_frequency * frequency * 2.2 / denominator
        return score

    def _structural_ranking(
        self,
        query: RetrievalQuery,
        lexical: list[tuple[float, str]],
    ) -> list[tuple[float, str]]:
        seeds = {key for _score, key in lexical}
        scores: dict[str, float] = defaultdict(float)
        for seed in seeds:
            for neighbor in self._neighbors.get(seed, ()):
                document = self._documents[neighbor]
                if self._eligible(document, query):
                    scores[neighbor] += 1.0
        return sorted(((score, key) for key, score in scores.items()), reverse=True)

    def _query_tokens(self, query: RetrievalQuery) -> set[str]:
        tokens = {
            token
            for value in (*query.terms, *query.seed_paths)
            for token in normalize_terms(value)
            if len(token) > 1
        }
        # Deployment verbs are useful when looking for a component's execution route.
        if query.seed_paths:
            tokens.update({"install", "deploy", "apply", "helm", "kubectl"})
        return tokens

    def _select_diverse(
        self,
        ranked: list[str],
        *,
        query: RetrievalQuery,
        limit: int,
    ) -> list[str]:
        if query.source_kind is not None or limit < 2 or not ranked:
            return ranked[:limit]
        selected = [ranked[0]]
        first_kind = self._documents[ranked[0]].source_kind
        contrasting = next(
            (key for key in ranked[1:] if self._documents[key].source_kind != first_kind),
            None,
        )
        if contrasting:
            selected.append(contrasting)
        selected.extend(key for key in ranked if key not in selected)
        return selected[:limit]

    @staticmethod
    def _eligible(document: RetrievalDocument, query: RetrievalQuery) -> bool:
        if query.source_kind and document.source_kind != query.source_kind:
            return False
        return not query.path_prefix or document.path.startswith(query.path_prefix.strip("/"))

    @staticmethod
    def _best_line(document: RetrievalDocument, query_tokens: set[str]) -> int:
        best = (0, document.start_line)
        for number, line in enumerate(document.text.splitlines(), start=document.start_line):
            score = len(set(normalize_terms(line)) & query_tokens)
            if score > best[0]:
                best = (score, number)
        return best[1]

    @staticmethod
    def _document_frequencies(
        indexed: list[_IndexedDocument],
    ) -> dict[str, Counter[str]]:
        result: dict[str, Counter[str]] = defaultdict(Counter)
        for item in indexed:
            for field_name, tokens in item.fields.items():
                result[field_name].update(tokens.keys())
        return result

    @staticmethod
    def _field_averages(indexed: list[_IndexedDocument]) -> dict[str, float]:
        if not indexed:
            return {}
        return {
            field_name: sum(item.lengths[field_name] for item in indexed) / len(indexed)
            for field_name in indexed[0].fields
        }

    @staticmethod
    def _build_neighbors(
        documents: list[RetrievalDocument],
        relations: tuple[tuple[str, str], ...],
    ) -> dict[str, set[str]]:
        by_file: dict[str, list[str]] = defaultdict(list)
        for document in documents:
            by_file[document.file_key].append(document.key)
        neighbors: dict[str, set[str]] = defaultdict(set)
        for passage_keys in by_file.values():
            for left, right in zip(passage_keys, passage_keys[1:], strict=False):
                neighbors[left].add(right)
                neighbors[right].add(left)
        for document in documents:
            path = PurePosixPath(document.path)
            structural_paths = set()
            parent = path.parent
            for name in _INSTALLER_NAMES:
                structural_paths.add(str(parent / name))
                if parent != PurePosixPath("."):
                    structural_paths.add(str(parent.parent / name))
            for raw in structural_paths:
                resolved = str(PurePosixPath(parent, raw)) if raw.startswith(".") else raw
                normalized = posixpath.normpath(resolved)
                for target in by_file.get(f"{document.source_id}:{normalized}", ()):
                    neighbors[document.key].add(target)
                    neighbors[target].add(document.key)
        for source, target in relations:
            for source_key in by_file.get(source, ()):
                for target_key in by_file.get(target, ()):
                    neighbors[source_key].add(target_key)
                    neighbors[target_key].add(source_key)
        return neighbors


def _passage_ranges(path: str, lines: list[str]) -> list[tuple[int, int]]:
    suffix = PurePosixPath(path).suffix.casefold()
    starts = [0]
    if suffix in _MARKDOWN_SUFFIXES:
        starts.extend(
            index
            for index, line in enumerate(lines[1:], start=1)
            if _MARKDOWN_HEADING.match(line)
        )
    elif suffix in _YAML_SUFFIXES:
        starts.extend(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    elif suffix in _SHELL_SUFFIXES:
        starts.extend(
            index
            for index, line in enumerate(lines[1:], start=1)
            if _SHELL_FUNCTION.match(line)
        )
    ranges = [
        (start, end)
        for start, end in zip(starts, [*starts[1:], len(lines)], strict=True)
    ]
    return [bounded for start, end in ranges for bounded in _bounded_ranges(start, end)]


def _bounded_ranges(start: int, end: int) -> list[tuple[int, int]]:
    if end - start <= _MAX_PASSAGE_LINES:
        return [(start, end)]
    step = _MAX_PASSAGE_LINES - _PASSAGE_OVERLAP_LINES
    return [
        (offset, min(offset + _MAX_PASSAGE_LINES, end))
        for offset in range(start, end, step)
        if offset < end
    ]


def _passage_kind(path: str) -> str:
    suffix = PurePosixPath(path).suffix.casefold()
    return (
        "documentation_section"
        if suffix in _MARKDOWN_SUFFIXES
        else "yaml_document"
        if suffix in _YAML_SUFFIXES
        else "shell_section"
        if suffix in _SHELL_SUFFIXES
        else "text_section"
    )
