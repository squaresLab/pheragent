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
_PATH_TOKEN = re.compile(r"(?:\./|\.\./)?[A-Za-z0-9_.${}/-]+")
_REFERENCE_SUFFIXES = (".sh", ".yaml", ".yml")
_MAX_REFERENCES_PER_DOCUMENT = 128
_INSTALLER_NAMES = frozenset({"install.sh", "deploy.sh", "setup.sh"})
_INSTALLER_NAME = re.compile(r"^(?:install|deploy|setup)(?:[-_.][a-z0-9_.-]+)?\.(?:sh|bash)$")


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
    references: tuple[str, ...] = ()

    @property
    def key(self) -> str:
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
        "path": 2.5,
        "body": 1.0,
    }

    def __init__(self, documents: list[RetrievalDocument]):
        self._documents = {document.key: document for document in documents}
        self._indexed = [self._index(document) for document in documents]
        self._index_by_key = {item.document.key: item for item in self._indexed}
        self._document_frequency = self._document_frequencies(self._indexed)
        self._average_lengths = self._field_averages(self._indexed)
        self._neighbors = self._build_neighbors(documents)

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
        feedback_tokens = self._feedback_tokens(lexical[:3], query_tokens)
        if feedback_tokens:
            lexical = sorted(
                (
                    (
                        score
                        + 0.25
                        * self._lexical_score(
                            self._index_by_key[key],
                            feedback_tokens,
                        ),
                        key,
                    )
                    for score, key in lexical
                ),
                reverse=True,
            )
        exact = sorted(
            ((self._exact_score(item.document, query), item.document.key) for item in eligible),
            reverse=True,
        )
        exact = [(score, key) for score, key in exact if score > 0]
        structural = self._structural_ranking(query, lexical[: max(limit, 3)])
        fused: dict[str, float] = defaultdict(float)
        reasons: dict[str, set[str]] = defaultdict(set)
        for label, ranking in (("lexical", lexical), ("exact", exact), ("structural", structural)):
            for rank, (_score, key) in enumerate(ranking, start=1):
                fused[key] += 1.0 / (40 + rank)
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
                line=self._best_line(self._documents[key].text, query_tokens),
                reasons=tuple(sorted(reasons[key])),
            )
            for key in ranked
        ]

    @classmethod
    def document(
        cls,
        *,
        source_id: str,
        source_kind: str,
        path: str,
        text: str,
        roles: tuple[str, ...] = (),
    ) -> RetrievalDocument:
        """Create a document and extract only explicit local file references."""
        references = _local_references(text)
        return RetrievalDocument(
            source_id=source_id,
            source_kind=source_kind,
            path=path,
            text=text,
            roles=roles,
            references=references,
        )

    def _index(self, document: RetrievalDocument) -> _IndexedDocument:
        path = PurePosixPath(document.path)
        commands = "\n".join(match.group(0) for match in _COMMAND.finditer(document.text))
        fields = {
            "filename": Counter(normalize_terms(path.name)),
            "path": Counter(normalize_terms(document.path)),
            "roles": Counter(normalize_terms(" ".join(document.roles))),
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

    @staticmethod
    def _exact_score(document: RetrievalDocument, query: RetrievalQuery) -> float:
        haystack = f"{document.path}\n{document.text}".casefold()
        return float(sum(term.casefold() in haystack for term in query.terms if term.strip()))

    def _structural_ranking(
        self,
        query: RetrievalQuery,
        lexical: list[tuple[float, str]],
    ) -> list[tuple[float, str]]:
        seeds = {key for _score, key in lexical} | {
            f"{document.source_id}:{path}"
            for document in self._documents.values()
            for path in query.seed_paths
            if f"{document.source_id}:{path}" in self._documents
        }
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

    def _feedback_tokens(
        self,
        initial: list[tuple[float, str]],
        query_tokens: set[str],
    ) -> set[str]:
        """Expand with repeated identifiers from top paths/commands, never arbitrary prose."""
        counts: Counter[str] = Counter()
        for _score, key in initial:
            item = self._index_by_key[key]
            counts.update(set(item.fields["filename"]) | set(item.fields["commands"]))
        ignored = query_tokens | {"install", "deploy", "apply", "helm", "kubectl", "sh"}
        return {
            token
            for token, count in counts.most_common(4)
            if count >= 2 and token not in ignored and len(token) > 2
        }

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
    def _best_line(text: str, query_tokens: set[str]) -> int:
        best = (0, 1)
        for number, line in enumerate(text.splitlines(), start=1):
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
    def _build_neighbors(documents: list[RetrievalDocument]) -> dict[str, set[str]]:
        by_source_path = {(item.source_id, item.path): item.key for item in documents}
        neighbors: dict[str, set[str]] = defaultdict(set)
        for document in documents:
            path = PurePosixPath(document.path)
            structural_paths = set(document.references)
            parent = path.parent
            for name in _INSTALLER_NAMES:
                structural_paths.add(str(parent / name))
                if parent != PurePosixPath("."):
                    structural_paths.add(str(parent.parent / name))
            for raw in structural_paths:
                resolved = str(PurePosixPath(parent, raw)) if raw.startswith(".") else raw
                normalized = posixpath.normpath(resolved)
                target = by_source_path.get((document.source_id, normalized))
                if target:
                    neighbors[document.key].add(target)
                    neighbors[target].add(document.key)
        return neighbors


def _local_references(text: str) -> tuple[str, ...]:
    """Extract path-like deployment references in linear time with a hard cap."""
    references: dict[str, None] = {}
    for match in _PATH_TOKEN.finditer(text):
        token = match.group(0)
        if not token.casefold().endswith(_REFERENCE_SUFFIXES):
            continue
        references.setdefault(token, None)
        if len(references) >= _MAX_REFERENCES_PER_DOCUMENT:
            break
    return tuple(references)
