from __future__ import annotations

import re

_SENSITIVE_NAME = (
    r"(?:password|passwd|secret|private[_-]?key|token|access[_-]?key|client[_-]?secret|"
    r"authorization)"
)
_KEY_VALUE_PATTERN = re.compile(
    rf"(?im)^(?P<prefix>\s*[^#\n:=]*{_SENSITIVE_NAME}[^\n:=]*\s*[:=]\s*)"
    r"(?P<value>[^\n#]*)(?P<suffix>\s*(?:#.*)?)$"
)
_FLAG_PATTERN = re.compile(rf"(?i)(?P<prefix>--{_SENSITIVE_NAME}(?:=|\s+))(?P<value>[^\s]+)")
_REFERENCE_PATTERN = re.compile(r"^(?:\$\{[^}]+\}|\{\{[^}]+\}\}|<[^>]+>|\$[A-Za-z_][A-Za-z0-9_]*)$")
_PRIVATE_KEY_BLOCK_PATTERN = re.compile(
    r"-----BEGIN [^-\n]*PRIVATE KEY-----.*?-----END [^-\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_QUERY_SECRET_PATTERN = re.compile(rf"(?i)(?P<prefix>[?&]{_SENSITIVE_NAME}=)(?P<value>[^&\s]+)")
_BEARER_PATTERN = re.compile(r"(?i)(?P<prefix>\bBearer\s+)(?P<value>[^\s]+)")
_JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_AWS_ACCESS_KEY_PATTERN = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")


def redact_secrets(text: str) -> str:
    redacted = _PRIVATE_KEY_BLOCK_PATTERN.sub("[REDACTED PRIVATE KEY]", text)
    redacted = _KEY_VALUE_PATTERN.sub(_replace_key_value, redacted)
    redacted = _FLAG_PATTERN.sub(_replace_flag, redacted)
    redacted = _QUERY_SECRET_PATTERN.sub(_replace_flag, redacted)
    redacted = _BEARER_PATTERN.sub(_replace_flag, redacted)
    redacted = _JWT_PATTERN.sub("[REDACTED JWT]", redacted)
    return _AWS_ACCESS_KEY_PATTERN.sub("[REDACTED ACCESS KEY]", redacted)


def _replace_key_value(match: re.Match[str]) -> str:
    value = match.group("value").strip().strip("\"'")
    if not value or _REFERENCE_PATTERN.fullmatch(value):
        return match.group(0)
    return f"{match.group('prefix')}[REDACTED]{match.group('suffix')}"


def _replace_flag(match: re.Match[str]) -> str:
    value = match.group("value").strip().strip("\"'")
    if _REFERENCE_PATTERN.fullmatch(value):
        return match.group(0)
    return f"{match.group('prefix')}[REDACTED]"
