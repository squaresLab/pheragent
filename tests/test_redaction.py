from pheragent.deployment.redaction import redact_secrets


def test_redaction_covers_common_secret_forms() -> None:
    text = "\n".join(
        (
            "password=actual-password",
            "token=${TOKEN_REFERENCE}",
            "url=https://example.test/hook?access_key=actual-key&mode=test",
            "Authorization: Bearer bearer-value",
            "aws_access_key_id=AKIAIOSFODNN7EXAMPLE",
            "-----BEGIN PRIVATE KEY-----",
            "private-key-material",
            "-----END PRIVATE KEY-----",
        )
    )

    redacted = redact_secrets(text)

    assert "actual-password" not in redacted
    assert "actual-key" not in redacted
    assert "bearer-value" not in redacted
    assert "AKIAIOSFODNN7EXAMPLE" not in redacted
    assert "private-key-material" not in redacted
    assert "token=${TOKEN_REFERENCE}" in redacted
    assert "[REDACTED PRIVATE KEY]" in redacted


def test_redaction_removes_jwt_like_values() -> None:
    token = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJmaXh0dXJlIn0.signaturevalue"

    assert redact_secrets(f"value={token}") == "value=[REDACTED JWT]"
