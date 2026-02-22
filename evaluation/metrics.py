def is_hit(
    payload: dict,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
) -> bool:
    source = payload.get("source") or ""
    text = payload.get("text") or ""
    if expected_source_substr.lower() not in source.lower():
        return False
    if expected_phrase_in_text is not None:
        if expected_phrase_in_text.lower() not in text.lower():
            return False
    return True


def recall_at_k(
    retrieved: list,
    expected_source_substr: str,
    expected_phrase_in_text: str | None = None,
    k: int | None = None,
) -> bool:
    if k is not None:
        retrieved = retrieved[:k]
    for point in retrieved:
        payload = getattr(point, "payload", point) if not isinstance(point, dict) else point
        if is_hit(payload, expected_source_substr, expected_phrase_in_text):
            return True
    return False