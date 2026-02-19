import json

from utils.llm_wrapper import llm_chat


def _build_context_block(chunks, max_chunks=20, max_chars=12000):
    selected = []
    total = 0
    for c in chunks:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        if len(selected) >= max_chunks:
            break
        if total + len(text) > max_chars:
            break
        selected.append(text)
        total += len(text)
    return "\n\n".join(selected)


def mapreduce_answer(query, chunks_file, map_count=10, map_max_chunks=20, map_max_chars=12000):
    with open(chunks_file, encoding="utf-8") as r:
        all_chunks = [json.loads(l) for l in r]

    chunk_splits = [all_chunks[i::map_count] for i in range(map_count)]
    partial = []
    for sp in chunk_splits:
        if not sp:
            continue
        context_block = _build_context_block(
            sp,
            max_chunks=map_max_chunks,
            max_chars=map_max_chars,
        )
        msgs = [
            {
                "role": "system",
                "content": "You answer strictly based on provided context. If unsure, say insufficient evidence.",
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nContexts:\n{context_block}",
            },
        ]
        partial.append(llm_chat(msgs))

    msgs = [
        {"role": "system", "content": "Combine partial answers into one concise final answer."},
        {"role": "user", "content": f"Question:\n{query}\n\nPartial answers:\n{partial}"},
    ]
    return llm_chat(msgs)
