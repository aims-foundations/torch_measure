"""Pure prompt-construction functions, verbatim from the ADeLe paper pipeline."""
from __future__ import annotations


def get_full_instruction(dimension: str, rubric_content: str, item_text: str) -> str:
    """Verbatim prompt template from adgomant/delean-batch-manager/src/.../files.py."""
    return (
        f"The following rubric describes six distinct levels of *{dimension}*"
        f" required by different tasks:\n"
        f"{rubric_content}\n"
        f"\nTASK INSTANCE: {item_text}\n"
        f"\nINSTRUCTION: Score the level of *{dimension}* demanded by the given"
        f" TASK INSTANCE using a discrete value from 0 to 5. Use CHAIN-OF-THOUGHTS"
        f" REASONING to reason step by step before assigning the score. After the"
        f" CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the"
        f' statement: "Thus, the level of *{dimension}* demanded by the given TASK'
        f' INSTANCE is: SCORE", where SCORE is an integer score you have determined.\n'
        f"\nCHAIN-OF-THOUGHTS REASONING STEPS to score the level of *{dimension}*"
        f" demanded by the given TASK INSTANCE above:\n"
    )


def get_ug_instruction(item_text: str, reference_answer: str, ug_rubric_content: str) -> str:
    """Best-faith reconstruction of the UG prompt (prepend format undocumented in paper repos)."""
    return f"{item_text}\n\nReference answer: {reference_answer}\n\n{ug_rubric_content}"
