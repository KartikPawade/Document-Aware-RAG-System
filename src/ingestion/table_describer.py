"""
ingestion/table_describer.py
─────────────────────────────
LLM-based table description generator for the schema registry.

Given a filename, sheet name, column names, and a small data sample (≤15 rows),
asks the LLM to produce a concise description (30-40 words) of what the table
contains — without echoing column names back.

Uses LCEL: prompt | llm | PydanticOutputParser
Falls back to a plain filename-based description if the LLM call fails,
so ingestion is never blocked.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Output Model
# ─────────────────────────────────────────────────────────────────────────────

class TableDescriptionOutput(BaseModel):
    """Structured output for LLM-generated table descriptions."""
    description: str = Field(
        ...,
        description=(
            "A concise 30-40 word description of what the table contains. "
            "Must NOT mention column names. Focus on the subject matter, "
            "time period, and business domain of the data."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

_parser = PydanticOutputParser(pydantic_object=TableDescriptionOutput)

TABLE_DESCRIPTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a data catalogue assistant. Your job is to write a short, clear description
of what a database table contains, so that a query router can decide whether to use it.

Rules:
- Keep the description between 30 and 40 words.
- Describe the subject matter, domain, time period, and granularity of the data.
- Do NOT mention column names or list fields.
- Do NOT start with "This table contains" — be direct and descriptive.
- Return ONLY valid JSON matching the format instructions below.

{format_instructions}""",
    ),
    (
        "human",
        """File: {filename}
Sheet: {sheet_name}
Columns ({col_count} total): {column_names}
Sample data ({sample_rows} rows):
{sample_data}

Write a description of what this table holds.""",
    ),
])


# ─────────────────────────────────────────────────────────────────────────────
# Describer
# ─────────────────────────────────────────────────────────────────────────────

class TableDescriber:
    """
    Generates LLM-based descriptions for ingested tables.
    Lazy-loads the LLM on first use.
    """

    _SAMPLE_ROWS = 15       # max rows sent to LLM
    _MAX_COLS_PREVIEW = 20  # cap columns shown in prompt (avoids token bloat)

    def __init__(self):
        self._chain = None

    def _get_chain(self):
        if self._chain is None:
            llm = ChatOpenAI(
                model=settings.openai_enrichment_model,
                temperature=0.0,
                api_key=settings.openai_api_key,
            )
            self._chain = (
                TABLE_DESCRIPTION_PROMPT.partial(
                    format_instructions=_parser.get_format_instructions()
                )
                | llm
                | _parser
            )
        return self._chain

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def describe(
        self,
        df: pd.DataFrame,
        filename: str,
        sheet_name: str,
    ) -> str:
        """
        Generate a short description for a table.

        Args:
            df:          The full DataFrame (we sample internally).
            filename:    Original source filename (e.g. "sales_q3.xlsx").
            sheet_name:  Sheet or table label (e.g. "Q3", "default").

        Returns:
            A 30-40 word description string. Falls back gracefully on failure.
        """
        try:
            col_names = list(df.columns)
            col_count = len(col_names)

            # Cap columns shown in prompt to avoid token bloat
            cols_for_prompt = col_names[:self._MAX_COLS_PREVIEW]
            col_names_str = ", ".join(cols_for_prompt)
            if col_count > self._MAX_COLS_PREVIEW:
                col_names_str += f" ... and {col_count - self._MAX_COLS_PREVIEW} more"

            # Sample rows — take from spread of DataFrame, not just head
            sample_df = df.head(self._SAMPLE_ROWS)
            sample_data = sample_df.to_string(index=False, max_cols=self._MAX_COLS_PREVIEW)

            result: TableDescriptionOutput = await self._get_chain().ainvoke({
                "filename":    filename,
                "sheet_name":  sheet_name,
                "col_count":   col_count,
                "column_names": col_names_str,
                "sample_rows": len(sample_df),
                "sample_data": sample_data,
            })

            logger.info(
                "Table description generated",
                filename=filename,
                sheet=sheet_name,
                description=result.description,
            )
            return result.description

        except Exception as exc:
            logger.warning(
                "Table description LLM call failed, using fallback",
                filename=filename,
                sheet=sheet_name,
                error=str(exc),
            )
            # Graceful fallback — never block ingestion
            return _fallback_description(filename, sheet_name, df)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_description(filename: str, sheet_name: str, df: pd.DataFrame) -> str:
    """
    Plain text fallback when the LLM call fails.
    Avoids column names per the same rules.
    """
    sheet_part = f" (sheet: {sheet_name})" if sheet_name and sheet_name != "default" else ""
    return (
        f"Structured data from {filename}{sheet_part}. "
        f"Contains {len(df)} rows of tabular records."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_describer: Optional[TableDescriber] = None


def get_table_describer() -> TableDescriber:
    global _describer
    if _describer is None:
        _describer = TableDescriber()
    return _describer