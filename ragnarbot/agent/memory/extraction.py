"""LLM-based fact extraction from conversations."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from ragnarbot.agent.memory.models import ExtractionType, Fact, Source

EXTRACTION_PROMPT = """Ти — аналітичний екстрактор знань для дослідницької онтології.
Твоє завдання — виявити структурні факти про світ: як організовані компанії,
установи, армії, партії; як влаштовані технології, доктрини, методології;
як функціонують схеми впливу та фінансових потоків.
Ти аналізуєш текст або діалог між Користувачем та AI-асистентом.

## Що ВИТЯГУВАТИ (ЗБЕРІГАТИ):

### Організації та структури
- Хто кому підпорядковується, хто кого контролює
- Хто є власником, засновником, бенефіціаром
- Структура підрозділів, дочірніх компаній, філій
- Партійна приналежність, коаліції, альянси
- Членство у радах, комітетах, організаціях

### Люди як вузли структури
- Посади та ролі (функція в системі, не особисті дані)
- Зв'язки між особами через структури
- Призначення, звільнення, переходи між структурами
- Причетність до схем, рішень, контрактів

### Технології як самостійні сутності
- Що являє собою технологія/система: клас, тип, призначення
- Технічні характеристики зі стратегічним значенням
- Походження, виробник, країна розробки
- Еволюція: що замінює, що є розвитком чого, нові модифікації
- Вразливості та контрзаходи
- Де і як застосовується

### Концепції, доктрини, методології
- Що являє собою концепція: до якої галузі належить
- Ключові принципи якщо є у тексті
- Чим відрізняється від суміжних концепцій
- Зв'язки між концепціями: є_розвитком, суперечить, доповнює

### Схеми та техніки
- Опис механізму схеми як такої
- Техніки маніпуляцій, атак, обходу
- Методи фінансових схем (як саме працює)

### Фінансові потоки
- Хто кому платить, через які структури
- Контракти, тендери, офшори, посередники
- Джерела фінансування

### Геополітика
- Союзи, угоди, протистояння
- Санкції, обмеження, залежності
- Вплив однієї держави/структури на іншу

## Що НЕ ВИТЯГУВАТИ (ПРОПУСКАТИ):
- Дії AI-асистента в розмові
- Одноразові запити та завдання розмови
- Загальновідомі факти без аналітичної цінності
- Емоційні оцінки без фактичного підґрунтя
- Привітання, підтвердження, метадискусія

## Ключове питання:
«Чи допоможе цей факт зрозуміти структуру, зв'язки або механізм роботи
якоїсь системи — зараз або через рік?»

## Напрямок трійки (КРИТИЧНО):
Трійка читається як речення: "subject [predicate] object"
subject — це ХТО РОБИТЬ / ХТО ВОЛОДІЄ / БІЛЬША сутність
object — це НАД ЧИМ / ЩО НАЛЕЖИТЬ / МЕНША сутність

Приклади ПРАВИЛЬНО:
  Palantir Technologies | develops | Gotham     → "Palantir розробляє Gotham"
  Росія | controls | ФСБ                        → "Росія контролює ФСБ"
  Lockheed Martin | manufactured_in | США        → "Lockheed Martin виробляється в США"
  HIMARS | is_type_of | РСЗВ                    → "HIMARS є типом РСЗВ"
  Залужний | appointed_to | Посол у Великій Британії

Приклади НЕПРАВИЛЬНО (інвертовані):
  ✗ Gotham | develops | Palantir Technologies   → продукт не розробляє компанію
  ✗ ФСБ | controls | Росія                      → підрозділ не контролює державу
  ✗ РСЗВ | is_type_of | HIMARS                  → клас не є типом конкретної системи

Мнемоніка: subject завжди БІЛЬШИЙ/АКТИВНІШИЙ, object — МЕНШИЙ/ПАСИВНІШИЙ.
Виняток: reports_to, is_subsidiary_of, is_member_of — тут subject МЕНШИЙ (бо він підпорядковується).

## Невизначеність:
- 0.9+ : факт прямо заявлений
- 0.7–0.89 : логічно випливає з контексту
- 0.5–0.69 : припущення, позначити «(імовірно)» в object
- < 0.5 : не витягувати

## Текст для аналізу:
{text}

## Стандартні predicate (snake_case, англійською — це технічні ідентифікатори):
# Організаційні:
controls, reports_to, owns, finances, is_member_of, leads,
appointed_to, related_to, founded, is_beneficiary_of,
funded_by, has_offshore_in, contracted_with, is_subsidiary_of

# Технологічні:
uses, develops, supplies, is_type_of, is_class_of,
used_for, deployed_in, evolved_from, replaces,
built_on, has_countermeasure, vulnerable_to,
has_range, has_precision, manufactured_in

# Концептуальні:
is_doctrine, is_methodology, is_concept, is_technique,
consists_of, implemented_via, contradicts, complements,
adopted_in, emerged_as_response_to, is_variant_of

# Геополітичні:
allied_with, opposes, influences, depends_on,
under_sanctions, supplies_weapons_to, funds_regime

## Нормалізація:
- subject та object — власні назви українською в офіційній формі
- predicate — завжди snake_case англійською
- Скорочення допустимі якщо загальновідомі (НАТО, ФСБ, ЗСУ)
- Якщо суб'єкт невідомий — не вигадуй, пропусти факт

Ліміт: до 25 фактів. Пріоритет при переповненні:
1. Структурні зв'язки між організаціями
2. Технологічні факти зі стратегічним значенням
3. Концептуальні зв'язки
4. Персональні деталі — лише якщо розкривають структуру

КРИТИЧНО ВАЖЛИВО: імена полів JSON — строго англійською.
Порожній масив [] — коректна відповідь якщо нема структурних фактів.

[{{"subject": "назва сутності", "predicate": "тип_звязку_snake_case", "object": "значення або повʼязана сутність", "confidence": 0.9, "category": "organization|person|technology|finance|geopolitics|scheme|event", "time_context": null}}]"""


ENTITY_EXTRACTION_PROMPT = """From these facts, identify distinct entities and their types.

Facts:
{facts_json}

Return JSON array of entities:
[
  {{"name": "...", "entity_type": "person|project|tool|technology|organization|location|concept|other", "aliases": ["..."]}}
]

Rules:
- Normalize names (proper capitalization)
- Include aliases if the same entity is referred to differently
- entity_type should be one of: person, project, tool, technology, organization, location, concept, other"""


def _parse_json_response(text: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code block
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "facts" in result:
            return result["facts"]
        return [result] if isinstance(result, dict) else []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse extraction JSON: {text[:200]}")
        return []


class FactExtractor:
    """Extracts facts from conversation text using LLM."""

    def __init__(self, provider: Any, model: str = "anthropic/claude-haiku-4-5-20251001"):
        self.provider = provider
        self.model = model

    async def extract_facts(
        self,
        text: str,
        source: Source | None = None,
        extraction_type: ExtractionType = ExtractionType.AUTO,
    ) -> list[Fact]:
        """Extract facts from text using LLM.

        Args:
            text: Conversation text to extract from.
            source: Source metadata for provenance.
            extraction_type: How the extraction was triggered.

        Returns:
            List of extracted Fact objects (not yet validated/stored).
        """
        if not text or len(text.strip()) < 20:
            return []

        try:
            response = await self.provider.chat(
                messages=[{
                    "role": "user",
                    "content": EXTRACTION_PROMPT.format(text=text[:4000]),
                }],
                model=self.model,
                max_tokens=4000,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"Extraction LLM call failed: {e}")
            return []

        raw_facts = _parse_json_response(response.content or "[]")
        facts = []
        for raw in raw_facts:
            if not all(k in raw for k in ("subject", "predicate", "object")):
                continue
            fact = Fact(
                subject=raw["subject"].strip(),
                predicate=raw["predicate"].strip(),
                object=raw["object"].strip(),
                confidence=float(raw.get("confidence", 0.7)),
                original_confidence=float(raw.get("confidence", 0.7)),
                extraction_type=extraction_type,
                source_session=source.session_key if source else None,
                category=raw.get("category", "other"),
                time_context=raw.get("time_context"),
            )
            # Validate category
            from ragnarbot.agent.memory.models import FactCategory
            try:
                fact.category = FactCategory(fact.category)
            except ValueError:
                fact.category = FactCategory.OTHER
            facts.append(fact)

        logger.debug(f"Extracted {len(facts)} facts from {len(text)} chars")
        return facts

    async def extract_entities(self, facts: list[Fact]) -> list[dict]:
        """Extract entities from a list of facts."""
        if not facts:
            return []

        facts_json = json.dumps([
            {"subject": f.subject, "predicate": f.predicate, "object": f.object}
            for f in facts
        ], indent=2)

        try:
            response = await self.provider.chat(
                messages=[{
                    "role": "user",
                    "content": ENTITY_EXTRACTION_PROMPT.format(facts_json=facts_json),
                }],
                model=self.model,
                max_tokens=500,
                temperature=0.0,
            )
        except Exception as e:
            logger.error(f"Entity extraction LLM call failed: {e}")
            return []

        return _parse_json_response(response.content or "[]")
