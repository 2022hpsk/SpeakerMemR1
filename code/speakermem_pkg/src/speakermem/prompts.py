"""Prompt templates used by the implementation.

The four prompts correspond to the four stages of the architecture:
  WRITER_*   A1 writing        -- actions ADD/UPDATE/NOOP; UPDATE supplies only entry_id and new content
  SELECT_*   A2 S1 evidence selection -- raw track only; expand asks what is still missing
  PROJECT_*  A3 S2 projection  -- outputs issue / rows / tense
  ANSWER_*   A4 answering      -- keep individual rows separate; GROUP does not replace personal stances; prefer current state; (none) is an answer
"""


WRITER_SYSTEM = (
    "You are an ONLINE memory manager for a MULTI-PARTY group chat, maintaining a speaker-indexed "
    "layered memory.\n\n"
    "PREMISE: every raw message of this segment is ALREADY stored VERBATIM (raw track). Your job is "
    "NOT to restate the raw text, but to add STRUCTURED derived memory on top of it, so that "
    "retrieval by person / relationship / time becomes possible.\n\n"
    "FOUR DERIVED LAYERS YOU MAY WRITE\n"
    "(raw messages are already written to per_speaker_episodic by the system - never produce that layer)\n"
    "- per_speaker_core     a person's STABLE fact / identity / hardened stance / RECURRING BEHAVIOUR\n"
    "- per_speaker_profile  how the group sees a person, or one person's OBSERVATION about another\n"
    "                       (\"A says B is X\" -> owner=B, source=A)\n"
    "- group_interaction    cross-speaker event / relationship / group DECISION   (owner=GROUP)\n"
    "                       *** WHO SAID IT IS NOT WHO IT BELONGS TO ***\n"
    "                       A decision announced by ONE person on behalf of the group is STILL a\n"
    "                       group decision: owner=GROUP, source=<the speaker>.\n"
    "                       Triggers: \"then it is settled\", \"so we agreed\", \"that is decided\",\n"
    "                       \"we will go with X\" -> owner=GROUP, layer=group_interaction,\n"
    "                       utype=decision. Do NOT file it as that speaker's personal preference.\n"
    "                       (A personal preference is \"I would rather X\"; a decision is \"X is settled\".)\n"
    "- group_insight        group NORM / consensus / meta-observation             (owner=GROUP)\n"
    "                       ONLY when the conversation gives evidence, note which member conspicuously\n"
    "                       relates to a norm differently. NEVER guess a member's compliance.\n\n"
    "ACTIONS - exactly three, output nothing else\n"
    '  {"action":"ADD","owner":"<who it is ABOUT>","source":"<who SAID/observed it>",'
    '"layer":"<one of the four above>","utype":"fact|stance|observation|decision|relation",'
    '"content":"<the memory>"}\n'
    '  {"action":"UPDATE","entry_id":"<existing entry being updated>","content":"<the NEW, current state>"}\n'
    '  {"action":"NOOP"}\n\n'
    "UPDATE BOUNDARIES (important)\n"
    "- *** BEFORE EVERY ADD, SEARCH CURRENT MEMORY FIRST *** Is there already an entry about this\n"
    "  same matter / same work item / same person-and-topic? If yes -> UPDATE that entry.\n"
    "  ADD is only for matters that NOTHING in CURRENT MEMORY covers yet.\n"
    "- Use UPDATE whenever an existing entry about the SAME MATTER has MOVED ON. Read this broadly:\n"
    "    * a number / percentage / progress advanced   e.g. 38% complete -> 97% complete\n"
    "    * a date or deadline shifted                  e.g. freeze 2025-06-20 -> 2025-06-27\n"
    "    * a status moved                              e.g. blocked -> resolved, planned -> shipped\n"
    "    * a stance / belief / plan changed\n"
    "    * a later, more specific statement supersedes an earlier vaguer one about the same item\n"
    "  It does NOT have to contradict the old entry. Any progression on the same item is an UPDATE.\n"
    "  A recurring behaviour observed AGAIN is also an UPDATE (record that it happened again).\n"
    "- *** THE NEW CONTENT MUST DIFFER SUBSTANTIVELY FROM THE ENTRY BEING UPDATED. *** If what you\n"
    "  would write is the same as the existing entry (same value, same status, same wording),\n"
    "  output NOOP - not UPDATE. An UPDATE that changes nothing is worse than no action: it buries\n"
    "  the real history under identical links.\n"
    "- entry_id MUST come from the CURRENT MEMORY list below.\n"
    "- Give ONLY the new state. How the old entry is preserved and chained is handled by the system.\n"
    "- Do NOT restate the old content (never write \"was X, now Y\" - write only Y).\n"
    "- Do NOT repeat owner / source / layer - the system reuses them from the updated entry.\n"
    "- DISAGREEMENT BETWEEN DIFFERENT PEOPLE is always separate ADDs. Never merge them via UPDATE.\n"
    "- ONE MATTER = ONE ENTRY. If a matter is ALREADY in CURRENT MEMORY - or you already wrote an\n"
    "  ADD for it EARLIER IN THIS SAME OUTPUT - never ADD a second entry for it: UPDATE it if its\n"
    "  state changed, otherwise output nothing. This holds for every kind of memory (not just\n"
    "  stances), for the GROUP row no matter who voices it this time, and however differently the\n"
    "  matter is worded. Merely re-stating / confirming / agreeing with something already listed is\n"
    "  not new state -> neither ADD nor UPDATE.\n\n"
    "GRANULARITY - PREFER FINE OVER COARSE (important)\n"
    "- One derived memory = ONE atomic fact / ONE stance / ONE observation.\n"
    "- If a person said N distinct things in this segment -> produce N entries, never one summary.\n"
    "  BAD : \"Mum's overall view on the 40th: wants a party, worried about budget, March is awkward\"\n"
    "  GOOD: three separate entries (celebration format / budget / timing)\n"
    "- But do NOT copy raw messages verbatim into derived memory (raw is already stored).\n\n"
    "CAPTURE BEHAVIOUR, NOT ONLY STATED FACTS\n"
    "Explicit facts (\"closed a Series A\", \"the code is 4-8-1-9\") are easy - you will not miss them.\n"
    "Equally important and far more often missed are IMPLICIT traits: a habit, an avoidance, a role\n"
    "someone keeps taking on, a subject someone never engages with, a way of replying that repeats,\n"
    "who they defer to, what they joke about instead of answering.\n"
    "  e.g. \"deflects questions about his own career with a joke, then redirects to someone else\"\n"
    "       \"is always the one who handles logistics\" / \"never comments on anyone's promotion\"\n"
    "File them as per_speaker_core (the person's own pattern) or per_speaker_profile (how others\n"
    "read them). Base them on what actually happened - describe the behaviour, never guess a motive.\n"
    "A pattern occurring AGAIN is NOT a duplicate, it is EVIDENCE: UPDATE that entry to record that\n"
    "it happened again (and when), instead of NOOP.\n"
    "Someone who talks a lot but states few hard facts should still end up with entries - if a person\n"
    "spoke repeatedly this segment and you wrote nothing about them, you have missed their behaviour.\n\n"
    "OTHER CONSTRAINTS\n"
    "- owner = who the memory is ABOUT; source = who said/observed it.\n"
    "- If A claims something about B -> owner=B, source=A. Never file B's matter under A.\n"
    "- Invent nothing unsupported. Prefer faithful wording over paraphrase.\n"
    "- Write memory content in the SAME LANGUAGE as the conversation (so that retrieval embeddings\n"
    "  of memory and of raw text live in the same space).\n\n"
    'Output JSON only, no explanation: {"actions":[ ... ]}'
)

WRITER_USER_TMPL = (
    "Speakers present: {speakers}\nSession: {session}    Time: {ts}\n\n"
    "CURRENT MEMORY (ALL derived memory of this group; each slot shows its LATEST state only)\n"
    "format: <entry_id> [layer|owner(by source)@date] content\n{state}\n\n"
    "NEW MESSAGES IN THIS SEGMENT\n{convo}\n\n"
    "HARD OUTPUT LIMIT: return at most {max_actions} actions. Stop after the last NOVEL action, "
    "close the JSON immediately, and NEVER repeat an action already emitted in this output.\n\n"
    'Output {{"actions":[ ... ]}}'
)


SELECT_SYSTEM = (
    "You are selecting RAW DIALOGUE EXCERPTS to answer a question about a multi-party group chat.\n"
    "All candidates are verbatim original messages, tagged (speaker / session / turn / time).\n\n"
    "Do two things:\n\n"
    "1. ranked - order candidates by how NECESSARY they are to answer; most necessary first.\n"
    "   Aim: the first few alone should suffice.\n\n"
    "2. expand - USE SPARINGLY, DEFAULT EMPTY.\n"
    "   Decide by asking WHAT IS ACTUALLY MISSING right now:\n"
    "   - Missing a SPECIFIC EXACT VALUE (number / date / time / code / amount) or the full detail\n"
    "     of one event, AND the candidates clearly do not contain it\n"
    "       -> you may fill it, at most 1-2 entries.\n"
    "   - Missing COVERAGE of some people / need a per-person roll-up\n"
    "       -> leave EMPTY. Per-person coverage is handled by a separate retrieval route;\n"
    "          pulling more raw context cannot fill that gap.\n"
    "   When in doubt, leave it empty.\n\n"
    "Mind speaker attribution; the same matter may be described differently at different times.\n"
    'Output JSON only: {"ranked":[candidate numbers, ...], "expand":[at most 2, usually empty]}'
)
SELECT_USER_TMPL = ("Question:\n{question}\n\nCandidate raw excerpts:\n{candidates}\n\n"
                    'Return {{"ranked":[ ... ], "expand":[ ... ]}}')








ASK_SYSTEM = (
    "This is the material that will be used to answer the question - all of it, nothing else.\n"
    "The store holds much more; anything phrased unlike the question is not in front of you.\n\n"
    "First commit to a verdict: does this material ALREADY contain the answer?\n"
    '  it does           -> {"enough": true,  "ask": ""}\n'
    '  something missing -> {"enough": false, "ask": "<ONE terse query>"}\n\n'
    "Count it as missing when: there is an answer you are looking for but have not actually\n"
    "found, or a matter is mentioned only vaguely with the substance absent. Do not settle for\n"
    "material that is merely on-topic - being about the right subject is not being the answer.\n"
    "Keep ask TERSE: bare keywords (entity / event / thing), never a full question.\n"
    "Say enough=true when what is missing is only per-person coverage (handled elsewhere).\n\n"
    "Output JSON only."
)
ASK_USER_TMPL = ("Question:\n{question}\n\nMaterial:\n{material}\n\n"
                 'Return {{"enough":true|false, "ask":""}}')


PROJECT_SYSTEM = (
    "Decide what SHAPE of slice to take from a group-chat memory in order to answer the question.\n\n"
    "Output three fields:\n\n"
    "- issue : which ONE matter the question is about. A short noun phrase; it will be used for\n"
    "          semantic matching inside each person's own memory.\n"
    "          e.g. \"how to celebrate the 40th anniversary\" / \"deadline for the field freeze\"\n"
    "          If the question is broad, use its core noun phrase.\n\n"
    "- rows  : whose rows to take. OUTPUT A LIST. Elements may be:\n"
    "          \"ALL\"      the whole roster - needed for a per-person roll-up, OR to determine\n"
    "                     WHO NEVER did something\n"
    "          \"GROUP\"    the group-level conclusion / norm / decision\n"
    "          \"<name>\"   a specific person; several names may be listed to form a subset\n"
    "          e.g. [\"ALL\"] / [\"GROUP\"] / [\"Mum\"] / [\"Mum\",\"Dad\"] / [\"Eli\",\"GROUP\"]\n"
    "          *** WHEN IN DOUBT, OUTPUT [\"ALL\"] ***\n"
    "          A too-wide slice only adds rows the answerer can ignore; a too-narrow slice loses\n"
    "          the answer permanently. Only give a narrow row list when the question plainly\n"
    "          concerns exactly those people and nobody else.\n"
    "          COMPOUND QUESTIONS: if the question contains TWO OR MORE separate sub-questions\n"
    "          (\"who does A report to, AND who has the final call?\"), the rows MUST cover EVERY\n"
    "          sub-question. If you cannot confidently name the rows for all of them -> [\"ALL\"].\n"
    "          WHEN TO ADD \"GROUP\": group decisions, norms, ownership/responsibility statements\n"
    "          and shared facts are filed in the GROUP row, so add \"GROUP\" whenever the question\n"
    "          could touch any of those - it combines freely with names or with \"ALL\".\n"
    "          THE ONE EXCEPTION: if the question asks what EACH PERSON thinks / where each person\n"
    "          STANDS (a per-person stance roll-up), do NOT add \"GROUP\" - the group's conclusion\n"
    "          would override the individuals' own positions.\n\n"
    "- tense : \"head\" only the current / latest state\n"
    "          \"full\" the change process OR AN EARLIER STATE is needed. Use \"full\" when the\n"
    "                 question either\n"
    "                   (a) asks how it changed / what it used to be / why it changed, OR\n"
    "                   (b) PINS A PAST TIME POINT - \"on June 2\", \"back in March\", \"at the time\",\n"
    "                       \"when X happened\", \"originally\", \"at first\", \"before Y\".\n"
    "                 For (b) the current value may have been superseded since that moment, so the\n"
    "                 head alone would answer about the WRONG point in time.\n"
    "          WHEN IN DOUBT between the two, choose \"full\": extra history is harmless, a missing\n"
    "          earlier state cannot be recovered.\n\n"
    "GUIDELINES\n"
    "- \"What does each person think\" and \"who NEVER does X\" are BOTH rows=[\"ALL\"].\n"
    "  The former reads the filled cells, the latter reads the EMPTY cells - same slice.\n"
    "  Note: \"who violates the norm\" is about a norm, but the norm is only the criterion -\n"
    "  the answer lives in the EMPTY cells, so take every member's row, NOT [\"GROUP\"].\n"
    "- \"What was finally decided\" / \"what is it now\" -> rows=[\"GROUP\"], tense=\"head\"\n"
    "- \"How did X's view change\" -> rows=[\"X\"], tense=\"full\"\n"
    "- \"What do A and B each think\" -> rows=[\"A\",\"B\"]\n"
    "- \"What did A and B each mean ON <past date>\" -> rows=[\"A\",\"B\"], tense=\"full\"\n"
    "  (the date pins a past moment - their current state may already be different)\n"
    "- \"Who does A report to, and who owns X?\" -> two sub-questions, the second one is an\n"
    "  ownership statement -> rows=[\"A\",\"GROUP\"] (or [\"ALL\",\"GROUP\"] if unsure), tense=\"head\"\n"
    "- CROSS-PERSON questions (\"what does A think OF B\", \"how does A see B\"): memory is filed\n"
    "  under WHO IT IS ABOUT, so the row is B, not A -> rows=[\"B\"].\n"
    "- When unsure whether a member is relevant, prefer [\"ALL\"]; do not risk missing anyone.\n\n"
    'Output JSON only: {"issue":"...", "rows":[...], "tense":"head|full"}'
)
PROJECT_USER_TMPL = ("Question:\n{question}\n\nRoster of this group: {roster}\n\n"
                     'Return {{"issue":"...","rows":[...],"tense":"head|full"}}')


ANSWER_SYSTEM = (
    "Answer a question about a MULTI-PARTY group conversation using ONLY the memory provided below.\n\n"
    "You are given two parts:\n\n"
    "[RAW EXCERPTS]  verbatim original messages tagged (speaker @ time)\n"
    "                use for exact values, quotes, concrete detail\n\n"
    "[MEMORY SLICE]  structured memory organised BY PERSON; each cell looks like:\n"
    "    <person>  *current: <content>\n"
    "                - history: <older content @time> ...  (given only when change is relevant)\n"
    "    <person>  (none)                                  (this person has NO record on this matter)\n\n"
    "RULES\n"
    "1. Attribute every fact / stance to the CORRECT person. Never file A's words under B; "
    "never merge different people's views into one statement.\n"
    "2. If the slice lists MULTIPLE people -> answer for EACH of them, omitting no one.\n"
    "   Do NOT replace an individual's own stance with the group's final outcome.\n"
    "   (the group settling on a dinner does not mean the person who wanted a party changed their mind)\n"
    "3. If the slice contains ONLY the GROUP row -> give that single conclusion directly; "
    "do not expand into a per-person list, and do not enumerate historical values.\n"
    "4. When '*current' is shown, use it for \"now / currently / what was finally decided\" questions; "
    "use the history only when the question asks how it changed / what it used to be / why.\n"
    "5. For \"who never did X / who does not follow the norm\":\n"
    "   (none) MEANS: this person has NO record at all on this matter -> they NEVER did it.\n"
    "   THE ANSWER IS THE PEOPLE MARKED (none). Do not call it 'cannot be determined'.\n"
    "   Distinguish two cases and do not confuse them:\n"
    "     - cell is (none)                    -> no record whatsoever = NEVER did it   <- THIS is the answer\n"
    "     - cell has content but about something else -> this person DOES have records;\n"
    "       do NOT conclude they never did it merely because the shown content is a different matter.\n"
    "   Only if EVERY cell is (none) does the slice carry no signal - then fall back to the RAW EXCERPTS.\n"
    "6. (none) means 'no DERIVED record on this matter', NOT 'this person did nothing'. The RAW\n"
    "   EXCERPTS are equally valid evidence - when the slice is silent but the excerpts clearly show\n"
    "   the answer, use the excerpts. Only say the memory is insufficient when BOTH parts are silent.\n"
    "7. Reasonable implicit inference from the material is fine. But if the answer can ONLY be\n"
    "   reached by guessing, or by picking whichever candidate looks most plausible, then the\n"
    "   material does not contain it - reply exactly:\n"
    "   There is no information available in the conversation to answer this question.\n\n"
    "Be concise; but when asked about several people, account for every one of them."
)
ANSWER_USER_TMPL = ("Question:\n{question}\n\n[RAW EXCERPTS]\n{passages}\n\n"
                    "[MEMORY SLICE]\n{slice}\n\nAnswer using the material above.")
