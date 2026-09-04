"""
Teacher Schedule Generator
A Streamlit app that uses Google's Gemini Flash model to auto-generate
a weekly teacher timetable from user-supplied inputs (timings, teachers,
qualifications, load, and constraints).
"""

import io
import json
import re
from dataclasses import dataclass, field, asdict

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError


# --------------------------------------------------------------------------
# Page configuration & styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Teacher Schedule Generator",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1100px;}
    h1 {font-weight: 700; letter-spacing: -0.02em;}
    h2, h3 {font-weight: 600;}
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.4rem;
    }
    .primary-generate button {
        background-color: #2563eb;
        color: white;
        border: none;
    }
    div[data-testid="stMetricValue"] {font-size: 1.4rem;}
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .app-subtitle {color: #64748b; margin-top: -0.6rem; margin-bottom: 1.4rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px;}
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px 8px 0 0;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

DEFAULT_MODEL = "gemini-flash-latest"  # Google-maintained alias -> current Gemini Flash model


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------
@dataclass
class Teacher:
    name: str
    qualifications: str
    subjects: str  # comma separated
    classes_per_week: int
    max_periods_per_day: int = 4
    unavailable: str = ""  # free text e.g. "Mon 1st period, Fri afternoon"


def _default_teachers():
    return [
        Teacher("Ayesha Khan", "MSc Mathematics", "Mathematics, Statistics", 15, 4, ""),
        Teacher("Bilal Ahmed", "MSc Physics", "Physics", 12, 4, "Fri afternoon"),
    ]


if "teachers" not in st.session_state:
    st.session_state.teachers = _default_teachers()
if "schedule_result" not in st.session_state:
    st.session_state.schedule_result = None
if "raw_model_text" not in st.session_state:
    st.session_state.raw_model_text = None


# --------------------------------------------------------------------------
# Sidebar — API configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Gemini API settings")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        help="Get a free key from Google AI Studio (aistudio.google.com/apikey). "
             "It is only kept in this browser session and never saved to disk.",
    )
    model_name = st.text_input(
        "Model name",
        value=DEFAULT_MODEL,
        help="Default uses Google's 'latest Flash' alias so it keeps working as "
             "models are updated. You can pin an exact model name instead, "
             "e.g. gemini-2.5-flash.",
    )
    st.caption(
        "Your key is stored only in this session's memory (`st.session_state`), "
        "sent directly to Google's API, and is cleared when you close the tab."
    )
    st.divider()
    st.markdown("### ℹ️ About")
    st.caption(
        "Fill in class timings, add teachers, set constraints, then click "
        "**Generate Schedule**. The AI drafts a clash-free weekly timetable "
        "which you can review and download."
    )


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🗓️ Teacher Schedule Generator")
st.markdown(
    '<p class="app-subtitle">Describe your school day, add your teaching staff, '
    'set any constraints — Gemini drafts a full weekly timetable.</p>',
    unsafe_allow_html=True,
)

tab_setup, tab_teachers, tab_constraints, tab_generate = st.tabs(
    ["1 · Class Timings", "2 · Teachers", "3 · Constraints", "4 · Generate & Download"]
)


# --------------------------------------------------------------------------
# Tab 1 — Class timings / structure
# --------------------------------------------------------------------------
with tab_setup:
    st.subheader("School week & period structure")
    col1, col2 = st.columns(2)
    with col1:
        working_days = st.multiselect(
            "Working days",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
            default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        )
        periods_per_day = st.number_input(
            "Periods per day", min_value=1, max_value=12, value=6, step=1
        )
    with col2:
        period_length = st.number_input(
            "Period length (minutes)", min_value=20, max_value=120, value=45, step=5
        )
        start_time = st.time_input("First period starts at")

    st.markdown("##### Break periods (optional)")
    break_periods_raw = st.text_input(
        "Which period numbers are breaks/lunch? (comma separated, e.g. 3)",
        value="",
        help="These period slots will be excluded from teaching assignments.",
    )

    st.markdown("##### Classes / Sections")
    sections_raw = st.text_area(
        "List class sections that need a timetable (one per line or comma separated)",
        value="Grade 9-A, Grade 9-B, Grade 10-A",
        height=90,
    )

    st.markdown("##### Subjects offered")
    subjects_raw = st.text_area(
        "List all subjects taught in the school",
        value="Mathematics, Physics, Chemistry, Biology, English, Urdu, Computer Science",
        height=80,
    )


# --------------------------------------------------------------------------
# Tab 2 — Teachers
# --------------------------------------------------------------------------
with tab_teachers:
    st.subheader("Teaching staff")
    st.caption("Add every teacher, their qualification, subjects they can teach, "
               "and how many classes/periods they need per week.")

    for idx, t in enumerate(st.session_state.teachers):
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 2])
            t.name = c1.text_input("Name", value=t.name, key=f"name_{idx}")
            t.qualifications = c2.text_input(
                "Qualification", value=t.qualifications, key=f"qual_{idx}"
            )
            t.subjects = c3.text_input(
                "Subject(s) — comma separated", value=t.subjects, key=f"subj_{idx}"
            )
            c4, c5, c6 = st.columns([2, 2, 3])
            t.classes_per_week = c4.number_input(
                "Classes / week", min_value=1, max_value=60,
                value=t.classes_per_week, key=f"cpw_{idx}"
            )
            t.max_periods_per_day = c5.number_input(
                "Max periods / day", min_value=1, max_value=12,
                value=t.max_periods_per_day, key=f"mpd_{idx}"
            )
            t.unavailable = c6.text_input(
                "Unavailable slots (optional)", value=t.unavailable,
                key=f"unavail_{idx}", placeholder="e.g. Mon 1st period, Fri afternoon"
            )
            if st.button("🗑️ Remove teacher", key=f"remove_{idx}"):
                st.session_state.teachers.pop(idx)
                st.rerun()

    if st.button("➕ Add teacher"):
        st.session_state.teachers.append(
            Teacher("", "", "", 10, 4, "")
        )
        st.rerun()

    st.info(f"Total teachers added: **{len(st.session_state.teachers)}**")


# --------------------------------------------------------------------------
# Tab 3 — Constraints & preferences
# --------------------------------------------------------------------------
with tab_constraints:
    st.subheader("Preferences & hard constraints")
    st.caption(
        "Describe anything the schedule must respect — specific teacher-to-subject "
        "assignments, avoiding back-to-back periods for a teacher, keeping "
        "double-periods for labs, etc. Plain English is fine."
    )
    constraints_text = st.text_area(
        "Constraints (one per line)",
        value=(
            "Assign Bilal Ahmed to all Physics classes for Grade 10-A.\n"
            "No teacher should have more than 2 consecutive periods without a gap.\n"
            "Computer Science classes should be scheduled in the first half of the day."
        ),
        height=160,
    )
    prioritize_no_gaps = st.checkbox(
        "Prefer minimizing free/idle periods for teachers", value=True
    )
    balance_load = st.checkbox(
        "Balance daily teaching load evenly across the week", value=True
    )


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------
def parse_list(raw: str):
    parts = re.split(r"[,\n]", raw)
    return [p.strip() for p in parts if p.strip()]


def build_prompt() -> str:
    days = working_days
    sections = parse_list(sections_raw)
    subjects = parse_list(subjects_raw)
    breaks = parse_list(break_periods_raw)

    teacher_lines = []
    for t in st.session_state.teachers:
        if not t.name.strip():
            continue
        teacher_lines.append(
            f"- Name: {t.name} | Qualification: {t.qualifications} | "
            f"Subjects: {t.subjects} | Classes needed per week: {t.classes_per_week} | "
            f"Max periods/day: {t.max_periods_per_day} | "
            f"Unavailable: {t.unavailable or 'none'}"
        )

    constraint_lines = [c.strip() for c in constraints_text.splitlines() if c.strip()]

    prompt = f"""
You are an expert school timetable scheduler. Build a clash-free weekly class
schedule that satisfies as many constraints as possible. If a hard constraint
cannot be fully satisfied (e.g. not enough periods for a teacher's required
load), do your best, and note the conflict in the "warnings" field instead of
silently dropping requirements.

SCHOOL STRUCTURE
- Working days: {", ".join(days) if days else "Monday-Friday"}
- Periods per day: {periods_per_day}
- Period length: {period_length} minutes, first period starts at {start_time}
- Break/lunch period numbers (do not assign classes here): {", ".join(breaks) if breaks else "none"}
- Class sections requiring a timetable: {", ".join(sections)}
- Subjects offered: {", ".join(subjects)}

TEACHERS
{chr(10).join(teacher_lines) if teacher_lines else "- (no teachers provided)"}

CONSTRAINTS / PREFERENCES
{chr(10).join(f"- {c}" for c in constraint_lines) if constraint_lines else "- none specified"}
- {"Minimize idle/free periods for teachers between classes." if prioritize_no_gaps else ""}
- {"Balance each teacher's daily load evenly across the week." if balance_load else ""}

RULES
1. A teacher cannot teach two different sections in the same day+period.
2. A section cannot have two different subjects in the same day+period.
3. Respect each teacher's "Unavailable" slots and "Max periods/day".
4. Only assign a teacher to a subject listed in their Subjects.
5. Try to satisfy every teacher's "Classes needed per week" as closely as possible.

OUTPUT FORMAT
Return ONLY valid JSON (no markdown fences, no commentary) matching exactly
this schema:

{{
  "schedule": [
    {{
      "day": "Monday",
      "period": 1,
      "section": "Grade 9-A",
      "subject": "Mathematics",
      "teacher": "Ayesha Khan"
    }}
  ],
  "warnings": [
    "Short free-text notes about any constraint that could not be fully met."
  ]
}}

Include one entry in "schedule" for every (day, non-break period, section)
combination. If a slot truly cannot be filled, set "subject" and "teacher" to
"Free Period" for that entry instead of omitting it.
""".strip()
    return prompt


# --------------------------------------------------------------------------
# Gemini call
# --------------------------------------------------------------------------
def call_gemini(prompt: str, key: str, model: str) -> dict:
    client = genai.Client(api_key=key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json",
        ),
    )
    text = response.text or ""
    st.session_state.raw_model_text = text
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip())
    cleaned = re.sub(r"```$", "", cleaned.strip())
    return json.loads(cleaned)


def validate_inputs():
    problems = []
    if not api_key:
        problems.append("Enter your Gemini API key in the sidebar.")
    if not working_days:
        problems.append("Select at least one working day.")
    if not parse_list(sections_raw):
        problems.append("Add at least one class section.")
    if not any(t.name.strip() for t in st.session_state.teachers):
        problems.append("Add at least one teacher with a name.")
    return problems


def schedule_to_dataframe(schedule_json: dict) -> pd.DataFrame:
    rows = schedule_json.get("schedule", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    expected_cols = ["day", "period", "section", "subject", "teacher"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = ""
    return df[expected_cols].sort_values(["section", "day", "period"]).reset_index(drop=True)


def pivot_for_section(df: pd.DataFrame, section: str) -> pd.DataFrame:
    sub = df[df["section"] == section]
    pivot = sub.pivot_table(
        index="period", columns="day",
        values="subject", aggfunc="first"
    )
    teacher_pivot = sub.pivot_table(
        index="period", columns="day",
        values="teacher", aggfunc="first"
    )
    combined = pivot.copy()
    for col in pivot.columns:
        combined[col] = pivot[col].fillna("") + combined[col].apply(
            lambda x: ""
        )
    # merge subject + teacher into one cell for display
    display = pivot.astype(str)
    for col in pivot.columns:
        display[col] = [
            f"{s}\n({tt})" if pd.notna(s) and s else ""
            for s, tt in zip(pivot[col], teacher_pivot[col])
        ]
    # reorder columns by working day order
    ordered_cols = [d for d in working_days if d in display.columns]
    if ordered_cols:
        display = display[ordered_cols]
    return display


def to_excel_bytes(df: pd.DataFrame, per_section: dict) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Full Schedule")
        for section, pivot_df in per_section.items():
            sheet_name = re.sub(r"[\[\]\:\*\?/\\]", "-", section)[:31] or "Section"
            pivot_df.to_excel(writer, sheet_name=sheet_name)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Tab 4 — Generate & Download
# --------------------------------------------------------------------------
with tab_generate:
    st.subheader("Generate the schedule")

    colA, colB, colC = st.columns(3)
    colA.metric("Working days", len(working_days))
    colB.metric("Teachers", sum(1 for t in st.session_state.teachers if t.name.strip()))
    colC.metric("Class sections", len(parse_list(sections_raw)))

    problems = validate_inputs()
    if problems:
        for p in problems:
            st.warning(p)

    generate_clicked = st.button(
        "🚀 Generate Schedule", type="primary", disabled=bool(problems)
    )

    if generate_clicked:
        prompt = build_prompt()
        with st.spinner("Gemini is building your timetable..."):
            try:
                result = call_gemini(prompt, api_key, model_name)
                st.session_state.schedule_result = result
                st.success("Schedule generated successfully.")
            except json.JSONDecodeError:
                st.error(
                    "The model did not return valid JSON. Try clicking Generate "
                    "again, or simplify your constraints."
                )
                with st.expander("Show raw model output"):
                    st.code(st.session_state.raw_model_text or "")
            except (ClientError, ServerError) as e:
                st.error(f"Gemini API error: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Unexpected error: {e}")

    result = st.session_state.schedule_result
    if result:
        df = schedule_to_dataframe(result)

        warnings = result.get("warnings") or []
        if warnings:
            with st.expander(f"⚠️ {len(warnings)} scheduling note(s)", expanded=False):
                for w in warnings:
                    st.write(f"- {w}")

        if df.empty:
            st.warning("The model returned no schedule rows. Try generating again.")
        else:
            sections = [s for s in parse_list(sections_raw) if s in df["section"].unique()]
            per_section_pivots = {s: pivot_for_section(df, s) for s in sections}

            st.markdown("##### Timetable by section")
            section_tabs = st.tabs(sections) if sections else []
            for tab, section in zip(section_tabs, sections):
                with tab:
                    st.dataframe(
                        per_section_pivots[section],
                        use_container_width=True,
                    )

            st.markdown("##### Full schedule (flat table)")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("##### Download")
            dcol1, dcol2 = st.columns(2)
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            dcol1.download_button(
                "⬇️ Download CSV",
                data=csv_bytes,
                file_name="teacher_schedule.csv",
                mime="text/csv",
                use_container_width=True,
            )
            excel_bytes = to_excel_bytes(df, per_section_pivots)
            dcol2.download_button(
                "⬇️ Download Excel (per-section sheets)",
                data=excel_bytes,
                file_name="teacher_schedule.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.caption("No schedule generated yet. Fill in the tabs above and click Generate.")
