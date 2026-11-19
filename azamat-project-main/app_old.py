import re
import statistics
import json
import urllib.error
import urllib.parse
import urllib.request
import pandas as pd
import streamlit as st

from src.features import extract_features
from src.normalize import calculate_readability_score
from src.parsers import ParserManager

# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------
# Если сгенерируете в Higgsfield картинку для шапки, вставьте https-ссылку сюда.
# Пустая строка = шапка рисуется только SVG-схемой разбора предложения.
HERO_IMAGE_URL = ""

PAPER = "#EEF2F6"
INK = "#1B2440"
TEAL = "#0F8B8D"
AMBER = "#D98E04"
ROSE = "#B8433A"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,700&display=swap&subset=cyrillic,cyrillic-ext');

:root {{
    --paper: {PAPER};
    --ink: {INK};
    --teal: {TEAL};
    --line: rgba(27, 36, 64, 0.14);
}}

.stApp {{
    background-color: var(--paper);
    background-image:
        linear-gradient(rgba(27, 36, 64, 0.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(27, 36, 64, 0.045) 1px, transparent 1px);
    background-size: 28px 28px;
    color: var(--ink);
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}

header[data-testid="stHeader"] {{ background: transparent; }}

.block-container {{ max-width: 1080px; padding-top: 2.2rem; }}

h1, h2, h3 {{
    font-family: 'Source Serif 4', Georgia, 'Times New Roman', serif;
    color: var(--ink);
    letter-spacing: -0.01em;
}}
h1 {{ font-weight: 700; }}
h2, h3 {{ font-weight: 600; }}

/* Шапка */
.hero {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 1.6rem 1.8rem 1.2rem;
    margin-bottom: 1.6rem;
    background-size: cover;
    background-position: center;
}}
.hero h1 {{ margin: 0 0 0.3rem 0; padding: 0; font-size: 2.3rem; line-height: 1.15; }}
.hero p {{ margin: 0 0 0.8rem 0; max-width: 46ch; line-height: 1.55; opacity: 0.8; }}
.hero svg {{ width: 100%; height: auto; display: block; color: var(--teal); }}
.hero svg text {{ font-family: 'Source Serif 4', Georgia, serif; font-size: 20px; fill: var(--ink); }}

/* Поля ввода */
textarea, input {{
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border-radius: 8px !important;
}}
div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    border-radius: 8px !important;
    border-color: var(--line) !important;
    background: #FFFFFF !important;
}}
textarea:focus-visible, input:focus-visible {{ outline: 2px solid var(--teal) !important; }}

/* Кнопки */
.stButton > button {{
    border-radius: 8px;
    border: 1px solid var(--ink);
    background: transparent;
    color: var(--ink);
    font-weight: 600;
    padding: 0.55rem 1.3rem;
    transition: background 0.15s, color 0.15s;
}}
.stButton > button:hover {{ background: var(--ink); color: #FFFFFF; border-color: var(--ink); }}
.stButton > button:focus-visible {{ outline: 2px solid var(--teal); outline-offset: 2px; }}
.stButton > button[kind="primary"] {{
    background: var(--teal);
    border-color: var(--teal);
    color: #FFFFFF;
}}
.stButton > button[kind="primary"]:hover {{ background: #0B6E70; border-color: #0B6E70; }}

/* Метрики и таблицы */
[data-testid="stMetric"] {{
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}}
[data-testid="stMetricValue"] {{ font-family: 'Source Serif 4', Georgia, serif; }}
[data-testid="stTable"] table, [data-testid="stDataFrame"] {{
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #FFFFFF;
}}

/* Боковая панель */
[data-testid="stSidebar"] {{
    background: #FFFFFF;
    border-right: 1px solid var(--line);
}}

/* Карточка балла */
.score-card {{
    display: flex;
    align-items: center;
    gap: 1.4rem;
    background: #FFFFFF;
    border: 1px solid var(--line);
    border-left: 6px solid var(--tone);
    border-radius: 8px;
    padding: 1.1rem 1.4rem;
}}
.score-num {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 3.6rem;
    font-weight: 700;
    line-height: 1;
    color: var(--tone);
    min-width: 5.5rem;
}}
.score-meta {{ flex: 1; }}
.score-label {{ font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem; }}
.score-bar {{ height: 8px; background: rgba(27, 36, 64, 0.1); border-radius: 4px; overflow: hidden; }}
.score-bar span {{ display: block; height: 100%; background: var(--tone); border-radius: 4px; }}
.score-hint {{ font-size: 0.82rem; opacity: 0.65; margin-top: 0.45rem; }}

hr {{ border-color: var(--line); }}

label, [data-testid="stWidgetLabel"] p, .stMarkdown p, .stMarkdown li, .stCaption, small {{
    color: var(--ink) !important;
}}
[data-testid="stSidebar"] * {{ color: var(--ink); }}
[data-testid="stAlert"] * {{ color: inherit; }}

@media (max-width: 640px) {{
    .hero h1 {{ font-size: 1.7rem; }}
    .score-card {{ flex-direction: column; align-items: flex-start; }}
}}
</style>
"""

# Схема зависимостей: корень «читается», дуги идут к зависимым словам.
HERO_SVG = """
<svg viewBox="0 0 640 190" role="img" aria-label="Схема синтаксических связей в предложении">
<defs>
<marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto">
<path d="M0,0 L10,5 L0,10 z" fill="currentColor"/>
</marker>
</defs>
<g fill="none" stroke="currentColor" stroke-width="1.8" marker-end="url(#arr)">
<path d="M140,155 Q90,85 40,150"/>
<path d="M140,155 Q195,79 250,150"/>
<path d="M550,155 Q345,-5 140,150" opacity="0.55"/>
<path d="M550,155 Q445,45 340,150" opacity="0.55"/>
<path d="M550,155 Q495,79 440,150" opacity="0.55"/>
</g>
<g>
<text x="40" y="178" text-anchor="middle">Текст</text>
<text x="140" y="178" text-anchor="middle">читается</text>
<text x="250" y="178" text-anchor="middle">легко,</text>
<text x="340" y="178" text-anchor="middle">когда</text>
<text x="440" y="178" text-anchor="middle">структура</text>
<text x="550" y="178" text-anchor="middle">ясна</text>
</g>
</svg>
"""


def render_hero():
    bg = ""
    if HERO_IMAGE_URL:
        bg = (
            f' style="background-image: linear-gradient(rgba(255,255,255,0.86), '
            f'rgba(255,255,255,0.86)), url(\'{HERO_IMAGE_URL}\');"'
        )
    html = (
        f'<div class="hero"{bg}>'
        "<h1>Оценка читаемости и упрощение текста</h1>"
        "<p>Вставьте учебный текст на английском, русском или казахском. "
        "Сервис измерит сложность и при необходимости перепишет его проще.</p>"
        f"{HERO_SVG}"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_score_card(score: float):
    value = max(0.0, min(100.0, float(score)))
    if value >= 70:
        label, tone = "Читается легко", TEAL
    elif value >= 40:
        label, tone = "Средняя сложность", AMBER
    else:
        label, tone = "Сложный текст", ROSE
    html = (
        f'<div class="score-card" style="--tone:{tone}">'
        f'<div class="score-num">{value:.0f}</div>'
        '<div class="score-meta">'
        f'<div class="score-label">{label}</div>'
        f'<div class="score-bar"><span style="width:{value:.0f}%"></span></div>'
        '<div class="score-hint">100 — очень легко, 0 — очень сложно</div>'
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Приложение
# ---------------------------------------------------------------------------
@st.cache_resource
def get_parser_manager():
    return ParserManager()


st.set_page_config(page_title="Анализ и упрощение текста", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Боковая панель для настроек ИИ (Gemini)
with st.sidebar:
    st.header("Настройки ИИ")
    st.caption("Google Gemini")
    api_key = st.text_input("Gemini API Key (AIza...):", type="password")
    st.markdown("[Получить ключ Gemini](https://aistudio.google.com/app/apikey)")
    st.info("Ключ нужен только для перефразирования. Оценка читаемости работает локально.")

render_hero()

col_lang, _ = st.columns([1, 2])
with col_lang:
    lang = st.selectbox(
        "Язык текста",
        options=["en", "ru", "kk"],
        format_func=lambda x: {"en": "English", "ru": "Русский", "kk": "Қазақша"}[x],
    )

text_input = st.text_area("Текст для анализа", height=200)

# Блок 1: Расчет сложности текста (локальный анализ)
if st.button("Рассчитать сложность", type="primary"):
    if text_input.strip():
        with st.spinner("Анализ синтаксиса и расчет метрик..."):
            pm = get_parser_manager()
            doc = pm.parse(text_input, lang)

            sentence_features = extract_features(doc, lang)

            if sentence_features:
                total_tokens = sum(f.get("token_count", 0) for f in sentence_features)
                avg_tree_depth = statistics.mean(f.get("tree_depth", 0) for f in sentence_features)
                avg_char_density = statistics.mean(f.get("char_density", 0) for f in sentence_features)
                avg_syllables = statistics.mean(f.get("avg_syllables", 0) for f in sentence_features)
                avg_pos_ratio = statistics.mean(f.get("pos_ratio", 0) for f in sentence_features)

                aggregated_features = {
                    "token_count": total_tokens,
                    "tree_depth": avg_tree_depth,
                    "char_density": avg_char_density,
                    "avg_syllables": avg_syllables,
                    "pos_ratio": avg_pos_ratio,
                }

                score = calculate_readability_score(aggregated_features, lang)

                st.subheader("Результат анализа")
                c1, c2 = st.columns([2, 1])
                with c1:
                    render_score_card(score)
                with c2:
                    st.metric("Всего токенов/слов", total_tokens)

                st.subheader("Сводные метрики текста")
                df_summary = pd.DataFrame({
                    "Метрика": [
                        "Глубина синтаксического дерева (средняя)",
                        "Длина слова в символах (средняя)",
                        "Количество слогов в слове (среднее)",
                        "Доля знаменательных слов (POS ratio)",
                    ],
                    "Значение": [
                        f"{avg_tree_depth:.2f}",
                        f"{avg_char_density:.2f}",
                        f"{avg_syllables:.2f}",
                        f"{avg_pos_ratio:.2f}",
                    ],
                })
                st.table(df_summary)

                st.subheader("Детализация по предложениям")
                st.dataframe(pd.DataFrame(sentence_features), use_container_width=True)
            else:
                st.warning("Не удалось разобрать текст на предложения.")

st.divider()

# Блок 2: Интеграция LLM для перефразирования (через Gemini API)
MAX_CHARS_PER_CHUNK = 1500   # длинный текст режется на куски, чтобы не упереться в лимит токенов
MAX_OUTPUT_TOKENS = 2000

GEMINI_MODEL = "gemini-2.5-flash"

LANG_NAMES = {"en": "английский", "ru": "русский", "kk": "казахский"}

# Ошибки, после которых имеет смысл попробовать другую модель
RETRYABLE_MARKERS = ["413", "429", "rate_limit", "too large", "model_not_found",
                     "does not exist", "decommissioned", "not found"]


def split_into_chunks(paragraph, limit=MAX_CHARS_PER_CHUNK):
    """Режет абзац на куски не длиннее limit символов по границам предложений."""
    sentences = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
    chunks, current = [], ""
    for s in sentences:
        while len(s) > limit:  # очень длинное предложение режем по пробелу
            cut = s.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if current:
                chunks.append(current)
                current = ""
            chunks.append(s[:cut])
            s = s[cut:].lstrip()
        if current and len(current) + len(s) + 1 > limit:
            chunks.append(current)
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


def simplify_chunk(api_key, chunk, lang):
    """Упрощает один кусок текста через Gemini."""
    system_prompt = (
        "Ты — эксперт в образовании. Адаптируй и упрости учебный текст, чтобы снизить "
        "языковой барьер для ученика. Сделай сложные синтаксические конструкции проще, "
        "разбей слишком длинные предложения, но обязательно сохрани академический смысл, "
        "числа и ссылки вида [46]. Пиши только готовый перефразированный текст, без пояснений. "
        f"Язык ответа: {LANG_NAMES.get(lang, lang)}."
    )
    model = urllib.parse.quote(GEMINI_MODEL, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": chunk}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": MAX_OUTPUT_TOKENS},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8"))
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini не вернул текст. Запрос мог быть заблокирован.")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()


st.subheader("Автоматическое упрощение текста")
if st.button("Перефразировать"):
    if not api_key:
        st.error("Введите API ключ Gemini в боковой панели слева (начинается с AIza).")
    elif not text_input.strip():
        st.warning("Введите текст для упрощения.")
    else:
        try:
            paragraphs = [p for p in re.split(r"\n\s*\n", text_input) if p.strip()]
            plan = [split_into_chunks(p) for p in paragraphs]
            total = sum(len(c) for c in plan)

            progress = st.progress(0.0, text="Упрощаем текст...")
            done, result_paragraphs = 0, []
            for chunks in plan:
                parts = []
                for chunk in chunks:
                    text = simplify_chunk(api_key.strip(), chunk, lang)
                    parts.append(text)
                    done += 1
                    progress.progress(done / total, text=f"Готово {done} из {total}")
                result_paragraphs.append(" ".join(parts))
            progress.empty()

            st.success(f"Текст упрощен. Модель: {GEMINI_MODEL}")
            with st.container(border=True):
                st.markdown("\n\n".join(result_paragraphs))

        except Exception as e:
            st.error(f"Сбой сервера Gemini. Детали ошибки: {e}")
