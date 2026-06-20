"""
Google Maps Review Summarizer — Backend Python
Scraping dinâmico do Google Maps com Playwright + resumo via GroqCloud.

Uso:
    pip install -r requirements.txt
    playwright install chromium
    crie .env com GROQ_API_KEY=gsk_...
    python app.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from groq import Groq

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b").strip()
MAX_REVIEWS_LIMIT = int(os.environ.get("MAX_REVIEWS_LIMIT", "10000"))  # limite de segurança para o modo "todas as avaliações"
DEFAULT_REVIEWS_LIMIT = 30
AI_SAMPLE_LIMIT = int(os.environ.get("AI_SAMPLE_LIMIT", "300"))  # máximo de reviews enviadas para a IA
AI_HIGHLIGHT_POOL_LIMIT = int(os.environ.get("AI_HIGHLIGHT_POOL_LIMIT", "45"))
AI_REVIEW_BODY_CHARS = int(os.environ.get("AI_REVIEW_BODY_CHARS", "420"))

GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"
GROQ_CHAT_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
    "groq/compound",
    "groq/compound-mini",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
]

# ---------------------------------------------------------------------------
# Utilidades gerais
# ---------------------------------------------------------------------------


def clean_text(value: Any) -> str:
    """Remove espaços repetidos e símbolos privados comuns do Google Maps."""
    if value is None:
        return ""
    text = str(value).replace("\xa0", " ")
    text = re.sub(r"[\ue000-\uf8ff]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_probable_place_name(value: Any) -> bool:
    """Evita usar o texto inteiro do painel do Google Maps como nome."""
    text = clean_text(value)
    if not text:
        return False

    lower = text.lower()
    generic_titles = {"resultados", "results", "result", "google maps", "visão geral", "overview", "avaliações", "reviews"}
    if lower in generic_titles:
        return False

    # Quando o seletor pega o painel inteiro, aparecem muitos termos de navegação/reviews.
    noise_terms = [
        "visão geral", "avaliações", "sobre", "pedir para", "ver cardápios",
        "avaliar", "ordenar", "gostei", "compartilhar", "local guide",
        "avaliações analisadas", "resumo executivo",
    ]
    noise_hits = sum(1 for term in noise_terms if term in lower)

    # Nomes de estabelecimento costumam ser curtos. Um painel inteiro passa fácil de 80 chars.
    if len(text) > 80:
        return False
    if noise_hits >= 2:
        return False
    if re.search(r"\b\d+\s+avalia(?:ç|c)ões\b|\b\d+\s+reviews\b", lower):
        return False
    if len(text.split()) > 10:
        return False

    return len(text) > 2


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        n = int(value)
    except Exception:
        return default
    return max(minimum, min(n, maximum))


def parse_review_limit(value: Any, default: int = DEFAULT_REVIEWS_LIMIT) -> int | None:
    """Converte o valor vindo do frontend. None significa: tentar coletar todas."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"all", "todas", "todos", "tudo", "*"}:
            return None
    if value is None:
        return default
    return clamp_int(value, default, 1, MAX_REVIEWS_LIMIT)


def parse_total_reviews_count(value: Any) -> int | None:
    """Extrai números como '113 avaliações' ou '1.234 reviews'."""
    text = clean_text(value).lower()
    if not text:
        return None
    match = re.search(r"([0-9][0-9\.\, ]*)\s*(?:avalia(?:ç|c)ões|reviews?)", text)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def extract_json_object(raw: str) -> dict:
    """Extrai o primeiro objeto JSON válido mesmo se a IA devolver texto extra."""
    if not raw:
        raise json.JSONDecodeError("Resposta vazia", raw, 0)

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Nenhum objeto JSON válido encontrado", text, 0)


def mask_secret(value: str) -> str:
    value = clean_text(value)
    if not value:
        return ""
    if len(value) <= 12:
        return "***"
    return f"{value[:7]}...{value[-4:]}"


def read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return str(exc)


def groq_json_request(url: str, *, method: str = "GET", payload: dict | None = None, timeout: int = 20) -> tuple[dict, dict]:
    """Chamada HTTP direta à API Groq para validar chave, listar modelos e ler headers de limite."""
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY não definida.")

    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "google-maps-summarizer/1.1",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in response.headers.items()}
            return json.loads(raw or "{}"), headers
    except urllib.error.HTTPError as exc:
        body = read_http_error_body(exc)
        if exc.code == 401:
            raise PermissionError(
                "GROQ_API_KEY inválida ou sem permissão. Confira se o valor secreto da chave foi copiado inteiro, "
                "se não há variável de ambiente antiga sobrepondo o .env e reinicie o servidor."
            ) from exc
        if exc.code == 429:
            headers = {k.lower(): v for k, v in exc.headers.items()}
            try:
                parsed_body = json.loads(body)
            except Exception:
                parsed_body = {"error": body}
            parsed_body["_status_code"] = 429
            return parsed_body, headers
        raise ValueError(f"Erro HTTP {exc.code} na API Groq: {body}") from exc


def _obj_to_dict(value: Any) -> dict:
    """Converte objetos Pydantic/SDK em dict quando possível."""
    if isinstance(value, dict):
        return value
    for attr in ("model_dump", "dict"):
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    return {}


def _extract_model_ids(models_response: Any) -> list[str]:
    data = getattr(models_response, "data", None)
    if data is None and isinstance(models_response, dict):
        data = models_response.get("data", [])
    ids: set[str] = set()
    for item in data or []:
        item_id = getattr(item, "id", None)
        if not item_id and isinstance(item, dict):
            item_id = item.get("id")
        item_id = str(item_id or "").strip()
        if item_id:
            ids.add(item_id)
    return sorted(ids)


def fetch_groq_models() -> list[str]:
    """Lista modelos pelo SDK oficial. Evita o erro Cloudflare 1010 do urllib."""
    try:
        client = build_client()
        response = client.models.list()
        models = _extract_model_ids(response)
        return models or FALLBACK_MODELS
    except Exception as exc:
        log.warning("Não foi possível listar modelos via SDK Groq: %s", exc)
        return FALLBACK_MODELS


def test_groq_chat(model_id: str) -> tuple[dict, dict, int]:
    """Valida chave/modelo com SDK Groq e tenta capturar headers quando o SDK permite."""
    client = build_client()
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Responda apenas: ok"}],
        "max_tokens": 3,
        "temperature": 0,
    }

    try:
        raw_client = getattr(client, "with_raw_response", None)
        if raw_client is not None:
            raw = raw_client.chat.completions.create(**kwargs)
            parsed = raw.parse() if hasattr(raw, "parse") else raw
            headers_obj = getattr(raw, "headers", {}) or {}
            headers = {str(k).lower(): str(v) for k, v in dict(headers_obj).items()}
            return _obj_to_dict(parsed), headers, 200
    except Exception as exc:
        # Se a interface raw do SDK falhar, cai no modo normal logo abaixo.
        log.debug("SDK raw_response indisponível/falhou: %s", exc)

    response = client.chat.completions.create(**kwargs)
    return _obj_to_dict(response), {}, 200


def groq_exception_to_status(exc: Exception) -> tuple[int, str]:
    text = str(exc)
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if not isinstance(status, int):
        m = re.search(r"\b(401|403|404|429|500|502|503)\b", text)
        status = int(m.group(1)) if m else 500
    if status == 401:
        return 401, "GROQ_API_KEY inválida/revogada ou diferente da chave esperada. Confira o .env e reinicie o servidor."
    if status == 403:
        return 403, "A chave foi reconhecida, mas não tem permissão para este modelo/projeto ou a conta está bloqueando a requisição."
    if status == 429:
        return 429, "Limite de uso/rate limit atingido. Aguarde o reset ou troque para um modelo com limite disponível."
    return status, text


def parse_rate_limit_headers(headers: dict) -> dict:
    keys = [
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-limit-input-tokens",
        "x-ratelimit-remaining-input-tokens",
        "x-ratelimit-limit-output-tokens",
        "x-ratelimit-remaining-output-tokens",
    ]
    return {key: headers.get(key) for key in keys if headers.get(key) is not None}


def validate_model_id(model_id: str, fallback: str = MODEL) -> str:
    model_id = clean_text(model_id or fallback)
    if not re.fullmatch(r"[A-Za-z0-9._:/-]+", model_id):
        raise ValueError("Modelo Groq inválido.")
    return model_id


# ---------------------------------------------------------------------------
# Helpers de URL
# ---------------------------------------------------------------------------


def resolve_short_url(url: str) -> str:
    """Resolve URLs curtas do Google sem depender do navegador."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        res = urllib.request.urlopen(req, timeout=12)
        final = res.url
        log.info("URL resolvida: %s", final)
        return final
    except Exception as exc:
        log.warning("Não foi possível resolver URL curta (%s); usando a original.", exc)
        return url


def normalize_url(url: str) -> str:
    """Normaliza e valida URLs aceitas pelo app."""
    url = clean_text(url)
    if not url:
        raise ValueError("Campo 'url' é obrigatório.")

    if not re.match(r"^https?://", url, flags=re.IGNORECASE):
        url = "https://" + url

    if any(domain in url for domain in ("share.google", "maps.app.goo.gl", "goo.gl/maps")):
        url = resolve_short_url(url)

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    if "google." not in host or "/maps" not in path:
        raise ValueError(
            "URL não reconhecida. Use uma URL do Google Maps "
            "(google.com/maps/place/...) ou um link curto do Google Maps."
        )
    return url


def is_search_results_url(url: str) -> bool:
    """
    Detecta URL de lista de busca.

    O bug anterior marcava qualquer /maps/place/ com !1m2!2m1! como busca.
    Esse trecho também aparece em URLs diretas copiadas após uma pesquisa, então
    agora só tratamos como busca quando o caminho é realmente /maps/search/ ou
    quando a URL é /maps?output=search sem /maps/place/.
    """
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.lower()
    query = urllib.parse.parse_qs(parsed.query)

    if "/maps/search/" in path:
        return True
    if "/maps/place/" in path:
        return False
    return path.rstrip("/") in {"/maps", "/maps/"} and (
        query.get("output") == ["search"] or "q" in query
    )


def build_reviews_url(url: str) -> str:
    """Tenta transformar uma URL /maps/place/ na aba de avaliações (!9m1!1b1)."""
    if "/maps/place/" not in url or "!9m1!1b1" in url:
        return url

    main, sep, query = url.partition("?")
    main = re.sub(r"!4m6!3m5!", "!4m8!3m7!", main, count=1)
    if "!16s" in main:
        main = main.replace("!16s", "!9m1!1b1!16s", 1)
    elif "/data=" in main:
        main += "!9m1!1b1"
    else:
        main += "/data=!9m1!1b1"
    return main + (sep + query if sep else "")


# ---------------------------------------------------------------------------
# Playwright helpers
# ---------------------------------------------------------------------------


def safe_inner_text(locator, timeout: int = 1200) -> str:
    try:
        return clean_text(locator.first.inner_text(timeout=timeout))
    except Exception:
        return ""


def safe_attr(locator, attr: str, timeout: int = 1200) -> str:
    try:
        return clean_text(locator.first.get_attribute(attr, timeout=timeout))
    except Exception:
        return ""


def click_first_visible(page, selectors: list[str], timeout: int = 1500) -> bool:
    for selector in selectors:
        try:
            element = page.locator(selector).first
            if element.is_visible(timeout=timeout):
                element.click(timeout=3000)
                log.info("Clique efetuado via seletor: %s", selector)
                return True
        except Exception:
            continue
    return False


def accept_google_consent(page) -> None:
    for selector in [
        "#L2AGLb",
        "button[aria-label*='Aceitar']",
        "button[aria-label*='Accept']",
        "button:has-text('Aceitar tudo')",
        "button:has-text('I agree')",
        "form[action*='consent'] button",
    ]:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1500):
                btn.click(timeout=3000)
                log.info("Cookies/consentimento aceitos via: %s", selector)
                time.sleep(1.5)
                return
        except Exception:
            pass


def wait_page_settle(page, seconds: float = 2.0) -> None:
    try:
        page.wait_for_load_state("domcontentloaded", timeout=8000)
    except Exception:
        pass
    time.sleep(seconds)


# ---------------------------------------------------------------------------
# Scraping com Playwright
# ---------------------------------------------------------------------------


def click_first_organic_result(page) -> bool:
    """Clica no primeiro resultado de busca que não pareça anúncio."""
    log.info("URL de busca detectada — procurando primeiro resultado orgânico...")

    card_selectors = ["div.Nv2PK", "div[role='article']", "div[jsaction*='mouseover']"]
    for card_selector in card_selectors:
        try:
            cards = page.locator(card_selector).all()
        except Exception:
            cards = []

        for card in cards[:10]:
            try:
                card_text = clean_text(card.inner_text(timeout=1000))
                if re.search(r"patrocinad|sponsor|anúncio|ad\b", card_text, re.IGNORECASE):
                    log.info("Resultado patrocinado/anúncio ignorado.")
                    continue
                link = card.locator("a[href*='/maps/place/']").first
                href = link.get_attribute("href", timeout=1000) or ""
                if href:
                    link.click(timeout=3500)
                    log.info("Resultado orgânico aberto: %s", href[:120])
                    wait_page_settle(page, 3)
                    return True
            except Exception:
                continue

    # Fallback genérico, ainda filtrando links óbvios de rotas.
    try:
        links = page.locator("a[href*='/maps/place/']").all()
        for link in links[:20]:
            href = link.get_attribute("href", timeout=1000) or ""
            text = clean_text(link.inner_text(timeout=1000))
            if not href or "/maps/dir/" in href:
                continue
            if re.search(r"patrocinad|sponsor|anúncio|ad\b", text, re.IGNORECASE):
                continue
            link.click(timeout=3500)
            log.info("Resultado aberto via fallback: %s", href[:120])
            wait_page_settle(page, 3)
            return True
    except Exception:
        pass

    log.warning("Não foi possível abrir um resultado orgânico da busca.")
    return False


def extract_place_info(page) -> dict:
    place_info = {
        "name": "Estabelecimento",
        "rating": None,
        "total_ratings": None,
        "category": None,
        "address": None,
    }

    generic_titles = {"resultados", "results", "result", "google maps"}

    # Nome: use apenas seletores realmente curtos. Nunca use inner_text de painéis amplos,
    # porque a aba de avaliações pode colocar todos os reviews dentro do mesmo texto.
    name_candidates: list[str] = []
    for selector in ["h1.DUwDvf", "div.DUwDvf"]:
        try:
            loc = page.locator(selector)
            count = min(loc.count(), 5)
            for i in range(count):
                name_candidates.append(clean_text(loc.nth(i).inner_text(timeout=800)))
                name_candidates.append(clean_text(loc.nth(i).get_attribute("aria-label", timeout=800)))
        except Exception:
            pass

    # h1 genérico como fallback, mas só se o texto passar no filtro anti-painel inteiro.
    try:
        loc = page.locator("h1")
        count = min(loc.count(), 5)
        for i in range(count):
            name_candidates.append(clean_text(loc.nth(i).inner_text(timeout=800)))
            name_candidates.append(clean_text(loc.nth(i).get_attribute("aria-label", timeout=800)))
    except Exception:
        pass

    # Em alguns layouts, o nome vem no aria-label das tabs: "Avaliações de X".
    try:
        tabs = page.locator("[role='tab'][aria-label]").all()
        for tab in tabs:
            aria = clean_text(tab.get_attribute("aria-label", timeout=800))
            match = re.search(r"(?:Avaliações|Reviews|Visão geral|Overview)\s+(?:de|of)\s+(.+)", aria, re.IGNORECASE)
            if match:
                name_candidates.append(match.group(1))
    except Exception:
        pass

    try:
        title = clean_text(page.title())
        if title:
            name_candidates.append(re.sub(r"\s*-\s*Google Maps\s*$", "", title, flags=re.IGNORECASE))
    except Exception:
        pass

    for candidate in name_candidates:
        candidate = clean_text(candidate)
        if is_probable_place_name(candidate):
            place_info["name"] = candidate
            log.info("Nome detectado: %s", candidate)
            break

    # Nota média.
    rating_selectors = [
        "div.F7nice span[aria-hidden='true']",
        "div.jANrlb div.fontDisplayLarge",
        "span.ceNzKf",
        "span.Aq14fc",
        "div.F7nice",
        "div[role='img'][aria-label*='estrelas']",
        "div[role='img'][aria-label*='stars']",
    ]
    for selector in rating_selectors:
        try:
            loc = page.locator(selector)
            count = min(loc.count(), 4)
            for i in range(count):
                txt = clean_text(loc.nth(i).get_attribute("aria-label", timeout=800)) or clean_text(loc.nth(i).inner_text(timeout=800))
                match = re.search(r"(\d+(?:[,.]\d+)?)\s*(?:estrela|star)?", txt, re.IGNORECASE)
                if match:
                    place_info["rating"] = match.group(1).replace(",", ".")
                    log.info("Rating detectado: %s", place_info["rating"])
                    raise StopIteration
        except StopIteration:
            break
        except Exception:
            continue

    # Total de avaliações — evitar pegar "4,3 estrelas" como total.
    total_selectors = [
        "button[jsaction*='pane.rating.moreReviews']",
        "div.jANrlb div.fontBodySmall",
        "span[aria-label*='avaliações']",
        "span[aria-label*='reviews']",
    ]
    for selector in total_selectors:
        try:
            loc = page.locator(selector)
            count = min(loc.count(), 8)
            for i in range(count):
                txt = clean_text(loc.nth(i).get_attribute("aria-label", timeout=800)) or clean_text(loc.nth(i).inner_text(timeout=800))
                if not re.search(r"avalia|review", txt, re.IGNORECASE):
                    continue
                if re.search(r"estrela|star", txt, re.IGNORECASE) and not re.search(r"avalia|review", txt, re.IGNORECASE):
                    continue
                # Prefere o trecho total, não uma linha de distribuição por estrela.
                match = re.search(r"(\d[\d.\s,]*\s*(?:avaliações|reviews))", txt, re.IGNORECASE)
                if match:
                    place_info["total_ratings"] = clean_text(match.group(1))
                else:
                    place_info["total_ratings"] = txt
                log.info("Total de avaliações detectado: %s", place_info["total_ratings"])
                raise StopIteration
        except StopIteration:
            break
        except Exception:
            continue

    for selector in ["button[jsaction*='category']", "button[jsaction*='pane.rating.category']", "span.DkEaL"]:
        txt = safe_inner_text(page.locator(selector), timeout=1000)
        if txt:
            place_info["category"] = txt
            break

    for selector in ["button[data-item-id='address']", "div[data-item-id='address']", "button[aria-label*='Endereço']"]:
        txt = safe_inner_text(page.locator(selector), timeout=1000)
        if txt:
            place_info["address"] = txt
            break

    return place_info


def count_review_cards(page) -> int:
    try:
        return page.locator("div[data-review-id]").count()
    except Exception:
        return 0


def unique_review_count(page) -> int:
    try:
        return int(page.locator("div[data-review-id]").evaluate_all(
            """
            els => new Set(els.map((el, i) =>
                el.getAttribute('data-review-id') || (el.innerText || '').slice(0, 140) || String(i)
            )).size
            """
        ))
    except Exception:
        return count_review_cards(page)


def click_reviews_tab(page) -> bool:
    """Abre a aba de avaliações por texto, aria-label ou botão de contagem."""
    try:
        tab = page.get_by_role("tab", name=re.compile(r"avalia|review", re.IGNORECASE)).first
        if tab.is_visible(timeout=1800):
            tab.click(timeout=3500)
            log.info("Aba de avaliações clicada via role/tab.")
            wait_page_settle(page, 2.5)
            return True
    except Exception:
        pass

    if click_first_visible(page, [
        "[role='tab'][aria-label*='Avaliações']",
        "[role='tab'][aria-label*='Reviews']",
        "button[aria-label*='Avaliações']",
        "button[aria-label*='Reviews']",
    ]):
        wait_page_settle(page, 2.5)
        return True

    # Fallback por JavaScript: útil quando o texto está em aria-label e não visível.
    try:
        clicked = page.evaluate(
            """
            () => {
                const re = /(avalia|review)/i;
                const elements = [...document.querySelectorAll('button,[role="tab"]')];
                const el = elements.find(e => re.test(e.innerText || '') || re.test(e.getAttribute('aria-label') || ''));
                if (!el) return false;
                el.click();
                return true;
            }
            """
        )
        if clicked:
            log.info("Aba/botão de avaliações clicado via JS.")
            wait_page_settle(page, 2.5)
            return True
    except Exception:
        pass

    # Em alguns layouts, clicar no número de avaliações abre a seção correta.
    if click_first_visible(page, [
        "button[jsaction*='pane.rating.moreReviews']",
        "button[aria-label*='avaliaç']",
        "button[aria-label*='review']",
    ], timeout=1200):
        wait_page_settle(page, 2.5)
        return True

    return False


def ensure_reviews_open(page, original_url: str) -> bool:
    if count_review_cards(page) > 0:
        return True

    if click_reviews_tab(page):
        try:
            page.wait_for_selector("div[data-review-id]", timeout=8000)
        except Exception:
            pass
        if count_review_cards(page) > 0:
            return True

    # Último recurso: navegar diretamente para a URL da aba de avaliações.
    reviews_url = build_reviews_url(page.url or original_url)
    if reviews_url != (page.url or original_url):
        try:
            log.info("Tentando URL direta da aba de avaliações: %s", reviews_url[:180])
            page.goto(reviews_url, wait_until="domcontentloaded", timeout=40000)
            wait_page_settle(page, 4)
            accept_google_consent(page)
            if count_review_cards(page) == 0:
                click_reviews_tab(page)
            page.wait_for_selector("div[data-review-id]", timeout=10000)
        except Exception:
            pass

    return count_review_cards(page) > 0


def expand_visible_reviews(page) -> None:
    """Expande botões 'Ver mais' somente dentro dos cards de review."""
    selectors = [
        "button.w8nwRe",
        "button[aria-label='Ver mais']",
        "button[aria-label='More']",
        "button:has-text('Mais')",
        "button:has-text('More')",
    ]
    try:
        cards = page.locator("div[data-review-id]").all()
    except Exception:
        cards = []

    # Se ainda não há cards, não clique em botões globais de 'Mais' para evitar
    # abrir menus errados do Google Maps.
    for card in cards[:40]:
        for selector in selectors:
            try:
                buttons = card.locator(selector).all()
                for button in buttons[:3]:
                    try:
                        if button.is_visible(timeout=300):
                            button.click(timeout=500)
                    except Exception:
                        pass
            except Exception:
                pass


def scroll_reviews(page, max_reviews: int, collect_all: bool = False) -> None:
    """Rola o contêiner correto de avaliações, com fallback por mouse wheel."""
    scroll_js = """
    () => {
        const cards = [...document.querySelectorAll('div[data-review-id]')];
        const seeds = [
            ...cards,
            ...document.querySelectorAll('div[role="feed"], div.m6QErb, div[role="main"]')
        ];
        const candidates = [];
        const seen = new Set();
        for (const seed of seeds) {
            let el = seed;
            while (el && el !== document.body && el !== document.documentElement) {
                if (!seen.has(el)) {
                    seen.add(el);
                    const style = window.getComputedStyle(el);
                    const delta = el.scrollHeight - el.clientHeight;
                    if (delta > 80 && /(auto|scroll|overlay)/i.test(style.overflowY + ' ' + style.overflow)) {
                        candidates.push({el, delta});
                    }
                }
                el = el.parentElement;
            }
        }
        candidates.sort((a, b) => b.delta - a.delta);
        const target = candidates[0]?.el;
        if (!target) return {ok:false, reason:'no-scroll-target'};
        const before = target.scrollTop;
        target.scrollTop = target.scrollHeight;
        target.dispatchEvent(new WheelEvent('wheel', {deltaY: 5000, bubbles: true, cancelable: true}));
        target.dispatchEvent(new Event('scroll', {bubbles: true}));
        return {ok:true, before, after:target.scrollTop, scrollHeight:target.scrollHeight, clientHeight:target.clientHeight};
    }
    """

    previous_unique = unique_review_count(page)
    stalled = 0
    # No modo "todas", damos mais iterações e paramos quando não aparecem novos reviews.
    max_iterations = 180 if collect_all else max(28, min(70, max_reviews + 10))

    for iteration in range(max_iterations):
        expand_visible_reviews(page)
        current_unique = unique_review_count(page)
        current_raw = count_review_cards(page)
        log.info("Iteração %s: %s cards / %s reviews únicos", iteration + 1, current_raw, current_unique)

        if current_unique >= max_reviews:
            log.info("Quantidade desejada atingida%s.", " ou limite de segurança" if collect_all and max_reviews >= MAX_REVIEWS_LIMIT else "")
            break

        if current_unique <= previous_unique:
            stalled += 1
        else:
            stalled = 0
        previous_unique = max(previous_unique, current_unique)

        if stalled >= 8 and current_unique > 0:
            log.info("Sem novos reviews após várias tentativas. Parando scroll.")
            break

        try:
            page.evaluate(scroll_js)
        except Exception:
            pass

        try:
            last = page.locator("div[data-review-id]").last
            box = last.bounding_box(timeout=800)
            if box:
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.mouse.wheel(0, 5000)
        except Exception:
            try:
                page.mouse.wheel(0, 5000)
            except Exception:
                pass

        time.sleep(1.8)


def extract_reviews(page, max_reviews: int) -> list[dict]:
    """
    Extrai reviews em lote via JavaScript.

    A versão antiga fazia várias chamadas Playwright por card e isso podia levar
    dezenas de minutos quando havia centenas/milhares de avaliações. Aqui a
    extração é feita dentro do navegador em uma única chamada page.evaluate(),
    retornando uma lista já quase pronta para o Python.
    """
    expand_visible_reviews(page)

    extraction_js = """
    (maxReviews) => {
      const clean = (value) => String(value || '')
        .replace(/[\uE000-\uF8FF]/g, '')
        .replace(/\u00a0/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();

      const firstText = (root, selectors) => {
        for (const selector of selectors) {
          const el = root.querySelector(selector);
          const txt = clean(el?.innerText || el?.textContent || '');
          if (txt) return txt;
        }
        return '';
      };

      const firstAttr = (root, selectors, attr) => {
        for (const selector of selectors) {
          const els = [...root.querySelectorAll(selector)].slice(0, 8);
          for (const el of els) {
            const val = clean(el.getAttribute(attr) || '');
            if (val) return val;
          }
        }
        return '';
      };

      const cards = [...document.querySelectorAll('div[data-review-id]')];
      const out = [];
      const seen = new Set();

      for (const card of cards) {
        if (out.length >= maxReviews) break;

        const review_id = clean(card.getAttribute('data-review-id') || '');

        let author = firstText(card, [
          'div.d4r55',
          'div[class*="fontTitleSmall"]',
          '.kvMYJc'
        ]);
        if (!author || /estrela|star/i.test(author)) author = 'Anônimo';

        let stars = '?';
        const starLabel = firstAttr(card, [
          'span[role="img"]',
          'span[aria-label*="estrela"]',
          'span[aria-label*="star"]'
        ], 'aria-label');
        const starMatch = starLabel.match(/(\d+)/);
        if (starMatch && /estrela|star/i.test(starLabel)) stars = starMatch[1];

        let date = firstText(card, [
          'span.rsqaWe',
          'span.hvjdm',
          'span[class*="fontBodySmall"]'
        ]);
        if (/avaliaç|review|foto|photo|local guide/i.test(date)) date = '';

        let body = firstText(card, [
          'span.wiI7pd',
          'div.MyEned span.wiI7pd',
          'div.MyEned'
        ]).replace(/\s+Mais$/i, '').trim();
        if (/^(Mais|More)$/i.test(body)) body = '';

        let visit_info = firstText(card, [
          'span.RfnDt'
        ]);
        if (/local guide|foto|photo/i.test(visit_info)) visit_info = '';

        const key = review_id || [author, date, stars, body.slice(0, 160)].join('|');
        if (seen.has(key)) continue;
        seen.add(key);

        if (body || stars !== '?') {
          out.push({review_id, author, stars, date, body, visit_info});
        }
      }

      return {raw_cards: cards.length, reviews: out};
    }
    """

    try:
        result = page.evaluate(extraction_js, max_reviews) or {}
        raw_cards = int(result.get("raw_cards") or 0)
        raw_reviews = result.get("reviews") or []
        log.info("Extração em lote: até %s reviews de %s cards DOM.", max_reviews, raw_cards)

        reviews: list[dict] = []
        seen: set[str] = set()
        for item in raw_reviews:
            if len(reviews) >= max_reviews:
                break
            if not isinstance(item, dict):
                continue
            review = {
                "review_id": clean_text(item.get("review_id")),
                "author": clean_text(item.get("author")) or "Anônimo",
                "stars": clean_text(item.get("stars")) or "?",
                "date": clean_text(item.get("date")),
                "body": clean_text(item.get("body")),
                "visit_info": clean_text(item.get("visit_info")),
            }
            key = review.get("review_id") or "|".join([
                review.get("author", ""),
                review.get("date", ""),
                review.get("stars", ""),
                review.get("body", "")[:160],
            ])
            if key in seen:
                continue
            seen.add(key)
            if review.get("body") or review.get("stars") != "?":
                reviews.append(review)

        log.info("Reviews únicos extraídos com sucesso: %s", len(reviews))
        return reviews

    except Exception as exc:
        log.warning("Extração em lote falhou; usando extração antiga como fallback: %s", exc)

    # Fallback antigo, limitado e com timeouts menores, só para casos em que o JS falhe.
    review_els = page.locator("div[data-review-id]").all()
    log.info("Fallback: extraindo até %s reviews de %s cards...", max_reviews, len(review_els))

    reviews: list[dict] = []
    seen: set[str] = set()
    for el in review_els:
        if len(reviews) >= max_reviews:
            break
        try:
            review_id = clean_text(el.get_attribute("data-review-id", timeout=250))
            review: dict[str, str] = {"review_id": review_id} if review_id else {}

            author = safe_inner_text(el.locator("div.d4r55"), timeout=250) or safe_inner_text(el.locator("div[class*='fontTitleSmall']"), timeout=250)
            review["author"] = author if author and not re.search(r"estrela|star", author, re.IGNORECASE) else "Anônimo"

            label = clean_text(el.locator("span[role='img']").first.get_attribute("aria-label", timeout=250))
            match = re.search(r"(\d+)", label)
            review["stars"] = match.group(1) if match and re.search(r"estrela|star", label, re.IGNORECASE) else "?"

            date = safe_inner_text(el.locator("span.rsqaWe"), timeout=250) or safe_inner_text(el.locator("span.hvjdm"), timeout=250)
            review["date"] = "" if re.search(r"avaliaç|review|foto|photo|local guide", date, re.IGNORECASE) else date

            body = safe_inner_text(el.locator("span.wiI7pd"), timeout=300) or safe_inner_text(el.locator("div.MyEned"), timeout=300)
            review["body"] = re.sub(r"\s+Mais\s*$", "", body).strip()
            review["visit_info"] = safe_inner_text(el.locator("span.RfnDt"), timeout=250)

            key = review_id or "|".join([review.get("author", ""), review.get("date", ""), review.get("stars", ""), review.get("body", "")[:160]])
            if key in seen:
                continue
            seen.add(key)
            if review.get("body") or review.get("stars") != "?":
                reviews.append(review)
        except Exception as exc:
            log.debug("Erro ao extrair review no fallback: %s", exc)

    log.info("Reviews únicos extraídos com sucesso: %s", len(reviews))
    return reviews


def scrape_google_maps_reviews(url: str, max_reviews: int | None = DEFAULT_REVIEWS_LIMIT) -> dict:
    from playwright.sync_api import TimeoutError as PWTimeout
    from playwright.sync_api import sync_playwright

    normalized_url = normalize_url(url)
    collect_all = max_reviews is None
    target_reviews = MAX_REVIEWS_LIMIT if collect_all else clamp_int(max_reviews, DEFAULT_REVIEWS_LIMIT, 1, MAX_REVIEWS_LIMIT)
    log.info("URL normalizada: %s", normalized_url)
    log.info("É URL de busca: %s", is_search_results_url(normalized_url))

    place_info = {
        "name": "Estabelecimento",
        "rating": None,
        "total_ratings": None,
        "category": None,
        "address": None,
    }
    reviews: list[dict] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--lang=pt-BR",
            ],
        )
        context = browser.new_context(
            locale="pt-BR",
            timezone_id="America/Sao_Paulo",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1366, "height": 900},
        )
        page = context.new_page()

        try:
            log.info("Navegando para a URL...")
            page.goto(normalized_url, wait_until="domcontentloaded", timeout=45000)
            wait_page_settle(page, 4)
            accept_google_consent(page)

            if is_search_results_url(normalized_url):
                click_first_organic_result(page)

            # Extrai dados básicos antes de mudar para a aba de reviews.
            place_info = extract_place_info(page)

            if not ensure_reviews_open(page, normalized_url):
                log.warning("Aba de avaliações não encontrada ou sem cards carregados.")
            else:
                # Depois de abrir a aba, alguns layouts têm dados melhores de nota/total.
                refreshed_info = extract_place_info(page)
                for key, value in refreshed_info.items():
                    if not value:
                        continue
                    if key == "name":
                        # Não sobrescreve um nome bom com o texto gigante da aba de avaliações.
                        if place_info.get("name") == "Estabelecimento" and is_probable_place_name(value):
                            place_info["name"] = value
                        continue
                    place_info[key] = value
                if collect_all:
                    total_count = parse_total_reviews_count(place_info.get("total_ratings"))
                    if total_count:
                        target_reviews = min(total_count, MAX_REVIEWS_LIMIT)
                        log.info("Modo todas as avaliações: tentando coletar %s de %s avaliações.", target_reviews, total_count)
                    else:
                        target_reviews = MAX_REVIEWS_LIMIT
                        log.info("Modo todas as avaliações: total não detectado; coletando até parar ou atingir %s.", MAX_REVIEWS_LIMIT)

                scroll_reviews(page, target_reviews, collect_all=collect_all)
                reviews = extract_reviews(page, target_reviews)

        except PWTimeout as exc:
            log.error("Timeout do Playwright: %s", exc)
        except Exception as exc:
            log.error("Erro durante scraping: %s", exc, exc_info=True)
        finally:
            context.close()
            browser.close()

    return {"place_info": place_info, "reviews": reviews, "total_scraped": len(reviews), "requested_all_reviews": collect_all}


# ---------------------------------------------------------------------------
# Análise com Groq
# ---------------------------------------------------------------------------


def build_client() -> Groq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY não definida. Crie um arquivo .env com GROQ_API_KEY=gsk_... "
            "ou defina a variável de ambiente antes de executar o servidor."
        )
    return Groq(api_key=GROQ_API_KEY)


def call_groq_with_retry(
    client: Groq,
    model: str,
    prompt: str,
    max_tokens: int = 2800,
    max_retries: int = 4,
    json_mode: bool = False,
) -> str:
    """Chama Groq com retry e mensagens de erro mais claras."""
    for attempt in range(max_retries):
        try:
            log.info("Tentativa %s/%s para chamar Groq...", attempt + 1, max_retries)
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            error = str(exc)
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if not isinstance(status_code, int):
                m = re.search(r"\b(401|403|413|429|500|502|503)\b", error)
                status_code = int(m.group(1)) if m else None
            is_payload_too_large = status_code == 413 or "payload too large" in error.lower()
            is_rate_limit = status_code == 429 or "too many requests" in error.lower()
            is_auth = status_code == 401 or "invalid_api_key" in error.lower() or "unauthorized" in error.lower()
            unsupported_json_mode = json_mode and "response_format" in error.lower() and attempt == 0

            if unsupported_json_mode:
                log.warning("Modelo/API não aceitou json_mode; repetindo sem response_format.")
                json_mode = False
                continue
            if is_auth:
                raise PermissionError(
                    "GROQ_API_KEY inválida ou sem permissão. Gere uma chave nova no GroqCloud, "
                    "coloque no .env e reinicie o servidor."
                ) from exc
            if is_payload_too_large:
                raise ValueError(
                    "A requisição enviada à Groq ficou grande demais mesmo após otimização. "
                    "Reduza AI_SAMPLE_LIMIT/AI_REVIEW_BODY_CHARS no .env ou colete menos avaliações."
                ) from exc
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = min(45, (2 ** attempt) * 6)
                log.warning("Rate limit detectado. Aguardando %ss antes de tentar novamente...", wait_time)
                time.sleep(wait_time)
                continue
            raise ValueError(f"Erro ao chamar Groq: {error}") from exc

    raise ValueError("Erro ao chamar Groq após várias tentativas.")


def deterministic_star_distribution(reviews: list[dict]) -> dict[str, int]:
    counts = Counter(str(r.get("stars", "?")).strip() for r in reviews)
    total = sum(counts.get(str(i), 0) for i in range(1, 6)) or 1
    return {str(i): round(counts.get(str(i), 0) * 100 / total) for i in range(5, 0, -1)}


def _review_quality_score(review: dict) -> int:
    """Pontua reviews úteis para resumo/destaques sem precisar mandar tudo para a IA."""
    body = clean_text(review.get("body"))
    stars = clean_text(review.get("stars"))
    score = len(body)
    if stars in {"1", "2", "4", "5"}:
        score += 80
    if stars == "3":
        score += 40
    # Dá preferência a avaliações específicas, não só longas.
    if re.search(r"atendimento|servi[cç]o|ambiente|pre[cç]o|pizza|pedido|entrega|espera|qualidade|sabor|cliente|funcion", body, re.IGNORECASE):
        score += 120
    if len(body) < 25:
        score -= 200
    return score


def build_ai_review_sample(reviews: list[dict], sample_limit: int = AI_SAMPLE_LIMIT) -> list[dict]:
    """
    Monta uma amostra pequena para a IA.

    Para 1000+ reviews, enviar tudo causa 413 Payload Too Large. A análise usa:
    - as primeiras reviews carregadas, que normalmente são as mais relevantes/visíveis;
    - as reviews com mais texto e mais sinais específicos;
    - algumas reviews negativas, positivas e neutras para equilibrar.
    """
    if not reviews:
        return []

    chosen: dict[int, dict] = {}

    def add(idx: int) -> None:
        if 0 <= idx < len(reviews) and len(chosen) < sample_limit:
            chosen.setdefault(idx, reviews[idx])

    # 1) primeiras avaliações, mas não todas
    for i in range(min(35, len(reviews))):
        add(i)

    # 2) avaliações mais informativas por tamanho/especificidade
    ranked = sorted(range(len(reviews)), key=lambda i: _review_quality_score(reviews[i]), reverse=True)
    for i in ranked[:80]:
        add(i)

    # 3) diversidade por estrelas
    for star in ["1", "2", "3", "4", "5"]:
        star_ranked = [i for i in ranked if clean_text(reviews[i].get("stars")) == star]
        for i in star_ranked[:10]:
            add(i)

    # Se ainda sobrou espaço, completa com ordem de qualidade.
    for i in ranked:
        add(i)

    # Ordena pelo índice original para ficar legível, mantendo no máximo sample_limit.
    selected = sorted(chosen.items(), key=lambda pair: pair[0])[:sample_limit]
    result = []
    for original_index, review in selected:
        result.append({
            "index": original_index,
            "stars": clean_text(review.get("stars")) or "?",
            "author": clean_text(review.get("author")),
            "date": clean_text(review.get("date")),
            "body": clean_text(review.get("body"))[:AI_REVIEW_BODY_CHARS],
        })
    return result


def fallback_highlights(reviews: list[dict], amount: int = 3) -> list[dict]:
    """Seleciona avaliações úteis sem IA, usando só score local e diversidade de estrelas."""
    indexed = list(enumerate(reviews))
    indexed.sort(key=lambda pair: _review_quality_score(pair[1]), reverse=True)
    highlights = []
    seen_stars = set()
    for idx, review in indexed:
        stars = clean_text(review.get("stars")) or "?"
        body = clean_text(review.get("body"))
        if len(body) < 20:
            continue
        if stars in seen_stars and len(seen_stars) < min(3, amount):
            continue
        item = review.copy()
        item["motivo_selecao"] = "Avaliação detalhada, específica e útil para representar a experiência dos clientes."
        highlights.append(item)
        seen_stars.add(stars)
        if len(highlights) >= amount:
            break

    if len(highlights) < amount:
        for _, review in indexed:
            if review in highlights:
                continue
            item = review.copy()
            item["motivo_selecao"] = "Avaliação representativa do conjunto coletado."
            highlights.append(item)
            if len(highlights) >= amount:
                break

    return highlights[:amount]


def build_local_analysis(reviews: list[dict]) -> dict:
    star_values = [int(r["stars"]) for r in reviews if str(r.get("stars", "")).isdigit()]
    avg = sum(star_values) / len(star_values) if star_values else 0
    sentiment = "positivo" if avg >= 4 else "misto" if avg >= 3 else "negativo"
    return {
        "sentimento_geral": sentiment,
        "score_sentimento": round(avg * 2, 1) if avg else 5,
        "pontos_positivos": [],
        "pontos_negativos": [],
        "temas_recorrentes": [],
        "perfil_visitante": "Perfil não determinado automaticamente.",
        "distribuicao_estrelas": deterministic_star_distribution(reviews),
        "alertas": [],
    }


def agent_generate_report(client: Groq, place_info: dict, reviews: list[dict], model_id: str) -> dict:
    """Gera análise, resumo e destaques em uma única chamada para evitar 429."""
    reviews_for_prompt = build_ai_review_sample(reviews, sample_limit=AI_SAMPLE_LIMIT)
    log.info(
        "Amostra enviada para IA: %s de %s reviews coletadas; limite de caracteres por review: %s",
        len(reviews_for_prompt),
        len(reviews),
        AI_REVIEW_BODY_CHARS,
    )

    local_distribution = deterministic_star_distribution(reviews)
    prompt = f"""Você é um analista de reputação de estabelecimentos e experiência do cliente.

DADOS DO ESTABELECIMENTO:
{json.dumps(place_info, ensure_ascii=False, indent=2)}

AVALIAÇÕES PARA ANÁLISE DA IA ({len(reviews_for_prompt)} selecionadas de {len(reviews)} coletadas):
{json.dumps(reviews_for_prompt, ensure_ascii=False, indent=2)}

Distribuição real calculada a partir das avaliações coletadas:
{json.dumps(local_distribution, ensure_ascii=False)}

Retorne APENAS um JSON válido com EXATAMENTE esta estrutura:
{{
  "analysis": {{
    "sentimento_geral": "positivo" | "neutro" | "negativo" | "misto",
    "score_sentimento": <número de 0 a 10>,
    "pontos_positivos": ["ponto 1", "ponto 2"],
    "pontos_negativos": ["ponto 1", "ponto 2"],
    "temas_recorrentes": ["tema 1", "tema 2"],
    "perfil_visitante": "descrição do tipo de cliente/visitante típico",
    "distribuicao_estrelas": {{"5": <percentual>, "4": <percentual>, "3": <percentual>, "2": <percentual>, "1": <percentual>}},
    "alertas": ["alerta importante 1", "alerta importante 2"]
  }},
  "summary": "Resumo executivo em português, entre 220 e 450 palavras, em parágrafos fluidos, terminando com Recomendado, Recomendado com ressalvas ou Não recomendado.",
  "highlights": [
    {{"index": <índice original da avaliação>, "motivo": "por que esta avaliação foi selecionada"}}
  ]
}}

Regras:
- Use somente evidências das avaliações fornecidas na amostra otimizada.
- Não invente fatos fora das avaliações.
- Em distribuicao_estrelas, use a distribuição real calculada acima.
- Selecione no máximo 3 highlights usando apenas índices presentes na amostra otimizada.
- Priorize avaliações com texto específico e útil; não escolha avaliações curtas só por terem 5 estrelas.
"""

    raw = call_groq_with_retry(client, model_id, prompt, max_tokens=3200, max_retries=4, json_mode=True)
    parsed = extract_json_object(raw)

    analysis = parsed.get("analysis") or build_local_analysis(reviews)
    if not isinstance(analysis, dict):
        analysis = build_local_analysis(reviews)
    analysis.setdefault("distribuicao_estrelas", local_distribution)

    summary = clean_text(parsed.get("summary"))
    if not summary:
        summary = "Não foi possível gerar um resumo executivo detalhado, mas as avaliações foram coletadas com sucesso."

    highlights: list[dict] = []
    for item in parsed.get("highlights", []) if isinstance(parsed.get("highlights"), list) else []:
        try:
            idx = int(item.get("index"))
        except Exception:
            continue
        if 0 <= idx < len(reviews):
            review = reviews[idx].copy()
            review["motivo_selecao"] = clean_text(item.get("motivo"))
            highlights.append(review)
    if not highlights:
        highlights = fallback_highlights(reviews, amount=3)

    return {"analysis": analysis, "summary": summary, "highlights": highlights[:3]}


# ---------------------------------------------------------------------------
# Rotas da API
# ---------------------------------------------------------------------------


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "api_key_configured": bool(GROQ_API_KEY),
        "api_key_masked": mask_secret(GROQ_API_KEY),
        "model": MODEL,
        "dotenv_override": True,
    })


@app.route("/api/models", methods=["GET"])
def models():
    try:
        models_list = fetch_groq_models()
        return jsonify({
            "current_model": MODEL,
            "models": models_list,
            "api_key_configured": bool(GROQ_API_KEY),
            "api_key_masked": mask_secret(GROQ_API_KEY),
        })
    except PermissionError as exc:
        return jsonify({"error": str(exc), "models": FALLBACK_MODELS, "current_model": MODEL}), 401
    except Exception as exc:
        return jsonify({"error": str(exc), "models": FALLBACK_MODELS, "current_model": MODEL}), 400


@app.route("/api/groq-status", methods=["GET"])
def groq_status():
    try:
        model_id = validate_model_id(request.args.get("model") or MODEL)
        models_list = fetch_groq_models()
        data, headers, status_code = test_groq_chat(model_id)

        return jsonify({
            "ok": status_code == 200,
            "model": model_id,
            "api_key_configured": bool(GROQ_API_KEY),
            "api_key_masked": mask_secret(GROQ_API_KEY),
            "models_count": len(models_list),
            "models": models_list,
            "limits": parse_rate_limit_headers(headers),
            "usage": data.get("usage") if isinstance(data, dict) else None,
            "note": (
                "Este teste valida a chave com uma chamada real pelo SDK oficial da Groq. "
                "Quando a biblioteca expõe headers de limite, eles aparecem aqui; o histórico diário completo fica no console/logs da Groq."
            ),
        }), status_code
    except EnvironmentError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    except Exception as exc:
        status_code, message = groq_exception_to_status(exc)
        return jsonify({
            "ok": False,
            "error": message,
            "api_key_masked": mask_secret(GROQ_API_KEY),
            "raw_error": str(exc),
        }), status_code


@app.route("/", methods=["GET"])
def index():
    base_dir = os.path.dirname(__file__)
    return send_file(os.path.join(base_dir, "index.html"))


@app.route("/api/summarize", methods=["POST"])
def summarize():
    try:
        body = request.get_json(silent=True) or {}
        url = clean_text(body.get("url"))
        max_reviews = parse_review_limit(body.get("max_reviews", DEFAULT_REVIEWS_LIMIT))
        model_id = validate_model_id(body.get("model") or MODEL)

        if not url:
            return jsonify({"error": "Campo 'url' é obrigatório."}), 400

        # Valida chave antes de gastar tempo no scraping usando o SDK oficial.
        # Evita o erro Cloudflare 1010 que pode ocorrer com urllib.
        client = build_client()
        try:
            client.models.list()
        except Exception as exc:
            status_code, message = groq_exception_to_status(exc)
            if status_code in {401, 403, 429}:
                return jsonify({"error": message, "raw_error": str(exc)}), status_code
            raise

        log.info("Iniciando scraping: %s", url)
        scraped = scrape_google_maps_reviews(url, max_reviews=max_reviews)

        if not scraped["reviews"]:
            return jsonify({
                "place_info": scraped.get("place_info", {}),
                "total_scraped": 0,
                "error": (
                    "Nenhuma avaliação foi coletada. Tente copiar a URL direta do estabelecimento "
                    "ou abrir a aba Avaliações no Google Maps antes de copiar o link."
                ),
            }), 422

        report = agent_generate_report(client, scraped["place_info"], scraped["reviews"], model_id)

        return jsonify({
            "place_info": scraped["place_info"],
            "total_scraped": scraped["total_scraped"],
            "requested_all_reviews": scraped.get("requested_all_reviews", False),
            "analysis": report["analysis"],
            "summary": report["summary"],
            "highlights": report["highlights"],
        })

    except EnvironmentError as exc:
        return jsonify({"error": str(exc)}), 500
    except PermissionError as exc:
        return jsonify({"error": str(exc)}), 401
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Erro ao processar JSON da IA: {exc}"}), 500
    except Exception as exc:
        log.exception("Erro inesperado")
        return jsonify({"error": f"Erro interno: {exc}"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    if not GROQ_API_KEY:
        print("\n⚠️  AVISO: GROQ_API_KEY não definida!")
        print("   Crie um arquivo .env com: GROQ_API_KEY=gsk_...\n")
    else:
        print(f"\n✅  Chave API configurada ({mask_secret(GROQ_API_KEY)}) | Modelo: {MODEL}")
    print("🚀  Servidor iniciando em http://localhost:5000\n")
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1", host="127.0.0.1", port=5000, use_reloader=False)
