import re
import math
from collections import Counter
from urllib.parse import urlparse

SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "cutt.ly", "rb.gy", "shorturl.at",
}
SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "bank",
    "paypal", "confirm", "password", "free", "bonus", "lucky", "win",
    "click", "webscr", "ebay", "wallet", "support", "invoice", "alert",
]
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "click", "work",
    "zip", "icu", "buzz", "country", "su",
}
RISKY_EXTENSIONS = (".exe", ".zip", ".rar", ".apk", ".scr", ".bat", ".php", ".js")
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

FEATURE_NAMES = [
    "url_len", "host_len", "path_len", "query_len",
    "n_dots", "n_hyphens", "n_underscores", "n_slashes", "n_at",
    "n_question", "n_equal", "n_amp", "n_percent",
    "n_digits", "n_letters", "digit_ratio", "letter_ratio", "n_special",
    "has_ip", "n_subdomains", "tld_len", "suspicious_tld", "is_shortener",
    "n_susp_words", "host_digits", "host_hyphens", "path_depth", "has_port",
    "entropy_url", "entropy_host", "longest_token", "risky_extension",
    "double_slash_in_path", "host_is_long",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _normalize(url: str) -> str:
    """Remove the scheme so the model doesn't rely on http/https (dataset bias)."""
    url = str(url).strip()
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url)


def extract_features(url: str) -> list:
    raw = _normalize(url)
    try:
        parsed = urlparse("http://" + raw)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        try:
            has_port = 1 if parsed.port else 0
        except ValueError:
            has_port = 0
    except ValueError:
        host, path, query, has_port = "", "", "", 0

    low = raw.lower()
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    n_digits = sum(c.isdigit() for c in raw)
    n_letters = sum(c.isalpha() for c in raw)
    n = max(len(raw), 1)
    tokens = re.split(r"[^a-zA-Z0-9]+", low)

    feats = {
        "url_len": len(raw),
        "host_len": len(host),
        "path_len": len(path),
        "query_len": len(query),
        "n_dots": raw.count("."),
        "n_hyphens": raw.count("-"),
        "n_underscores": raw.count("_"),
        "n_slashes": raw.count("/"),
        "n_at": raw.count("@"),
        "n_question": raw.count("?"),
        "n_equal": raw.count("="),
        "n_amp": raw.count("&"),
        "n_percent": raw.count("%"),
        "n_digits": n_digits,
        "n_letters": n_letters,
        "digit_ratio": n_digits / n,
        "letter_ratio": n_letters / n,
        "n_special": len(re.findall(r"[^a-zA-Z0-9]", raw)),
        "has_ip": 1 if IP_RE.match(host) else 0,
        "n_subdomains": max(len(parts) - 2, 0),
        "tld_len": len(tld),
        "suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "is_shortener": 1 if host in SHORTENERS else 0,
        "n_susp_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "host_digits": sum(c.isdigit() for c in host),
        "host_hyphens": host.count("-"),
        "path_depth": len([p for p in path.split("/") if p]),
        "has_port": has_port,
        "entropy_url": _entropy(raw),
        "entropy_host": _entropy(host),
        "longest_token": max((len(t) for t in tokens), default=0),
        "risky_extension": 1 if path.lower().endswith(RISKY_EXTENSIONS) else 0,
        "double_slash_in_path": 1 if "//" in path else 0,
        "host_is_long": 1 if len(host) > 30 else 0,
    }
    return [float(feats[name]) for name in FEATURE_NAMES]