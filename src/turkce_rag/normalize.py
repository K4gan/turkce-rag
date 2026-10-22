"""Türkçe metin normalizasyonu.

Python'un ``str.lower()`` fonksiyonu dilden bağımsızdır: ``"I".lower()`` → ``"i"`` (Türkçede
``"ı"`` olmalı) ve ``"İ".lower()`` → ``"i̇"`` (``i`` + U+0307 birleşik nokta) üretir. İkincisi
görünüşte doğru olsa da farklı bir kod noktası dizisidir ve arama/eşleştirmeyi sessizce bozar.
Bu modül bu tuzakları kapatır ve PDF'ten gelen tipik kirlilikleri temizler.
"""

from __future__ import annotations

import re
import unicodedata

_LOWER_MAP = str.maketrans({"I": "ı", "İ": "i"})
_UPPER_MAP = str.maketrans({"i": "İ", "ı": "I"})

# Türkçe harfler + Türkçe metinlerde sık görülen şapkalı ünlüler (â, î, û).
_FOLD_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "â": "a",
        "î": "i",
        "û": "u",
        "Ç": "C",
        "Ğ": "G",
        "İ": "I",
        "Ö": "O",
        "Ş": "S",
        "Ü": "U",
        "Â": "A",
        "Î": "I",
        "Û": "U",
    }
)

_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})

# Kesme işareti varyantları → düz apostrof (İstanbul’da / İstanbul`da / İstanbul'da).
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'", "ʼ": "'"})
_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})

_SOFT_HYPHEN = "\u00ad"
_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
# Satır sonunda tireyle bölünmüş kelime: "bilgi-\nsayar" → "bilgisayar".
# Yalnızca küçük harfle devam eden durumlarda birleştir ("Ankara-\nİstanbul" korunur).
_HYPHENATION = re.compile(r"(\w)-\n(?=[a-zçğıöşü])")
_SPACES = re.compile(r"[ \t\u00a0]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def turkish_lower(text: str) -> str:
    """Türkçe kurallarıyla küçük harfe çevirir (I→ı, İ→i)."""
    return unicodedata.normalize("NFC", text).translate(_LOWER_MAP).lower()


def turkish_upper(text: str) -> str:
    """Türkçe kurallarıyla büyük harfe çevirir (i→İ, ı→I)."""
    return unicodedata.normalize("NFC", text).translate(_UPPER_MAP).upper()


def fold_diacritics(text: str) -> str:
    """Türkçe karakterleri ASCII karşılıklarına indirger (ş→s, ı→i, ğ→g ...).

    Türkçe klavyesi olmayan kullanıcıların sorgularını ("sikayet" ↔ "şikâyet") eşleştirmek için
    kullanılır. Geri dönüşsüzdür; yalnızca eşleştirme anahtarı üretmek içindir, gösterim için değil.
    """
    text = unicodedata.normalize("NFC", text).translate(_FOLD_MAP)
    # Kalan birleşik aksanları da (ör. é, ñ) temizle.
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_text(text: str) -> str:
    """Belge metnini indekslemeden önce temizler.

    * Unicode NFC (ayrık yazılmış ``i`` + ``̇`` gibi dizileri tek kod noktasına birleştirir)
    * Bağlaç ligatürleri (ﬁ → fi), yumuşak tire ve sıfır genişlikli karakterler
    * Satır sonu tirelemesi, kesme işareti ve tırnak varyantları
    * Fazla boşluklar (paragraf sınırı olan boş satırlar korunur)
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_LIGATURES).translate(_APOSTROPHES).translate(_QUOTES)
    text = text.replace(_SOFT_HYPHEN, "")
    text = _ZERO_WIDTH.sub("", text)
    text = _HYPHENATION.sub(r"\1", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()
