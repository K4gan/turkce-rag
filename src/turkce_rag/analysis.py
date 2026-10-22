"""Türkçe için sözcük ayırma, durak sözcükler ve hafif kök bulma (light stemming).

Türkçe eklemeli (agglutinative) bir dildir: "kitap", "kitabı", "kitaplarımızdan" aynı köke
sahiptir. Klasik boşlukla ayırma + BM25 bu biçimleri farklı terimler sayar ve geri çağırma
(recall) çöker. Tam bir morfolojik çözümleyici (ör. Zemberek) yerine burada bilinçli olarak
*hafif* bir yaklaşım kullanılıyor:

1. Özel adlardaki kesme işaretinden sonrasını at: ``İstanbul'da`` → ``istanbul``.
2. Sözcük sonundaki çekim eklerini (çoğul, iyelik, hâl, ek-fiil) en uzundan başlayarak, kök en az
   3 harf kalacak şekilde en fazla birkaç tur soy.
3. Ünsüz yumuşamasını geri al (b→p, c→ç, d→t, g/ğ→k): ``kitab`` → ``kitap``.
   Kural sondaki her harfe uygulanır; sorguya ve belgeye aynı şekilde uygulandığı için tutarlıdır.
4. İsteğe bağlı olarak Türkçe karakterleri katla (``ş``→``s``): ASCII klavyeyle yazılan
   sorguları yakalar.

Yapım eklerine (-lık, -cı, -sız ...) ve fiil çekimlerine (-dı, -mış ...) dokunulmaz: anlamı
değiştirirler ya da isim köklerini bozarlar (``yeraltı`` ↛ ``yeral``). Bu kök bulucu dilbilimsel
olarak "doğru" kök üretmeyi değil, aynı sözcüğün çekimli biçimlerini *tutarlı* biçimde aynı
anahtara indirmeyi hedefler.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from turkce_rag.normalize import fold_diacritics, turkish_lower

_WORD = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)

STOPWORDS: frozenset[str] = frozenset(
    [
        "acaba",
        "ama",
        "ancak",
        "aslında",
        "az",
        "bana",
        "bazı",
        "belki",
        "ben",
        "beni",
        "benim",
        "beri",
        "bile",
        "bir",
        "birçok",
        "biri",
        "birkaç",
        "birşey",
        "biz",
        "bize",
        "bizi",
        "bizim",
        "bu",
        "buna",
        "bunda",
        "bundan",
        "bunlar",
        "bunları",
        "bunların",
        "bunu",
        "bunun",
        "burada",
        "çok",
        "çünkü",
        "da",
        "daha",
        "dahi",
        "de",
        "defa",
        "değil",
        "diğer",
        "diye",
        "dolayı",
        "en",
        "fakat",
        "gibi",
        "göre",
        "hangi",
        "hatta",
        "hem",
        "hep",
        "hepsi",
        "her",
        "hiç",
        "için",
        "ile",
        "ise",
        "işte",
        "kadar",
        "ki",
        "kim",
        "kime",
        "kimi",
        "mi",
        "mı",
        "mu",
        "mü",
        "nasıl",
        "ne",
        "neden",
        "nerde",
        "nerede",
        "nereye",
        "niçin",
        "niye",
        "o",
        "olan",
        "olarak",
        "oldu",
        "olduğu",
        "olduğunu",
        "olmak",
        "olur",
        "ona",
        "ondan",
        "onlar",
        "onları",
        "onların",
        "onu",
        "onun",
        "orada",
        "öyle",
        "pek",
        "rağmen",
        "sadece",
        "şey",
        "şu",
        "şuna",
        "şunda",
        "şundan",
        "şunu",
        "siz",
        "sizi",
        "sizin",
        "ve",
        "veya",
        "ya",
        "yani",
        "yine",
        "zaten",
        "çünkü",
        "hangisi",
        "kaç",
        "midir",
        "mıdır",
        "mudur",
        "müdür",
        "nedir",
        "hangisidir",
        "nelerdir",
        "kaçtır",
        "kimdir",
        "nerededir",
        "nereden",
        "nereye",
        "neresi",
        "neler",
        "hangileri",
        "hangileridir",
        "kendi",
        "kendim",
        "kendimi",
        "kendine",
        "gerek",
        "gerekir",
        "gerekiyor",
        "lazım",
        "olmalı",
        "olmalıdır",
        "olması",
        "olsun",
    ]
)

# ASCII yazılmış sorgular için katlanmış biçimler de durak sözcüktür ("olmali", "cunku").
_STOPWORDS_FOLDED: frozenset[str] = frozenset(fold_diacritics(w) for w in STOPWORDS)

# Çekim ekleri; ünlü uyumu varyantlarıyla. Uzunluğa göre azalan sırada denenir.
_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        set(
            [
                "lar",
                "ler",
                "ları",
                "leri",
                "larını",
                "lerini",
                "larının",
                "lerinin",
                "larında",
                "lerinde",
                "larından",
                "lerinden",
                "larına",
                "lerine",
                "larla",
                "lerle",
                "ımız",
                "imiz",
                "umuz",
                "ümüz",
                "ınız",
                "iniz",
                "unuz",
                "ünüz",
                "ımızın",
                "imizin",
                "ımızda",
                "imizde",
                "ımızdan",
                "imizden",
                "sı",
                "si",
                "su",
                "sü",
                "sını",
                "sini",
                "sunu",
                "sünü",
                "sının",
                "sinin",
                "sında",
                "sinde",
                "sından",
                "sinden",
                "sına",
                "sine",
                "ın",
                "in",
                "un",
                "ün",
                "nın",
                "nin",
                "nun",
                "nün",
                "ının",
                "inin",
                "unun",
                "ünün",
                "ım",
                "im",
                "um",
                "üm",
                "ı",
                "i",
                "u",
                "ü",
                "yı",
                "yi",
                "yu",
                "yü",
                "nı",
                "ni",
                "nu",
                "nü",
                "ya",
                "ye",
                "na",
                "ne",
                "a",
                "e",
                "da",
                "de",
                "ta",
                "te",
                "nda",
                "nde",
                "dan",
                "den",
                "tan",
                "ten",
                "ndan",
                "nden",
                "la",
                "le",
                "yla",
                "yle",
                "daki",
                "deki",
                "taki",
                "teki",
                "ndaki",
                "ndeki",
                "dır",
                "dir",
                "dur",
                "dür",
                "tır",
                "tir",
                "tur",
                "tür",
            ]
        ),
        key=len,
        reverse=True,
    )
)

_VOWELS = set("aeıioöuüâîû")
_CIRCUMFLEX = str.maketrans({"â": "a", "î": "i", "û": "u"})
_DESOFTEN = {"b": "p", "c": "ç", "d": "t", "g": "k", "ğ": "k"}
MIN_STEM = 3
MAX_PASSES = 3


def _has_vowel(s: str) -> bool:
    return any(ch in _VOWELS for ch in s)


def stem(word: str) -> str:
    """Küçük harfli tek bir Türkçe sözcüğün hafif kökünü döndürür.

    >>> stem("kitaplarımızdan")
    'kitap'
    >>> stem("istanbul'da")
    'istanbul'
    """
    if "'" in word:
        word = word.split("'", 1)[0]
    if word.isdigit() or len(word) <= MIN_STEM:
        return word
    for _ in range(MAX_PASSES):
        for suf in _SUFFIXES:
            if not word.endswith(suf):
                continue
            rest = word[: -len(suf)]
            # Tek harfli ekler çok belirsiz: kök en az 4 harf kalmalı (elma ↛ elm).
            min_len = MIN_STEM + 1 if len(suf) == 1 else MIN_STEM
            if len(rest) >= min_len and _has_vowel(rest):
                word = rest
                break
        else:
            break
    if len(word) > MIN_STEM and word[-1] == word[-2] and word[-1] not in _VOWELS:
        word = word[:-1]  # ikiz ünsüz: hakkı → hakk → hak
    if word and word[-1] in _DESOFTEN:
        word = word[:-1] + _DESOFTEN[word[-1]]
    return word


@dataclass(frozen=True)
class Analyzer:
    """Metni BM25 terimlerine çeviren yapılandırılabilir boru hattı.

    ``Analyzer.plain()`` karşılaştırma için dil bilgisi olmayan taban çizgisidir.
    """

    stem: bool = True
    fold: bool = True
    stopwords: bool = True
    turkish_case: bool = True

    @classmethod
    def plain(cls) -> Analyzer:
        return cls(stem=False, fold=False, stopwords=False, turkish_case=False)

    def words(self, text: str) -> list[str]:
        lowered = turkish_lower(text) if self.turkish_case else text.lower()
        if self.turkish_case:
            # Şapka (â, î, û) yazım tercihidir, anlam farkı yaratmaz: Resmî = Resmi.
            lowered = lowered.translate(_CIRCUMFLEX)
        return _WORD.findall(lowered)

    def __call__(self, text: str) -> list[str]:
        out: list[str] = []
        for w in self.words(text):
            if self.stopwords:
                base = w.split("'", 1)[0]
                if base in STOPWORDS or (self.fold and base in _STOPWORDS_FOLDED):
                    continue
            term = stem(w) if self.stem else w
            if self.fold:
                term = fold_diacritics(term)
            if term:
                out.append(term)
        return out


DEFAULT_ANALYZER = Analyzer()


def content_terms(text: str, analyzer: Analyzer = DEFAULT_ANALYZER) -> set[str]:
    """Sorgunun içerik terimleri (durak sözcükler hariç, köklenmiş)."""
    return set(analyzer(text))


def _term_matches(q: str, doc_terms: set[str]) -> bool:
    if q in doc_terms:
        return True
    if len(q) < 4:
        return False
    # Önek toleransı: hafif kök bulucunun kaçırdığı fiil çekimlerini yakalar
    # ("yayimlant" ↔ "yayimlanmistir", "kuru" ↔ "kurul").
    prefix = q[: max(4, len(q) - 2)]
    return any(d.startswith(prefix) for d in doc_terms)


def coverage(
    query_terms: Iterable[str],
    doc_terms: set[str],
    idf: dict[str, float] | None = None,
    default_idf: float = 1.0,
) -> float:
    """Sorgu terimlerinin parçada geçme oranı; ``idf`` verilirse IDF ağırlıklı.

    IDF ağırlığı sayesinde derlemde hiç geçmeyen ayırt edici bir terim ("espresso") eksik
    kaldığında, "hazırlanır" gibi genel bir terimin eşleşmesi tek başına güven oluşturmaz.
    """
    q = list(dict.fromkeys(query_terms))
    if not q:
        return 0.0
    weights = [idf.get(t, default_idf) if idf is not None else 1.0 for t in q]
    total = sum(weights)
    matched = sum(w for t, w in zip(q, weights, strict=True) if _term_matches(t, doc_terms))
    return matched / total if total else 0.0
