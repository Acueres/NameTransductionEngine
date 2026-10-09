"""Arabic: BGN/PCGN-style romanization, as used for Arab place names in
GeoNames (Ar-Riyād, Al-Qāhirah, Madīnat al-Kuwayt, Al-Jazāʾir), without the
dots under letters. Spoken varieties (Egyptian, Levantine...) are read the
same way.

Arabic script leaves short vowels unwritten and uses و and ي both for
consonants and long vowels, so a word is read the way Persian is (see
abjad.py), most reliable first:

1. written vowel marks, when the word has them: exact;
2. a lexicon: hand-checked generic terms and international names, then words
   attested in GeoNames with their BGN romanization;
3. a letter model learned from the same data: for each letter, its reading
   together with the short vowel before it, chosen by the letters around it.

When the entity has Latin-script names (its GeoNames name, English names),
each word's reading allowed by its letters is matched against them. This
settles the vowels of a native name from its BGN form, and of a foreign name
from its original spelling: نوتنغهامشير with "Nottinghamshire" is
Nūtinghāmshīr, where no Arabic model could know the vowels.

Grammar the letters do not show is handled between words:
- the article ال is written al-, assimilated before the "sun letters"
  (ash-Shāriqah, ar-Riyād), and read apart from the word it is attached to;
- tāʾ marbūṭa (ة) is -ah, and -at when the word heads a construct (iḍāfa):
  Madīnat al-Kuwayt but Al-Madīnah al-Munawwarah. Whether a word heads a
  construct is learned from GeoNames: the head, the word after it, and
  whether either has the article.

The learned parts are a model file built from GeoNames by `nte init`
(models/arabic_romanization/). Without it this provider steps aside with a
warning, and the rule-based Arabic-script reader (arabic_script.py) is used.
"""

import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from name_transduction_engine.paths import ARABIC_MODEL_PATH

from . import abjad
from .abjad import fold_latin
from .base import Context, ProviderUnavailable, Rendering
from .text import capitalize_first

# --------------------------------------------------------------------------- #
# Letters
# --------------------------------------------------------------------------- #

_FATHA, _DAMMA, _KASRA = "\u064e", "\u064f", "\u0650"
_TANWIN = frozenset("\u064b\u064c\u064d")
_SUKUN, _SHADDA = "\u0652", "\u0651"
_SUPERSCRIPT_ALEF = "\u0670"
_MARKS = frozenset([_FATHA, _DAMMA, _KASRA, *_TANWIN, _SUKUN, _SHADDA])

_NORMALIZE = str.maketrans(
    {
        # Persian and Urdu forms of Arabic letters
        "ی": "ي", "ې": "ي", "ک": "ك", "ہ": "ه", "ھ": "ه", "ۀ": "ة", "ۃ": "ة",
        "ٱ": "ا",
        # Joiners, tatweel, direction marks
        "\u200c": None, "\u200d": None, "ـ": None, "\u200e": None, "\u200f": None,
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6",
        "٧": "7", "٨": "8", "٩": "9", "۰": "0", "۱": "1", "۲": "2", "۳": "3",
        "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }
)  # fmt: skip

CONSONANTS = {
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h", "خ": "kh", "د": "d",
    "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh", "ص": "s", "ض": "d",
    "ط": "t", "ظ": "z", "ع": "ʿ", "غ": "gh", "ف": "f", "ق": "q", "ك": "k",
    "ل": "l", "م": "m", "ن": "n",
    # Letters for foreign sounds (Persian, Maghrebi, Iraqi spellings)
    "پ": "p", "چ": "ch", "ژ": "zh", "گ": "g", "ڤ": "v", "ڥ": "v", "ڨ": "g",
    "ڭ": "g", "ڢ": "f", "ڧ": "q",
}  # fmt: skip

# The article's l is assimilated before these (ash-Shams, an-Nīl)
SUN_LETTERS = frozenset("تثدذرزسشصضطظلن")

_VOWEL_LETTERS = frozenset("اآأإءؤئويىة")
LETTERS = frozenset([*CONSONANTS, *_VOWEL_LETTERS, "ه"])
TA_MARBUTA = "ة"


def normalize(text: str) -> str:
    """NFC, Arabic letter forms, and the superscript alef (dagger alef)
    written out as an alef: الرحمٰن -> الرحمان"""
    text = unicodedata.normalize("NFC", text).translate(_NORMALIZE)
    if _SUPERSCRIPT_ALEF in text:
        text = text.replace(_FATHA + _SUPERSCRIPT_ALEF, "ا").replace(
            _SUPERSCRIPT_ALEF, "ا"
        )
    return text


class ArabicOrthography(abjad.Orthography):
    name = "arabic"
    letters = LETTERS
    consonants = CONSONANTS
    mark_vowels = {_FATHA: "a", _KASRA: "i", _DAMMA: "u"}
    sukun = _SUKUN
    shadda = _SHADDA
    marks = _MARKS
    carriers = frozenset("اأإ")
    # Letter pairs for one foreign sound: تش ch, تس ts/z, دج dj, دز dz, كس x
    units = frozenset({"تش", "تس", "دج", "دز", "كس"})
    near_pairs = abjad.Orthography.near_pairs | frozenset(
        frozenset(p)
        # Arabic has no p, v, e, o, hard g: foreign names write them with
        # b, f, ī, ū and gh / j / k / q. In a BGN hint, a vowel's length
        # and ʿ / ʾ are written, and count for half a letter
        for p in ("pb", "vf", "gk", "gq", "ie", "uo", "ea", "fw", "jy", "xk",
                  "aā", "iī", "uū", "aá", "āá", "ʿʾ")
    )  # fmt: skip

    def letter_options(self, word: str, i: int) -> tuple[str, ...]:
        """Readings in the BGN form the model learns from: tāʾ marbūṭa and a
        final ه are "ah" (pausal), ي and و after their own vowel are "īy",
        "ūw" (Janūbīyah, Qūwah)"""
        ch = word[i]
        first = i == 0
        last = i == len(word) - 1
        if ch in CONSONANTS:
            c = CONSONANTS[ch]
            return (c, c + c)
        # A word-initial alef before و or ي may stand for the long vowel
        # they write, which are then silent (أوروبا Ūrubbā, إيران Īrān)
        # (a short u before و, i before ي, would be that long vowel)
        nxt = word[i + 1 : i + 2]
        carried = {"و": ("ū",), "ي": ("ī",)}.get(nxt, ())
        short = tuple(
            v for v in ("a", "i", "u") if (v, nxt) not in (("u", "و"), ("i", "ي"))
        )
        if ch == "ا":
            return (*short, "ā", *carried) if first else ("ā", "")
        if ch == "آ":
            return ("ā",) if first else ("ʾā", "ā")
        if ch == "أ":
            return (*(v for v in short if v != "i"), *carried) if first else ("ʾ", "")
        if ch == "إ":
            return (*(v for v in short if v == "i"), *carried) if first else ("ʾ", "")
        if ch == "ء":
            return ("ʾ", "")
        if ch == "ؤ":
            return ("ʾ", "ʾū", "w", "ū", "")
        if ch == "ئ":
            return ("ʾ", "ʾī", "y", "ī", "")
        # A written و or ي is a glide or a long vowel, never a short one
        # (كوروني Kūrūnī, not Kurūnī; أديس Adīs, not Adis). It is silent only
        # after a word-initial alef that reads its long vowel (أوروبا), and
        # و at the end after a consonant (عمرو ʿAmr)
        if ch == "و":
            if first:
                return ("w", "ū")
            silent = self.may_be_silent(word, i) or (
                last and word[i - 1] in CONSONANTS
            )
            return ("w", "ū", "ww", "ūw", *(("",) if silent else ()))
        if ch == "ي":
            if first:
                return ("y", "ī")
            silent = self.may_be_silent(word, i)
            return ("y", "ī", "yy", "īy", *(("",) if silent else ()))
        if ch == "ى":
            # Egyptian and Sudanese spelling also use it for final ي
            return ("á", "ī", "y")
        if ch == TA_MARBUTA:
            # After a long ā only the h or t is left (حماة Ḩamāh)
            return ("h", "t") if word[i - 1 : i] == "ا" else ("ah", "at", "a")
        if ch == "ه":
            return ("h", "hh", "ah", "a") if last and not first else ("h", "hh")
        return (ch,)

    def slot_options(self, word: str, i: int) -> tuple[str, ...]:
        if i == 0:
            return ("",)
        if word[i] in "اآى" or word[i] == TA_MARBUTA:
            return ("",)  # the letter is itself the vowel after the consonant
        return ("", "a", "i", "u")

    def allowed(self, prefix: str, out: str, strict: bool = False) -> bool:
        """Arabic syllables are CV, CVV, CVC (CVVC, CVCC at the end of a
        word): no vowel meets another (a glide or hamza stands between), and
        a native word starts with one consonant. Foreign names start with
        clusters (Stūkhūlm), so the hint-guided search allows them"""
        if not out:
            return True
        vowels = abjad.LATIN_VOWELS
        text = prefix[-2:] + out
        for k in range(max(len(prefix[-2:]) - 1, 0), len(text) - 1):
            a, b = text[k], text[k + 1]
            if a in vowels and b in vowels:
                return False  # كويلار is Kuwaylār, not Kūīlār
            if k > 0 and _stranded_glide(text[k - 1], a, b):
                return False  # سليفن is Slīfin, not Slyfin
            if k > 0 and _open_geminate(text[k - 1], a, b):
                return False
            if k > 0 and text[k - 1] == a and a not in vowels and b not in vowels:
                return False  # a doubled consonant is followed by a vowel
        starts_cluster = (
            bool(prefix)
            and out[0] not in vowels
            and not any(c in vowels for c in prefix)
        )
        if strict:
            # Foreign names start with clusters, but not with ʿ or ʾ in them
            return not (starts_cluster and ({"ʿ", "ʾ"} & set(prefix + out[0])))
        return not starts_cluster

    def is_doubled(self, reading: str) -> bool:
        return super().is_doubled(reading) or reading.endswith(("ūw", "īy"))

    def may_be_silent(self, word: str, i: int) -> bool:
        # و or ي after a word-initial alef that reads their long vowel
        return i == 1 and word[0] in "اأإ" and word[1] in "وي"

    def fully_voweled(self, bare: str, after: dict[int, str]) -> bool:
        # Arabic names often carry a stray damma or shadda (الدُّليميَّة):
        # a vowel or sukun on more than half the consonants makes a voweled
        # word (دِمَشْق: the last consonant is never marked in pause)
        consonants = [k for k, ch in enumerate(bare) if ch in self.consonants]
        marked = sum(1 for k in consonants if k in after)
        return bool(consonants) and 2 * marked > len(consonants)

    def forced_vowel_fits(self, vowel: str, letter: str, forced: str) -> bool:
        # A marked damma or kasra before a doubled و or ي is uww, iyy
        if letter in ("ūw", "īy"):
            return False
        return super().forced_vowel_fits(vowel, letter, forced)

    def allowed_end(self, shape: str) -> bool:
        # A glide may end a word after a consonant (ظبي Zaby, نحو Nahw), but
        # not after i or u, nor doubled with a long vowel (Jārūw)
        return len(shape) < 2 or not _open_geminate(shape[-2], shape[-1], "")

    def shape(self, prefix: str) -> str:
        # The last two letters: enough to see a glide between consonants
        base = abjad.shape(prefix)
        if base == "ab":
            return "ab" + prefix[-2:]
        return base

    def _sample_positions(self, ch: str):
        yield from super()._sample_positions(ch)
        yield ("after_hamza", "ء" + ch, 1)


def _stranded_glide(before: str, glide: str, after: str) -> bool:
    """y or w read as a consonant with no vowel on either side"""
    vowels = abjad.LATIN_VOWELS
    return (
        glide in "yw"
        and before not in vowels
        and before != glide
        and after not in vowels
    )


def _open_geminate(before: str, glide: str, after: str) -> bool:
    """A glide closing a syllable after any vowel but a: Arabic has the
    diphthongs ay and aw (and āy, āw in foreign names), while iy and uw
    before a consonant are ī and ū, and īy, ūw (BGN for iyy, uww) need a
    vowel after them"""
    return (
        glide in "yw"
        and before in "iuīū"
        and after not in abjad.LATIN_VOWELS
        and not (after == glide and before in "iu")  # iyy, uww
    )


ORTHOGRAPHY = ArabicOrthography()


def strip_marks(word: str) -> str:
    return "".join(ch for ch in word if ch not in _MARKS)


# Words that start with ال without it being the article
_NOT_ARTICLE = frozenset({"الله", "اللاه"})  # the second: اللّٰه normalized


def split_article(word: str) -> tuple[bool, str]:
    """(has article, the word after it). The article is ال before at least
    two more letters; marks on its letters go with it"""
    bare = strip_marks(word)
    if not (bare.startswith("ال") and len(bare) > 3) or bare in _NOT_ARTICLE:
        return False, word
    seen = 0
    for k, ch in enumerate(word):
        if ch in _MARKS:
            continue
        seen += 1
        if seen == 2:
            j = k + 1
            while j < len(word) and word[j] in _MARKS:
                j += 1
            body = word[j:]
            # A shadda on a sun letter after the article marks the
            # assimilation (التّين at-tīn), not a doubled letter
            if body[:1] in SUN_LETTERS and _SHADDA in body[1:3]:
                cut = body.index(_SHADDA, 1)
                body = body[:cut] + body[cut + 1 :]
            return True, body
    return False, word


def article_form(body_bare: str) -> str:
    """al-, or assimilated before a sun letter: ash-, ar-, an-..."""
    first = body_bare[:1]
    if first in SUN_LETTERS and first != "ل":
        return "a" + CONSONANTS[first] + "-"
    return "al-"


# --------------------------------------------------------------------------- #
# Rendering a reading
# --------------------------------------------------------------------------- #


def pausal(bare: str, outputs: list[str]) -> list[str]:
    """The readings of a word as said alone: a final tāʾ marbūṭa (or a
    final ه written for it) is -ah, or -h after a long ā (Ḩamāh)"""
    if not outputs or not bare or bare[-1] not in (TA_MARBUTA, "ه"):
        return outputs
    last = outputs[-1]
    if bare[-1] == "ه" and last in ("h", "hh"):
        return outputs
    out = list(outputs)
    out[-1] = "h" if last in ("h", "t") else "ah"
    return out


def render(bare: str, outputs: list[str]) -> str:
    """A word's readings (one per letter of `bare`) as display text, in its
    pausal form. A word-initial hamza is not written"""
    text = "".join(pausal(bare, outputs))
    return text[1:] if text.startswith("ʾ") else text


def construct(bare: str, text: str) -> str:
    """The construct form of a word read in pausal form: -ah -> -at"""
    if bare.endswith(TA_MARBUTA) and text.endswith("h"):
        return text[:-1] + "t"
    return text


# --------------------------------------------------------------------------- #
# Model data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArabicModel:
    version: str
    table: dict[str, str]  # context key -> joint output
    ranked: dict[str, list[str]]  # smallest contexts: outputs by frequency
    lexicon: dict[str, str]  # word without the article -> reading (pausal)
    # A word ending in tāʾ marbūṭa heads a construct, as log-odds: a prior,
    # plus how much these raise or lower them: the head word, the word after
    # it, and whether each has the article ("head:1", "next:0"...)
    construct_prior: float
    construct_feature: dict[str, float]
    construct_head: dict[str, float]
    construct_next: dict[str, float]


@lru_cache(maxsize=1)
def load_model() -> ArabicModel | None:
    """The built model, None when it has not been built. Not checked for
    staleness here (`nte data status` does that)"""
    try:
        with gzip.open(ARABIC_MODEL_PATH, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return make_model(raw)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def make_model(raw: dict) -> ArabicModel:
    return ArabicModel(
        raw["version"],
        raw["table"],
        raw.get("ranked", {}),
        raw["lexicon"],
        raw["construct_prior"],
        raw["construct_feature"],
        raw["construct_head"],
        raw["construct_next"],
    )


# --------------------------------------------------------------------------- #
# Hand lexicon: generic terms and international names, which GeoNames' Arab
# data does not teach (or teaches in local forms). Keys are written as they
# usually appear, with the article; values are display readings without it.
# --------------------------------------------------------------------------- #

HAND_LEXICON_SOURCE: dict[str, str] = {
    # States and administrative units
    "جمهورية": "jumhūrīyah", "مملكة": "mamlakah", "إمارة": "imārah",
    "إمارات": "imārāt", "سلطنة": "saltanah", "دولة": "dawlah",
    "إمبراطورية": "imbarāṭūrīyah", "خلافة": "khilāfah", "اتحاد": "ittihād",
    "ولاية": "wilāyah", "ولايات": "wilāyāt", "محافظة": "muhāfazah",
    "مقاطعة": "muqātaʿah", "إقليم": "iqlīm", "منطقة": "mintaqah",
    "مدينة": "madīnah", "بلدية": "baladīyah", "قضاء": "qadāʾ", "ناحية": "nāhiyah",
    "مديرية": "mudīrīyah", "قرية": "qaryah", "عاصمة": "ʿāsimah",
    "كونتية": "kawntīyah", "مقاطعات": "muqātaʿāt",
    "المتحدة": "muttahidah", "العربية": "ʿarabīyah", "الإسلامية": "islāmīyah",
    "الديمقراطية": "dīmuqrātīyah", "الشعبية": "shaʿbīyah",
    "الاشتراكية": "ishtirākīyah", "الاتحادية": "ittihādīyah",
    "السوفيتية": "sūfiyītīyah", "الفيدرالية": "fīdirālīyah",
    # Geography
    "جزيرة": "jazīrah", "جزر": "juzur", "شبه": "shibh", "بحر": "bahr",
    "بحيرة": "buhayrah", "نهر": "nahr", "جبل": "jabal", "جبال": "jibāl",
    "وادي": "wādī", "صحراء": "sahrāʾ", "خليج": "khalīj", "مضيق": "madīq",
    "محيط": "muhīt", "ميناء": "mīnāʾ", "قلعة": "qalʿah", "واحة": "wāhah",
    "سهل": "sahl", "هضبة": "hadabah", "رأس": "raʾs",
    # Religious names
    "الله": "allāh", "اللاه": "allāh", "عبد": "ʿabd", "أبو": "abū", "بن": "bin",
    # Directions and common modifiers
    "شمال": "shamāl", "جنوب": "janūb", "شرق": "sharq", "غرب": "gharb",
    "الشمالية": "shamālīyah", "الجنوبية": "janūbīyah", "الشرقية": "sharqīyah",
    "الغربية": "gharbīyah", "الوسطى": "wustá", "العليا": "ʿulyā",
    "السفلى": "suflá", "الكبرى": "kubrá", "الصغرى": "sughrá",
    "الكبير": "kabīr", "الصغير": "saghīr", "الجديدة": "jadīdah",
    "الجديد": "jadīd", "القديمة": "qadīmah", "القديم": "qadīm",
    "الشمالي": "shamālī", "الجنوبي": "janūbī", "الشرقي": "sharqī",
    "الغربي": "gharbī", "الأوسط": "awsat", "الأعلى": "aʿlá", "الأدنى": "adná",
    # Continents and countries
    "آسيا": "āsiyā", "أوروبا": "ūrubbā", "أفريقيا": "afrīqiyā",
    "إفريقيا": "ifrīqiyā", "أمريكا": "amrīkā", "أستراليا": "usturāliyā",
    "فرنسا": "faransā", "ألمانيا": "almāniyā", "إيطاليا": "īṭāliyā",
    "إسبانيا": "isbāniyā", "البرتغال": "burtughāl", "بريطانيا": "barīṭāniyā",
    "إنجلترا": "injiltirā", "إنكلترا": "inkiltirā", "اسكتلندا": "iskutlandā",
    "أيرلندا": "īrlandā", "روسيا": "rūsiyā", "الصين": "sīn",
    "اليابان": "yābān", "الهند": "hind", "إيران": "īrān", "تركيا": "turkiyā",
    "اليونان": "yūnān", "مصر": "misr", "العراق": "ʿirāq", "سوريا": "sūriyā",
    "سورية": "sūrīyah", "لبنان": "lubnān", "الأردن": "urdun",
    "فلسطين": "filastīn", "السعودية": "saʿūdīyah", "اليمن": "yaman",
    "الكويت": "kuwayt", "قطر": "qatar", "البحرين": "bahrayn",
    "ليبيا": "lībiyā", "تونس": "tūnis", "الجزائر": "jazāʾir",
    "المغرب": "maghrib", "السودان": "sūdān", "موريتانيا": "mūrītāniyā",
    "الصومال": "sūmāl", "كندا": "kanadā", "المكسيك": "miksīk",
    "البرازيل": "barāzīl", "الأرجنتين": "arjantīn", "النمسا": "nimsā",
    "سويسرا": "swīsrā", "هولندا": "hūlandā", "بلجيكا": "baljīkā",
    "السويد": "suwayd", "النرويج": "narwīj", "الدنمارك": "dānimārk",
    "فنلندا": "finlandā", "بولندا": "būlandā", "أوكرانيا": "ūkrāniyā",
    "المجر": "majar", "رومانيا": "rūmāniyā", "بلغاريا": "bulghāriyā",
    "صربيا": "sirbiyā", "كرواتيا": "kurwātiyā", "ألبانيا": "albāniyā",
    "التشيك": "tishīk", "سلوفاكيا": "slūfākiyā", "أفغانستان": "afghānistān",
    "باكستان": "bākistān", "إندونيسيا": "indūnīsiyā", "ماليزيا": "mālīziyā",
    "كوريا": "kūriyā", "تايلاند": "tāylānd", "فيتنام": "fiyitnām",
    "إثيوبيا": "ithyūbiyā", "نيجيريا": "nayjīriyā", "كينيا": "kīniyā",
    "أرمينيا": "armīniyā", "جورجيا": "jūrjiyā", "أذربيجان": "adharbayjān",
    "كازاخستان": "kāzākhistān", "أوزبكستان": "ūzbakistān",
    "الأندلس": "andalus", "بيزنطة": "bīzanṭah", "الروم": "rūm",
    # Cities outside the Arab world often seen in Arabic
    "لندن": "landan", "باريس": "bārīs", "برلين": "barlīn", "روما": "rūmā",
    "موسكو": "mūskū", "أثينا": "athīnā", "إسطنبول": "istanbūl",
    "القسطنطينية": "qustantīnīyah", "مدريد": "madrīd", "لشبونة": "lishbūnah",
    "فيينا": "fiyīnā", "طهران": "tihrān", "كابول": "kābūl", "دلهي": "dilhī",
    "بكين": "bikīn", "طوكيو": "tūkyū", "واشنطن": "wāshinṭun",
    "نيويورك": "nyūyūrk",
    # Arab capitals and holy cities, whose GeoNames data may be local forms
    "القاهرة": "qāhirah", "دمشق": "dimashq", "بغداد": "baghdād",
    "بيروت": "bayrūt", "الرياض": "riyād",
    "مكة": "makkah", "المكرمة": "mukarramah", "المنورة": "munawwarah",
    "القدس": "quds", "الإسكندرية": "iskandarīyah", "حلب": "halab",
    "الموصل": "mawsil", "البصرة": "basrah", "صنعاء": "sanʿāʾ",
    "طرابلس": "tarābulus", "الخرطوم": "khartūm", "الدوحة": "dawhah",
    "أبوظبي": "abūzaby", "دبي": "dubayy", "مسقط": "masqat",
    "المنامة": "manāmah", "الرباط": "ribāt", "مراكش": "marrākush",
}  # fmt: skip


def _strip_hand_dots(text: str) -> str:
    """The hand lexicon may be written with ALA-LC dots; display has none"""
    decomposed = unicodedata.normalize("NFD", text)
    kept = "".join(ch for ch in decomposed if ch not in "\u0323\u0331\u0327")
    return unicodedata.normalize("NFC", kept)


@lru_cache(maxsize=1)
def hand_lexicon() -> dict[str, str]:
    """Hand lexicon keyed by the word without the article"""
    out = {}
    for key, value in HAND_LEXICON_SOURCE.items():
        _, body = split_article(normalize(key))
        out[strip_marks(body)] = _strip_hand_dots(value)
    return out


# --------------------------------------------------------------------------- #
# Reading a word
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WordReading:
    article: bool
    body: str  # the word after the article, normalized, without harakat
    text: str  # reading of the body, lowercase, pausal
    attested: bool  # from a lexicon or written marks, not the letter model
    voweled: bool  # fully voweled: read from its marks
    construct: bool | None = None  # a hint writes it -at (True) or -ah (False)


def read_word(model: ArabicModel, word: str) -> WordReading:
    article, body_marked = split_article(word)
    body = strip_marks(body_marked)
    voweled = _fully_voweled(body_marked)
    if not ORTHOGRAPHY.has_vowel_marks(body_marked):
        hand = hand_lexicon()
        if body in hand:
            return WordReading(article, body, hand[body], True, False)
        if body in model.lexicon:
            return WordReading(
                article, body, _lexicon_reading(body, model.lexicon[body]), True, False
            )
    outputs = abjad.read_letters(ORTHOGRAPHY, model, body_marked)
    return WordReading(article, body, render(body, outputs), voweled, voweled)


def _fully_voweled(word: str) -> bool:
    bare, after, _ = ORTHOGRAPHY.split_marks(word)
    return ORTHOGRAPHY.fully_voweled(bare, after)


@lru_cache(maxsize=1 << 14)
def _lexicon_reading(body: str, stored: str) -> str:
    """A learned lexicon entry in display form: aligned to the letters again
    so the same rendering rules apply"""
    outputs = abjad.align(ORTHOGRAPHY, body, stored)
    return render(body, outputs) if outputs is not None else stored


# --------------------------------------------------------------------------- #
# Reading guided by the entity's Latin names
# --------------------------------------------------------------------------- #

# How far a hint word may be from the reading, as a share of its length (and
# at most): a word the lexicon knows needs a close hint to change, a word the
# letter model guessed needs less, since the guess is weak evidence
_GUIDE_MAX_RATIO = {True: 0.25, False: 0.4}  # by attested
_GUIDE_MAX_COST = 4.0
_GUIDE_COSTS = abjad.GuideCosts(
    # The letter model knows native names, not the foreign ones hints help
    # with most, so departing from it costs little
    off_model=0.05,
    # Gemination is unwritten in a foreign name's Arabic spelling: a double
    # letter in the hint (Buccheri, Milazzo) is not evidence for it
    doubling=1.0,
    # و ي mostly write the long vowels ū ī in foreign names; reading them as
    # consonants needs the hint to say so
    reading_costs=(("و", "w", 0.4), ("ي", "y", 0.4)),
    # و ي and a medial ا are nearly always sounded
    silenced=1.0,
    # How the letters Arabic uses for foreign sounds are spelled in the
    # names they transcribe (readings folded to ASCII)
    equivalences=(
        ("sh", "ch", 0.3), ("sh", "sch", 0.2), ("sh", "sz", 0.3), ("sh", "s", 0.5),
        ("sh", "x", 0.5), ("sh", "h", 0.5),
        ("kh", "ch", 0.2), ("kh", "h", 0.4), ("kh", "x", 0.3), ("kh", "j", 0.5),
        ("kh", "k", 0.5),
        ("gh", "g", 0.2), ("gh", "gu", 0.3), ("gh", "r", 0.6),
        ("th", "t", 0.3), ("dh", "d", 0.3), ("dh", "th", 0.3),
        ("j", "g", 0.3), ("j", "gi", 0.2), ("j", "dj", 0.2),
        ("j", "dzh", 0.3), ("j", "dz", 0.3), ("j", "zh", 0.3), ("j", "dg", 0.3),
        ("k", "ck", 0.1), ("k", "c", 0.2), ("k", "ch", 0.3), ("k", "qu", 0.2),
        ("k", "g", 0.4), ("q", "k", 0.3), ("q", "c", 0.3), ("q", "qu", 0.2),
        ("q", "g", 0.4),
        ("ks", "x", 0.2), ("s", "z", 0.3), ("s", "c", 0.3), ("s", "ss", 0.1),
        ("s", "zz", 0.3), ("z", "s", 0.3), ("t", "tt", 0.1), ("t", "z", 0.5),
        ("f", "ph", 0.2), ("f", "v", 0.3), ("f", "w", 0.4), ("b", "p", 0.3),
        ("b", "pp", 0.3), ("w", "v", 0.3), ("w", "u", 0.3), ("w", "o", 0.4),
        ("y", "j", 0.3), ("y", "i", 0.3), ("y", "ie", 0.3), ("i", "y", 0.2),
        ("i", "e", 0.3), ("i", "ie", 0.2), ("i", "ei", 0.3), ("u", "o", 0.3),
        ("u", "ou", 0.2), ("u", "oo", 0.2), ("u", "oe", 0.2), ("u", "eu", 0.3),
        ("u", "ue", 0.3), ("i", "ee", 0.2), ("i", "ea", 0.3), ("a", "aa", 0.1),
        ("a", "e", 0.4), ("ay", "ei", 0.2),
        ("ay", "ai", 0.2), ("ay", "e", 0.4), ("aw", "au", 0.2), ("aw", "o", 0.4),
        # Letter units
        ("tsh", "ch", 0.1), ("tsh", "c", 0.3), ("tsh", "cz", 0.2), ("tsh", "tsch", 0.1),
        ("tsh", "tch", 0.1), ("tsh", "cs", 0.3), ("ts", "z", 0.2), ("ts", "c", 0.3),
        ("ts", "tz", 0.1), ("ts", "zz", 0.2), ("dj", "j", 0.2), ("dj", "g", 0.3),
        ("dz", "z", 0.3), ("ks", "x", 0.1),
    ),
)  # fmt: skip

# BGN marks: a hint written with them is a romanization of the Arabic itself
# (macrons, letters only BGN dots, ‘ for ʿayn; not á, which Spanish and
# Czech names have, nor ’ and ', which Shlissel’burg has)
_ROMANIZATION_MARKS = frozenset("āīūĀĪŪḩḨḑḐẓẒḤḥṢṣṬṭḌḍ‘ʿ")
# BGN's ş and ţ, unless the name is Turkish or Romanian
_CEDILLAS = frozenset("şŞţŢ")
_TURKISH_ROMANIAN = frozenset("ğĞıİçÇöÖüÜăĂâÂîÎșȘțȚ")


def _is_romanized(hint: str) -> bool:
    letters = set(hint)
    if letters & _ROMANIZATION_MARKS:
        return True
    return bool(letters & _CEDILLAS) and not letters & _TURKISH_ROMANIAN


_BGN_APOSTROPHES = str.maketrans(
    {"‘": "ʿ", "`": "ʿ", "ʻ": "ʿ", "’": "ʾ", "'": "ʾ", "ʼ": "ʾ"}
)
_BGN_KEEP = frozenset("abcdefghijklmnopqrstuvwxyzāīūáʿʾ")


@lru_cache(maxsize=1 << 14)
def fold_bgn(text: str) -> str:
    """A BGN-style romanization (or a reading) for matching it letter for
    letter: lowercase, dots and cedillas dropped, vowel length and ʿ ʾ kept
    (Ḩaţţīn -> hattīn)"""
    decomposed = unicodedata.normalize("NFD", text.translate(_BGN_APOSTROPHES).lower())
    kept = "".join(
        ch for ch in decomposed if ch not in "\u0323\u0327\u0331\u0326\u0306\u0307"
    )
    composed = unicodedata.normalize("NFC", kept)
    return "".join(ch for ch in composed if ch in _BGN_KEEP)


_ARTICLE_TOKEN = re.compile(r"^(?:[ae]l|[ae](?:th|dh|sh|ch|[tdrzsnl]))$")


@dataclass(frozen=True)
class HintWord:
    text: str  # folded, without its article
    romanized: bool  # BGN-style: follow it letter for letter
    source: str  # the hint as given, for warnings
    # Only for a word read with the article: the hint wrote the article
    # into the word (Albacete for البسيط al-Basīt)
    glued_article: bool = False


def hint_words(hints: tuple[str, ...]) -> list[HintWord]:
    """Hint words without their articles (Ad Dākhilah -> dakhilah; Ech
    Chlef -> chlef), plus adjacent words run together (نيويورك ~ New York),
    plus words that start with an article written into them (Alicante ->
    icante, for القنت)"""
    out: list[HintWord] = []
    for hint in hints:
        romanized = _is_romanized(hint)
        fold = fold_bgn if romanized else fold_latin
        tokens = [fold(t) for t in re.split(r"[\s\-]+", hint)]
        tokens = [t for t in tokens if t]
        words: list[str] = []
        for k, t in enumerate(tokens):
            nxt = tokens[k + 1] if k + 1 < len(tokens) else ""
            plain, plain_next = fold_latin(t), fold_latin(nxt)
            if (
                nxt
                and _ARTICLE_TOKEN.match(plain)
                and (plain.endswith("l") or plain_next.startswith(plain[1:]))
            ):
                continue
            words.append(t)
        for w in words:
            out.append(HintWord(w, romanized, hint))
            if len(w) > 4 and w[:2] in ("al", "el") and not romanized:
                out.append(HintWord(w[2:], romanized, hint, glued_article=True))
        for a, b in zip(words, words[1:], strict=False):
            out.append(HintWord(a + b, romanized, hint))
    return out


@dataclass(frozen=True)
class _Guide:
    article: bool  # the word is read with the article
    body: str
    text: str  # display reading of the body
    hint: str
    construct: bool | None  # the hint writes -at (True) or -ah (False)
    romanized: bool  # matched to a BGN-style hint


def _guide_word(
    model: ArabicModel, word: str, candidates: list[HintWord], ratio: float
) -> _Guide | None:
    """The reading closest to one of the hint words, within `ratio` of the
    hint word's length. A word starting with ال is tried with the article
    and as a whole (البانيا may be Albania rather than al-Bāniyā)"""
    bare = strip_marks(word)
    article, body = split_article(bare)
    parses = [(article, body)]
    if article:
        parses.append((False, bare))
    best: tuple[float, _Guide] | None = None
    for has_article, form in parses:
        if not form:
            continue
        choice = abjad.read_letters(ORTHOGRAPHY, model, form)
        # Reading ال as part of the name needs evidence
        extra = 0.5 if article and not has_article else 0.0
        for hw in candidates:
            if not hw.text or (hw.glued_article and not has_article):
                continue
            limit = min(_GUIDE_MAX_COST, ratio * len(hw.text)) - extra
            if best is not None:
                limit = min(limit, best[0] - extra)
            if limit < 0:
                continue
            fold = fold_bgn if hw.romanized else fold_latin
            cost, path = abjad.guided_reading(
                ORTHOGRAPHY,
                form,
                hw.text,
                choice,
                hw.romanized,
                _GUIDE_COSTS,
                limit,
                fold,
            )
            cost += extra
            if cost <= limit + extra and (best is None or cost < best[0]):
                # A BGN hint also says whether the word heads a construct
                construct_written = None
                if hw.romanized and form.endswith(TA_MARBUTA) and path:
                    construct_written = path[-1] in ("at", "t")
                guide = _Guide(
                    has_article,
                    form,
                    render(form, list(path)),
                    hw.source,
                    construct_written,
                    hw.romanized,
                )
                best = (cost, guide)
    return best[1] if best else None


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

_CONJUNCTION = "و"


def construct_score(model: ArabicModel, head: WordReading, nxt: WordReading) -> float:
    """Log-odds that `head`, a word ending in tāʾ marbūṭa, heads a construct
    with the word after it"""
    return (
        model.construct_prior
        + model.construct_feature.get(f"head:{int(head.article)}", 0.0)
        + model.construct_feature.get(f"next:{int(nxt.article)}", 0.0)
        + model.construct_head.get(head.body, 0.0)
        + model.construct_next.get(nxt.body, 0.0)
    )


def romanize_arabic(text: str, ctx: Context) -> Rendering | None:
    if ctx.lang not in ("ar", None):
        return None
    model = load_model()
    if model is None:
        # The rule-based Arabic-script reader takes over
        raise ProviderUnavailable(
            "Arabic model not built or unreadable; run `nte init`"
        )
    rendering = romanize_with_model(model, text, ctx.hints)
    if ctx.lang is None:
        warnings = rendering.warnings + ("language unknown; Arabic assumed",)
        return Rendering(
            rendering.text,
            rendering.transform,
            rendering.confidence,
            warnings,
            rendering.engine,
        )
    return rendering


def romanize_with_model(
    model: ArabicModel, text: str, hints: tuple[str, ...] = ()
) -> Rendering:
    tokens = abjad.tokenize(normalize(text).strip())
    candidates = hint_words(hints)
    readings: dict[int, WordReading] = {}
    guided: list[str] = []
    for k, (is_word, word) in enumerate(tokens):
        if not is_word or word == _CONJUNCTION:
            continue
        if not any(ch in LETTERS for ch in word):
            readings[k] = WordReading(False, word, word, True, False)
            continue
        reading = read_word(model, word)
        guide = None
        if candidates and not reading.voweled:
            ratio = _GUIDE_MAX_RATIO[reading.attested]
            guide = _guide_word(model, word, candidates, ratio)
        # The hand lexicon keeps its reading (لندن Landan, not London) unless
        # the entity's own BGN name reads the word otherwise (مضيق is
        # madīq "strait", but Qalʿat al-Muḍīq)
        if guide is not None and (
            guide.romanized or reading.body not in hand_lexicon()
        ):
            reading = WordReading(
                guide.article, guide.body, guide.text, True, False, guide.construct
            )
            guided.append(guide.hint)
        readings[k] = reading

    out: list[str] = []
    guessed = False
    all_voweled = True
    for pos, (is_word, tok) in enumerate(tokens):
        if not is_word:
            out.append(abjad.separator(tok))
            continue
        if tok == _CONJUNCTION:
            out.append("wa")
            continue
        reading = readings[pos]
        guessed |= not reading.attested
        all_voweled &= reading.voweled
        body = reading.text
        nxt = _next_word(tokens, pos)
        if nxt is not None and reading.body.endswith(TA_MARBUTA):
            is_construct = reading.construct
            if is_construct is None:
                is_construct = construct_score(model, reading, readings[nxt]) > 0
            if is_construct:
                body = construct(reading.body, body)
        word = capitalize_first(body)
        if reading.article:
            word = article_form(reading.body) + word
        out.append(word)

    result = capitalize_first("".join(out).strip())
    engine = f"arabic model {model.version}"
    warnings: tuple[str, ...] = ()
    if guided:
        matched = ", ".join(dict.fromkeys(guided))
        warnings += (f"vowels chosen to match the entity's name ({matched})",)
    if all_voweled and readings:
        return Rendering(result, "nte:arabic-bgn", "high", warnings, engine=engine)
    if guessed:
        warnings += ("short vowels of unlisted words predicted by a letter model",)
        return Rendering(result, "nte:arabic-bgn", "low", warnings, engine=engine)
    return Rendering(result, "nte:arabic-bgn", "normal", warnings, engine=engine)


def _next_word(tokens: list[tuple[bool, str]], pos: int) -> int | None:
    """Index of the word right after `pos` across a single space, if any"""
    nxt = pos + 2
    if (
        nxt < len(tokens)
        and tokens[pos + 1][1].isspace()
        and tokens[nxt][0]
        and tokens[nxt][1] != _CONJUNCTION
    ):
        return nxt
    return None
