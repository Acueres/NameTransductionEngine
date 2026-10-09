"""Persian: BGN/PCGN-style romanization, as used for Iranian place names
(Kermānshāh, Bandar-e ʿAbbās, Orūmīyeh), without the dots under letters.

Persian script leaves short vowels unwritten and uses one letter for several
sounds (و is v, ū, o or ow; ی is y, ī or ey), so a word is read in three
steps, most reliable first:

1. a lexicon: hand-checked generic and international words (امپراتوری
   Emperātūrī, فرانسه Farānseh), then words attested in GeoNames with their
   BGN romanization;
2. compounds are split where a morpheme visibly starts (a medial آ, a
   zero-width non-joiner: خرم‌آباد Khorram + ābād) and the parts looked up;
3. anything else goes letter by letter through a context model learned from
   the same data: for each letter, its romanization together with the short
   vowel before it, chosen by the letters around it (backing off to smaller
   contexts when a context was not seen).

Between words, the unwritten ezāfe is added as -e / -ye (Daryā-ye Khazar,
Jomhūrī-ye Eslāmī-ye Īrān) when the word, and the word after it, usually
call for it (Kīāsar-e Bālā); a written ezāfe (ۀ, a final kasra, final ای on
a known word) is always honored.

Short vowels marked with harakat override the model.

The learned parts (the letter model, the attested words, the ezāfe tendencies)
are a model file built from GeoNames by `nte init` (models/persian/); the hand
lexicon lives here. Without the file this provider steps aside with a
warning, and the generic Arabic-script rules read the name.
"""

import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from name_transduction_engine.paths import PERSIAN_MODEL_PATH

from . import abjad
from .abjad import fold_latin
from .base import Context, ProviderUnavailable, Rendering
from .text import capitalize_first

# --------------------------------------------------------------------------- #
# Letters
# --------------------------------------------------------------------------- #

ZWNJ = "\u200c"
EZAFE_MARK = "\u0654"  # hamza above on final heh: هٔ = ۀ
_NORMALIZE = str.maketrans(
    {
        "ي": "ی", "ى": "ی", "ې": "ی", "ك": "ک", "ۀ": "ه" + EZAFE_MARK,
        "ٱ": "ا", "\u200d": None, "ـ": None, "\u200f": None, "\u200e": None,
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4", "٥": "5", "٦": "6",
        "٧": "7", "٨": "8", "٩": "9", "۰": "0", "۱": "1", "۲": "2", "۳": "3",
        "۴": "4", "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }
)  # fmt: skip

CONSONANTS = {
    "ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "ch", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "z", "ر": "r", "ز": "z", "ژ": "zh", "س": "s",
    "ش": "sh", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "غ": "gh", "ف": "f",
    "ق": "q", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n",
}  # fmt: skip

# Short vowels that may precede a letter (unwritten); BGN writes e and o for
# Persian kasra and zamma. i and u occur in some names (Arabic, Turkic).
SLOT_VOWELS = ("", "a", "e", "o", "i", "u")

_FATHA, _DAMMA, _KASRA = "\u064e", "\u064f", "\u0650"
_SUKUN, _SHADDA = "\u0652", "\u0651"
_TANWIN_FATH = "\u064b"
_MARK_VOWEL = {_FATHA: "a", _KASRA: "e", _DAMMA: "o"}
_MARKS = frozenset(
    [*_MARK_VOWEL, _SUKUN, _SHADDA, _TANWIN_FATH, "\u064c", "\u064d", "\u0670"]
)

LETTERS = frozenset(
    [*CONSONANTS, "ا", "آ", "و", "ی", "ه", "ع", "ء", "ئ", "ؤ", "أ", "إ", "ة", ZWNJ]
)


def letter_options(word: str, i: int) -> tuple[str, ...]:
    """Every way letter i of a normalized word can be romanized, most usual
    first (the order breaks ties when aligning training data)"""
    ch = word[i]
    first = i == 0 or word[i - 1] == ZWNJ
    if ch in CONSONANTS:
        c = CONSONANTS[ch]
        return (c, c + c)
    if ch == "ا":
        return ("a", "e", "o", "", "ā") if first else ("ā",)
    if ch == "آ":
        return ("ā", "ʾā")
    if ch == "و":
        return ("v", "ū", "o", "ow", "", "vv", "u")
    if ch == "ی":
        return ("y", "ī", "ey", "īy", "yy", "eyy", "ā", "i", "e", "ʾ")
    if ch == "ه":
        return ("h", "eh", "hh", "e") if not first else ("h",)
    if ch == "ع":
        return ("ʿ", "", "ʿʿ")
    if ch in ("ء", "أ", "إ"):
        return ("ʾ", "", "a", "e") if first else ("ʾ", "")
    if ch == "ئ":
        return ("ʾ", "y", "", "ʾī")
    if ch == "ؤ":
        return ("ʾ", "ʾū", "ū", "")
    if ch == "ة":
        return ("eh", "at", "e")
    if ch == ZWNJ:
        return ("",)
    return (ch,)


def slot_options(word: str, i: int) -> tuple[str, ...]:
    """Short vowels that may stand before letter i"""
    if i == 0 or word[i - 1] == ZWNJ or word[i] == ZWNJ:
        return ("",)
    if word[i] in ("ا", "آ") and word[i - 1] not in ("ء", "ئ", "ؤ", "ع"):
        return ("",)  # a long vowel follows; no short vowel before it
    return SLOT_VOWELS


def normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).translate(_NORMALIZE)


# --------------------------------------------------------------------------- #
# The orthography, as the shared abjad reader needs it
# --------------------------------------------------------------------------- #

# Persian syllables are (C)V(C)(C): a word starts with a single consonant
# (kh+v counts as one: Khvāf) and two short vowels never meet. Three
# consonants in a row do occur, across compound boundaries (Poshtkūh), so
# they are allowed. The hint-guided search, which can choose any reading,
# also keeps a short vowel from preceding a long one (a + ū is written ow,
# e + ī is ey), inside a letter's reading too; the letter model, which
# follows attested readings, does better without that rule.
_LATIN_VOWELS = frozenset("aeiouāīū")
_SHORT_VOWELS = frozenset("aeiou")


class PersianOrthography(abjad.Orthography):
    name = "persian"
    letters = LETTERS
    consonants = CONSONANTS
    mark_vowels = _MARK_VOWEL
    sukun = _SUKUN
    shadda = _SHADDA
    marks = _MARKS
    carriers = frozenset("اأإ")
    part_break = ZWNJ

    def letter_options(self, word: str, i: int) -> tuple[str, ...]:
        return letter_options(word, i)

    def slot_options(self, word: str, i: int) -> tuple[str, ...]:
        return slot_options(word, i)

    def forced_vowel_fits(self, vowel: str, letter: str, forced: str) -> bool:
        # Fatha before و or ی is the diphthong BGN writes ow / ey (دَولت Dowlat)
        if forced == "a" and not vowel and letter in ("ow", "ey"):
            return True
        return super().forced_vowel_fits(vowel, letter, forced)

    def allowed(self, prefix: str, out: str, strict: bool = False) -> bool:
        if not out:
            return True
        if strict:
            text = prefix[-1:] + out
            for a, b in zip(text, text[1:], strict=False):
                if a in _SHORT_VOWELS and b in _LATIN_VOWELS:
                    return False
                if a in _LATIN_VOWELS and b in _SHORT_VOWELS:
                    return False
        elif prefix and prefix[-1] in _LATIN_VOWELS and out[0] in _SHORT_VOWELS:
            return False
        if (
            not prefix
            or out[0] in _LATIN_VOWELS
            or any(c in _LATIN_VOWELS for c in prefix)
        ):
            return True
        # Only consonants so far: the word would start with a cluster
        return prefix == "kh" and out.startswith("v")


ORTHOGRAPHY = PersianOrthography()

# Context windows used by the letter model (shared with the trainer)
LEVELS = ORTHOGRAPHY.levels


def context_keys(word: str, i: int) -> list[str]:
    return ORTHOGRAPHY.context_keys(word, i)


# --------------------------------------------------------------------------- #
# Model data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PersianModel:
    version: str
    table: dict[str, str]  # context key -> joint output (vowel + letter)
    lexicon: dict[str, str]  # normalized word or part -> romanization
    # Ezāfe before a following word, as log-odds: a prior, plus how much a
    # word raises or lowers them as the head (with its usual form, "e" or
    # "ye", "" if unknown) and as the word that follows (Kīāsar-e Bālā)
    ezafe_prior: float
    ezafe_head: dict[str, tuple[float, str]]
    ezafe_next: dict[str, float]
    ranked: dict[str, list[str]]  # smallest contexts: readings by frequency


@lru_cache(maxsize=1)
def load_model() -> PersianModel | None:
    """The built model, None when it has not been built. Not checked for
    staleness here (`nte data status` does that)"""
    try:
        with gzip.open(PERSIAN_MODEL_PATH, "rt", encoding="utf-8") as f:
            raw = json.load(f)
        return make_model(raw)
    except (OSError, ValueError, KeyError, TypeError):
        return None  # missing or unreadable: `nte data status` tells which


def make_model(raw: dict) -> PersianModel:
    """Combine the learned data with the hand lexicon (which wins)"""
    lexicon = {**raw["lexicon"], **HAND_LEXICON}
    head = {w: (d, f) for w, (d, f) in raw["ezafe_head"].items()}
    for word in HAND_EZAFE:
        if head.get(word, (0.0, ""))[0] <= 0:
            head[word] = (_HAND_EZAFE_WEIGHT, ezafe_form(lexicon.get(word, "")))
    return PersianModel(
        raw["version"],
        raw["table"],
        lexicon,
        raw["ezafe_prior"],
        head,
        raw["ezafe_next"],
        raw.get("ranked", {}),
    )


_HAND_EZAFE_WEIGHT = 5.0


# --------------------------------------------------------------------------- #
# Hand lexicon: generic terms and names outside Iran, which GeoNames' Iranian
# data does not teach. BGN conventions: e/o short vowels, eh for final silent
# heh, ey/ow diphthongs, īā/īy rather than iyā.
# --------------------------------------------------------------------------- #

HAND_LEXICON: dict[str, str] = {
    # States and administrative units
    "امپراتوری": "emperātūrī", "امپراتور": "emperātūr", "جمهوری": "jomhūrī",
    "پادشاهی": "pādeshāhī", "کشور": "keshvar", "ایالت": "eyālat",
    "ایالات": "eyālāt", "استان": "ostān", "شهرستان": "shahrestān",
    "شهر": "shahr", "بخش": "bakhsh", "دهستان": "dehestān", "روستا": "rūstā",
    "منطقه": "mantaqeh", "ناحیه": "nāhīyeh", "امارات": "emārāt",
    "خلافت": "khelāfat", "سلطنت": "saltanat", "دولت": "dowlat",
    "فدرال": "fedrāl", "متحده": "mottahedeh", "متحد": "mottahed",
    "اسلامی": "eslāmī", "خلق": "khalq", "دموکراتیک": "demokrātīk",
    "سوسیالیستی": "sosīālīstī", "شوروی": "showravī", "اتحاد": "ettehād",
    "جماهیر": "jamāhīr", "پایتخت": "pāytakht", "حکومت": "hokūmat",
    "قلمرو": "qalamrow", "شاهنشاهی": "shāhanshāhī", "دودمان": "dūdmān",
    "سلسله": "selseleh", "آل": "āl",
    # Dynasties and historical peoples
    "هخامنشی": "hakhāmaneshī", "هخامنشیان": "hakhāmaneshīān",
    "اشکانی": "ashkānī", "اشکانیان": "ashkānīān", "ساسانی": "sāsānī",
    "ساسانیان": "sāsānīān", "سامانی": "sāmānī", "سامانیان": "sāmānīān",
    "غزنوی": "ghaznavī", "غزنویان": "ghaznavīān", "سلجوقی": "saljūqī",
    "سلجوقیان": "saljūqīān", "خوارزمشاهی": "khvārazmshāhī",
    "خوارزمشاهیان": "khvārazmshāhīān", "مغول": "moghūl", "ایلخانی": "īlkhānī",
    "ایلخانان": "īlkhānān", "تیموری": "teymūrī", "تیموریان": "teymūrīān",
    "صفوی": "safavī", "صفویه": "safavīyeh", "افشاریه": "afshārīyeh",
    "زندیه": "zandīyeh", "قاجار": "qājār", "قاجاریه": "qājārīyeh",
    "پهلوی": "pahlavī", "عثمانی": "ʿosmānī", "بویه": "būyeh",
    "عباسی": "ʿabbāsī", "عباسیان": "ʿabbāsīān", "اموی": "omavī",
    "امویان": "omavīān", "فاطمی": "fātemī", "فاطمیان": "fātemīān",
    "ماد": "mād", "پارت": "pārt", "سغد": "soghd", "باختر": "bākhtar",
    "بلخ": "balkh", "مرو": "marv", "نیشابور": "neyshābūr",
    # Geography
    "دریا": "daryā", "دریاچه": "daryācheh", "اقیانوس": "oqyānūs",
    "خلیج": "khalīj", "تنگه": "tangeh", "جزیره": "jazīreh", "جزایر": "jazāyer",
    "کوه": "kūh", "کوهستان": "kūhestān", "رشته‌کوه": "reshteh-kūh",
    "رود": "rūd", "رودخانه": "rūdkhāneh", "بیابان": "bīābān", "کویر": "kavīr",
    "دشت": "dasht", "جلگه": "jolgeh", "قاره": "qāreh", "آبشار": "ābshār",
    "جنگل": "jangal", "دره": "darreh", "بندر": "bandar", "شبه‌جزیره": "shebh-e Jazīreh",
    # Directions and common modifiers
    "شمال": "shomāl", "جنوب": "jonūb", "شرق": "sharq", "غرب": "gharb",
    "شمالی": "shomālī", "جنوبی": "jonūbī", "شرقی": "sharqī", "غربی": "gharbī",
    "مرکزی": "markazī", "میانه": "mīāneh", "بزرگ": "bozorg", "کوچک": "kūchek",
    "بالا": "bālā", "پایین": "pāʾīn", "علیا": "ʿolyā", "سفلی": "soflā",
    "وسطی": "vostā", "نو": "now", "کهنه": "kohneh", "قدیم": "qadīm",
    "باستان": "bāstān", "سرخ": "sorkh", "سیاه": "sīāh", "سفید": "sefīd",
    "سبز": "sabz", "عربی": "ʿarabī", "و": "va",
    # Continents, countries, regions
    "آسیا": "āsīā", "اروپا": "orūpā", "آفریقا": "āfrīqā", "آمریکا": "āmrīkā",
    "امریکا": "āmrīkā", "استرالیا": "ostrālīā", "اقیانوسیه": "oqyānūsīyeh",
    "ایران": "īrān", "افغانستان": "afghānestān", "پاکستان": "pākestān",
    "هند": "hend", "هندوستان": "hendūstān", "چین": "chīn", "ژاپن": "zhāpon",
    "کره": "koreh", "روسیه": "rūsīyeh", "ترکیه": "torkīyeh", "عراق": "ʿerāq",
    "سوریه": "sūrīyeh", "لبنان": "lobnān", "اردن": "ordon", "مصر": "mesr",
    "عربستان": "ʿarabestān", "یمن": "yaman", "عمان": "ʿommān", "کویت": "koveyt",
    "قطر": "qatar", "بحرین": "bahreyn", "یونان": "yūnān", "روم": "rūm",
    "ایتالیا": "ītālīā", "اسپانیا": "espānīā", "پرتغال": "porteghāl",
    "فرانسه": "farānseh", "آلمان": "ālmān", "انگلستان": "engelestān",
    "انگلیس": "engelīs", "بریتانیا": "berītānīā", "هلند": "holand",
    "بلژیک": "belzhīk", "اتریش": "otrīsh", "سوئیس": "sūʾīs", "سوئد": "sūʾed",
    "نروژ": "norvezh", "دانمارک": "dānmārk", "لهستان": "lahestān",
    "اوکراین": "ūkrāyn", "ارمنستان": "armanestān", "گرجستان": "gorjestān",
    "آذربایجان": "āzarbāyjān", "ترکمنستان": "torkamanestān",
    "ازبکستان": "ozbakestān", "تاجیکستان": "tājīkestān",
    "قزاقستان": "qazzāqestān", "قرقیزستان": "qerqīzestān",
    "مغولستان": "moghūlestān", "کانادا": "kānādā", "مکزیک": "mekzīk",
    "برزیل": "berezīl", "آرژانتین": "ārzhāntīn", "بیزانس": "bīzāns",
    "خراسان": "khorāsān", "خوارزم": "khvārazm",
    # Cities outside Iran
    "لندن": "landan", "پاریس": "pārīs", "برلین": "berlīn", "مسکو": "mosko",
    "رم": "rom", "آتن": "āten", "قسطنطنیه": "qostantanīyeh",
    "استانبول": "estānbūl", "بغداد": "baghdād", "دمشق": "dameshq",
    "قاهره": "qāhereh", "مکه": "makkeh", "مدینه": "madīneh", "کابل": "kābol",
    "هرات": "herāt", "بخارا": "bokhārā", "سمرقند": "samarqand",
    "دهلی": "dehlī", "اورشلیم": "ūrshalīm", "بیت‌المقدس": "beyt ol-Moqaddas",
    "بندرعباس": "bandar-e ʿAbbās", "خزر": "khazar", "کبیر": "kabīr",
    "ونیز": "venīz", "فلورانس": "florāns", "میلان": "mīlān", "ناپل": "nāpl",
    "لیسبون": "līsbon", "مادرید": "mādrīd", "بارسلونا": "bārselonā",
    "پراگ": "perāg", "ورشو": "varshow", "بوداپست": "būdāpest",
    "آمستردام": "āmsterdām", "بروکسل": "boroksel", "ژنو": "zhenev",
    "استکهلم": "estokholm", "کپنهاگ": "kopenhāg", "اسلو": "oslo",
    "هلسینکی": "helsīnkī", "توکیو": "tokyo", "پکن": "pekan", "سئول": "seʾūl",
    "بمبئی": "bombeʾī", "کراچی": "karāchī", "لاهور": "lāhūr", "باکو": "bākū",
    "تفلیس": "teflīs", "ایروان": "īravān", "تاشکند": "tāshkand",
    "دوشنبه": "dūshanbeh", "آنکارا": "ānkārā", "ازمیر": "ezmīr",
    "قونیه": "qūnīyeh", "حلب": "halab", "بیروت": "beyrūt", "ریاض": "rīāz",
    "دبی": "dobey", "ابوظبی": "abūzabī", "دوحه": "dowheh",
    "شیکاگو": "shīkāgo", "لس‌آنجلس": "los Ānjeles", "تورنتو": "torento",
    "مونترال": "montreāl", "سیدنی": "sīdnī", "ونزوئلا": "venezūʾelā",
    "کلمبیا": "kolombīā", "شیلی": "shīlī", "کوبا": "kūbā",
    "نیجریه": "nījerīyeh", "کانتی": "kāntī",
    # US states, common as qualifiers (ونیز، ایلینوی)
    "آلاباما": "ālābāmā", "آلاسکا": "ālāskā", "آریزونا": "ārīzonā",
    "آرکانزاس": "ārkānzās", "کالیفرنیا": "kālīfornīā", "کلرادو": "kolorādo",
    "دلاویر": "delāver", "فلوریدا": "florīdā", "جورجیا": "jorjīā",
    "هاوایی": "hāvāyī", "آیداهو": "āydāho", "ایلینوی": "īlīnoy",
    "ایندیانا": "īndīānā", "آیووا": "āyovā", "کانزاس": "kānzās",
    "کنتاکی": "kentākī", "لوئیزیانا": "lūʾīzīānā", "مریلند": "merīland",
    "ماساچوست": "māsāchūset", "میشیگان": "mīshīgān", "مینه‌سوتا": "mīnesotā",
    "میسیسیپی": "mīsīsīpī", "میزوری": "mīzūrī", "مونتانا": "montānā",
    "نبراسکا": "nebrāskā", "نوادا": "nevādā", "نیوهمپشایر": "nīū Hampshāyer",
    "نیوجرسی": "nīū Jersī", "نیومکزیکو": "nīū Mekzīko", "نیویورک": "nīū York",
    "کارولینا": "kārolīnā", "داکوتا": "dākotā", "اوهایو": "ohāyo",
    "اوکلاهما": "oklāhomā", "اورگن": "oregon", "پنسیلوانیا": "pensīlvānīā",
    "تنسی": "tenesī", "تگزاس": "tegzās", "یوتا": "yūtā", "ورمانت": "vermont",
    "ویرجینیا": "vīrjīnīā", "واشینگتن": "vāshangton",
    "ویسکانسین": "vīskānsīn", "وایومینگ": "vāyomīng",
}  # fmt: skip

# Heads that take the ezāfe before a following word
HAND_EZAFE = frozenset(
    {
        "امپراتوری", "جمهوری", "پادشاهی", "کشور", "ایالت", "ایالات", "استان",
        "شهرستان", "شهر", "بخش", "دهستان", "منطقه", "ناحیه", "امارات",
        "خلافت", "سلطنت", "دولت", "اتحاد", "جماهیر", "دریا", "دریاچه",
        "اقیانوس", "خلیج", "تنگه", "جزیره", "جزایر", "کوه", "کوهستان",
        "رشته‌کوه", "رود", "رودخانه", "بیابان", "کویر", "دشت", "جلگه",
        "قاره", "جنگل", "دره", "شبه‌جزیره", "اسلامی", "خلق", "متحده",
        "دموکراتیک", "سوسیالیستی", "شوروی", "حکومت", "قلمرو", "شاهنشاهی",
        "دودمان", "سلسله", "آل", "شمال", "جنوب", "شرق", "غرب", "کره",
        "آفریقا", "آمریکا", "امریکا", "آسیا", "اروپا",
    }
)  # fmt: skip


# --------------------------------------------------------------------------- #
# Reading a word
# --------------------------------------------------------------------------- #


def joint_outputs(word: str, i: int) -> list[str]:
    """Every short vowel + letter reading possible for letter i"""
    return ORTHOGRAPHY.joint_outputs(word, i)


def romanize_letters(model: PersianModel, word: str) -> list[str]:
    """Letter-model reading of one normalized word (no lexicon), one output
    per letter of the word without harakat"""
    return abjad.read_letters(ORTHOGRAPHY, model, word)


def part_spans(word: str) -> list[tuple[int, int]]:
    """Where morphemes visibly start in a word without harakat: after a
    zero-width non-joiner, and at a medial آ (خرم‌آباد, حیدرآباد). Returns
    (start, end) index pairs; the ZWNJ itself belongs to no part"""
    spans: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(word):
        if ch == ZWNJ:
            if i > start:
                spans.append((start, i))
            start = i + 1
        elif ch == "آ" and i > start:
            spans.append((start, i))
            start = i
    if start < len(word):
        spans.append((start, len(word)))
    return spans


@dataclass(frozen=True)
class WordReading:
    text: str  # lowercase romanization
    attested: bool  # from a lexicon, not the letter model
    ezafe: bool  # the ezāfe is written (ۀ, a final kasra)


def read_word(model: PersianModel, word: str) -> WordReading:
    bare = "".join(ch for ch in word if ch not in _MARKS)
    written_ezafe = False
    if bare.endswith("ه" + EZAFE_MARK):
        bare = bare[:-1]
        word = word.replace(EZAFE_MARK, "")
        written_ezafe = True
    if word.endswith(_KASRA) and len(bare) > 1 and bare[-1] in CONSONANTS:
        written_ezafe = True  # گالِشِ بالا Gālesh-e Bālā
    # Written vowels outrank the lexicon, which may record another reading
    voweled = any(ch in _MARK_VOWEL for ch in word)

    if bare in model.lexicon and not voweled:
        return WordReading(model.lexicon[bare], True, written_ezafe)
    # Written ezāfe after a final long ā or ū: دریای = دریا + -ye
    if bare.endswith(("ای", "وی")) and bare[:-1] in model.lexicon and not voweled:
        return WordReading(model.lexicon[bare[:-1]], True, True)

    # The letter model reads the whole word (its contexts cross morpheme
    # boundaries, as in training); parts found in the lexicon replace it
    letters = romanize_letters(model, word)
    pieces: list[str] = []
    attested = True
    for start, end in part_spans(bare):
        part = bare[start:end]
        if part in model.lexicon and (start, end) != (0, len(bare)) and not voweled:
            pieces.append(model.lexicon[part])
        else:
            pieces.append("".join(letters[start:end]))
            attested = False
    return WordReading(_join_parts(pieces), attested, written_ezafe)


def _join_parts(pieces: list[str]) -> str:
    out = ""
    for p in pieces:
        # A vowel-initial part after a vowel keeps the syllables apart
        if out and p and out[-1] in _LATIN_VOWELS and p[0] in _LATIN_VOWELS:
            out += "ʾ"
        out += p
    return out


# --------------------------------------------------------------------------- #
# Reading guided by the entity's own Latin names
# --------------------------------------------------------------------------- #
#
# A Persian spelling allows many readings (ونیز: vanīz, venīz, vonīz...). When
# the entity has Latin-script names (Venice), the reading closest to one of
# them is taken, if the letters clearly correspond: its distance to the
# hint word must stay within a third of the word's length. For Iranian places
# the hint is the BGN name itself, so the result is attested; for foreign
# places the vowels follow the original name.

_GUIDE_MAX_RATIO = 0.2
_GUIDE_MAX_COST = 3.0
_GUIDE_COSTS = abjad.GuideCosts()


@dataclass(frozen=True)
class HintWord:
    text: str  # folded
    ezafe: str | None  # "e"/"ye" written after it, "" if none, None if last
    # The hint is itself a romanization of Persian (macrons, ‘ ’: BGN), so
    # its doubled letters are real; in a foreign name (Illinois) they are not
    romanized: bool = False
    source: str = ""  # the hint as given, for warnings


_ROMANIZATION_MARKS = frozenset("āīūĀĪŪ‘’ʿʾ")


def hint_words(hints: tuple[str, ...]) -> list[list[HintWord]]:
    """Each hint as words with the ezāfe written after them:
    Bandar-e ʿAbbās -> [bandar (e), abbas]"""
    out: list[list[HintWord]] = []
    for hint in hints:
        romanized = any(ch in _ROMANIZATION_MARKS for ch in hint)
        tokens = [fold_latin(t) for t in re.split(r"[\s\-]+", hint)]
        words: list[HintWord] = []
        for t in tokens:
            if not t:
                continue
            if t in ("e", "ye") and words:
                words[-1] = HintWord(words[-1].text, t, romanized, hint)
            else:
                if words and words[-1].ezafe is None:
                    words[-1] = HintWord(words[-1].text, "", romanized, hint)
                words.append(HintWord(t, None, romanized, hint))
        if words:
            out.append(words)
    return out


def guided_reading(
    model: PersianModel, word: str, target: str, romanized: bool = False
) -> tuple[float, str]:
    """The reading of `word` (no harakat) closest to the folded hint word
    `target`: (cost, reading); see `abjad.guided_reading`"""
    cost, path = abjad.guided_reading(
        ORTHOGRAPHY,
        word,
        target,
        romanize_letters(model, word),
        romanized,
        _GUIDE_COSTS,
        max_cost=min(_GUIDE_MAX_COST, _GUIDE_MAX_RATIO * len(target)),
    )
    return cost, "".join(path)


def _accepts(cost: float, target: str) -> bool:
    return cost <= _GUIDE_MAX_COST and cost <= _GUIDE_MAX_RATIO * len(target)


@dataclass(frozen=True)
class _Guide:
    base: str  # the word matched, without a written ezāfe ی
    reading: str
    hint: str  # the hint the word was matched to, as given
    ezafe: str | None  # ezāfe written after it in the hint
    written_ezafe: bool  # the Persian writes one: کارولینای = Kārolīnā-ye


def _guide_word(
    model: PersianModel, word: str, candidates: list[HintWord]
) -> _Guide | None:
    bare = "".join(ch for ch in word if ch not in _MARKS)
    if not bare or any(ch in _MARK_VOWEL for ch in word):
        return None
    # A final ی after ا or و may be a written ezāfe (دریای): try without it
    forms = [(bare, False)]
    if len(bare) > 2 and bare.endswith(("ای", "وی")):
        forms.append((bare[:-1], True))
    best: tuple[float, _Guide] | None = None
    for hw in candidates:
        if not hw.text:
            continue
        for form, written in forms:
            cost, reading = guided_reading(model, form, hw.text, hw.romanized)
            if _accepts(cost, hw.text) and (best is None or cost < best[0]):
                guide = _Guide(form, reading, hw.source, hw.ezafe, written)
                best = (cost, guide)
    return best[1] if best else None


def _hint_candidates(hints: tuple[str, ...]) -> list[HintWord]:
    """Hint words, plus adjacent pairs run together (نیویورک ~ New York)"""
    out: list[HintWord] = []
    for words in hint_words(hints):
        out.extend(words)
        for a, b in zip(words, words[1:], strict=False):
            out.append(HintWord(a.text + b.text, b.ezafe, b.romanized, b.source))
    return out


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

_CONJUNCTION = "و"
_SUFFIX_WORDS = frozenset({"آباد"})  # written apart but joined in BGN


def _is_word_char(ch: str) -> bool:
    return abjad.is_word_char(ch) or ch == ZWNJ


def _tokenize(text: str) -> list[tuple[bool, str]]:
    return abjad.tokenize(text, _is_word_char)


def _merge_suffix_words(tokens: list[tuple[bool, str]]) -> list[tuple[bool, str]]:
    """Join words like آباد written apart to the word before: خرم آباد"""
    out: list[tuple[bool, str]] = []
    for tok in tokens:
        if (
            tok[0]
            and tok[1] in _SUFFIX_WORDS
            and len(out) >= 2
            and out[-2][0]
            and out[-1][1].isspace()
        ):
            out.pop()
            out[-1] = (True, out[-1][1] + ZWNJ + tok[1])
        else:
            out.append(tok)
    return out


_separator = abjad.separator


def ezafe_form(roman: str) -> str:
    """-ye after a vowel or a silent final heh (Daryā-ye, Qalʿeh-ye), -e
    after a consonant (Shahr-e); ده Deh-e, a pronounced h, needs the lexicon"""
    return "ye" if roman.endswith(("ā", "ī", "ū", "o", "a", "eh")) else "e"


def ezafe_between(
    model: PersianModel, word: str, reading: WordReading, nxt: str
) -> str:
    """The ezāfe to write after `word` before `nxt`: "e", "ye" or "" """
    bare = "".join(ch for ch in word if ch not in _MARKS)
    head_delta, form = model.ezafe_head.get(bare, (0.0, ""))
    if not reading.ezafe:
        nxt_bare = "".join(ch for ch in nxt if ch not in _MARKS)
        score = model.ezafe_prior + head_delta + model.ezafe_next.get(nxt_bare, 0.0)
        if score <= 0:
            return ""
    return form or ezafe_form(reading.text)


def romanize_persian(text: str, ctx: Context) -> Rendering | None:
    if ctx.lang != "fa":
        return None
    model = load_model()
    if model is None:
        # The generic Arabic-script rules take over
        raise ProviderUnavailable(
            "Persian model not built or unreadable; run `nte init`"
        )
    return romanize_with_model(model, text, ctx.hints)


def romanize_with_model(
    model: PersianModel, text: str, hints: tuple[str, ...] = ()
) -> Rendering:
    tokens = _merge_suffix_words(_tokenize(normalize(text).strip()))
    candidates = _hint_candidates(hints)
    word_idx = [k for k, (is_word, _) in enumerate(tokens) if is_word]

    readings: dict[int, WordReading] = {}
    guides: dict[int, _Guide] = {}
    guided: set[int] = set()  # words whose reading comes from a hint
    for k in word_idx:
        word = tokens[k][1]
        if word == _CONJUNCTION:
            continue
        reading = read_word(model, word)
        guide = _guide_word(model, word, candidates) if candidates else None
        if guide is not None:
            guides[k] = guide
            # The hand lexicon keeps its reading (لندن Landan, not London);
            # the match still tells where the hint writes an ezāfe
            if guide.base not in HAND_LEXICON:
                written = reading.ezafe or guide.written_ezafe
                reading = WordReading(guide.reading, True, written)
                guided.add(k)
        readings[k] = reading

    out: list[str] = []
    guessed = False
    for pos, (is_word, tok) in enumerate(tokens):
        if not is_word:
            out.append(_separator(tok))
            continue
        if tok == _CONJUNCTION:
            out.append("va")
            continue
        reading = readings[pos]
        guessed |= not reading.attested
        roman = capitalize_first(reading.text)
        nxt = pos + 2
        if (
            nxt < len(tokens)
            and tokens[pos + 1][1].isspace()
            and tokens[nxt][0]
            and tokens[nxt][1] != _CONJUNCTION
            and any(ch.isalpha() for ch in reading.text)
        ):
            form = _guided_ezafe(guides.get(pos), guides.get(nxt), reading)
            if form is None:
                form = ezafe_between(model, tok, reading, tokens[nxt][1])
            if form:
                roman += "-" + form
        out.append(roman)

    result = "".join(out).strip()
    engine = f"persian model {model.version}"
    warnings: tuple[str, ...] = ()
    if guided:
        matched = ", ".join(dict.fromkeys(guides[k].hint for k in sorted(guided)))
        warnings += (f"vowels chosen to match the entity's name ({matched})",)
    if guessed:
        warnings += ("short vowels of unlisted words predicted by a letter model",)
        return Rendering(result, "nte:persian-bgn", "low", warnings, engine=engine)
    return Rendering(result, "nte:persian-bgn", "normal", warnings, engine=engine)


def _guided_ezafe(
    guide: "_Guide | None", next_guide: "_Guide | None", reading: WordReading
) -> str | None:
    """The ezāfe as the hint writes it, when both words were matched to it"""
    if reading.ezafe or guide is None or next_guide is None or guide.ezafe is None:
        return None
    return guide.ezafe
