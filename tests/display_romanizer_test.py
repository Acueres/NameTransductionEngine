"""Golden cases for the display romanizer: (text, lang tag, expected).

Run: uv run python tests/display_romanizer_test.py   (prints mismatches)
"""

import sys

from name_transduction_engine.transliteration.display_romanizer import DisplayRomanizer

CASES: list[tuple[str, str | None, str]] = [
    # Latin and neutral input
    ("München", "de", "München"),
    ("123", None, "123"),
    ("", None, ""),
    # Cyrillic
    ("Москва", "ru", "Moskva"),
    ("Санкт-Петербург", "ru", "Sankt-Peterburg"),
    ("Ярославль", "ru", "Yaroslavl"),
    ("Нижний Новгород", "ru", "Nizhniy Novgorod"),
    ("Київ", "uk", "Kyiv"),
    ("Запоріжжя", "uk", "Zaporizhzhia"),
    ("Львів", "uk", "Lviv"),
    ("Згурівка", "uk", "Zghurivka"),
    ("Яготин", "uk", "Yahotyn"),
    ("Гродна", "be", "Hrodna"),
    ("Пловдив", "bg", "Plovdiv"),
    ("Београд", "sr", "Beograd"),
    ("Чачак", "sr", "Čačak"),
    ("Алматы", "kk", "Almaty"),
    ("Улаанбаатар", "mn", "Ulaanbaatar"),
    ("Тошкент", "uz", "Toshkent"),
    ("Душанбе", "tg", "Dushanbe"),
    ("Казань", "tt", "Kazan"),
    ("Москва", None, "Moskva"),
    # Greek, Armenian, Georgian
    ("Αθήνα", "el", "Athina"),
    ("Θεσσαλονίκη", "el", "Thessaloniki"),
    ("Χανιά", "el", "Chania"),
    ("Երևան", "hy", "Yerevan"),
    ("თბილისი", "ka", "Tbilisi"),
    ("მცხეთა", "ka", "Mtskheta"),
    # Japanese
    ("東京", "ja", "Tōkyō"),
    ("東京都", "ja", "Tōkyō-to"),
    ("大阪府", "ja", "Ōsaka-fu"),
    ("北海道", "ja", "Hokkaidō"),
    ("横浜市", "ja", "Yokohama-shi"),
    ("新宿区", "ja", "Shinjuku-ku"),
    ("神戸", "ja", "Kōbe"),
    ("札幌", "ja", "Sapporo"),
    ("日本橋", "ja", "Nihonbashi"),
    ("三宮", "ja", "Sannomiya"),
    ("とうきょう", "ja", "Tōkyō"),
    ("さいたま市", "ja", "Saitama-shi"),
    ("霞ヶ関", "ja", "Kasumigaseki"),
    ("六本木ヒルズ", "ja", "Roppongi Hiruzu"),
    # Japanese word division: words split, toponyms kept whole
    ("東ローマ帝国", "ja", "Higashi Rōma Teikoku"),
    ("神聖ローマ帝国", "ja", "Shinsei Rōma Teikoku"),
    ("大日本帝国", "ja", "Dainippon Teikoku"),
    ("関西国際空港", "ja", "Kansai Kokusai Kūkō"),
    ("国立国会図書館", "ja", "Kokuritsu Kokkai Toshokan"),
    ("南アフリカ共和国", "ja", "Minami Afurika Kyōwakoku"),
    ("アメリカ合衆国", "ja", "Amerika Gasshūkoku"),
    ("伏見稲荷大社", "ja", "Fushimi Inari Taisha"),
    ("日光東照宮", "ja", "Nikkō Tōshō-gū"),
    ("東京都千代田区", "ja", "Tōkyō-to Chiyoda-ku"),
    ("東京ビッグサイト", "ja", "Tōkyō Biggusaito"),
    ("東大阪市", "ja", "Higashiōsaka-shi"),
    ("西東京市", "ja", "Nishitōkyō-shi"),
    ("天の橋立", "ja", "Amanohashidate"),
    ("御茶ノ水", "ja", "Ochanomizu"),
    ("ボスニア・ヘルツェゴビナ", "ja", "Bosunia Herutsegobina"),
    ("さいたまスーパーアリーナ", "ja", "Saitama Sūpāarīna"),
    # Katakana pieces of one foreign name stay one word
    ("レンツブルク", "ja", "Rentsuburuku"),
    ("スファックス", "ja", "Sufakkusu"),
    # Spoken Arabic varieties use the Arabic rules
    ("ياليسوس", "arz", "Yālīsūs"),
    ("東京-大阪", "ja", "Tōkyō-Ōsaka"),
    # Japanese kana, read over the whole word: sokuon and small kana across
    # the analyzer's token boundaries
    ("ブラックリー", "ja", "Burakkurī"),
    ("サトゥ・マーレ", "ja", "Satu Māre"),
    ("ウィッチェ", "ja", "Witche"),
    # extended katakana composed by rule; a small vowel spelling a syllable
    # the syllabary already has is a vowel of its own
    ("リトムニェジツェ", "ja", "Ritomunyejitse"),
    ("スィノプ", "ja", "Sinopu"),
    ("ブィドゴシュチュ", "ja", "Buidogoshuchu"),
    ("クラスノダル", "ja", "Kurasunodaru"),  # not "Kurasu no Daru"
    ("アピンへダム", "ja", "Apinhedamu"),  # hiragana へ typed for katakana ヘ
    # long vowels: UniDic's pronunciation, katakana as written but for ou
    ("大野", "ja", "Ōno"),
    ("井上", "ja", "Inoue"),
    ("ヴァードウス", "ja", "Vādōsu"),
    ("クズルウルマク", "ja", "Kuzuruurumaku"),
    ("しおやまち", "ja", "Shioyamachi"),
    # generic terms after a name: hyphenated, read as in a compound
    ("エルベ川", "ja", "Erube-gawa"),
    ("利根川", "ja", "Tone-gawa"),
    ("キトノス島", "ja", "Kitonosu-tō"),
    ("本州島", "ja", "Honshū-tō"),
    ("琵琶湖", "ja", "Biwa-ko"),
    ("東京駅", "ja", "Tōkyō-eki"),
    ("大阪城", "ja", "Ōsaka-jō"),
    ("チワワ州", "ja", "Chiwawa-shū"),
    ("キョンサン道", "ja", "Kyonsan-dō"),
    ("ハプスブルク家", "ja", "Hapusuburuku-ke"),
    ("ナイル川デルタ", "ja", "Nairu-gawa Deruta"),
    ("クラショヴァ人", "ja", "Kurashovajin"),  # written solid, read -jin
    ("Dingwall市", "ja", "Dingwall-shi"),
    # particles between words, prefixes before katakana
    ("アテナイのアクロポリス", "ja", "Atenai no Akuroporisu"),
    ("古代エジプトの都市", "ja", "Kodai Ejiputo no Toshi"),
    ("高アトラス", "ja", "Kō Atorasu"),
    # a Chinese generic term marks a Chinese name: Sino-Japanese readings
    ("吉隆鎮", "ja", "Kichiryū-chin"),
    ("外務省", "ja", "Gaimushō"),  # a word the dictionary knows
    ("壌塘県", "ja", "壌塘県"),  # no reading, no sign of a Chinese name
    ("信濃", "ja", "Shinano"),
    ("群馬", "ja", "Gunma"),
    ("東京", None, "Dongjing"),  # no tag, no kana: Chinese
    # Chinese
    ("北京", "zh", "Beijing"),
    ("北京市", "zh", "Beijing Shi"),
    ("厦门", "zh", "Xiamen"),
    ("西安", "zh", "Xi'an"),
    ("六安", "zh", "Lu'an"),
    ("河北省", "zh", "Hebei Sheng"),
    ("张家口市", "zh", "Zhangjiakou Shi"),
    ("中山", "zh", "Zhongshan"),
    ("广西壮族自治区", "zh", "Guangxi Zhuangzu Zizhiqu"),
    ("臺北", "zh-Hant", "Taibei"),
    ("绿春", "zh", "Lüchun"),
    # Korean
    ("서울", "ko", "Seoul"),
    ("종로", "ko", "Jongno"),
    ("종로구", "ko", "Jongno-gu"),
    ("신라", "ko", "Silla"),
    ("한라산", "ko", "Hallasan"),
    (
        "왕십리",
        "ko",
        "Wangsip-ri",
    ),  # administrative -ri: hyphen, no assimilation across it
    ("청량리", "ko", "Cheongnyang-ri"),
    ("선릉", "ko", "Seolleung"),
    ("속리산", "ko", "Songnisan"),
    ("압록강", "ko", "Amnokgang"),
    ("별내", "ko", "Byeollae"),
    ("설악산", "ko", "Seoraksan"),
    ("묵호", "ko", "Mukho"),
    ("울릉도", "ko", "Ulleungdo"),
    ("독도", "ko", "Dokdo"),
    ("경기도", "ko", "Gyeonggi-do"),
    ("충청북도", "ko", "Chungcheongbuk-do"),
    ("삼죽면", "ko", "Samjuk-myeon"),
    ("대구", "ko", "Daegu"),
    ("중구", "ko", "Jung-gu"),
    ("부산광역시", "ko", "Busan-gwangyeoksi"),
    ("여의도", "ko", "Yeouido"),
    ("의정부", "ko", "Uijeongbu"),
    ("평양", "ko-KP", "P’yŏngyang"),
    # Indic
    ("राजस्थान", "hi", "Rajasthan"),
    ("भरतपुर", "hi", "Bharatpur"),
    ("कानपुर", "hi", "Kanpur"),
    ("गोरखपुर", "hi", "Gorakhpur"),
    ("लखनऊ", "hi", "Lakhnau"),
    ("पटना", "hi", "Patna"),
    ("अहमदाबाद", "hi", "Ahmadabad"),
    ("आगरा", "hi", "Agra"),
    ("दिल्ली", "hi", "Dilli"),
    ("नई दिल्ली", "hi", "Nai Dilli"),
    ("सिंह", "hi", "Singh"),
    ("मुंबई", "mr", "Mumbai"),
    ("पुणे", "mr", "Pune"),
    ("श्रीनगर", "hi", "Shrinagar"),
    ("ऋषिकेश", "hi", "Rishikesh"),
    ("ढाका", "bn", "Dhaka"),
    ("ঢাকা", "bn", "Dhaka"),
    ("সিলেট", "bn", "Silet"),
    ("খুলনা", "bn", "Khulna"),
    ("ਅੰਮ੍ਰਿਤਸਰ", "pa", "Amritsar"),
    ("ਲੁਧਿਆਣਾ", "pa", "Ludhiana"),
    ("સુરત", "gu", "Surat"),
    ("ଭୁବନେଶ୍ୱର", "or", "Bhubaneshwar"),
    ("சென்னை", "ta", "Chennai"),
    ("மதுரை", "ta", "Madurai"),
    ("தஞ்சாவூர்", "ta", "Thanjavur"),
    ("கோயம்புத்தூர்", "ta", "Koyambuttur"),
    ("சிதம்பரம்", "ta", "Chidambaram"),
    ("விஜயவாడ", "te", None),  # mixed Tamil/Telugu junk: must not crash
    ("విజయవాడ", "te", "Vijayavada"),
    ("ಬೆಂಗಳೂರು", "kn", "Bengaluru"),
    ("ಮೈಸೂರು", "kn", "Maisuru"),
    ("തിരുവനന്തപുരം", "ml", "Thiruvananthapuram"),
    ("കൊച്ചി", "ml", "Kochchi"),
    # Arabic script. Arabic: BGN/PCGN without dots, learned model (needs the
    # Arabic model built by `nte init`)
    ("بغداد", "ar", "Baghdād"),
    ("الرياض", "ar", "Ar-Riyād"),
    ("القاهرة", "ar", "Al-Qāhirah"),
    ("دمشق", "ar", "Dimashq"),
    ("القَاهِرَة", "ar", "Al-Qāhirah"),
    ("دِمَشْق", "ar", "Dimashq"),
    ("مكة المكرمة", "ar", "Makkah al-Mukarramah"),
    ("مدينة الكويت", "ar", "Madīnat al-Kuwayt"),
    ("المدينة المنورة", "ar", "Al-Madīnah al-Munawwarah"),
    ("المملكة العربية السعودية", "ar", "Al-Mamlakah al-ʿArabīyah as-Saʿūdīyah"),
    ("محافظة القاهرة", "ar", "Muhāfazat al-Qāhirah"),
    ("شرم الشيخ", "ar", "Sharm ash-Shaykh"),
    ("دير الزور", "ar", "Dayr az-Zawr"),
    ("الدار البيضاء", "ar", "Ad-Dār al-Baydāʾ"),
    ("اللاذقية", "ar", "Al-Lādhiqīyah"),
    ("الجنوبية", "ar", "Al-Janūbīyah"),
    ("أسوان", "ar", "Aswān"),
    ("التّين", "ar", "At-Tīn"),  # shadda of the assimilated article
    ("عَبْدُ اللّٰه", "ar", "ʿAbd Allāh"),
    ("القاهرة", None, "Al-Qāhirah"),
    ("تهران", "fa", None),
    ("شیراز", "fa", "Shīrāz"),
    ("تبریز", "fa", "Tabrīz"),
    ("مشهد", "fa", "Mashhad"),
    ("کراچی", "ur", "Karāchī"),
    ("اسلام آباد", "ur", "Islām Ābād"),
    ("ئۈرۈمچى", "ug", "Ürümchi"),
    ("قەشقەر", "ug", "Qeshqer"),
    ("هەولێر", "ckb", "Hewlêr"),
    ("سلێمانی", "ckb", "Slêmanî"),
    # Hebrew
    ("תל אביב", "he", None),
    ("חיפה", "he", None),
    ("אילת", "he", "Eilat"),
    ("יְרוּשָׁלַיִם", "he", "Yerushalayim"),
    ("תֵּל אָבִיב", "he", "Tel Aviv"),
    ("ווילנע", "yi", "Vilne"),
    # Thai
    ("เชียงใหม่", "th", "Chiangmai"),
    ("ขอนแก่น", "th", "Khonkaen"),
    ("ภูเก็ต", "th", "Phuket"),
    ("พัทยา", "th", "Phatya"),
    ("หาดใหญ่", "th", "Hatyai"),
    ("อุดรธานี", "th", "Udonthani"),
    ("ชลบุรี", "th", "Chonburi"),
    ("สุโขทัย", "th", "Sukhothai"),
    ("นนทบุรี", "th", "Nonthaburi"),
    ("สมุทรปราการ", "th", "Samutprakan"),
    ("หัวหิน", "th", "Huahin"),
    ("กรุงเทพมหานคร", "th", "Krungthepmahanakhon"),
    ("ลำปาง", "th", "Lampang"),
    ("ปทุมธานี", "th", "Pathumthani"),
    ("จันทบุรี", "th", "Chanthaburi"),
    ("กาญจนบุรี", "th", "Kanchanaburi"),
    ("สุพรรณบุรี", "th", "Suphanburi"),
    ("นครพนม", "th", "Nakhonphanom"),
    ("สกลนคร", "th", "Sakonnakhon"),
    ("นครสวรรค์", "th", "Nakhonsawan"),
    ("สุราษฎร์ธานี", "th", "Suratthani"),
    ("เกาะสมุย", "th", "Kosamui"),
    ("นราธิวาส", "th", "Narathiwat"),
    ("สีลม", "th", "Silom"),
    ("สาทร", "th", "Sathon"),
    ("สมุทรสาคร", "th", "Samutsakhon"),
    ("ฉะเชิงเทรา", "th", "Chachoengsao"),
    ("จังหวัด", "th", "Changwat"),
    ("ธนบุรี", "th", "Thonburi"),
    # Others
    ("አዲስ አበባ", "am", None),
    ("ጎንደር", "am", "Gonder"),
    ("މާލެ", "dv", "Maale"),
    ("ཐིམ་ཕུ", "dz", "Thimphu"),  # uroman fallback
    ("ᐃᖃᓗᐃᑦ", "iu", None),
    ("ရန်ကုန်", "my", None),
    # Mixed and edge cases
    ("富士山", "ja", "Fuji-san"),
    ("新大阪", "ja", "Shin'ōsaka"),
    ("東京", "ko", "東京"),  # hanja: no Korean reader, native kept
    ("कच्छ", "gu", "Kachchh"),
    ("ज्ञानपुर", "hi", "Gyanpur"),
    ("मित्र", "hi", "Mitra"),
    ("ময়মনসিংহ", "bn", "Maymansingh"),
    ("തൃശ്ശൂർ", "ml", "Thrissur"),
    ("کرمانشاه", "fa", "Kermānshāh"),
    # Persian: lexicon, letter model, ezāfe
    ("امپراتوری بیزانس", "fa", "Emperātūrī-ye Bīzāns"),
    ("یزد", "fa", "Yazd"),
    ("رشت", "fa", "Rasht"),
    ("کرج", "fa", "Karaj"),
    ("کابل", "fa", "Kābol"),
    ("ارومیه", "fa", "Orūmīyeh"),
    ("خرم‌آباد", "fa", "Khorramābād"),
    ("خرم آباد", "fa", "Khorramābād"),
    ("بندرعباس", "fa", "Bandar-e ʿAbbās"),
    ("دریای خزر", "fa", "Daryā-ye Khazar"),
    ("خلیج فارس", "fa", "Khalīj-e Fārs"),
    ("جمهوری اسلامی ایران", "fa", "Jomhūrī-ye Eslāmī-ye Īrān"),
    ("ایالات متحده آمریکا", "fa", "Eyālāt-e Mottahedeh-ye Āmrīkā"),
    ("آل بویه", "fa", "Āl-e Būyeh"),
    ("بیت‌المقدس", "fa", "Beyt ol-Moqaddas"),
    ("نَسل", "fa", "Nasl"),
    ("اِمامزادِه هَفت بَرادَر", "fa", "Emāmzādeh Haft Barādar"),
    ("گالِشِ بالا", "fa", "Gālesh-e Bālā"),
    ("ونیز، ایلینوی", "fa", "Venīz, Īlīnoy"),
    ("برلین (آلمان)", "fa", "Berlīn (Ālmān)"),
    ("شهرک ۲۲ بهمن", "fa", "Shahrak-e 22 Bahman"),
    ("حیدرآباد", "ur", "Hīdrābād"),
    ("عمان", "ar", "ʿAmmān"),
    ("بَيْرُوت", "ar", "Bayrūt"),
    ("مَكَّة", "ar", "Makkah"),
    ("חֵיפָה", "he", "Heifa"),
    ("בְּאֵר שֶׁבַע", "he", "Be'er Sheva"),
    ("אמסטערדאם", "yi", "Amsterdam"),
    ("Café 東京", "ja", "Café Tōkyō"),
    ("東京 Tower", "ja", "Tōkyō Tower"),
    ("Москва-2", "ru", "Moskva-2"),
    ("(Київ)", "uk", "(Kyiv)"),
    ("  Москва  ", "ru", "  Moskva  "),
    ("МОСКВА", "ru", "MOSKVA"),
    ("ЩУКА", "ru", "SHCHUKA"),
    ("Щука", "ru", "Shchuka"),
    ("Алматы / Almaty", "kk", "Almaty / Almaty"),
    ("서울 (Seoul)", "ko", "Seoul (Seoul)"),
    ("丿乀", "ja", "丿乀"),  # no reading: native kept
    ("🙂", None, "🙂"),
    ("Київ", None, "Kyiv"),
    ("Київ", "x-foo", "Kyiv"),
]


# Cases with reading hints: other names of the same entity
HINT_CASES: list[tuple[str, str | None, tuple[str, ...], str]] = [
    ("ونیس، فلوریدا", "fa", ("Venice",), "Venīs, Florīdā"),
    ("ونیس", "fa", (), "Vanīs"),  # no hint: the letter model's guess
    ("مونیخ", "fa", ("Munich", "München"), "Mūnīkh"),
    ("بندر عباس", "fa", ("Bandar-e ‘Abbās", "Bandar Abbas"), "Bandar-e ʿAbbās"),
    ("کارولینای شمالی", "fa", ("North Carolina",), "Kārolīnā-ye Shomālī"),
    ("لندن", "fa", ("London",), "Landan"),  # the hand lexicon wins
    ("آلمان", "fa", ("Germany",), "Ālmān"),  # unrelated hint: ignored
    # Arabic: a BGN name settles the vowels and the construct state...
    ("الداخلة", "ar", ("Al Wāḩāt ad Dākhilah", "Dakhla Oasis"), "Ad-Dākhilah"),
    ("وحدة الضحاكي", "ar", ("Waḩdat aḑ Ḑaḩākī",), "Wahdat ad-Dahākī"),
    ("جبل أم حَطين", "ar", ("Jabal Umm Ḩaţţīn",), "Jabal Umm Hattīn"),
    # ...a foreign name its vowels, read as the Arabic letters allow
    ("نوتنغهامشير", "ar", ("Nottinghamshire",), "Nūtinghhāmshīr"),
    ("ستوكهولم", "ar", ("Stockholm",), "Stūkhūlm"),
    ("ستراين", "ar", ("Strijen", "Gemeente Strijen"), "Strāyin"),
    ("تشاسلاف", "ar", ("Čáslav",), "Tshāslāf"),  # تش as one sound
    ("ميلاتسو", "ar", ("Milazzo",), "Mīlātsū"),
    ("بيرتينورو", "ar", ("Bertinoro",), "Bīrtīnūrū"),
    ("سانتا جوستا", "ar", ("Santa Giusta",), "Sāntā Jūstā"),
    ("البسيط", "ar", ("Albacete",), "Al-Basīt"),  # the hint writes the article in
    ("السويس", "ar", ("Suez",), "As-Suways"),  # unrelated hint: ignored
    # A written و or ي is never a short vowel
    ("أديس أبابا", "ar", ("Addis Ababa",), "Adīs Abābā"),
    ("كوروني", "ar", ("Koroni",), "Kūrūnī"),
    ("فرنسا", "ar", ("France",), "Faransā"),  # the hand lexicon wins
    # Japanese: the reading of a kanji name that spells the entity's name...
    ("羽田空港", "ja", ("Haneda Airport",), "Haneda Kūkō"),
    ("国立市", "ja", ("Kunitachi",), "Kunitachi-shi"),
    ("清水寺", "ja", ("Kiyomizu-dera",), "Kiyomizu-dera"),
    ("石垣島", "ja", ("Ishigaki-jima",), "Ishigaki-jima"),
    ("東京", "ja", ("Tokyo",), "Tōkyō"),
    ("チアパス州", "ja", ("Chiapas",), "Chiapasu-shū"),  # not 州 "su"
    # ...and a pinyin name marks a Chinese place: Sino-Japanese readings,
    # unless the dictionary knows the place's Japanese name
    ("南充市", "ja", ("Nanchong",), "Nanjū-shi"),
    ("九寨溝", "ja", ("Jiuzhaigou Valley",), "Kyūsaikō"),
    ("小金県", "ja", ("Xiaojin Xian",), "Shōkin-ken"),
    ("遼東半島", "ja", ("Liaodong Peninsula",), "Ryōtō Hantō"),
    ("曲麻莱県", "ja", ("Qumalai Xian",), "Kyokumarai-ken"),
    ("北京市", "ja", ("Beijing",), "Pekin-shi"),
    ("青島市", "ja", ("Qingdao",), "Chintao-shi"),
]


def main() -> int:
    r = DisplayRomanizer()
    bad = 0
    cases = [(t, lang, (), want) for t, lang, want in CASES] + HINT_CASES
    for text, lang, hints, expected in cases:
        res = r.romanize(text, lang, hints)
        ok = expected is None or res.text == expected
        mark = "  " if ok else "✗ "
        if not ok:
            bad += 1
        if not ok or "-v" in sys.argv:
            print(
                f"{mark}{text!r:22} {str(lang):8} -> {res.text!r:28}"
                f" want {expected!r:24} [{res.transform}, {res.confidence}] {res.warnings}"
            )
    print(f"{len(cases) - bad}/{len(cases)} as expected")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
