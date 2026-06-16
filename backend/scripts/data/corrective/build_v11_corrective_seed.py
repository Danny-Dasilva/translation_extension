#!/usr/bin/env python3
"""
Build the v11it corrective SFT seed dataset.

Per plan: thoughts/shared/plans/2026-06-14_translation-finetune-scoping.md (sections 4.1-4.2).

Schema MUST match data_v10.parquet exactly:
    [jp:large_string, en:large_string, src:large_string, register_tag:large_string, gold_flag:bool]

This produces the FIRST corrective batch (model-capability error classes the
orchestration fixes can't reach):
  Class 2 -- comparative grammar (より / の方がいい)         [HIGHEST priority]
  Class 1 -- 3rd-person self-reference (mother -> Mommy/I)
  Class 3 -- adult/garment vocab + intimate register
  Class 4 -- confabulation suppression (faithful, no padding)

src convention (plan 4.1): "corrective_v11:<class>:<id>"
gold_flag = True ; register_tag in {vn_eroge, manga_dialog, ...} matching v10.

The English is generated here (strong JP->EN translator, no teacher model).
Phrasing is VARIED so the model learns the PATTERN, not memorized strings.

Output:
  scripts/data/corrective/v11_corrective_seed.parquet
  scripts/data/corrective/v11_corrective_seed.jsonl   (human review)

Does NOT merge into data_v10 or train.
"""
import json
import random
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

random.seed(42)

OUT_DIR = Path(__file__).resolve().parent
PARQUET_OUT = OUT_DIR / "v11_corrective_seed.parquet"
JSONL_OUT = OUT_DIR / "v11_corrective_seed.jsonl"

rows = []  # list of dict(jp, en, src, register_tag, gold_flag)


def add(jp, en, cls, idx, register_tag, gold=True):
    rows.append(
        {
            "jp": jp,
            "en": en,
            "src": f"corrective_v11:{cls}:{idx:05d}",
            "register_tag": register_tag,
            "gold_flag": gold,
        }
    )


# ---------------------------------------------------------------------------
# CLASS 2 -- COMPARATIVE GRAMMAR (HIGHEST priority, drill set ~300-500)
# Construction truth:
#   X より Y (が)いい   = "Y is better than X"  (Y wins, X is the baseline)
#   X の方がいい         = "X is the better one / X is better"  (X wins)
#   X より A の方がいい  = "A is better than X"
# The v10 model INVERTS direction / mis-assigns the subject.
# We over-represent the construction with CORRECT directionality + varied
# vocab and registers so the pattern dominates.
# ---------------------------------------------------------------------------

# (label, possessive/subject-phrasing) for the thing being compared.
# These supply natural English on BOTH sides of the comparison.
COMP_ITEMS = [
    # jp_noun, en_subject(for "X is better"), en_object(for "than X")
    ("姉ちゃん", "doing it with big sis", "big sis"),
    ("お姉ちゃん", "being with big sis", "big sis"),
    ("母さん", "Mom's", "Mom"),
    ("お母さん", "Mom's", "Mom"),
    ("こっち", "this one", "that one"),
    ("そっち", "that one", "this one"),
    ("こっちの方", "this one", "the other"),
    ("前の", "the old one", "the new one"),
    ("こいつ", "this guy", "him"),
    ("お前のやつ", "yours", "the others'"),
    ("コーヒー", "coffee", "tea"),
    ("紅茶", "tea", "coffee"),
    ("電車", "the train", "the bus"),
    ("バス", "the bus", "the train"),
    ("夏", "summer", "winter"),
    ("冬", "winter", "summer"),
    ("犬派", "being a dog person", "being a dog person"),
    ("猫", "the cat", "the cat"),
    ("海", "the sea", "the mountains"),
    ("山", "the mountains", "the sea"),
    ("赤い方", "the red one", "the blue one"),
    ("こっちの服", "this outfit", "that outfit"),
    ("この味", "this flavor", "the other one"),
    ("家", "home", "going out"),
    ("外", "going out", "staying home"),
    ("昼間", "the daytime", "the night"),
    ("夜", "the night", "the daytime"),
    ("私", "mine", "yours"),
    ("あなた", "yours", "mine"),
    ("生", "raw", "cooked"),
    ("右", "the right one", "the left one"),
    ("左", "the left one", "the right one"),
    ("新しいの", "the new one", "the old one"),
    ("安いの", "the cheap one", "the expensive one"),
    ("地味な方", "the plain one", "the flashy one"),
    ("こっちの色", "this color", "that color"),
    ("この方法", "this way", "the other way"),
    ("お風呂", "a bath", "a shower"),
    ("シャワー", "a shower", "a bath"),
    ("ビール", "beer", "wine"),
    ("こっちの店", "this shop", "that one"),
]

# Register-flavored sentence-final variations and their natural EN realizations.
# tag, jp_suffix(after stem), en_template_for_X_better, en_template_for_Y_better_than_X
# We build two families: "Xの方がいい" (X wins) and "XよりY (がいい)" (Y wins).

# ---- Family A: X の方がいい  => "X is better"
A_SUFFIX = [
    ("の方がいい", "{x} is better"),
    ("の方がいいです", "{x} is better."),
    ("の方がいいですぅ!!", "{x} is better!!"),
    ("の方がいいかな", "I think {x} is better"),
    ("の方がいいよ", "{x} is better, you know"),
    ("の方がいいわ", "{x} is better"),
    ("の方がいいわね", "{x} is the better one, isn't it"),
    ("の方がいいだろ", "{x} is better, right?"),
    ("の方が好き", "I like {x} better"),
    ("の方が好きだなぁ", "I like {x} better"),
    ("の方がずっといい", "{x} is way better"),
    ("の方が断然いい", "{x} is by far better"),
    ("の方がマシ", "{x} is better than nothing"),
    ("の方が気持ちいい♡", "{x} feels better ♡"),
    ("の方が好きなの♡", "I like {x} better ♡"),
]

# ---- Family B: X より (も) ... いい  => "better than X" / "Y is better than X"
B_SUFFIX = [
    ("よりいい", "better than {x}"),
    ("よりいいです", "It's better than {x}."),
    ("よりいいですぅ!!", "This is better than {x}!!"),
    ("よりいいよ", "It's better than {x}"),
    ("よりずっといい", "way better than {x}"),
    ("よりマシだ", "better than {x}, at least"),
    ("よりこっちの方がいい", "this one is better than {x}"),
    ("よりそっちの方がいい", "that one is better than {x}"),
    ("よりいいに決まってる", "Of course it's better than {x}"),
    ("よりも気持ちいい♡", "It feels better than {x} ♡"),
    ("よりこっちの方が好き", "I like this better than {x}"),
    ("より楽しい", "more fun than {x}"),
    ("より全然いい", "totally better than {x}"),
    ("よりおいしい", "tastier than {x}"),
    ("より大事だ", "more important than {x}"),
]

# ---- Family C: X より Y の方がいい  => "Y is better than X" (both named)
C_TEMPLATES = [
    ("{xj}より{yj}の方がいい", "{ye} is better than {xe}"),
    ("{xj}より{yj}の方がいいです", "{ye} is better than {xe}."),
    ("{xj}より{yj}の方が好き", "I like {ye} better than {xe}"),
    ("{xj}より{yj}の方がずっといい", "{ye} is way better than {xe}"),
    ("{xj}より{yj}の方が気持ちいい♡", "{ye} feels better than {xe} ♡"),
    ("{xj}より{yj}がいい", "{ye} is better than {xe}"),
    ("{xj}よりも{yj}の方がいいよ", "{ye} is better than {xe}, you know"),
]

cls2_register_cycle = ["manga_dialog", "vn_eroge", "vn", "manga_dialog", "novel"]


def cls2_register(i):
    return cls2_register_cycle[i % len(cls2_register_cycle)]


c2 = 0
# Family A: X-wins  ("{x} is better")
for noun, en_subj, _ in COMP_ITEMS:
    for suf, tmpl in A_SUFFIX:
        # capitalize sentence-initial subject
        subj = en_subj
        en = tmpl.format(x=subj)
        en = en[0].upper() + en[1:]
        add(noun + suf, en, "comparative", c2, cls2_register(c2))
        c2 += 1

# Family B: Y-wins-over-X  ("better than {x}")
for noun, _, en_obj in COMP_ITEMS:
    for suf, tmpl in B_SUFFIX:
        en = tmpl.format(x=en_obj)
        en = en[0].upper() + en[1:]
        add(noun + suf, en, "comparative", c2, cls2_register(c2))
        c2 += 1

# Family C: both named  ("Y is better than X").
# Skip pairs whose English referents collide (e.g. 姉ちゃん/お姉ちゃん both "big sis",
# 母さん/お母さん both "Mom") so we never emit "big sis is better than big sis".
def _core(s):
    return s.lower().replace("the ", "").replace("a ", "").replace("'s", "").strip()

pair_seed = list(zip(COMP_ITEMS, COMP_ITEMS[2:] + COMP_ITEMS[:2]))
for (xnoun, _, xobj), (ynoun, ysubj, _) in pair_seed:
    if _core(xobj) == _core(ysubj):
        continue
    for jt, et in C_TEMPLATES:
        en = et.format(xe=xobj, ye=ysubj)
        en = en[0].upper() + en[1:]
        add(jt.format(xj=xnoun, yj=ynoun), en, "comparative", c2, cls2_register(c2))
        c2 += 1

# Real failures (anchor the exact inspection cases with CORRECT directionality).
add("姉ちゃんよりいいですぅ!!", "This is better than [doing it with] big sister!!",
    "comparative", c2, "vn_eroge"); c2 += 1
add("姉ちゃんよりいいですぅ!", "It's better than with big sis!!",
    "comparative", c2, "vn_eroge"); c2 += 1
add("母さんの方がいいですぅ!!", "Mom's is better!!", "comparative", c2, "vn_eroge"); c2 += 1
add("お母さんの方がいいですぅ!!", "Mom's is better!!", "comparative", c2, "vn_eroge"); c2 += 1
add("姉ちゃんより母さんの方がいい", "Mom is better than big sis", "comparative", c2, "vn_eroge"); c2 += 1


# ---------------------------------------------------------------------------
# CLASS 1 -- 3rd-PERSON SELF-REFERENCE (~300-500)
# A mother referring to HERSELF as お母さん / 母さん with feminine self-predicate
# (...わ / ...の / ...ちゃう + feminine endings) => "Mommy" or "I", NOT "my mom".
# Contrastive set: the SON saying お母さん about her => "Mom".
# ---------------------------------------------------------------------------

# Self-reference (mother speaking of herself). varied predicates + EN realization.
# Each: jp, en. We sweep "Mommy" / "I" renderings to teach the mapping, not a string.
MOM_SELF = [
    ("お母さん先にイッちゃうわねぇ♡", "Mommy's gonna cum first ♡"),
    ("お母さん先にイッちゃうわ♡", "Mommy's gonna cum first ♡"),
    ("お母さんもう我慢できないわ♡", "Mommy can't hold back anymore ♡"),
    ("お母さんがしてあげる♡", "Mommy'll do it for you ♡"),
    ("お母さんに任せなさい♡", "Just leave it to Mommy ♡"),
    ("お母さんがいいって言うまでダメよ♡", "Not until Mommy says it's okay ♡"),
    ("お母さんのこと好き?", "Do you love Mommy?"),
    ("お母さんもう限界なの♡", "Mommy's already at her limit ♡"),
    ("お母さんが教えてあげる♡", "Mommy'll teach you ♡"),
    ("お母さんの言うこと聞きなさい", "Listen to what Mommy says"),
    ("お母さんも気持ちいいわよ♡", "Mommy feels good too ♡"),
    ("お母さんに見せてごらん♡", "Show it to Mommy ♡"),
    ("お母さんのところへおいで♡", "Come to Mommy ♡"),
    ("お母さんが全部してあげるからね♡", "Mommy'll do everything for you ♡"),
    ("お母さんイッちゃう♡イッちゃう♡", "Mommy's cumming ♡ cumming ♡"),
    ("母さんに任せておきなさい", "Leave it to me"),
    ("母さんがちゃんとしてあげるから", "I'll take good care of you"),
    ("母さんもう待てないわ♡", "I can't wait anymore ♡"),
    ("母さんのこと忘れちゃダメよ", "Don't you dare forget about me"),
    ("母さんが守ってあげるわ", "I'll protect you"),
    ("母さん先に行くわね", "I'm going on ahead"),
    ("母さんがいるから大丈夫よ", "It's okay, I'm right here"),
    ("母さんもうおかしくなっちゃう♡", "I'm losing my mind ♡"),
    ("母さんのこと見て♡", "Look at me ♡"),
    ("母さんだってまだまだ現役よ", "I've still got it, you know"),
    ("お母さんがこんなにしてあげてるのに♡", "After Mommy's doing all this for you ♡"),
    ("お母さんの方がもっと気持ちよくしてあげる♡", "Mommy'll make you feel even better ♡"),
    ("お母さんに任せておけばいいの♡", "Just leave everything to Mommy ♡"),
    ("お母さんがいいようにしてあげるわ♡", "Mommy'll make it nice for you ♡"),
    ("お母さんもイッちゃいそう♡", "Mommy's about to cum too ♡"),
    ("お母さんのおっぱい好きでしょ?♡", "You love Mommy's breasts, don't you? ♡"),
    ("お母さんが慰めてあげる♡", "Mommy'll comfort you ♡"),
    ("お母さんの中に出していいのよ♡", "You can cum inside Mommy ♡"),
    ("お母さんと一緒に気持ちよくなろ♡", "Let's feel good together with Mommy ♡"),
    ("お母さんもう止まらないの♡", "Mommy can't stop anymore ♡"),
    ("母さんがやってあげるって言ってるでしょ", "I said I'd do it for you, didn't I"),
    ("母さんに甘えていいのよ♡", "You can spoil yourself with me ♡"),
    ("母さんのこと困らせないで", "Don't give me a hard time"),
    ("母さんがついててあげるから", "I'll stay right by your side"),
    ("母さんもこんなの初めてよ♡", "This is a first for me too ♡"),
]

# Mother self-ref expanded by light variation to reach drill volume.
MOM_SELF_EXTRA_STEMS = [
    ("お母さんが{v}♡", "Mommy'll {e} ♡"),
    ("お母さんが{v}わ♡", "Mommy'll {e} ♡"),
    ("お母さんが{v}わね♡", "Mommy'll {e}, okay? ♡"),
    ("お母さんも{v}の♡", "Mommy's gonna {e} too ♡"),
    ("お母さんが{v}からね♡", "Mommy'll {e}, okay ♡"),
    ("母さんが{v}から", "I'll {e}"),
    ("母さんが{v}わよ", "I'll {e}"),
    ("母さんが{v}からね", "I'll {e}, okay"),
    ("お母さんに{v}せて♡", "Let Mommy {e} ♡"),
]
MOM_VERBS = [
    ("慰めてあげる", "comfort you"),
    ("綺麗にしてあげる", "clean you up"),
    ("温めてあげる", "warm you up"),
    ("優しくしてあげる", "be gentle with you"),
    ("ご褒美あげる", "give you a reward"),
    ("もっとしてあげる", "do more for you"),
    ("抱きしめてあげる", "hold you tight"),
    ("舐めてあげる", "lick it for you"),
    ("可愛がってあげる", "love you up"),
    ("受け止めてあげる", "take all of it"),
    ("気持ちよくしてあげる", "make you feel good"),
    ("全部飲んであげる", "swallow it all"),
    ("ぎゅっとしてあげる", "give you a big hug"),
    ("お世話してあげる", "take care of you"),
    ("ぜんぶ見ててあげる", "watch over everything"),
]

# Contrastive: SON (or others) referring to the mother in 3rd person => "Mom".
MOM_THIRD = [
    ("姉ちゃんがトイレに行ってる間に母さんからメッセージが..",
     "Mom messaged me while big sis was in the bathroom..."),
    ("母さんは今出かけてるよ", "Mom's out right now"),
    ("母さんに怒られちゃうよ", "Mom's gonna get mad at us"),
    ("お母さんが呼んでるよ", "Mom's calling you"),
    ("お母さんはまだ帰ってこない", "Mom's not back yet"),
    ("母さんには内緒だぞ", "Don't tell Mom"),
    ("お母さんに聞いてみなよ", "Go ask Mom"),
    ("母さんの作ったご飯うまいな", "The food Mom made is delicious"),
    ("母さんが心配するから早く帰ろう", "Let's head home, Mom'll worry"),
    ("お母さんってば過保護なんだから", "Mom is so overprotective"),
    ("母さんの部屋に入っちゃダメだろ", "You're not supposed to go in Mom's room"),
    ("お母さんがそう言ってたよ", "That's what Mom said"),
    ("母さんはもう寝たのか?", "Has Mom gone to bed already?"),
    ("お母さんに見つかったら終わりだ", "If Mom finds us, we're done for"),
    ("母さんがまた何か買ってきた", "Mom bought something again"),
]

# Second feminine-self-predicate family: pin the cue (sentence-final わ/の/ちゃう)
# that signals SELF-reference, across the same verb pool but different surfaces.
MOM_SELF_PRED = [
    ("お母さん{v}ちゃう♡", "Mommy's gonna {e} ♡"),
    ("お母さんもう{v}そう♡", "Mommy's about to {e} ♡"),
    ("お母さん{v}たいの♡", "Mommy wants to {e} ♡"),
    ("母さん{v}ちゃうわ♡", "I'm gonna {e} ♡"),
    ("母さんもう{v}そうなの♡", "I'm about to {e} ♡"),
]
MOM_SELF_PRED_VERBS = [
    ("イッ", "cum"),
    ("おかしくなっ", "lose my mind"),
    ("溶け", "melt"),
    ("我慢できなくなっ", "lose control"),
    ("感じ", "feel it"),
    ("乱れ", "come undone"),
    ("達し", "climax"),
    ("蕩け", "go limp"),
]

c1 = 0
for jp, en in MOM_SELF:
    add(jp, en, "selfref", c1, "vn_eroge"); c1 += 1
for stem_j, stem_e in MOM_SELF_EXTRA_STEMS:
    for vj, ve in MOM_VERBS:
        add(stem_j.format(v=vj), (stem_e.format(e=ve))[0].upper() + (stem_e.format(e=ve))[1:],
            "selfref", c1, "vn_eroge"); c1 += 1
for stem_j, stem_e in MOM_SELF_PRED:
    for vj, ve in MOM_SELF_PRED_VERBS:
        en = stem_e.format(e=ve)
        add(stem_j.format(v=vj), en[0].upper() + en[1:], "selfref", c1, "vn_eroge"); c1 += 1
# contrastive 3rd-person (tagged selfref_contrast so it can be ablated)
cc = 0
for jp, en in MOM_THIRD:
    add(jp, en, "selfref_contrast", cc, "manga_dialog"); cc += 1
# expand contrastive volume with neutral 3rd-person kinship usages
THIRD_EXTRA = [
    ("お姉ちゃんがトイレに行ってる", "Big sis is in the bathroom"),
    ("姉ちゃんはもう先に行ったよ", "Big sis already went on ahead"),
    ("お姉ちゃんに相談してみたら?", "Why not talk to your big sister?"),
    ("お父さんが帰ってきた", "Dad's home"),
    ("お父さんには言わないで", "Don't tell Dad"),
    ("お兄ちゃんがゲームしてる", "Big bro is playing games"),
    ("お婆ちゃんが待ってるよ", "Grandma's waiting"),
    ("お母さんとお父さんは出かけた", "Mom and Dad went out"),
    ("姉ちゃんの方が背が高い", "Big sis is taller"),
    ("母さんと姉ちゃん、どっちに似てる?", "Who do I look more like, Mom or big sis?"),
]
for jp, en in THIRD_EXTRA:
    add(jp, en, "selfref_contrast", cc, "manga_dialog"); cc += 1
# More 3rd-person contrast (son/others talking ABOUT the mother => "Mom"),
# paired against the self-ref drills above so the model learns the cue.
THIRD_MORE = [
    ("母さんが作った弁当忘れた", "I forgot the lunch Mom made"),
    ("お母さんに頼んでみるよ", "I'll go ask Mom"),
    ("母さんはまだ仕事だよ", "Mom's still at work"),
    ("お母さんが買い物に行った", "Mom went shopping"),
    ("母さんの携帯鳴ってるよ", "Mom's phone is ringing"),
    ("お母さんが起こしてって言ってた", "Mom said to wake her up"),
    ("母さんが先に風呂入ってる", "Mom's taking a bath first"),
    ("お母さんに怒られる前に片付けよう", "Let's clean up before Mom gets mad"),
    ("母さんがそろそろ帰ってくる", "Mom'll be home soon"),
    ("お母さんって本当に料理上手だよね", "Mom's really good at cooking, huh"),
    ("母さんに見られたらまずい", "It'd be bad if Mom saw this"),
    ("お母さんが呼んでたよ、早く行きな", "Mom was calling you, hurry up"),
    ("母さんの分も取っておこう", "Let's save a portion for Mom"),
    ("お母さんはどこ行ったの?", "Where did Mom go?"),
    ("母さんが心配するから連絡しとけ", "Text Mom, she'll worry"),
]
for jp, en in THIRD_MORE:
    add(jp, en, "selfref_contrast", cc, "manga_dialog"); cc += 1


# ---------------------------------------------------------------------------
# CLASS 3 -- ADULT / GARMENT VOCAB + INTIMATE REGISTER (~200-300)
# Fix lexical gaps and register: ブラ=bra, パンツ=panties, バカ♡=affectionate
# "Silly ♡ / Idiot ♡" not "Kid", clinical -> intimate.
# ---------------------------------------------------------------------------

# Short lexicon anchors (term-in-context, varied), plus the real failures.
VOCAB_PAIRS = [
    # ブラ = bra (real failure: rendered "top")
    ("お母さんの匂いがたぁ〜っぷり染みついたブラ..",
     "A bra that's soaked through with Mom's scent..."),
    ("これお母さんのブラ?", "Is this Mom's bra?"),
    ("ブラのホックが外れない", "I can't get the bra's hook undone"),
    ("新しいブラ買ったの♡", "I bought a new bra ♡"),
    ("ブラ越しでも感じちゃう♡", "I can feel it even through my bra ♡"),
    ("ブラがきつくなってきた", "My bra's getting tight"),
    ("ブラとパンツお揃いなの♡", "My bra and panties match ♡"),
    ("黒いブラが透けてる", "Her black bra is showing through"),
    # パンツ = panties (women's underwear in this register)
    ("あとパンツぅ〜..", "And my panties..."),
    ("パンツは!?", "Where are my panties?!"),
    ("パンツが思いっきりはみ出てるし!!", "My panties are sticking way out!!"),
    ("濡れたパンツ脱がせて♡", "Take off my wet panties ♡"),
    ("パンツの中もう濡れ濡れ♡", "My panties are already soaking ♡"),
    ("お母さんのパンツ盗ったでしょ", "You took Mom's panties, didn't you"),
    ("パンツ一枚で歩き回らないの", "Don't walk around in just your panties"),
    ("脱いだパンツどこやったの?", "Where did you put your panties after taking them off?"),
    # バカ♡ = affectionate "Silly ♡ / Idiot ♡" (real failure: "Kid" / garbled)
    ("バ.カ♡", "Silly ♡"),
    ("バカ♡", "Idiot ♡"),
    ("もう、バカ♡", "Honestly, you dummy ♡"),
    ("ばかぁ♡", "You big dummy ♡"),
    ("ホントにバカなんだから♡", "You really are such an idiot ♡"),
    ("このスケベ♡", "You perv ♡"),
    ("えっち♡", "So naughty ♡"),
    ("変態♡", "You pervert ♡"),
    ("いじわる♡", "You meanie ♡"),
    ("もぅ♡知らない♡", "Hmph ♡ I don't care anymore ♡"),
    # garments / body register lexicon
    ("ブラジャー外してあげる♡", "Let me undo your bra ♡"),
    ("下着姿で出てこないで", "Don't come out in just your underwear"),
    ("ストッキング破けちゃった", "My stockings tore"),
    ("ネグリジェが薄すぎる♡", "This negligee is way too thin ♡"),
    ("おっぱい揉んでもいいよ♡", "You can squeeze my breasts ♡"),
    ("胸の先がもう硬くなってる♡", "My nipples are already hard ♡"),
    ("乳首つままないで♡", "Don't pinch my nipples ♡"),
    ("お尻もっと突き出して♡", "Stick your butt out more ♡"),
    ("太ももがすべすべ♡", "Your thighs are so smooth ♡"),
    ("うなじにキスして♡", "Kiss the nape of my neck ♡"),
    # intimate predicates (register, not clinical)
    ("もう我慢できない♡", "I can't hold back anymore ♡"),
    ("そこダメぇ♡", "Not there ♡"),
    ("気持ちよすぎて変になっちゃう♡", "It feels so good I'm losing it ♡"),
    ("もっと奥まで♡", "Deeper ♡"),
    ("優しくして♡", "Be gentle ♡"),
    ("はやくぅ♡", "Hurry ♡"),
    ("やめないで♡", "Don't stop ♡"),
    ("いっぱい出していいのよぉお♡", "Cum as much as you want ♡"),
    ("出していいのよぉお♡", "You can cum now ♡"),
    ("中に出して♡", "Cum inside me ♡"),
    ("一緒にイこ♡", "Let's cum together ♡"),
    ("だったら次はちゃんとオマンコに出しなさい",
     "Then next time, make sure you cum inside my pussy"),
    ("イキたいんなら自分で動きなさい!", "If you want to cum, then move on your own!"),
    ("グズねっ!!", "What a slowpoke!!"),
    ("トモキに騎乗位したいっっ♡", "I want to ride you, Tomoki ♡"),
]

# Lexicon drill: term + varied carrier sentences so the WORD maps, not a string.
# (jp_term, en_term) with carrier templates around it.
# (jp_term, en_term, is_plural). is_plural drives copula/article agreement so we
# never emit "Your panties is cute" or "I bought new nipples".
LEXICON = [
    ("ブラ", "bra", False),
    ("ブラジャー", "bra", False),
    ("パンツ", "panties", True),
    ("ショーツ", "panties", True),
    ("下着", "underwear", True),     # mass noun -> plural agreement
    ("ストッキング", "stockings", True),
    ("ネグリジェ", "negligee", False),
    ("水着", "swimsuit", False),
    ("キャミソール", "camisole", False),
    ("スリップ", "slip", False),
]
# Number-safe carriers. {e}=term, {be}=is/are, {dem}=this/these for the garment.
LEX_CARRIERS = [
    ("{j}見せて♡", "Show me your {e} ♡"),
    ("その{j}可愛いね♡", "{dem} {e} {be} cute ♡"),
    ("{j}脱いじゃおっか♡", "Why don't we take off your {e} ♡"),
    ("新しい{j}買ったの", "I bought a new {e}"),
    ("{j}が濡れちゃった♡", "My {e} got wet ♡"),
]
# Affectionate-insult lexicon (the バカ♡ class): term-only -> term + endearment.
AFFECTION = [
    ("バカ♡", "Idiot ♡"),
    ("ばか♡", "Dummy ♡"),
    ("あほ♡", "Silly ♡"),
    ("えっち♡", "Naughty ♡"),
    ("スケベ♡", "Perv ♡"),
    ("変態♡", "Pervert ♡"),
    ("いじわる♡", "Meanie ♡"),
    ("もう♡", "Honestly ♡"),
]

c3 = 0
for jp, en in VOCAB_PAIRS:
    add(jp, en, "vocab_register", c3, "vn_eroge"); c3 += 1
for jterm, eterm, plural in LEXICON:
    be = "are" if plural else "is"
    dem = "Those" if plural else "This"
    for jc, ec in LEX_CARRIERS:
        # plural/mass nouns don't take "a new"
        if "a new {e}" in ec and plural:
            en = ec.replace("a new {e}", "new {e}").format(e=eterm)
        else:
            en = ec.format(e=eterm, be=be, dem=dem)
        add(jc.format(j=jterm), en, "vocab_register", c3, "vn_eroge"); c3 += 1
for jp, en in AFFECTION:
    add(jp, en, "vocab_register", c3, "vn_eroge"); c3 += 1
    # also a doubled emphatic form, common in these bubbles
    add(jp.replace("♡", "♡♡"), en + "♡", "vocab_register", c3, "vn_eroge"); c3 += 1


# ---------------------------------------------------------------------------
# CLASS 4 -- CONFABULATION SUPPRESSION (~100-200)
# Faithful references for long emotive lines and short interjections, with
# NO added content. Directly counters the "breast milk" hallucination class
# and the padding of short SFX into invented sentences.
# ---------------------------------------------------------------------------

# Long emotive lines: faithful (no invention). These mirror the corpus tails.
FAITHFUL_LONG = [
    ("もともとオナニーではイケない射精障害だった息子はイケるのが母だけになってしまった",
     "Her son, who had a disorder that kept him from cumming through masturbation, "
     "could now only cum with his mother."),
    ("母も息子のその近況を知り自分しかイカせられないことに優越感を感じるようになる",
     "Learning of her son's situation, the mother came to feel a sense of superiority "
     "that she was the only one who could make him cum."),
    ("その優越感が母の自信につながり",
     "That sense of superiority fed into the mother's confidence,"),
    ("姉でイケなくなってしまった息子",
     "Her son, who could no longer cum with his sister,"),
    ("ただ好きな男を奪おうとする邪魔な存在でしかなくなってしまった",
     "had become nothing more than a nuisance trying to steal the man she loved."),
    ("もはや母にとって娘は邪魔な存在でしかなかった",
     "To the mother, her daughter was now nothing but an obstacle."),
    ("その言葉が呪いのようにからみつき",
     "Those words clung to him like a curse,"),
    ("姉と性行為中、射精間近の自身にささやきかけ",
     "While having sex with his sister, on the verge of climax, he whispered to himself,"),
    ("息子に対して積極的にアプローチするようになる",
     "She began to come on to her son more and more aggressively."),
    ("でも母さんのキレイな顔を汚したみたいで興奮する..!",
     "But it's like I've dirtied Mom's beautiful face, and that turns me on...!"),
    ("このままあの娘のところに行ってあげましょうか?",
     "Should I go to that girl just like this?"),
    ("ほんとに欲しいのはこっちってわかってるから♡",
     "Because I know this is the one you really want ♡"),
    ("あなたに特別にあげようかしら♡",
     "Maybe I'll give it to you as something special ♡"),
    ("見ないで姉ちゃ〜ん", "Don't look, big sis~"),
    ("この女のマンコ味わってるとこ見ないでぇえ",
     "Don't look at me tasting this woman's pussy..."),
]

# Short interjections / SFX -> faithful short output (NO padding into sentences).
# Real failures padded these: e.g. "ちょっと僕もトイレ!" -> "..." (dropped),
# "課題とかは...?" -> hallucinated "Past records create sensitivity".
FAITHFUL_SHORT = [
    ("ちょっと僕もトイレ!", "Hold on, I gotta pee too!"),
    ("見とけったって..", "You say keep watch, but..."),
    ("課題とかは...?", "What about your homework...?"),
    ("そんなの帰ってからするわよ", "I'll do that after I get home"),
    ("ご飯とか食べに行ったりさ", "Like, we could go grab a bite to eat"),
    ("ゲーセンとかで遊び", "We could hang out at the arcade or something"),
    ("お母さんと顔合わせたくないし", "And I don't wanna run into your mom"),
    ("そう..だね..", "Yeah... you're right..."),
    ("見つけてくれてありがとね", "Thanks for finding it for me"),
    ("何アレ?", "What was that?"),
    ("てっ..手拭い!", "A-A hand towel!"),
    ("探してたんだって!", "He said he was looking for it!"),
    ("髪の毛も!!", "My hair too!!"),
    ("枕の下に隠しとくわね", "I'll hide it under the pillow"),
    ("母さん!!", "Mom!!"),
    ("見つかった?", "Did you find it?"),
    ("それは勘弁して!!", "Give me a break with that!!"),
    ("言わないとぉおお", "I have to say it...!"),
    ("言います!", "I'll say it!"),
    ("ふふふっ", "Hehehe."),
    ("ただいまぁ〜", "I'm hooome~"),
    ("大丈夫よぉ", "It's okay~"),
    ("はあ", "Haah..."),
    ("ギク", "*flinch*"),
    ("ガタ", "*rattle*"),
    ("んふふ〜〜♡", "Hehe~ ♡"),
]

# More faithful long emotive lines (narration/dialogue) — stay exactly on-source.
FAITHFUL_LONG_2 = [
    ("どうしても忘れられなくてつい連絡してしまった",
     "I just couldn't get it out of my head, so I ended up reaching out."),
    ("こんなことしちゃいけないってわかってるのに止められない♡",
     "I know I shouldn't be doing this, but I can't stop myself ♡"),
    ("あの日のことを思い出すだけで体が熱くなる",
     "Just remembering that day makes my body hot."),
    ("最初はただの好奇心だったはずなのに",
     "It was supposed to be nothing but curiosity at first,"),
    ("気づいたときにはもう後戻りできないところまで来ていた",
     "by the time I realized it, there was no turning back."),
    ("誰にも言えない秘密がまた一つ増えてしまった",
     "Now I have one more secret I can never tell anyone."),
    ("罪悪感よりも気持ちよさの方が勝ってしまう♡",
     "The pleasure wins out over the guilt ♡"),
    ("ずっとこの瞬間が続けばいいのにと思った",
     "I wished this moment could last forever."),
    ("二人だけの時間がたまらなく愛おしかった",
     "Our time alone together was unbearably precious."),
    ("もう元の関係には戻れないとわかっていた",
     "I knew we could never go back to how things were."),
]
# More short interjections/SFX — faithful, no padding, no invention.
FAITHFUL_SHORT_2 = [
    ("えっっ!?", "Huh?!"),
    ("あぁ..", "Ahh..."),
    ("うん", "Yeah"),
    ("か..", "Wha—"),
    ("は?", "Huh?"),
    ("コレ", "Here"),
    ("ほらっ!", "There!"),
    ("はいぃ〜〜!!", "Y-Yes!!"),
    ("あっっ..", "Ah..."),
    ("ちゅ♪", "*smooch* ♪"),
    ("ブチ", "*snap*"),
    ("ギク", "*flinch*"),
    ("ニキキ", "*grin*"),
    ("わくっっっ...!", "*throb*...!"),
    ("おおーぶつっ", "*splurt*"),
    ("夜", "Night"),
    ("カ", "*twitch*"),
    ("!?", "?!"),
    ("そうねぇ〜〜..", "Let's see now..."),
    ("やだもう♡", "Stop it ♡"),
    ("ね?", "Right?"),
    ("だめ?", "No good?"),
    ("ほんと?", "Really?"),
    ("いいの?", "Is it okay?"),
    ("待って", "Wait"),
    ("やめて", "Stop"),
    ("こっち向いて", "Look at me"),
    ("おいで", "Come here"),
    ("そっか", "I see"),
    ("うそ..", "No way..."),
    ("どうしよう", "What do I do"),
    ("行こ?", "Shall we go?"),
    ("もういいでしょ", "That's enough, right?"),
    ("聞こえてる?", "Can you hear me?"),
    ("ばれた?", "Did they find out?"),
    ("内緒ね♡", "It's a secret, okay? ♡"),
    ("ちょっとだけ♡", "Just a little ♡"),
    ("約束だよ", "It's a promise"),
    ("怒ってる?", "Are you mad?"),
    ("平気平気", "I'm fine, I'm fine"),
    ("もう遅いよ", "It's already late"),
    ("先に寝てて", "Go to sleep without me"),
    ("おやすみ", "Goodnight"),
    ("バレないように", "So we don't get caught,"),
    ("静かにして", "Be quiet"),
]

c4 = 0
for jp, en in FAITHFUL_LONG:
    add(jp, en, "faithful", c4, "vn_eroge"); c4 += 1
for jp, en in FAITHFUL_LONG_2:
    add(jp, en, "faithful", c4, "vn_eroge"); c4 += 1
for jp, en in FAITHFUL_SHORT:
    add(jp, en, "faithful", c4, "manga_dialog"); c4 += 1
for jp, en in FAITHFUL_SHORT_2:
    add(jp, en, "faithful", c4, "manga_dialog"); c4 += 1


# ---------------------------------------------------------------------------
# Assemble, dedup on jp (keep first), write parquet (exact v10 schema) + jsonl.
# ---------------------------------------------------------------------------
df = pd.DataFrame(rows, columns=["jp", "en", "src", "register_tag", "gold_flag"])

before = len(df)
df = df.drop_duplicates(subset=["jp"], keep="first").reset_index(drop=True)
after = len(df)

# enforce exact arrow schema: large_string + bool
schema = pa.schema(
    [
        pa.field("jp", pa.large_string()),
        pa.field("en", pa.large_string()),
        pa.field("src", pa.large_string()),
        pa.field("register_tag", pa.large_string()),
        pa.field("gold_flag", pa.bool_()),
    ]
)
table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
pq.write_table(table, PARQUET_OUT)

with open(JSONL_OUT, "w", encoding="utf-8") as fh:
    for r in df.to_dict(orient="records"):
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---- report ----
def cls_of(src):
    return src.split(":")[1]

df["_cls"] = df["src"].map(cls_of)
print(f"rows before dedup: {before}  after: {after}  (dropped {before-after} dup jp)")
print("\nper-class counts:")
print(df["_cls"].value_counts().to_string())
print("\nregister_tag counts:")
print(df["register_tag"].value_counts().to_string())
print(f"\ngold_flag all True: {bool(df['gold_flag'].all())}")
print(f"\nwrote: {PARQUET_OUT}")
print(f"wrote: {JSONL_OUT}")
print("\narrow schema written:")
print(pq.read_schema(PARQUET_OUT))
