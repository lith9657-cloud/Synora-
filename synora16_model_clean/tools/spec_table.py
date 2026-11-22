"""spec_table (fix20): ~50 typed synthesis specs — the bridge from
natural language to synth_core. Each spec: AR/EN keyword regexes +
phrases, param type, fixed examples (given to the synthesizer),
3 hidden tests (never given — the proof), oracle, optional extra
literals and search size. Deterministic everywhere."""
import sys
sys.path.insert(0, "/home/z/my-project/scripts")
from synth_core import R_I, R_S, R_LI, R_LS  # noqa: E402

SP = []
def sp(sid, ar_kw, en_kw, ar_q, en_q, ptype, examples, hidden,
       oracle, extra=None, maxsize=7, style=None):
    SP.append({"id": sid, "ar": ar_kw, "en": en_kw, "ar_q": ar_q,
               "en_q": en_q, "ptype": ptype, "examples": examples,
               "hidden": hidden, "oracle": oracle,
               "extra": extra or [], "maxsize": maxsize,
               "style": style})

LI = R_LI
S = R_S
I = R_I
LS = R_LS

# ── list[int] ─────────────────────────────────────────────────
sp("sum_squares", r"مربعات", r"sum of squares",
   "اكتب دالة تجمع مربعات عناصر القائمة",
   "write a function returning the sum of squares of a list",
   LI, [([1, 2, 3], 14), ([2, 5], 29)],
   [([4], 16), ([1, 2, 3, 4, 5], 55), ([0, 0], 0)],
   lambda x: sum(v * v for v in x))
sp("sum_list", r"اجمع عناصر|مجموع عناصر", r"sum of the list",
   "اكتب دالة تجمع عناصر القائمة",
   "write a function returning the sum of a list",
   LI, [([1, 2, 3], 6), ([5, 5], 10)],
   [([4], 4), ([1, 2, 3, 4], 10), ([-1, 1], 0)],
   lambda x: sum(x))
sp("sum_even", r"زوجية|الأعداد الزوجية", r"sum of even numbers",
   "اكتب دالة تجمع الأعداد الزوجية في القائمة",
   "write a function returning the sum of even numbers",
   LI, [([1, 2, 3, 4], 6), ([2, 2], 4)],
   [([5], 0), ([2, 4, 7], 6), ([], 0)],
   lambda x: sum(v for v in x if v % 2 == 0))
sp("sum_odd", r"فردية|الأعداد الفردية", r"sum of odd numbers",
   "اكتب دالة تجمع الأعداد الفردية في القائمة",
   "write a function returning the sum of odd numbers",
   LI, [([1, 2, 3, 4], 4), ([7, 2], 7)],
   [([4], 0), ([1, 3, 6], 4), ([], 0)],
   lambda x: sum(v for v in x if v % 2 != 0))
sp("count_even", r"عدد الزوجية|كم عدد الزوجية", r"count even numbers",
   "اكتب دالة تحسب الأعداد الزوجية في القائمة",
   "write a function counting the even numbers",
   LI, [([1, 2, 3, 4], 2), ([2, 2], 2)],
   [([5], 0), ([2, 4, 7], 2), ([], 0)],
   lambda x: sum(1 for v in x if v % 2 == 0))
sp("max_list", r"أكبر عنصر", r"largest element",
   "اكتب دالة تعيد أكبر عنصر في القائمة",
   "write a function returning the largest element",
   LI, [([1, 9, 3], 9), ([5, 5], 5)],
   [([4], 4), ([1, 2, 8], 8), ([-3, -1], -1)],
   lambda x: max(x))
sp("min_list", r"أصغر عنصر", r"smallest element",
   "اكتب دالة تعيد أصغر عنصر في القائمة",
   "write a function returning the smallest element",
   LI, [([4, 9, 3], 3), ([5, 5], 5)],
   [([4], 4), ([1, 2, 0], 0), ([-3, -1], -3)],
   lambda x: min(x))
sp("second_max", r"ثاني أكبر", r"second largest",
   "اكتب دالة تعيد ثاني أكبر عنصر",
   "write a function returning the second largest element",
   LI, [([1, 9, 3], 3), ([5, 2], 2)],
   [([4, 8], 4), ([1, 2, 8, 9], 8), ([7, 7, 1], 7)],
   lambda x: sorted(x)[-2])
sp("sorted_asc", r"رتب تصاعدي|تصاعدياً", r"sort ascending",
   "اكتب دالة ترتب القائمة تصاعدياً",
   "write a function sorting the list ascending",
   LI, [([3, 1, 2], [1, 2, 3]), ([5, 5], [5, 5])],
   [([4], [4]), ([9, 1, 5], [1, 5, 9]), ([], [])],
   lambda x: sorted(x))
sp("sorted_desc", r"رتب تنازلي|تنازلياً", r"sort descending",
   "اكتب دالة ترتب القائمة تنازلياً",
   "write a function sorting the list descending",
   LI, [([3, 1, 2], [3, 2, 1]), ([5, 5], [5, 5])],
   [([4], [4]), ([9, 1, 5], [9, 5, 1]), ([], [])],
   lambda x: sorted(x)[::-1])
sp("reverse_list", r"اعكس القائمة|عكس ترتيب", r"reverse a list",
   "اكتب دالة تعكس ترتيب عناصر القائمة",
   "write a function reversing a list",
   LI, [([1, 2, 3], [3, 2, 1]), ([5], [5])],
   [([4, 8], [8, 4]), ([1, 2, 3, 4], [4, 3, 2, 1]), ([], [])],
   lambda x: x[::-1])
sp("unique_sorted", r"بلا تكرار|ازالة المكرر|إزالة المكرر",
   r"remove duplicates sorted",
   "اكتب دالة تعيد القائمة بلا تكرار ومرتبة",
   "write a function returning the sorted unique list",
   LI, [([3, 1, 3, 2], [1, 2, 3]), ([5, 5], [5])],
   [([4], [4]), ([2, 2, 1, 1], [1, 2]), ([], [])],
   lambda x: sorted(set(x)))
sp("squares_list", r"قائمة المربعات|تربيع", r"list of squares",
   "اكتب دالة تعيد قائمة مربعات العناصر",
   "write a function returning the list of squares",
   LI, [([1, 2, 3], [1, 4, 9]), ([2], [4])],
   [([4], [16]), ([1, 5], [1, 25]), ([], [])],
   lambda x: [v * v for v in x])
sp("evens_list", r"الزوجية فقط|استخرج الزوجية", r"even numbers only",
   "اكتب دالة تستخرج الأعداد الزوجية من القائمة",
   "write a function extracting the even numbers",
   LI, [([1, 2, 3, 4], [2, 4]), ([2], [2])],
   [([5], []), ([2, 4, 7], [2, 4]), ([], [])],
   lambda x: [v for v in x if v % 2 == 0])
sp("odds_list", r"الفردية فقط|استخرج الفردية", r"odd numbers only",
   "اكتب دالة تستخرج الأعداد الفردية من القائمة",
   "write a function extracting the odd numbers",
   LI, [([1, 2, 3, 4], [1, 3]), ([7], [7])],
   [([4], []), ([1, 3, 6], [1, 3]), ([], [])],
   lambda x: [v for v in x if v % 2 != 0])
sp("sum_pos", r"موجبة|الإيجابية", r"sum of positives",
   "اكتب دالة تجمع الأعداد الموجبة في القائمة",
   "write a function summing the positive numbers",
   LI, [([-1, 2, 3], 5), ([-5, -2], 0)],
   [([4], 4), ([-2, 5, -1], 5), ([], 0)],
   lambda x: sum(v for v in x if v > 0))
sp("count_pos", r"عدد الموجبة|كم عدد الموجبة", r"count positives",
   "اكتب دالة تحسب الأعداد الموجبة في القائمة",
   "write a function counting the positive numbers",
   LI, [([-1, 2, 3], 2), ([-5, -2], 0)],
   [([4], 1), ([-2, 5, -1], 1), ([], 0)],
   lambda x: sum(1 for v in x if v > 0))
sp("idx_of_max", r"موقع الأكبر|فهرس الأكبر", r"index of the maximum",
   "اكتب دالة تعيد موقع أكبر عنصر",
   "write a function returning the index of the maximum",
   LI, [([4, 9, 3], 1), ([5, 2], 0)],
   [([4], 0), ([1, 8, 3], 1), ([9, 1], 0)],
   lambda x: x.index(max(x)))
sp("first_elem", r"أول عنصر", r"first element",
   "اكتب دالة تعيد أول عنصر في القائمة",
   "write a function returning the first element",
   LI, [([4, 9, 3], 4), ([7], 7)],
   [([8, 1], 8), ([5, 6, 7], 5), ([2], 2)],
   lambda x: x[0])
sp("last_elem", r"آخر عنصر", r"last element",
   "اكتب دالة تعيد آخر عنصر في القائمة",
   "write a function returning the last element",
   LI, [([4, 9, 3], 3), ([7], 7)],
   [([8, 1], 1), ([5, 6, 7], 7), ([2], 2)],
   lambda x: x[-1])
sp("is_sorted", r"مرتبة تصاعدياً|هل مرتبة", r"is the list sorted",
   "اكتب دالة تفحص إن كانت القائمة مرتبة تصاعدياً",
   "write a function checking whether the list is sorted",
   LI, [([1, 2, 3], True), ([3, 1], False)],
   [([5], True), ([1, 3, 2], False), ([1, 1, 2], True)],
   lambda x: x == sorted(x))
sp("has_dup", r"مكرر|تكرار", r"has duplicates",
   "اكتب دالة تفحص إن كان بالقائمة تكرار",
   "write a function checking whether the list has duplicates",
   LI, [([1, 2, 2], True), ([1, 2], False)],
   [([5], False), ([1, 1], True), ([], False)],
   lambda x: len(x) != len(set(x)))
sp("range_list", r"قائمة من صفر|من صفر الى|من 0 إلى",
   r"list from zero to n",
   "اكتب دالة تنتج قائمة الأعداد من صفر إلى ما قبل n",
   "write a function producing the numbers from 0 to n-1",
   I, [(3, [0, 1, 2]), (1, [0])],
   [(0, []), (5, [0, 1, 2, 3, 4]), (2, [0, 1])],
   lambda x: list(range(x)), maxsize=6)
sp("sum_range", r"مجموع من صفر|اجمع من صفر", r"sum from zero to n",
   "اكتب دالة تجمع الأعداد من صفر إلى ما قبل n",
   "write a function summing the numbers from 0 to n-1",
   I, [(3, 3), (1, 0)],
   [(0, 0), (5, 10), (2, 1)],
   lambda x: sum(range(x)), maxsize=7)
sp("count_div3", r"مضاعفات الثلاثة|مضاعفات الـ3", r"multiples of three below n",
   "اكتب دالة تحسب مضاعفات الثلاثة من 1 إلى ما قبل n",
   "write a function counting multiples of 3 below n",
   I, [(10, 3), (3, 0)],
   [(1, 0), (16, 5), (6, 1)],
   lambda x: sum(1 for v in range(1, x) if v % 3 == 0),
   maxsize=8)
sp("double_list", r"ضاعف|الضعف", r"double every element",
   "اكتب دالة تضاعف كل عنصر في القائمة",
   "write a function doubling every element",
   LI, [([1, 2, 3], [2, 4, 6]), ([5], [10])],
   [([4], [8]), ([0, 3], [0, 6]), ([], [])],
   lambda x: [v * 2 for v in x])
sp("cubes_list", r"مكعبات", r"list of cubes",
   "اكتب دالة تعيد قائمة مكعبات العناصر",
   "write a function returning the list of cubes",
   LI, [([1, 2, 3], [1, 8, 27]), ([2], [8])],
   [([4], [64]), ([1, 5], [1, 125]), ([], [])],
   lambda x: [v * v * v for v in x])
sp("sum_gt10", r"أكبر من عشرة|أكبر من 10", r"sum of numbers above ten",
   "اكتب دالة تجمع الأعداد الأكبر من عشرة في القائمة",
   "write a function summing the numbers above ten",
   LI, [([5, 20, 3], 20), ([15, 10], 15)],
   [([40], 40), ([11, 9, 12], 23), ([], 0)],
   lambda x: sum(v for v in x if v > 10))

# ── list[str] ─────────────────────────────────────────────────
sp("join_words", r"اجمع الكلمات|وصل الكلمات", r"join words with spaces",
   "اكتب دالة تصل الكلمات بمسافة واحدة",
   "write a function joining words with a single space",
   LS, [(["مرحبا", "بكم"], "مرحبا بكم"), (["a"], "a")],
   [(["x"], "x"), (["ا", "ب", "ج"], "ا ب ج"), ([], "")],
   lambda x: " ".join(x))
sp("concat_words", r"الصق الكلمات|بدون مسافة", r"concatenate words",
   "اكتب دالة تلصق الكلمات بلا فاصل",
   "write a function concatenating words with no separator",
   LS, [(["مرحبا", "بكم"], "مرحبابكم"), (["a"], "a")],
   [(["x"], "x"), (["ا", "ب"], "اب"), ([], "")],
   lambda x: "".join(x))
sp("upper_words", r"كبّر الكلمات|حروف كبيرة", r"uppercase every word",
   "اكتب دالة تكبّر حروف كل كلمة في القائمة",
   "write a function uppercasing every word",
   LS, [(["hi", "ok"], ["HI", "OK"]), (["a"], ["A"])],
   [(["x"], ["X"]), (["ab", "cd"], ["AB", "CD"]), ([], [])],
   lambda x: [e.upper() for e in x])

# ── str ───────────────────────────────────────────────────────
sp("count_vowels", r"عدد حروف العلة|حروف العلة", r"count vowels",
   "اكتب دالة تحسب حروف العلة الإنجليزية في النص",
   "write a function counting the English vowels in a string",
   S, [("hello", 2), ("sky", 0)],
   [("a", 1), ("Education", 5), ("", 0)],
   lambda x: sum(1 for c in x if c in "aeiouAEIOU"))
sp("count_digits", r"عدد الأرقام|كم رقماً", r"count digits",
   "اكتب دالة تحسب الأرقام داخل النص",
   "write a function counting the digits in a string",
   S, [("abc123", 3), ("no", 0)],
   [("9", 1), ("a1b2c3", 3), ("", 0)],
   lambda x: sum(1 for c in x if c.isdigit()))
sp("digits_only", r"استخرج الأرقام|الأرقام فقط", r"extract digits only",
   "اكتب دالة تستخرج الأرقام من النص فقط",
   "write a function extracting only the digits",
   S, [("ab12cd", "12"), ("x", "")],
   [("9z", "9"), ("2026!", "2026"), ("", "")],
   lambda x: "".join([c for c in x if c.isdigit()]))
sp("no_spaces", r"ازل المسافات|بلا مسافات", r"remove all spaces",
   "اكتب دالة تزيل كل المسافات من النص",
   "write a function removing every space",
   S, [("a b c", "abc"), ("  x  ", "x")],
   [("n o", "no"), ("", ""), ("wow", "wow")],
   lambda x: "".join([c for c in x if c != " "]))
sp("upper_str", r"كبّر النص|حروف كبيرة النص", r"uppercase a string",
   "اكتب دالة تكبّر حروف النص",
   "write a function uppercasing a string",
   S, [("hi", "HI"), ("Ok", "OK")],
   [("x", "X"), ("abc", "ABC"), ("", "")],
   lambda x: x.upper())
sp("lower_str", r"صغّر النص|حروف صغيرة", r"lowercase a string",
   "اكتب دالة تصغّر حروف النص",
   "write a function lowercasing a string",
   S, [("HI", "hi"), ("Ok", "ok")],
   [("X", "x"), ("ABC", "abc"), ("", "")],
   lambda x: x.lower())
sp("title_case", r"أول حرف كبير لكل كلمة|عناوين", r"title case",
   "اكتب دالة تجعل أول حرف من كل كلمة كبيراً",
   "write a function capitalizing every word",
   S, [("hello world", "Hello World"), ("a b", "A B")],
   [("x y z", "X Y Z"), ("AB", "Ab"), ("", "")],
   lambda x: x.title())
sp("swap_case", r"اعكس حالة الحروف|بدل حالة", r"swap the letter case",
   "اكتب دالة تعكس حالة الحروف في النص",
   "write a function swapping the letter case",
   S, [("AbC", "aBc"), ("x", "X")],
   [("Hi", "hI"), ("OK", "ok"), ("", "")],
   lambda x: x.swapcase())
sp("reverse_str", r"اعكس النص|اقلب النص", r"reverse a string",
   "اكتب دالة تعكس النص",
   "write a function reversing a string",
   S, [("abc", "cba"), ("a", "a")],
   [("ab", "ba"), ("2026", "6202"), ("", "")],
   lambda x: x[::-1])
sp("is_palindrome", r"متناوبة|باليسار", r"is a palindrome",
   "اكتب دالة تفحص إن كان النص متناوباً (يُقرأ بالاتجاهين)",
   "write a function checking whether a string is a palindrome",
   S, [("level", True), ("abc", False)],
   [("a", True), ("abba", True), ("ab", False)],
   lambda x: x == x[::-1])
sp("word_count", r"عدد الكلمات|كم كلمة", r"count the words",
   "اكتب دالة تحسب كلمات النص",
   "write a function counting the words of a string",
   S, [("a b c", 3), ("واحد", 1)],
   [("x", 1), ("ا ب ج د", 4), ("", 0)],
   lambda x: len(x.split()))
sp("first_word", r"أول كلمة", r"first word",
   "اكتب دالة تعيد أول كلمة من النص",
   "write a function returning the first word",
   S, [("مرحبا بكم", "مرحبا"), ("واحد", "واحد")],
   [("x y", "x"), ("أ ب ج", "أ"), ("واحد", "واحد")],
   lambda x: x.split()[0])
sp("last_word", r"آخر كلمة", r"last word",
   "اكتب دالة تعيد آخر كلمة من النص",
   "write a function returning the last word",
   S, [("مرحبا بكم", "بكم"), ("واحد", "واحد")],
   [("x y", "y"), ("أ ب ج", "ج"), ("واحد", "واحد")],
   lambda x: x.split()[-1])
sp("char_a_count", r"عدد حرف ا|كم مرة حرف", r"count letter a",
   "اكتب دالة تحسب مرات ظهور الحرف a في النص",
   "write a function counting the letter a in a string",
   S, [("banana", 3), ("xyz", 0)],
   [("a", 1), ("baghdad", 2), ("", 0)],
   lambda x: x.count("a"), extra=["a"])

# ── int ───────────────────────────────────────────────────────
sp("is_even", r"زوجي|هل عدد زوجي", r"is even",
   "اكتب دالة تفحص إن كان العدد زوجياً",
   "write a function checking whether a number is even",
   I, [(4, True), (7, False)],
   [(0, True), (9, False), (2026, True)],
   lambda x: x % 2 == 0)
sp("div3", r"يقبل القسمة على ثلاثة|على 3", r"divisible by three",
   "اكتب دالة تفحص إن كان العدد يقبل القسمة على ثلاثة",
   "write a function checking divisibility by three",
   I, [(9, True), (10, False)],
   [(3, True), (7, False), (2026, False)],
   lambda x: x % 3 == 0)
sp("digit_sum", r"مجموع أرقام العدد|اجمع أرقام العدد",
   r"sum of the digits",
   "اكتب دالة تجمع أرقام العدد",
   "write a function summing the digits of a number",
   I, [(123, 6), (40, 4)],
   [(9, 9), (2026, 10), (0, 0)],
   lambda x: sum(int(c) for c in str(x)))
sp("reverse_int", r"اعكس العدد|اقلب العدد", r"reverse a number",
   "اكتب دالة تعكس أرقام العدد",
   "write a function reversing the digits of a number",
   I, [(123, 321), (40, 4)],
   [(9, 9), (2026, 6202), (10, 1)],
   lambda x: int(str(x)[::-1]))
sp("to_binary", r"ثنائي|بالصيغة الثنائية", r"binary representation",
   "اكتب دالة تعيد تمثيل العدد بالنظام الثنائي",
   "write a function returning the binary representation",
   I, [(5, "101"), (2, "10")],
   [(1, "1"), (2026, "11111101010"), (0, "0")],
   lambda x: bin(x)[2:])
sp("digit_count", r"عدد أرقام العدد|كم رقماً في العدد",
   r"number of digits",
   "اكتب دالة تحسب عدد أرقام العدد",
   "write a function counting the digits of a number",
   I, [(123, 3), (7, 1)],
   [(2026, 4), (0, 1), (99, 2)],
   lambda x: len(str(x)))
sp("power_two", r"قوة الثنائية|اثرا أس", r"two to the power n",
   "اكتب دالة تحسب 2 مرفوعة للقوة n",
   "write a function computing two to the power of n",
   I, [(3, 8), (0, 1)],
   [(1, 2), (10, 1024), (5, 32)],
   lambda x: 2 ** x)
sp("sign_fn", r"أشارة العدد|علامة العدد", r"the sign of a number",
   "اكتب دالة تعيد إشارة العدد: 1 موجب، -1 سالب، 0 صفر",
   "write a function returning the sign: 1, -1 or 0",
   I, [(5, 1), (-3, -1)],
   [(0, 0), (2026, 1), (-1, -1)],
   lambda x: 1 if x > 0 else (-1 if x < 0 else 0),
   extra=[-1])
sp("fizzbuzz_one", r"فزباز|fizzbuzz", r"fizzbuzz for one number",
   "اكتب دالة فزباز لعدد واحد: FizzBuzz لقابل القسمة على 15، Fizz للثلاثة، Buzz للخمسة، وإلا العدد نفسه",
   "write single-number fizzbuzz: FizzBuzz for 15, Fizz for 3, Buzz for 5, else the number",
   I, [(3, "Fizz"), (5, "Buzz"), (15, "FizzBuzz"), (7, "7")],
   [(30, "FizzBuzz"), (9, "Fizz"), (10, "Buzz"), (2026, "2026")],
   lambda x: "FizzBuzz" if x % 15 == 0 else
             ("Fizz" if x % 3 == 0 else
              ("Buzz" if x % 5 == 0 else str(x))),
   extra=["Fizz", "Buzz", "FizzBuzz"], maxsize=9, style="cond")
sp("is_leap", r"كبيسة", r"is a leap year",
   "اكتب دالة تفحص إن كانت السنة كبيسة",
   "write a function checking whether a year is a leap year",
   I, [(2024, True), (2026, False), (2000, True), (1900, False)],
   [(1996, True), (2100, False), (123, False), (0, True)],
   lambda x: (x % 4 == 0 and x % 100 != 0) or x % 400 == 0,
   extra=[], maxsize=8, style="bool")


# ── lookup: find a spec from a question ───────────────────────
_INTENT_RE = None


def find_spec(question: str):
    """AR/EN keyword match → spec (first hit wins). GATED behind
    explicit algorithm-writing intent (measured regression: broad
    keywords like رتب/اعكس hijacked KB questions like كيف أرتب
    قائمة تصاعدياً). Errors: none."""
    import re as _re
    global _INTENT_RE
    if _INTENT_RE is None:
        _INTENT_RE = _re.compile(
            r"اكتب دالة|دالة تعيد|دالة تحسب|دالة تفحص|دالة تجمع|"
            r"دالة تنتج|دالة تستخرج|دالة تضاعف|دالة ترتب|دالة تعكس|"
            r"دالة تحقق|دالة تعطي|دالة تحسب|خوارزمية|وظيفة تعيد|"
            r"write a function|write a single|a function that|"
            r"function to|function returning|function checking|"
            r"function counting|function summing|function producing|"
            r"function extracting|function capitalizing|"
            r"function reversing|function doubling|"
            r"function uppercasing|function lowercasing|"
            r"function swapping|function concatenating|"
            r"function joining|function removing|"
            r"function computing|algorithm", _re.I)
    if not _INTENT_RE.search(question):
        return None
    q = question.lower()
    for s in SP:
        if _re.search(s["ar"], question) or \
                _re.search(s["en"], q):
            return s
    return None


# ── adversarial enrichment: 4 examples + 6 hidden tests ───────
# (measured: 2 examples + 3 thin hidden tests let degenerate
# solutions like x%8 "pass" as leap years - wrong-type bools and
# coincidental compositions died only under adversarial sets)
_EX_POOL = {R_LI: [[7], [1, 2, 3, 4, 5], [2, 5, 8, 2], []],
            R_LS: [["x", "yy"], ["a"], []],
            R_S: ["Madam 2026", "ab3cd", "  HI  "],
            R_I: [40, 15, 90, 7]}
_HID_POOL = {R_LI: [[9, 1, 8, 7], [5, 5, 5], [1, 2, 3, 4, 5, 6, 7]],
             R_LS: [["ab", "cd", "ef"], ["q"], ["ok", "ok", "ok"]],
             R_S: ["Synora", "XYZ xyz 123", "racecar", ""],
             R_I: [1996, 1999, 2100, 123, 0, 999]}


def _eq(a, b):
    return a == b


for _s in SP:
    _seen = [i for i, _ in _s["examples"]]
    for _inp in _EX_POOL[_s["ptype"]]:
        if len(_s["examples"]) >= 4:
            break
        if any(_eq(_inp, u) for u in _seen):
            continue
        try:
            _out = _s["oracle"](_inp)
        except Exception:
            continue
        _s["examples"].append((_inp, _out))
        _seen.append(_inp)
    _seen_h = [i for i, _ in _s["hidden"]]
    for _inp in _HID_POOL[_s["ptype"]]:
        if len(_s["hidden"]) >= 6:
            break
        if any(_eq(_inp, u) for u in _seen_h):
            continue
        try:
            _out = _s["oracle"](_inp)
        except Exception:
            continue
        _s["hidden"].append((_inp, _out))
        _seen_h.append(_inp)
