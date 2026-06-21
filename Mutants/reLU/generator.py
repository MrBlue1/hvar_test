import os

os.makedirs("mutants", exist_ok=True)

# ===== 原始代码 =====
def build_original():
    return """def relu(x):
    if isinstance(x, list):
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
"""

# 写入 M00
with open("mutants/M00.py", "w", encoding="utf-8") as f:
    f.write(build_original())

operator_map = {"M00": "ORIG"}
counter = 1


def save(code, op_type):
    global counter
    name = f"M{counter:02d}"
    with open(f"mutants/{name}.py", "w", encoding="utf-8") as f:
        f.write(code)
    operator_map[name] = op_type
    counter += 1


# =========================================================
# 基础构造函数（关键：明确结构，不用 replace）
# =========================================================
def build_relu(list_cond="v > 0",
               list_expr="v",
               list_else="0",
               scalar_cond="x > 0",
               scalar_expr="x",
               scalar_else="0",
               pre_comment=""):
    return f"""def relu(x):
    {pre_comment}
    if isinstance(x, list):
        return [{list_expr} if {list_cond} else {list_else} for v in x]
    else:
        return {scalar_expr} if {scalar_cond} else {scalar_else}
"""


# =========================================================
# 1️⃣ ROR（关系运算）10个
# =========================================================
ror_list_ops = ["< 0", ">= 0", "<= 0", "== 0", "!= 0"]
for op in ror_list_ops:
    save(build_relu(
        list_cond=f"v {op}",
        pre_comment=f"# MUTATION: list分支 v > 0 -> v {op} （ROR）"
    ), "ROR")

ror_scalar_ops = ["< 0", ">= 0", "<= 0", "== 0", "!= 0"]
for op in ror_scalar_ops:
    save(build_relu(
        scalar_cond=f"x {op}",
        pre_comment=f"# MUTATION: 标量分支 x > 0 -> x {op} （ROR）"
    ), "ROR")


# =========================================================
# 2️⃣ AOR（算术）10个
# =========================================================
aor_list_exprs = ["v+1", "v-1", "v*2", "v/2", "v+0.1"]
for expr in aor_list_exprs:
    save(build_relu(
        list_expr=expr,
        pre_comment=f"# MUTATION: list分支 v -> {expr} （AOR）"
    ), "AOR")

aor_scalar_exprs = ["x+1", "x-1", "x*2", "x/2", "x+0.1"]
for expr in aor_scalar_exprs:
    save(build_relu(
        scalar_expr=expr,
        pre_comment=f"# MUTATION: 标量分支 x -> {expr} （AOR）"
    ), "AOR")


# =========================================================
# 3️⃣ LOR（逻辑）10个
# =========================================================
lor_list_conds = [
    "(v > 0 and True)",
    "(v > 0 and False)",
    "(v > 0 or True)",
    "(v > 0 or False)",
    "not (v <= 0)"
]

for cond in lor_list_conds:
    save(build_relu(
        list_cond=cond,
        pre_comment=f"# MUTATION: list条件 -> {cond} （LOR）"
    ), "LOR")

lor_scalar_conds = [
    "(x > 0 and True)",
    "(x > 0 and False)",
    "(x > 0 or True)",
    "(x > 0 or False)",
    "not (x <= 0)"
]

for cond in lor_scalar_conds:
    save(build_relu(
        scalar_cond=cond,
        pre_comment=f"# MUTATION: 标量条件 -> {cond} （LOR）"
    ), "LOR")


# =========================================================
# 4️⃣ COR（条件替换）10个
# =========================================================
cor_conditions = [
    "True",
    "False",
    "isinstance(x, int)",
    "isinstance(x, float)",
    "not isinstance(x, list)",
    "len(x) > 0",
    "len(x) == 0",
    "x is None",
    "x == []",
    "bool(x)"
]

for cond in cor_conditions:
    code = f"""def relu(x):
    # MUTATION: if条件 -> {cond} （COR）
    if {cond}:
        return [v if v > 0 else 0 for v in x]
    else:
        return x if x > 0 else 0
"""
    save(code, "COR")


# =========================================================
# 5️⃣ UOI（一元）10个
# =========================================================
uoi_list = ["-v", "+v", "abs(v)", "v*-1", "--v"]
for expr in uoi_list:
    save(build_relu(
        list_expr=expr,
        pre_comment=f"# MUTATION: list v -> {expr} （UOI）"
    ), "UOI")

uoi_scalar = ["-x", "+x", "abs(x)", "x*-1", "--x"]
for expr in uoi_scalar:
    save(build_relu(
        scalar_expr=expr,
        pre_comment=f"# MUTATION: scalar x -> {expr} （UOI）"
    ), "UOI")


# =========================================================
# 6️⃣ SDL（语句删除）10个
# =========================================================
sdl_cases = [
    "return x",
    "return 0",
    "return None",
    "return []",
    "pass",
    "return [v for v in x]",
    "return x if x else 0",
    "return 1",
    "return -1",
    "return True"
]

for stmt in sdl_cases:
    code = f"""def relu(x):
    # MUTATION: 删除主体逻辑 -> {stmt} （SDL）
    {stmt}
"""
    save(code, "SDL")


# =========================================================
# 7️⃣ ABS（5个）
# =========================================================
abs_cases = ["abs(v)", "abs(x)", "abs(v)+1", "abs(x)+1", "abs(v*2)"]

for expr in abs_cases:
    save(build_relu(
        list_expr=expr,
        pre_comment=f"# MUTATION: 引入abs -> {expr} （ABS）"
    ), "ABS")


# =========================================================
# 生成索引
# =========================================================
with open("MutantsIndex.py", "w", encoding="utf-8") as f:
    f.write("OPERATOR_MAPPING = {\n")
    for k, v in operator_map.items():
        f.write(f"    '{k}': '{v}',\n")
    f.write("}\n")

print(f"✅ 完成，共生成 {counter-1} 个高质量单点变异体")