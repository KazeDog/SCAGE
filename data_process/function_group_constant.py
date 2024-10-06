from functools import lru_cache, cache
from unittest import TestCase

from echo_logger import *

__FUNCTION_GROUP_LIST_FROM_GITHUB = [
    # C
    # # "[CX4]",
    "[$([CX2](=C)=C)]", "[$([CX3]=[CX3])]", "[$([CX2]#C)]",
    # C & O
    "[CX3]=[OX1]", "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]", "[CX3](=[OX1])C", "[OX1]=CN", "[CX3](=[OX1])O",
    "[CX3](=[OX1])[F,Cl,Br,I]", "[CX3H1](=O)[#6]", "[CX3](=[OX1])[OX2][CX3](=[OX1])", "[NX3][CX3](=[OX1])[#6]",
    "[NX3][CX3]=[NX3+]", "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]", "[NX3][CX3](=[OX1])[OX2H0]",
    "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]", "[CX3](=O)[O-]", "[CX3](=[OX1])(O)O", "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
    "C[OX2][CX3](=[OX1])[OX2]C", "[CX3](=O)[OX2H1]", "[CX3](=O)[OX1H0-,OX2H1]", "[NX3][CX2]#[NX1]",
    "[#6][CX3](=O)[OX2H0][#6]", "[#6][CX3](=O)[#6]", "[OD2]([#6])[#6]",
    # H
    "[H+]",
    "[+H]",
    # N
    "[n]1cccc1",  # we added this
    "[NX3;H2,H1;!$(NC=O)]", "[NX3][CX3]=[CX3]", "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
    "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]", "[NX3][$(C=C),$(cc)]", "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
    "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
    "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]", "[CH3X4]",
    "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]", "[CH2X4][CX3](=[OX1])[NX3H2]",
    "[CH2X4][CX3](=[OX1])[OH0-,OH]", "[CH2X4][SX2H,SX1H0-]", "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
    "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]", "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1", "[CHX4]([CH3X4])[CH2X4][CH3X4]",
    "[CH2X4][CHX4]([CH3X4])[CH3X4]", "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]", "[CH2X4][CH2X4][SX2][CH3X4]",
    "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
    # # "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]", "[CH2X4][OX2H]",
    "[NX3][CX3]=[SX1]", "[CHX4]([CH3X4])[OX2H]", "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
    "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1", "[CHX4]([CH3X4])[CH3X4]",
    "N[CX4H2][CX3](=[OX1])[O,N]", "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
    "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]", "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
    # "[#7]",
    "[NX2]=N", "[NX2]=[NX2]", "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
    "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]", "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]", "[NX3][NX3]",
    "[NX3][NX2]=[*]", "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
    "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]", "[NX3+]=[CX3]", "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])", "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
    "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]", "[NX1]#[CX2]", "[CX1-]#[NX2+]",
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
    "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]", "[NX2]=[OX1]",
    "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
    # O
    "[OX2H]", "[#6][OX2H]", "[OX2H][CX3]=[OX1]", "[OX2H]P", "[OX2H][#6X3]=[#6]", "[OX2H][cX3]:[c]",
    "[OX2H][$(C=C),$(cc)]", "[$([OH]-*=[!#6])]", "[OX2,OX1-][OX2,OX1-]",
    # P
    "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
    "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
    # S
    "[S-][CX3](=S)[#6]", "[#6X3](=[SX1])([!N])[!N]", "[SX2]", "[#16X2H]", "[#16!H0]", "[#16X2H0]", "[#16X2H0][!#16]",
    "[#16X2H0][#16X2H0]", "[#16X2H0][!#16].[#16X2H0][!#16]", "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
    "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
    "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]", "[SX4](C)(C)(=O)=N",
    "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
    "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]", "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
    "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
    "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
    "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
    "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
    "[#16X2][OX2H,OX1H0-]", "[#16X2][OX2H0]",
    # X
    "[#6][F,Cl,Br,I]", "[F,Cl,Br,I]", "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]", "[CX3](=[OX1])[F,Cl,Br,I]"
]


@cache
def _get_new_fn_groups(filepath: str = None):
    """Used for debug only"""
    filepath = Path(filepath) if filepath else os.path.dirname(os.path.realpath(__file__)) + "/new_fg_groups.txt"
    with open(filepath, 'r') as f:
        new_list = f.readlines()
    new_list = {smiles.strip() for smiles in new_list if smiles.strip()}
    new_list_dict = {f"{index + 144}": smiles for index, smiles in
                     enumerate(new_list) if smiles.strip()}
    return new_list_dict


# 最终方法
# 用原网址的 SMARTS 列表，然后做一些修改（去掉氨基酸），然后每个原子可以（先按照最长的，如果实在不行可以有多个匹配到的官能团，我们可以取最长的 k 个，比如 k=3）
# 把第三小节的列表的官能团加上去   手性的官能团不加了。 非氢用 3.1 杂化来取代。
# 连通性的需要加。链条和分支也需要加。
# 别忘了 check 一下，看看有没有错误的
# 优先级官能团
# 长的大于短的，环 > 链 > 杂化 > 单个.  更新：杂化、手性不做了。 三。 里面就加环，链，连通性。也可以多标签。

# TODO 大环单独考虑，不应该覆盖其他或者被覆盖。其他类似的官能团也考虑一下。（主要是大环和长链）
# TODO 去重！比如三个苯环相连
FUNCTION_GROUP_LIST_FROM_DAYLIGHT = sorted(list(set(__FUNCTION_GROUP_LIST_FROM_GITHUB + [
    # Connectivity
    "[$([NX4+]),$([NX4]=*)]",  # Quaternary Nitrogen
    "[$([SX3]=N)]",  # Tri coordinate S double bonded to N.
    "[$([SX1]=[#6])]",  # S double-bonded to Carbon
    "[$([NX1]#*)]",  # Triply bonded N
    "[$([OX2])]",  # Divalent Oxygen
    # Chains & Branching
    # # "[R0;D2][R0;D2][R0;D2][R0;D2]",  # Unbranched_alkane groups.
    # # "[R0;D2]~[R0;D2]~[R0;D2]~[R0;D2]",  # Unbranched_chain groups.
    "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]",  # Long_chain groups.
    # # "[!$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",  # Atom_fragment A fragment atom is a not an isolating carbon 我没用这个，因为基本上所有的匹配这个的原子，都能匹配一个别的。
    # # "[$([#6+0]);!$(C(F)(F)F);!$(c(:[!c]):[!c])!$([#6]=,#[!#6])]",  # Carbon_isolating WARN: I did not use this one.
    "[$([SX1]~P)]",  # Terminal S bonded to P
    "[$([NX3]C=N)]",  # Nitrogen on -N-C=N-
    "[$([NX3]N=C)]",  # Nitrogen on -N-N=C-
    "[$([NX3]N=N)]",  # Nitrogen on -N-N=N-
    "[$([OX2]C=N)]",  # Oxygen in -O-C=N-
    # Cyclic Features
    # # "[$([*R2]([*R])([*R])([*R]))].[$([*R2]([*R])([*R])([*R]))]",  # Bicyclic 不相邻的两个环，感觉不需要
    ""
    # # "*-!:aa-!:*",  # Ortho
    # # "*-!:aaa-!:*",  # Meta      # TODO 优先级 只有没有别的官能团的时候，才用这种环，因为会返回整个环的原子，也就是 match 的小 tuple 会很大。
    # 根据咱们的匹配算法，这会认为 环 的优先级特别大。一旦某个纯各种环分子中有一个超级大环，那所有的都会是环而不是别的。
    # # "*-!:aaaa-!:*",  # Para
    # # "*!@*",  # Acylic-bonds
    # # "*-!@*",  # Non-ring atom
    # # "[R0] or [!R]",  # Non-ring atom   WARN: I did not use this one.
    "[r;!r3;!r4;!r5;!r6;!r7]",  # Macrocycle
    "[sX2r5]",  # S in aromatic 5-ring with lone pair
    "[oX2r5]",  # Aromatic 5-Ring O with Lone Pair
    "[nX2r5]",  # N in 5-sided aromatic ring
    "[X4;R2;r4,r5,r6](@[r4,r5,r6])(@[r4,r5,r6])(@[r4,r5,r6])@[r4,r5,r6]",  # Spiro-ring center  # 试一下发现匹配了螺旋碳以及它附近的4个原子
    "[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]",  # N in 5-ring arom
    # # "*/,\\[R]=;@[R]/,\\*",  # CIS or TRANS double bond in a ring
    # # "*/,\\[R]=,:;@[R]/,\\*",  # CIS or TRANS double or aromatic bond in a ring
    "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # Unfused benzene ring
    "[cR1]1[cR1][cR1][cR1][cR1][cR1]1.[cR1]1[cR1][cR1][cR1][cR1][cR1]1",  # Multiple non-fused benzene rings
    "c12ccccc1cccc2",  # Fused benzene rings
    # Amino Acid
    "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",  # Amino Acid
    "c",
    "c1ccccc1"  # 苯环
    # 谁在ring里面, 区分一下分子

    # Add all the elements
]))) + list(_get_new_fn_groups().values()) + ['N', 'O', 'S']
# assert FUNCTION_GROUP_LIST_FROM_DAYLIGHT no dulicates
__dul_flag = FUNCTION_GROUP_LIST_FROM_DAYLIGHT == sorted(list(set(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)))
if __dul_flag:
    # print duplicates
    duplicate_list = []
    set_ = set(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)
    for item in set_:
        if FUNCTION_GROUP_LIST_FROM_DAYLIGHT.count(item) > 1:
            duplicate_list.append(item)
    raise ValueError(f"FUNCTION_GROUP_LIST_FROM_DAYLIGHT has duplicates:\n{dumps_json(duplicate_list)}")

INDEPENDENT_FUNCTION_GROUP_LIST = [
    "[r;!r3;!r4;!r5;!r6;!r7]",
    "[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]"
]

reference_fn_group = {
    str(index + 1): smarts for index, smarts in enumerate(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)
}


def get_independent_fn_group_ids() -> Dict[str, int]:
    # return INDEPENDENT_FUNCTION_GROUP_LIST element ids in FUNCTION_GROUP_LIST_FROM_DAYLIGHT
    return {fn_group: FUNCTION_GROUP_LIST_FROM_DAYLIGHT.index(fn_group) + 1 for fn_group in
            INDEPENDENT_FUNCTION_GROUP_LIST}


def get_independent_fn_group_ids_new() -> Dict[str, int]:
    # return INDEPENDENT_FUNCTION_GROUP_LIST element ids in FUNCTION_GROUP_LIST_FROM_DAYLIGHT
    return {fn_group: FUNCTION_GROUP_LIST_FROM_DAYLIGHT.index(fn_group) + 1 for fn_group in
            INDEPENDENT_FUNCTION_GROUP_LIST}


__fn_group_id_to_smarts = {}


@cache
def nfg() -> int:
    """Function Group Number"""
    return len(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)


@lru_cache()
def get_fn_group_id2smarts() -> Dict[int, str]:
    """
    start from 1
    Returns:

    """
    global __fn_group_id_to_smarts
    if not __fn_group_id_to_smarts:
        for i, line in enumerate(FUNCTION_GROUP_LIST_FROM_DAYLIGHT):
            __fn_group_id_to_smarts[i + 1] = line
        # __fn_group_id_to_smarts[0] = "None"
    return __fn_group_id_to_smarts


class TestFGConstant(TestCase):
    def test_get_independent_fn_group_ids(self):
        # assert dumps_json(FUNCTION_GROUP_LIST_FROM_DAYLIGHT).strip() == dumps_json(reference_fn_group).strip()
        self.maxDiff = None
        json_1 = dumps_json(
            {str(index + 1): smarts for index, smarts in enumerate(FUNCTION_GROUP_LIST_FROM_DAYLIGHT)}).strip()
        json_2 = dumps_json(reference_fn_group).strip()
        self.assertEqual(json_1, json_2)

    def test_new_fn_groups(self):
        new_list = _get_new_fn_groups()
        print_info(dumps_json(new_list, indent=2))

    def test_12367(self):
        print(_get_new_fn_groups().values())

    def test_87463278(self):
        print_info(dumps_json(get_fn_group_id2smarts(), indent=2))

    def test_nfg(self):
        print_info(nfg())


if __name__ == '__main__':
    FUNCTION_GROUP_LIST_FROM_DAYLIGHT_DICT = get_fn_group_id2smarts()
    print_info(dumps_json(FUNCTION_GROUP_LIST_FROM_DAYLIGHT_DICT, indent=2))
