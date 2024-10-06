from typing import Dict

__edge_type_general_frequency__ = {
    "from_dataset": "PubChem-10m-clean",
    "total_mol_num": 9999918,
    "total_edge_num": 269700166,
    "edge_type_statistics": {
        # name -> count
        "C~=C": 88870216,
        "C-C": 73211463,
        "C-N": 35615403,
        "C-O": 18153400,
        "C~=N": 16269546,
        "C=O": 11102874,
        "C-F": 3960082,
        "C-S": 3417297,
        "C~=S": 2510134,
        "O=S": 2401531,
        "C-Cl": 2195569,
        "C=C": 1816899,
        "N~=N": 1782108,
        "C~=O": 1519097,
        "N-S": 900862,
        "Br-C": 885673,
        "C=N": 772132,
        "N-O": 609306,
        "C#N": 535127,
        "N=O": 428469,
        "C-Si": 391361,
        "N-N": 388547,
        "N~=O": 281166,
        "O-P": 248539,
        "C=S": 208591,
        "C#C": 164015,
        "C-I": 130775,
        "O-Si": 130293,
        "C-P": 127504,
        "O-S": 112856,
        "O=P": 92410,
        "N=N": 77027,
        "N~=S": 59512,
        "B-O": 48497,
        "B-C": 39528,
        "N-P": 26615,
        "O-O": 24523,
        "S-S": 15545,
        "Cl-S": 14390,
        "C-Sn": 12803,
        "C-Se": 10066,
        "N-Si": 8606,
        "Si-Si": 8427,
        "P=S": 7050,
        "B-N": 6513,
        "F-S": 5513,
        "B-B": 5300,
        "Cl-Si": 5136,
        "P-S": 5132,
        "B-F": 4989,
        "N=S": 4926,
        "P-P": 3965,
        "C-Ge": 3692,
        "Cl-P": 3575,
        "As-C": 3504,
        "Cl-N": 3490,
        "N=P": 3442,
        "O=Si": 3033,
        "C~=Se": 2657,
        "C=P": 2304,
        "C-Te": 1979,
        "F-N": 1881,
        "S=S": 1855,
        "C~=P": 1728,
        "Cl-O": 1590,
        "I-N": 1471,
        "O-Sn": 1431,
        "F-P": 1411,
        "F-Si": 1370,
        "O~=P": 1255,
        "N#N": 1208,
        "Al-O": 1092,
        "I-O": 1049,
        "Al-C": 1002,
        "I-I": 975,
        "Br-N": 918,
        "S~=S": 742,
        "C-Hg": 719,
        "F-O": 715,
        "B-P": 647,
        "N~=Se": 638,
        "As-O": 618,
        "I-S": 595,
        "C-Sb": 583,
        "C-Pb": 530,
        "Ge-O": 492,
        "S-Si": 490,
        "Br-Si": 476,
        "N~=P": 474,
        "Br-P": 462,
        "I-P": 372,
        "Se-Se": 369,
        "S-Sn": 351,
        "Bi-C": 336,
        "C=Se": 325,
        "B-Cl": 323,
        "I=O": 321,
        "Cl-Sn": 310,
        "P-Si": 292,
        "B-S": 288,
        "As=O": 286,
        "P-Se": 276,
        "O-Sb": 263,
        "O-Pb": 262,
        "Br-S": 256,
        "C=W": 248,
        "Br-O": 245,
        "C~=Te": 245,
        "Cl-Ge": 233,
        "C-In": 231,
        "P=Se": 226,
        "N-Sn": 226,
        "C=Si": 224,
        "O=Se": 223,
        "As-S": 213,
        "P=P": 212,
        "C-Ga": 204,
        "N=Se": 197,
        "O-Te": 197,
        "C#P": 187,
        "O-Se": 178,
        "Cl-Ru": 177,
        "Hg-O": 165,
        "P~=P": 161,
        "Cl-Hg": 161,
        "C#S": 160,
        "I-Si": 155,
        "Ge-N": 155,
        "P#P": 154,
        "N-Se": 153,
        "B-Br": 153,
        "S-Se": 150,
        "B-I": 150,
        "F-Te": 146,
        "B=O": 144,
        "O-W": 133,
        "Al-N": 132,
        "B=C": 131,
        "Cl-Te": 130,
        "Bi-O": 127,
        "Ga-O": 122,
        "C=Ru": 116,
        "In-O": 116,
        "Br-Sn": 98,
        "C-Tl": 98,
        "Al-Cl": 98,
        "Ge-S": 97,
        "C=I": 95,
        "B=N": 94,
        "As-N": 93,
        "B~=C": 91,
        "S-Sb": 86,
        "As-Cl": 85,
        "C~=Si": 84,
        "Ga-N": 82,
        "C=Cr": 82,
        "O=W": 82,
        "B-Si": 77,
        "Cl-I": 69,
        "As=C": 68,
        "Si=Si": 66,
        "F-I": 66,
        "Se-Si": 64,
        "Te-Te": 63,
        "B=B": 63,
        "O-Tl": 60,
        "Hg-S": 60,
        "Cl-Se": 60,
        "O=Sn": 59,
        "Br-Te": 59,
        "O~=O": 57,
        "Br-Se": 54,
        "C=Te": 51,
        "P~=S": 50,
        "Br-Ge": 50,
        "C=V": 49,
        "Cl-Sb": 48,
        "N-Te": 47,
        "In-N": 45,
        "P#S": 45,
        "Br-Hg": 45,
        "Pb-S": 45,
        "I-Sn": 43,
        "O-V": 43,
        "As-As": 43,
        "Mo-O": 42,
        "N-Pb": 41,
        "Hg-N": 39,
        "I=N": 38,
        "I-Te": 37,
        "Bi-S": 36,
        "C=Zr": 33,
        "O=Sb": 32,
        "N#S": 32,
        "As-F": 30,
        "N=Si": 29,
        "N-Sb": 29,
        "Si-Te": 28,
        "O=V": 28,
        "Cl-In": 27,
        "As=S": 26,
        "Mo=O": 26,
        "F-Sn": 26,
        "B#S": 25,
        "O=Te": 24,
        "C=Ge": 24,
        "C=Pt": 24,
        "S-Te": 24,
        "Hg-I": 24,
        "F-Ge": 24,
        "Cl-Zr": 24,
        "Br-Sb": 24,
        "As-Br": 23,
        "Al-Br": 23,
        "B=S": 23,
        "As-I": 23,
        "B=P": 23,
        "Al-S": 21,
        "F-Se": 20,
        "Se~=Se": 19,
        "Cl-Ga": 19,
        "C=Fe": 19,
        "P~=Se": 19,
        "F-Os": 19,
        "Cl-W": 19,
        "C=Mo": 18,
        "Br-I": 17,
        "As-Si": 17,
        "Ge-I": 17,
        "Bi-Cl": 16,
        "I-Se": 16,
        "Au-Cl": 16,
        "C=Ti": 16,
        "Cr=O": 16,
        "Cr-O": 16,
        "N#P": 16,
        "C#Si": 16,
        "C-Po": 15,
        "N=W": 15,
        "Mo=N": 15,
        "Ge=O": 14,
        "N~=Te": 14,
        "F-Sb": 14,
        "B-Se": 14,
        "S=Sn": 13,
        "Si#Si": 13,
        "N=V": 13,
        "Al=O": 13,
        "C#O": 13,
        "Se=Se": 12,
        "C#W": 12,
        "Br-Ir": 12,
        "P-Te": 12,
        "As=N": 11,
        "As=As": 11,
        "B~=N": 11,
        "Cl-Ta": 11,
        "B#P": 11,
        "Si~=Si": 11,
        "S~=Se": 10,
        "C=In": 10,
        "Cl-Os": 10,
        "C#Mo": 10,
        "Cl-Fe": 10,
        "O~=S": 10,
        "C=Pd": 10,
        "Al-F": 10,
        "S=W": 10,
        "I-V": 10,
        "Au=P": 10,
        "O=U": 10,
        "As-P": 10,
        "Br-Re": 10,
        "Br-In": 9,
        "C=Zn": 9,
        "Ge=Ge": 9,
        "N~=Si": 9,
        "Cl-Tc": 9,
        "Fe-I": 9,
        "Cl-V": 9,
        "C-Xe": 9,
        "B#C": 9,
        "Cl-Tl": 9,
        "Ga-S": 9,
        "O-Pr": 9,
        "C=Rh": 9,
        "P=Si": 9,
        "Cl-Pt": 8,
        "Cl-Pb": 8,
        "P=Te": 8,
        "Se-Te": 8,
        "C=Sn": 8,
        "Si=Zr": 8,
        "Br-W": 7,
        "In-S": 7,
        "C=Ni": 7,
        "F-Ta": 7,
        "O-Xe": 7,
        "Br-Mn": 7,
        "Fe=N": 7,
        "O-U": 7,
        "F-Xe": 7,
        "O=Tc": 6,
        "N=U": 6,
        "Cl-Ti": 6,
        "Al~=N": 6,
        "Br-Ta": 6,
        "O=Ru": 6,
        "C=Co": 6,
        "Co-F": 6,
        "C=Cu": 6,
        "O#S": 6,
        "Cl-Mo": 6,
        "F-Tl": 6,
        "C=Sb": 6,
        "K-Se": 6,
        "S#Te": 6,
        "C=Ir": 6,
        "Cl-Mn": 6,
        "C=Hf": 6,
        "B~=B": 6,
        "Bi-N": 6,
        "Cl-Rh": 6,
        "I-Po": 5,
        "As=P": 5,
        "Bi-Br": 5,
        "Cl-Re": 5,
        "F-Th": 5,
        "N=Rh": 5,
        "Eu-F": 5,
        "C=Os": 5,
        "Na-Se": 5,
        "Cl-Ni": 5,
        "Br-Ni": 5,
        "Te~=Te": 5,
        "Li-Se": 5,
        "I-Nb": 5,
        "Al-I": 5,
        "S=V": 5,
        "O=Re": 5,
        "Fe=P": 5,
        "I-Sb": 5,
        "O~=Te": 5,
        "Al=N": 5,
        "F-Pb": 5,
        "F-U": 5,
        "O=O": 5,
        "Br-Mo": 5,
        "O=Os": 5,
        "F-Ru": 5,
        "At-C": 4,
        "N=Zr": 4,
        "Au=C": 4,
        "O-Ti": 4,
        "B~=O": 4,
        "F-W": 4,
        "Br-Co": 4,
        "Br-Os": 4,
        "F-Mn": 4,
        "N=Tc": 4,
        "Cl-Co": 4,
        "C=Mn": 4,
        "Cl-Ir": 4,
        "S=Si": 4,
        "Bi=O": 4,
        "N=Nb": 4,
        "N=Pt": 4,
        "I-Mo": 4,
        "Ir=O": 4,
        "Nb=O": 4,
        "Br-Rh": 4,
        "I-Ru": 4,
        "F-Zr": 4,
        "I-Zr": 4,
        "I-U": 4,
        "I-Re": 4,
        "Ca-Ca": 4,
        "F-Pa": 4,
        "P=W": 3,
        "S-Tl": 3,
        "N=Os": 3,
        "Br-Fe": 3,
        "Br-Cr": 3,
        "S=Se": 3,
        "C=Yb": 3,
        "C=Ga": 3,
        "In=N": 3,
        "Ce-I": 3,
        "Ge=S": 3,
        "Cl-Th": 3,
        "Br-Pb": 3,
        "F-Pt": 3,
        "O-Re": 3,
        "N=Te": 3,
        "O-Ta": 3,
        "Br-Gd": 3,
        "Am-F": 3,
        "Br-Tl": 3,
        "Br-Ru": 3,
        "C#Y": 3,
        "O-Po": 3,
        "Cl-No": 3,
        "C=Tl": 3,
        "I-Pt": 3,
        "S#S": 3,
        "Ra-Ra": 3,
        "Cs-Se": 3,
        "Au-Br": 3,
        "F-Sm": 3,
        "O=Pb": 3,
        "Br-Pt": 3,
        "Br-Y": 3,
        "P=Rh": 3,
        "Cl-Ho": 3,
        "C-Pr": 3,
        "I-Tl": 3,
        "Cl-Pm": 3,
        "F-Yb": 3,
        "N=Sn": 2,
        "As-Se": 2,
        "N=Ti": 2,
        "Ge=Si": 2,
        "O-Tc": 2,
        "I#N": 2,
        "Cl-Cr": 2,
        "O=Ti": 2,
        "C#Cr": 2,
        "Mg-Se": 2,
        "Br-Cl": 2,
        "Mo=Te": 2,
        "C=Re": 2,
        "I-Ni": 2,
        "Ge=P": 2,
        "C=Pb": 2,
        "N=Ta": 2,
        "O#P": 2,
        "I-Mn": 2,
        "C=Ta": 2,
        "N#Te": 2,
        "Mo#Si": 2,
        "Mo#N": 2,
        "Bi-I": 2,
        "P~=Te": 2,
        "Ga=O": 2,
        "Ga-I": 2,
        "Cl-Eu": 2,
        "Se=Si": 2,
        "C=Hg": 2,
        "Fe=V": 2,
        "N=Sb": 2,
        "Hf=S": 2,
        "N#Se": 2,
        "I=I": 2,
        "Cl-Hf": 2,
        "F-Ga": 2,
        "Re#Re": 2,
        "Ir-O": 2,
        "O-Ru": 2,
        "N#W": 2,
        "Se=Th": 2,
        "Nb-O": 2,
        "N#Si": 2,
        "Br-Yb": 2,
        "Ac-C": 2,
        "Se=W": 2,
        "S=Y": 2,
        "Bi=C": 2,
        "Mn=O": 2,
        "Mn-O": 2,
        "O=Zr": 2,
        "Cu-I": 2,
        "Fe=S": 2,
        "K-Te": 2,
        "I-Zn": 2,
        "Hf=O": 2,
        "Cl-Pd": 2,
        "N=Ru": 2,
        "Br-Zr": 2,
        "Cd-I": 2,
        "Nb=Se": 2,
        "S=Sb": 2,
        "Ni=O": 2,
        "Ni-O": 2,
        "Mo=Si": 2,
        "Ce=Si": 2,
        "Fe=Sb": 2,
        "Se=Te": 1,
        "Ge=Se": 1,
        "Ge=Zr": 1,
        "S=U": 1,
        "I-Pb": 1,
        "Ba=S": 1,
        "Hg=S": 1,
        "O=Tl": 1,
        "O#Si": 1,
        "P#Si": 1,
        "Na-Te": 1,
        "Ru=S": 1,
        "P=Pt": 1,
        "C#Os": 1,
        "S-Xe": 1,
        "Os=P": 1,
        "I-Ir": 1,
        "Ge=Ti": 1,
        "Se=Sn": 1,
        "P#Ta": 1,
        "O=Ta": 1,
        "N=Zn": 1,
        "Cl-Y": 1,
        "N=Y": 1,
        "As=Sb": 1,
        "N=Pd": 1,
        "Co=P": 1,
        "S#Si": 1,
        "O~=Si": 1,
        "N-Tl": 1,
        "Hf=N": 1,
        "Bi#C": 1,
        "F-V": 1,
        "Sb=Sb": 1,
        "Cu=O": 1,
        "Bi-F": 1,
        "F-Rh": 1,
        "Ge=Hf": 1,
        "I-In": 1,
        "B=I": 1,
        "I-W": 1,
        "Ge#Mo": 1,
        "Cl-Nb": 1,
        "C#Ge": 1,
        "Au-I": 1,
        "Hf=Si": 1,
        "Ge=N": 1,
        "In=In": 1,
        "I-Y": 1,
        "Cr=P": 1,
        "Br-Ga": 1,
        "As#Co": 1,
        "F-In": 1,
        "P#W": 1,
        "Fe=Zr": 1,
        "I#P": 1,
        "I=P": 1,
        "Pb=S": 1,
        "Co=S": 1,
        "I-Ti": 1,
        "At-O": 1,
        "Ge#P": 1,
        "Br-Pd": 1,
        "Ru=Ru": 1,
        "Mn=N": 1,
        "Ge#N": 1,
        "Sn=Te": 1,
        "I-Rh": 1,
        "N-Xe": 1,
        "Ag-F": 1,
        "Ag-I": 1,
        "Li-Te": 1,
        "Ni=P": 1,
        "P=Pd": 1,
        "N=Ni": 1,
        "P=Sn": 1,
        "C#Re": 1,
        "Ge=Te": 1,
        "Si=Ti": 1,
        "Rb-Te": 1,
        "Ra-Te": 1,
        "As#Fe": 1,
        "Br-Ti": 1,
        "Ag=C": 1,
        "Ga=S": 1,
        "P=V": 1,
        "Ge#Ge": 1,
        "N-Po": 1,
        "Pt=Sn": 1,
        "Cl-Cu": 1,
        "Br-Cu": 1,
        "B=Si": 1,
        "others": 0
    }
    # "total_edge_types": 579
}

from echo_logger import print_info

EDGE_TYPES_DICT = {
    name: id_ for id_, name in enumerate(__edge_type_general_frequency__["edge_type_statistics"].keys())
}

EDGE_TYPES_DICT_OLD: Dict[str, int] = {
    # name -> id
    # Here, `-` means single bond, `=` means double bond, `#` means triple bond, and `~=` means aromatic bond.
    "C-C": 0,
    "C~=C": 1,
    "C-N": 2,
    "C-O": 3,
    "C=O": 4,
    "C~=N": 5,
    "C-S": 6,
    "C=C": 7,
    "C-Cl": 8,
    "C-F": 9,
    "C=N": 10,
    "O=S": 11,
    "C~=S": 12,
    "C~=O": 13,
    "N~=N": 14,
    "N-O": 15,
    "N-S": 16,
    "N-N": 17,
    "Br-C": 18,
    "N=O": 19,
    "N~=O": 20,
    "C#N": 21,
    "O-P": 22,
    "C#C": 23,
    "O-S": 24,
    "C=S": 25,
    "O=P": 26,
    "C-P": 27,
    "N~=S": 28,
    "C-I": 29,
    "N=N": 30,
    "S-S": 31,
    "N-P": 32,
    "B-O": 33,
    "P=S": 34,
    "P-S": 35,
    "B-C": 36,
    "others": 37,
}

if __name__ == '__main__':
    print_info(EDGE_TYPES_DICT)
