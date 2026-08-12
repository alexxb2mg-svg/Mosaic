"""Pic RAM du build — le chiffre rejouable du plafond `pair_counts`.

Le build accumulait les comptages de cooccurrence dans un dict Python
{(i, j): float} : ~230 octets par paire de surcoût (tuple + boxing + table de
hachage) là où deux tableaux NumPy triés portent la même vérité pour 16 octets
par paire. Ce script mesure le PIC de working set du process pendant un build
réel — il a servi de juge avant/après la refonte en tableaux (Profiles.pair_keys/
pair_vals, consolidation par tampon borné).

Usage : python research/ram_build.py <corpus> <index_out>

Le même fichier tourne sur l'ancien code (dict) et le nouveau (tableaux) — le
champ `paires` lit l'une ou l'autre représentation — pour un A/B honnête :
    PYTHONPATH=<arbre>/src python research/ram_build.py <corpus> <out>
Windows uniquement (GetProcessMemoryInfo/psapi) ; la mesure inclut tout le
process (interpréteur, lexique, grilles), pas seulement le bloc épars — c'est
voulu : le plafond vécu est celui du process entier.
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaic.index import Index


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def pic_working_set_mo() -> int:
    pmc = _PROCESS_MEMORY_COUNTERS()
    pmc.cb = ctypes.sizeof(pmc)
    # Windows moderne : la fonction vit dans kernel32 (K32...) ; psapi.dll n'est
    # qu'un alias parfois absent. Types EXPLICITES : HANDLE est 64 bits, la
    # conversion c_int par défaut de ctypes corrompt le pseudo-handle (vécu).
    # Échec BRUYANT : un pic à 0 publié serait un mensonge.
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD,
    ]
    kernel32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = kernel32.K32GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
    )
    if not ok:
        raise OSError(f"K32GetProcessMemoryInfo a échoué ({ctypes.get_last_error()})")
    return int(pmc.PeakWorkingSetSize) // (1024 * 1024)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python research/ram_build.py <corpus> <index_out>")
        return 2
    corpus, out = Path(sys.argv[1]), Path(sys.argv[2])
    t0 = time.perf_counter()
    idx = Index.build(corpus, out, index_paths=False)
    duree = time.perf_counter() - t0
    profiles = idx.profiles
    if hasattr(profiles, "pair_keys"):  # représentation tableaux (post-refonte)
        paires = int(profiles.pair_keys.size)
    else:  # représentation dict (ancien code, pour l'A/B)
        paires = len(profiles.pair_counts)
    print(
        json.dumps(
            {
                "docs": len(idx.ids),
                "vocab": len(profiles.rows),
                "paires": paires,
                "pic_ram_mo": pic_working_set_mo(),
                "duree_s": round(duree, 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
