"""Download the upstream mesh files that this repository does NOT redistribute.

Open_Duck_Playground (https://github.com/apirrone/Open_Duck_Playground) has no license, so 28 simplified skeleton meshes that
the MuJoCo scenes and the URDF point at are not stored in this repository. This script fetches exactly those files from GitHub at
one pinned commit, checks every file against the SHA-256 table below, and only then writes it. A file with a wrong hash is never
written. Only the Python standard library is used.

Run once after cloning, from anywhere:
    python3 tools/fetch_upstream_meshes.py            # download missing files
    python3 tools/fetch_upstream_meshes.py --check    # only verify what is on disk

Targets: printed_face/mujoco/assets/ and lcd_face/mujoco/assets/ (all 28 files in each; the LCD scene uses the same 28 meshes), and
printed_face/urdf/assets/ (the 11 files listed in URDF_ALSO).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_COMMIT = "b9be205ac64488c23504ca42e5ec790337adeec3"
RAW_URL = ("https://raw.githubusercontent.com/apirrone/Open_Duck_Playground/{commit}/"
           "playground/open_duck_mini_v2/xmls/assets/{name}")
MUJOCO_DIR = REPO_ROOT / "printed_face" / "mujoco" / "assets"
LCD_MUJOCO_DIR = REPO_ROOT / "lcd_face" / "mujoco" / "assets"
URDF_DIR = REPO_ROOT / "printed_face" / "urdf" / "assets"
TIMEOUT_S = 60

# file name -> SHA-256 of the file at UPSTREAM_COMMIT
SHA256: dict[str, str] = {
    "antenna.stl": "8d18ff3ed5dc5beb89c15b95fc886e4d57f5fd8f76969532fe83f679eb0f08e6",
    "body_back.stl": "7166e8ccd5aa659566bf84c659a36c33248740a18d1eb04b0e34904dde71e733",
    "body_front.stl": "5c98c85f91a979df57ecda622b64f35689efd2d076df0ca584186daf773cb47c",
    "body_middle_bottom.stl": "3f91cfeeffc197d39a5e11e26811825cbffdaedcc3eea0f4195d701f994eb203",
    "body_middle_top.stl": "0929c3d0c45b9b41273b1333655c467bec994f6489df5ea2c7c1298e4c4d7d39",
    "foot_bottom_pla.stl": "4290199f08b3e8eb4ecd4f07c32c4d77ecf82839396ccd7d6062eea96cef8e6d",
    "foot_bottom_tpu.stl": "dfa3c58757e1d838b462d225c2dbfbb517e8fc57e3c6a9979c4894c801386e8a",
    "foot_side.stl": "3d564b42ef9c2bae340fd0b06cfadfe905c713ff84ad40b21d4181128d225138",
    "foot_top.stl": "0712dd6aec2e6d6c821b8bfcd0ba71bf7c8017f63889de5f6847e62814624a40",
    "head.stl": "e81b30c44d10d93194091d5fbb65bd4ccb6bd4e9504c027344c247d2ffc6eaf4",
    "head_bot_sheet.stl": "39cdafc528646e1ca627b357e8f9886a9f51df3d53dcfd566d3b7979fd592953",
    "head_pitch_to_yaw.stl": "a45c8dc5fb75ec001fd1e94185db635c60aa68f163482395d6c55db4ac678238",
    "head_yaw_to_roll.stl": "3a94ac2824e94326565a3e07f8794c9246c90f78c119a4577835b9c70745557b",
    "left_antenna_holder.stl": "01536add58f330d36fd9c63c4a6caeb414a1d931a27e501f63567a0785f93d23",
    "left_cache.stl": "8b50aa42f2f34b1e99c0702e6d6d64ee1b0a7fa7e856d97558b2388310930f85",
    "left_knee_to_ankle_left_sheet.stl": "e69aff1a7baeedd18676348ea72d185229722f8b1348bb5b2ffcafc7de9715bf",
    "left_knee_to_ankle_right_sheet.stl": "2518ec33ff2b018304b80026603d3f88e8e41f27474b5f43f1f7e4b5e10c1461",
    "left_roll_to_pitch.stl": "1af14b0e3bef3298d20f3d5b985848aaf9dc69097c8b449c43fc63181ee01d66",
    "leg_spacer.stl": "4f23745dac3598daf9bf6089e3864a318d091a5613af8544aecf3e68d6a6772e",
    "neck_left_sheet.stl": "2dc8e910a537509a018191c4e2a0b535b78c4ff822b6d2cf39f34427c83e38a6",
    "neck_right_sheet.stl": "afd4c04b1081a0a68d92a017fabc16d093f1ddeed52a212b5e50585cc2bb928e",
    "right_antenna_holder.stl": "39604b35cbe5f0070371fc460e51f89b09c20a06c003bff9cf87f7685234648a",
    "right_cache.stl": "391a1477b223cea4bbea37a64d154188294093c677cc52f62880ec7f001f061e",
    "right_roll_to_pitch.stl": "22d3e6f5e0afd386bc0bf288077ae99f3822f82ea4c1e6e47bd36fdcc21d1599",
    "roll_motor_bottom.stl": "3dc60095901ac2332cfb798dcd53149c2e98abeb636637f6bd7e8b9757fa9ecf",
    "roll_motor_top.stl": "89872314b73e926dc1a37dc4d0d4756bf74e82316a8f207cb160924ec0372e62",
    "trunk_bottom.stl": "d2382a154ccb241911225a560cf49985d363d476699a9198eb4ad9d5bfbb5e36",
    "trunk_top.stl": "943a7c75a81ede71fc985ecc37ccd8efec05b9be604302e04af8c9fa3a516c16",
}

# the subset that is also needed next to the URDF
URDF_ALSO: tuple[str, ...] = (
    "foot_bottom_pla.stl",
    "foot_bottom_tpu.stl",
    "foot_side.stl",
    "foot_top.stl",
    "left_knee_to_ankle_left_sheet.stl",
    "left_knee_to_ankle_right_sheet.stl",
    "leg_spacer.stl",
    "roll_motor_bottom.stl",
    "roll_motor_top.stl",
    "trunk_bottom.stl",
    "trunk_top.stl",
)


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download(name: str) -> bytes:
    url = RAW_URL.format(commit=UPSTREAM_COMMIT, name=name)
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return response.read()


def targets(name: str) -> list[Path]:
    dirs = [MUJOCO_DIR, LCD_MUJOCO_DIR] + ([URDF_DIR] if name in URDF_ALSO else [])
    return [d / name for d in dirs]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="only verify the files on disk, download nothing")
    args = parser.parse_args()
    bad = 0
    for name, expected in sorted(SHA256.items()):
        paths = targets(name)
        missing = [p for p in paths if not p.is_file() or sha256_of(p.read_bytes()) != expected]
        if not missing:
            continue
        if args.check:
            print(f"MISSING OR WRONG: {name}")
            bad += 1
            continue
        try:
            data = download(name)
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"DOWNLOAD FAILED: {name}: {error}")
            bad += 1
            continue
        if sha256_of(data) != expected:
            print(f"HASH MISMATCH, not written: {name}")
            bad += 1
            continue
        for path in missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f"ok {name} -> {len(missing)} place(s)")
    total = len(SHA256)
    print(f"{total - bad} of {total} files OK" + ("" if bad == 0 else f", {bad} problem(s)"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
