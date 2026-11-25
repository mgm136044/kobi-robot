# 코비 20축 애니메이션용 스켈레톤(리그) 파일

이 폴더에는 리타게팅과 애니메이션 팀이 Blender, Maya, Unity, MotionBuilder 같은 도구에서 바로 읽을 수 있는 코비 20축의 뼈대(스켈레톤) 파일이 들어 있습니다. 뼈대는 시뮬레이션에 쓰는 MuJoCo 모델 `kobi20/mujoco/kobi20_v8.xml`에서 자동으로 뽑았고, 관절 20개가 같은 이름의 뼈로 들어 있습니다. 그래서 시뮬레이션의 관절 각도를 그대로 뼈의 회전으로 옮길 수 있습니다. 인쇄 얼굴 판 코비의 같은 종류 파일(`printed_face/skeleton/`)과 같은 도구로 같은 방식으로 만들었습니다.

**먼저 알아 둘 점 두 가지입니다.**

1. **기준 자세(`rest`, 모든 관절 각도 0)는 이 로봇이 낼 수 없는 자세입니다.** 어깨 옆 관절(`left_shoulder_roll_joint`, `right_shoulder_roll_joint`)의 허용 범위가 19.25도부터라서 0도는 범위 밖입니다. `rest`는 뼈의 위치를 맞추는 기준 틀이며 로봇이 취하는 자세가 아닙니다. 로봇이 서 있는 자세는 `home` 동작에 따로 있습니다(4절). 보여 주거나 리타게팅의 시작 자세로 쓸 때는 `home`을 쓰세요.
2. **이 로봇에는 정책(학습된 신경망)이 없습니다.** 그래서 인쇄 얼굴 판 파일에 있던 "정책 행동 번호" 대신 **MuJoCo MJCF의 구동기(액추에이터) 번호**(`data.ctrl`의 순서)를 뼈마다 붙였습니다. 이름은 `actuator_index`입니다. 토크 한계는 MJCF 구동기의 힘 범위 상한(서보 정지 토크)입니다.

검증은 Blender 5.3.0에서의 재읽기와 파일 내부 직접 읽기로 했습니다. **Maya, Unity, MotionBuilder 프로그램에서 직접 열어 본 것은 아닙니다.** 검증 방법과 숫자는 7절에 있습니다.

## 1. 파일 목록

| 파일 | 크기 | 쓰는 도구 | 내용 |
|---|---|---|---|
| `kobi20_rig.blend` | 16 MB | Blender | 원본입니다. 뼈대, 겉모양 메시, 동작 2개(`rest`, `home`)가 모두 들어 있습니다. 다른 파일은 모두 이 파일에서 만들었습니다. |
| `kobi20_rig.fbx` | 19 MB | Maya, Unity, MotionBuilder 등 | 뼈대와 스킨 입힌 메시와 동작 2개입니다. |
| `kobi20_rig.glb` | 13.5 MB | 웹(three.js), Godot, Unity 플러그인, Blender 등 | glTF 2.0 바이너리입니다. 스킨과 동작 2개(`rest`, `home`)가 들어 있습니다. **줄인 메시를 씁니다**(삼각형 705,207개 -> 101,813개, 아래 1절). 뼈와 스킨 가중치와 동작은 다른 파일과 같습니다. |
| `kobi20_skeleton.bvh` | 4 KB | MotionBuilder, 리타게팅 도구 | 뼈대만 있고 메시는 없습니다. 1프레임이 기준 자세(`rest`), 2프레임이 홈 자세(`home`)입니다. |
| `skeleton.json` | 100 KB | 직접 만드는 도구 | 뼈와 관절의 위치, 회전축, 범위, 토크 한계, 메시 배치를 숫자로 적은 원본 데이터입니다. 위 파일들은 모두 이 파일에서 만듭니다. |

겉모양 메시는 한 개의 물체(`Kobi_Mesh`)로 합쳐져 있고, 꼭짓점 352,027개와 삼각형 705,207개이며 색 재질은 2개(빨강 섀시 부품, 흰색 외피)입니다. 질감 그림은 없습니다. `.blend`와 `.fbx`는 이 원본 메시(전체 해상도)를 그대로 담고 있습니다. **`.glb`만 줄인 메시를 씁니다.** 파일 크기를 93.6 MB에서 13.5 MB로 줄이려고, `kobi20/urdf/assets_light/`의 줄인 STL 81개(가벼운 URDF가 쓰는 것과 같은 파일)로 메시를 다시 만들어 내보냈습니다. 삼각형은 705,207개에서 101,813개(약 14%)로, 꼭짓점은 352,027개에서 50,484개로 줄었습니다. 줄이면서 생긴 표면 오차는 면 위 표본점 양방향 측정으로 최대 3.39 mm(얇은 화면 테두리 한 곳), 대부분 1.4 mm 이하, 상위 99% 오차 최대 1.35 mm입니다(`kobi20/urdf/results/light_mesh_error.json`). 뼈 21개, 뼈마다 꼭짓점 그룹과 가중치 1.0, 재질 2개, 동작 `rest`와 `home`은 다른 파일과 같고, 메시가 놓인 위치와 몸체 대응도 같습니다. 모양만 거칠어졌습니다. 전체 해상도 메시가 필요하면 `.blend`나 `.fbx`를 쓰세요. 뼈대를 만들고 읽을 때 쓴 이름은 인쇄 얼굴 판과 같게 물체 `Kobi_Armature`, 메시 `Kobi_Mesh`입니다. 두 로봇을 한 장면에 함께 불러오면 이름이 겹치니 하나는 이름을 바꾸세요.

## 2. 뼈 구조와 구동기 번호

뼈는 21개이고, 관절 하나가 뼈 하나를 움직입니다. 뼈 이름은 그 뼈를 돌리는 관절 이름(MJCF와 URDF의 관절 이름, 끝에 `_joint`가 붙음)과 같습니다. 뿌리 뼈 `pelvis`만 관절이 없고(바닥 위에서 자유롭게 움직이는 몸통) MJCF의 `pelvis` 몸체에 해당합니다. 허리(`torso_link`), 어깨 돌리기 자리(`*_shoulder_yaw_link`), 손목 자리(`*_wrist_roll_link`)는 20축에서 부모에 용접된 몸체라서 따로 뼈가 없고 부모 뼈에 붙어 함께 움직입니다(아래 5절).

```
pelvis
  left_hip_yaw_joint -> left_hip_roll_joint -> left_hip_pitch_joint -> left_knee_joint -> left_ankle_pitch_joint -> left_ankle_roll_joint
  right_hip_yaw_joint -> right_hip_roll_joint -> right_hip_pitch_joint -> right_knee_joint -> right_ankle_pitch_joint -> right_ankle_roll_joint
  neck_yaw_joint -> neck_pitch_joint
  left_shoulder_pitch_joint -> left_shoulder_roll_joint -> left_elbow_joint
  right_shoulder_pitch_joint -> right_shoulder_roll_joint -> right_elbow_joint
```

| 번호(구동기) | 뼈 이름 | 부모 뼈 | 회전축(월드, 기준 자세) | 범위(도) | 홈 자세 값(라디안) | 토크 한계(N·m) |
|---|---|---|---|---|---|---|
| 0 | left_hip_yaw_joint | pelvis | +z | -45 ~ 45 | 0 | 1.5 |
| 1 | left_hip_roll_joint | left_hip_yaw_joint | +x | -24 ~ 27 | 0 | 4.1 |
| 2 | left_hip_pitch_joint | left_hip_roll_joint | +y | -51 ~ 33 | -0.1081 | 1.5 |
| 3 | left_knee_joint | left_hip_pitch_joint | +y | -12 ~ 60 | 0.15 | 1.5 |
| 4 | left_ankle_pitch_joint | left_knee_joint | +y | -24 ~ 24 | -0.0419 | 1.5 |
| 5 | left_ankle_roll_joint | left_ankle_pitch_joint | +x | -9 ~ 9 | 0 | 1.5 |
| 6 | right_hip_yaw_joint | pelvis | -z | -45 ~ 45 | 0 | 1.5 |
| 7 | right_hip_roll_joint | right_hip_yaw_joint | -x | -24 ~ 27 | 0 | 4.1 |
| 8 | right_hip_pitch_joint | right_hip_roll_joint | +y | -51 ~ 33 | -0.1081 | 1.5 |
| 9 | right_knee_joint | right_hip_pitch_joint | +y | -12 ~ 60 | 0.15 | 1.5 |
| 10 | right_ankle_pitch_joint | right_knee_joint | +y | -24 ~ 24 | -0.0419 | 1.5 |
| 11 | right_ankle_roll_joint | right_ankle_pitch_joint | -x | -9 ~ 9 | 0 | 1.5 |
| 12 | neck_yaw_joint | pelvis | +z | -90 ~ 90 | 0 | 1.5 |
| 13 | neck_pitch_joint | neck_yaw_joint | +y | -16.75 ~ 12 | 0 | 1.5 |
| 14 | left_shoulder_pitch_joint | pelvis | +y | -96 ~ 3.75 | 0 | 0.52 |
| 15 | left_shoulder_roll_joint | left_shoulder_pitch_joint | +x | 19.25 ~ 92 | 0.3491 | 0.52 |
| 16 | left_elbow_joint | left_shoulder_roll_joint | +y | -61 ~ 3 | -0.3142 | 0.52 |
| 17 | right_shoulder_pitch_joint | pelvis | +y | -96 ~ 3.75 | 0 | 0.52 |
| 18 | right_shoulder_roll_joint | right_shoulder_pitch_joint | -x | 19.25 ~ 92 | 0.3491 | 0.52 |
| 19 | right_elbow_joint | right_shoulder_roll_joint | +y | -61 ~ 3 | -0.3142 | 0.52 |

- 번호는 MuJoCo MJCF의 구동기 순서(`data.ctrl`)이며, 왼다리 6, 오른다리 6, 목 2, 왼팔 3, 오른팔 3입니다. **정책 행동 번호가 아닙니다**(정책이 없습니다). 각 뼈에는 이 번호(`actuator_index`), 범위(`range_min_rad`, `range_max_rad`), 토크 한계(`torque_limit_nm`)가 Blender 뼈의 사용자 속성으로 붙어 있습니다. 다른 도구에서는 `skeleton.json`의 `joints`를 보세요(`index_meaning` 칸에 번호의 뜻이 적혀 있습니다).
- 토크 한계는 MJCF 구동기의 힘 범위 상한입니다(XL430-W250 1.5, XM430-W350 4.1, XL330-M288 0.52 N·m). 서보 정지 토크이며 연속으로 낼 수 있는 값이 아닙니다.
- 회전축 열은 모든 관절 각도가 0일 때의 월드 방향입니다. 오른쪽의 엉덩이 비틀기, 엉덩이 옆, 발목 옆, 어깨 옆은 축 방향이 왼쪽과 반대(-z, -x)라서, **같은 양의 각도를 주면 서로 거울 방향으로 돕니다.** 시뮬레이션 모델 그대로입니다. 사람 동작에서 뽑은 각도를 옮길 때 오른쪽의 이 네 관절은 부호를 맞춰야 합니다(`kobi20/README.md` 4절). 범위의 숫자는 왼쪽과 오른쪽이 같습니다.
- 범위는 소수점 둘째 자리로 줄여 적었습니다(정확한 라디안 값은 `skeleton.json`).
- 이름으로 찾을 수 있게 `skeleton.json`에는 MuJoCo 몸체 이름(`left_hip_roll_link`, `neck_pitch_link` 등)과 뼈의 대응도 들어 있습니다(`bodies`와 `joints`의 `child_body`).

## 3. 좌표축과 회전 규칙

### 3.1 모든 파일에 공통인 규칙

- **관절은 자기 뼈의 로컬 Z축을 중심으로 돕니다.** 뼈의 머리(시작점)는 관절 위치(MuJoCo의 관절 앵커)에 있고, 뼈를 만들 때 로컬 Z축이 정확히 관절 회전축과 같아지도록 롤을 맞췄습니다. 그래서 뼈의 로컬 Z 회전량이 곧 관절 각도(라디안)입니다. 양수 방향은 MuJoCo와 같습니다(회전축 방향으로 볼 때 오른손 법칙).
- **기준 자세(rest pose)는 모든 관절 각도가 0인 자세**입니다. 이때 모든 뼈의 회전값이 0입니다. **어깨 옆 범위 밖의 자세입니다**(맨 위 안내). 로봇이 서 있는 홈 자세와 다르니 헷갈리지 마세요.
- 뼈의 꼬리 방향(로컬 Y축)은 회전축과 수직이고, 자식 관절이 회전축과 수직으로 1 cm 이상 떨어져 있으면 그쪽을, 아니면 몸체 무게 중심 쪽을 향합니다. 자식이 없는 뼈(발목 옆, 목 끄덕임, 팔꿈치)는 몸체 무게 중심 쪽입니다. 꼬리는 모양을 보기 위한 것일 뿐 관절 위치를 정하지 않습니다.
- 뿌리 뼈 `pelvis`는 로컬 축이 월드 축과 같도록 만들었습니다. 그래서 위치 이동값을 월드 방향 그대로 읽을 수 있습니다.
- **길이 단위는 미터**입니다(FBX는 3.3절 참고). 각도는 Blender에서 라디안이고, glTF에서는 사원수이며, FBX와 BVH 파일 안에서는 도입니다.
- **로봇이 바라보는 방향은 +X입니다.** MuJoCo 월드를 그대로 썼기 때문입니다(x 앞, y 왼쪽, z 위). 다른 도구의 "앞쪽" 규칙(예: glTF는 +Z)과 다르므로, 보기 방향을 맞추려면 도구 쪽에서 몸 전체를 돌려야 합니다. 시뮬레이션 숫자와 직접 비교하려면 돌리지 않는 편이 안전합니다.

### 3.2 파일 형식별 축

| 파일 | 위쪽 | 이 파일에서 코비의 앞쪽 | 좌표 변환(MuJoCo에서 파일로) | 비고 |
|---|---|---|---|---|
| `.blend` | +Z | +X | 변환 없음 | Blender 기본 좌표입니다. |
| `.glb` | +Y | +X | (x, y, z) -> (x, z, -y) | glTF 규칙(Y 위)입니다. Blender의 glTF 내보내기 기본값(`export_yup=True`)을 썼습니다. 뼈 노드의 로컬 Z축이 관절 축이고, 단위는 미터입니다. |
| `.fbx` | +Y | +X | (x, y, z) -> (x, z, -y) | 아래 3.3절 |
| `.bvh` | +Z | +X | 변환 없음 | Blender가 쓰는 월드 좌표 그대로입니다. 각도는 도입니다. 아래 3.4절 |

### 3.3 FBX 설정과 읽는 법

Blender FBX 내보내기의 기본 축 설정을 그대로 썼습니다(Maya, Unity, MotionBuilder에서 열어 본 것은 아닙니다).

- 축: `axis_forward = -Z`, `axis_up = Y`. 파일 머리말에는 위쪽 Y, 앞쪽 +Z, 좌표 X로 적힙니다.
- 단위: `global_scale = 1.0`, `apply_unit_scale = True`, `apply_scale_options = FBX_SCALE_NONE`, `bake_space_transform = False`. 파일의 단위 배율은 1.0(센티미터)입니다.
- 이 설정의 결과로 최상위 물체 `Kobi_Armature`에 회전 (-90°, 0, 0)과 배율 100이 들어가고, 그 아래 뼈의 위치 값은 미터로 적힙니다. 즉 `Kobi_Armature` 안은 미터, 바깥에서 보면 센티미터 월드입니다.
- 뼈: `add_leaf_bones = False`(끝에 가짜 뼈를 붙이지 않음), `primary_bone_axis = Y`, `secondary_bone_axis = X`, 뼈 노드 종류는 `LimbNode`.
- **FBX 안의 뼈 회전은 기준 자세의 방향을 포함한 전체 오일러 각도(XYZ 순서, 도)로 저장됩니다.** 21개 뼈 중 18개에서 기준 자세 회전 값이 0이 아닙니다. 그래서 FBX에서 "Z 회전 값 = 관절 각도"가 성립하지 않습니다. 정확한 식은 `뼈의 기준 자세 회전 × (뼈 로컬 Z축 둘레로 관절 각도만큼 회전)`입니다. 관절 각도를 얻으려면 `rest` 동작을 기준으로 삼아 `rest` 대비 상대 회전을 구하세요.
- 동작 이름은 `Kobi_Armature|rest`, `Kobi_Armature|home`으로 들어갑니다.
- 이 설정의 장점은 Blender 기본값 그대로라는 점입니다. 단점은 위의 배율 100 때문에 일부 도구에서 최상위 물체의 배율이 100으로 보일 수 있다는 점입니다. 그럴 때는 `tools/skeleton/build_rig.py`의 `export_fbx`에서 `apply_scale_options`를 `FBX_SCALE_ALL`로 바꿔 다시 만들어 보세요(이 설정은 시험하지 않았고, 바꾸면 검증을 다시 돌려야 합니다).

### 3.4 BVH 읽는 법

- BVH는 뼈의 롤을 저장하지 못합니다. 모든 관절의 회전 틀이 월드 축과 나란하다고 보고, 기준 자세에서 0도인 상태를 기준으로 월드 축 둘레의 회전(X, Y, Z 오일러, 도)을 적습니다. 그래서 BVH의 값은 "뼈 로컬 Z 회전"이 아니라 월드 축 기준 회전입니다. 예를 들어 `left_knee_joint`의 회전축은 월드 +y이므로 Y 회전 채널에 관절 각도(도)가 들어가고, `right_hip_yaw_joint`의 축은 -z이므로 Z 회전 채널에 부호가 반대인 값이 들어갑니다.
- 뿌리 `pelvis`만 위치 채널(미터)이 있고, 나머지는 회전 채널만 있습니다. 프레임 간격은 1/30초입니다.
- MotionBuilder처럼 위쪽이 Y이고 단위가 센티미터인 도구에서는 불러온 뒤 좌표축과 배율(100배)을 맞추세요.

## 4. 동작(액션) 두 개

각 동작은 1프레임과 2프레임에 같은 자세가 키로 들어 있습니다(프레임 간격 1/30초). 한 프레임짜리 클립을 싫어하는 도구가 있어서 길이가 0이 아니게 했습니다.

- `rest`: 모든 관절 각도 0, 뿌리 위치 (0, 0, 0.185) m. 기준 자세이며 리타게팅의 기준 틀로 쓰되, **로봇이 낼 수 없는 자세**입니다(어깨 옆 0도는 범위 밖).
- `home`: MuJoCo 키프레임 `home`의 20개 관절 값(2절의 "홈 자세 값")과 뿌리 위치 (0, 0, 0.185802) m, 뿌리 방향은 회전 없음입니다. 로봇이 서 있는 자세입니다. 어깨 옆 20도, 팔꿈치 -18도, 엉덩이 앞뒤 -6.193도, 무릎 8.594도, 발목 앞뒤 -2.401도(좌우 같은 값)이고 나머지는 0입니다. `rest`에서 `home`으로 가면서 뿌리가 z로 +0.000802 m 움직이고 x와 y는 그대로입니다. 이 값은 뿌리 뼈의 위치 이동 채널에 들어 있습니다.

Blender에서는 두 동작이 가짜 사용자 표시(F)가 붙어 저장되어 있고 현재 동작은 `rest`입니다. glTF에서는 동작 이름이 `rest`, `home`인 애니메이션 두 개입니다. BVH는 한 클립에 `rest`(1프레임)와 `home`(2프레임)을 이어 붙였습니다.

## 5. 겉모양 메시와 스킨

- 한 물체 `Kobi_Mesh`에 뼈마다 꼭짓점 그룹이 하나씩 있고(21개), 모든 꼭짓점은 한 뼈에만 가중치 1.0으로 묶여 있습니다(몸체 하나가 뼈 하나이므로 단단한 스킨입니다). 부드럽게 휘는 스킨이 아닙니다. 팔다리 부품이 몸체째로 움직입니다.
- 뼈에 붙는 메시(모두 81조각): `pelvis` 8조각(골반 섀시 3, 허리 몸통과 재킷, 날개 5), `neck_yaw_joint` 2조각, `neck_pitch_joint` 11조각(머리), 어깨 앞뒤 두 뼈 각 3조각, 어깨 옆 두 뼈 각 6조각(위팔과 소매, 팔꿈치 공), 팔꿈치 두 뼈 각 6조각(아래팔, 소매, 손), 엉덩이 비틀기 두 뼈와 엉덩이 옆 두 뼈와 발목 앞뒤 두 뼈 각 1조각, 엉덩이 앞뒤 두 뼈와 무릎 두 뼈 각 3조각, 발목 옆 두 뼈 각 6조각(발과 덮개)입니다.
- **용접된 몸체는 부모 뼈에 붙어 있습니다.** 허리 몸통(`torso_link`)은 `pelvis` 뼈에, 어깨 돌리기 자리 링크(`*_shoulder_yaw_link`, 팔꿈치 공 외피가 붙은 링크)는 어깨 옆 뼈에, 손목 자리 링크(`*_wrist_roll_link`, 손 장착부와 장갑)는 팔꿈치 뼈에 붙습니다. 20축에서 이 몸체들의 관절이 없기 때문입니다. 허리, 팔 돌리기, 손목 돌리기 뼈는 없으므로 사람 동작의 그 각도는 버리거나 가장 가까운 관절로 근사해야 합니다.
- 모양은 기준 자세(모든 각도 0)에서 만들었습니다. MuJoCo가 읽은 겉모양과 위치가 같은지 확인했고(최대 차이 7.4e-9 m), 서로 닿는 꼭짓점은 한 조각 안에서만 합쳤습니다. 면의 음영은 평평(플랫) 셰이딩입니다.
- 색은 MJCF의 재질 색을 단순 색으로 옮겼습니다. 거칠기 0.6, 금속성 없음입니다.
- 서보 모터의 보이는 모양은 없습니다(모델에 서보는 무게만 있고 보이는 모양이 없습니다). 보이는 메시는 섀시 부품과 외피뿐입니다.

## 6. 메시에 대해: 모두 우리가 만든 것

이 로봇의 메시 81개는 전부 우리 팀이 만든 섀시 부품과 외피입니다. 인쇄 얼굴 판 스켈레톤에서는 라이선스가 없는 업스트림 메시 28개를 뺐지만, 이 로봇에는 그런 메시가 없어 **빠진 메시가 하나도 없습니다**(`skeleton.json`의 `excluded_upstream_geoms`는 빈 목록입니다). 외피에는 코리 캐릭터의 모양이 들어 있으므로 캐릭터 권리에 대한 안내(저장소 `NOTICE.md`)를 확인하세요.

## 7. 검증 결과

`tools/skeleton/verify_rig.py`가 만든 파일을 다시 읽어 MuJoCo의 순기구학(`mj_kinematics`) 결과와 비교합니다. 허용 오차는 1e-4 m(0.1 mm)입니다. 아래는 이 폴더의 최종 파일로 다시 돌린 결과입니다(Blender 5.3.0, MuJoCo 3.14.0).

- ① 한 일: 관절 범위 안에서 무작위로 고른 20가지 자세와 `rest`, `home`을 파일에 적용해, 모든 뼈 머리의 월드 위치를 MuJoCo의 관절 앵커 위치와 비교했습니다. 자세를 만들 때는 각 뼈의 로컬 관절 축을 중심으로 관절 각도만큼만 돌렸습니다.
- ② 값의 뜻: 아래 표의 오차는 가장 크게 어긋난 뼈 하나의 거리(미터)입니다. 모두 3e-7 m보다 작아서, 허용 오차보다 300배 이상 작습니다. 이 정도는 숫자 반올림 수준입니다. (참고로 `rest`는 로봇이 낼 수 없는 자세이지만, 이 검사는 계산(순기구학)만 비교하므로 문제가 없습니다.)
- ③ 왜 믿을 수 있나: 시험이 제대로 작동하는지 일부러 두 관절(`right_knee_joint`, `left_shoulder_pitch_joint`)의 정답 각도를 부호만 뒤집어 넣어 봤고, 그때 오차가 0.13~0.29 m로 커지며 실패로 나왔습니다.

| 검사한 대상 | 방법 | 뼈 머리 위치 최대 오차(m) | 스킨된 메시 꼭짓점 최대 오차(m) |
|---|---|---|---|
| `kobi20_rig.blend` | 원본 파일, 로컬 Z 오일러 회전 | 7.0e-8 (무작위 20가지), 1.5e-8 (rest), 1.6e-8 (home) | 2.0e-7 |
| `kobi20_rig.fbx` | Blender로 새로 불러옴 | 1.4e-7, 1.1e-7, 1.0e-7 | 2.6e-7 |
| `kobi20_rig.fbx` | 파일 내부를 직접 해석(Blender 없이), Y 위 변환과 100배 단위 포함 | 8.3e-8, rest 6.9e-8, home 6.0e-8 | 해당 없음 |
| `kobi20_rig.glb`(줄인 메시) | Blender로 새로 불러옴 | 1.8e-7, 1.4e-7, 9.6e-8 | 1.8e-7 |
| `kobi20_rig.glb`(줄인 메시) | 파일 내부를 직접 해석(JSON과 이진 데이터), (x, y, z) -> (x, z, -y) 포함 | 5.8e-8, rest 3.5e-8, home 5.6e-8 | 해당 없음 |
| `kobi20_skeleton.bvh` | 새로 불러와 1프레임(rest)과 2프레임(home) 비교 | 3.0e-8 | 해당 없음 |
| BVH 시험 파일 | 같은 자세 22개를 새로 구운 22프레임 BVH | 2.4e-7 | 해당 없음 |

- 뼈 개수 21개, 뼈 이름, 부모 관계는 `skeleton.json`과 모든 파일(blend, FBX와 GLB의 불러온 결과와 직접 해석, BVH)에서 일치했습니다.
- 모든 꼭짓점이 정확히 한 뼈에 가중치 1.0으로 묶여 있습니다(blend 352,027개 전부). 다시 불러온 FBX와 GLB에서도 꼭짓점마다 가중치가 있는 뼈가 하나뿐입니다(GLB는 줄인 메시이고, 불러올 때 꼭짓점이 280,660개로 늘어납니다. 면마다 법선을 쪼개기 때문이며 모양은 같습니다).
- 스킨 검사: 기준 자세에서 읽은 꼭짓점을 자세에 따라 움직인 뒤, MuJoCo의 해당 몸체 움직임(회전과 이동)을 같은 꼭짓점에 직접 적용한 값과 비교했습니다.
- 뼈 로컬 Z가 관절 축이라는 주장도 확인했습니다: blend와 다시 불러온 FBX와 GLB, 그리고 FBX와 GLB의 파일 내부 노드에서 기준 자세의 관절 축을 뼈 로컬 좌표로 바꾸면 모두 (0, 0, 1)입니다(오차 1e-6 이하, blend 1.5e-7, FBX 6.6e-7, GLB 5.5e-7). BVH는 롤이 없어서 이 검사를 하지 않았습니다.
- 도구를 20축에 맞게 고친 뒤에도 인쇄 얼굴 판 결과가 그대로인지 확인했습니다. 기본 선택지로 다시 만든 `printed_face/skeleton/skeleton.json`과 BVH는 저장소 파일과 바이트까지 같았고, GLB는 내보내기 프로그램 버전 문자열(5.2.40과 5.3.43)을 빼고 같았으며, 다시 만든 인쇄 얼굴 판 파일의 검증도 모두 통과했습니다(뼈 머리 위치 오차 최대 6.5e-7 m).
- 메모리: 가장 큰 Blender 실행(검증)의 최대 상주 메모리는 약 2.35 GB였고, 뼈대 만들기는 약 1.2 GB였습니다.

## 8. 다시 만드는 방법

모든 명령은 저장소 맨 위 폴더에서 실행합니다. Blender 5.3.0(이번에 쓴 판)과 `uv`가 필요합니다. 이 로봇은 업스트림 메시가 없으므로 메시 내려받기 단계가 없습니다.

```
# 1. MJCF에서 skeleton.json과 검증용 정답(MuJoCo 순기구학)을 뽑기
uv run --no-project --python 3.12 --with mujoco==3.14.0 --with numpy python -I tools/skeleton/extract_skeleton.py \
    --profile kobi20 --fk-reference out/skeleton_verify/fk_reference.json

# 2. 뼈대, 메시, 동작을 만들고 .blend, .fbx, .glb, .bvh 저장
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/build_rig.py -- \
    --skeleton kobi20/skeleton/skeleton.json --repo-root . --out-dir kobi20/skeleton --name-prefix kobi20 \
    --glb-mesh-dir kobi20/urdf/assets_light     # GLB만 줄인 메시로(없으면 GLB도 전체 해상도, 93.6 MB)

# 3. 검증(실패하면 마지막 줄에 FAIL, 성공하면 PASS)
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/verify_rig.py -- \
    --skeleton kobi20/skeleton/skeleton.json --fk-reference out/skeleton_verify/fk_reference.json \
    --rig-dir kobi20/skeleton --work-dir out/skeleton_verify --report out/skeleton_verify/report.json --name-prefix kobi20
```

- `out/`는 저장소에서 제외되는 폴더입니다(`.gitignore`). 검증용 정답과 시험 BVH, 보고서 JSON이 여기에 생깁니다.
- 1단계의 `--profile kobi20`이 이 로봇용 설정입니다. 기본값(`printed_face`)은 인쇄 얼굴 판 코비이고 그 결과는 바뀌지 않았습니다. `--samples`(기본 20)와 `--seed`로 무작위 자세 수와 난수 씨앗을 바꿀 수 있습니다.
- `--profile kobi20`은 MJCF의 모든 관절이 구동기 하나씩과 짝이 있는지 확인하고, 번호를 구동기 순서로 붙이고, 토크 한계를 구동기 힘 범위에서 읽고, 뿌리 뼈 이름을 `pelvis`로 하고, 메시를 하나도 빼지 않습니다.
- 이번 `.glb`는 2단계에 `--glb-mesh-dir kobi20/urdf/assets_light`를 더해서 만들었습니다(`.glb`만 그 폴더의 줄인 STL로 만들고 `.blend`와 `.fbx`는 전체 해상도입니다. 이 선택지를 빼면 `.glb`도 전체 해상도가 되어 93.6 MB입니다). 줄인 메시를 안 쓰고 싶을 때의 다른 방법인 `--max-export-triangles 300000` 같은 값은 FBX와 GLB 둘 다 줄이는데, 이번 산출물에는 쓰지 않았고 이 로봇에서 시험하지도 않았습니다).

## 9. 알려진 한계

- Maya, Unity, MotionBuilder 프로그램에서 직접 열어 본 것은 아닙니다. 위 검증은 Blender와, FBX와 GLB를 직접 읽는 코드로 한 것입니다.
- **기준 자세(`rest`)는 로봇이 낼 수 없는 자세입니다**(맨 위 안내, 어깨 옆 0도는 범위 밖). 로봇이 서 있는 자세는 `home`입니다.
- FBX의 뼈 회전은 기준 자세를 포함한 오일러 값이므로, 도구에서 "관절 각도"를 바로 읽을 수는 없습니다(3.3절). GLB와 Blender는 뼈 로컬 Z 회전이 그대로 관절 각도입니다.
- `kobi20_rig.glb`는 줄인 메시로 만들었습니다(1절). 모양이 `.blend`와 `.fbx`보다 거칠고 표면이 최대 3.39 mm(대부분 1.4 mm 이하) 어긋납니다. 뼈와 동작은 같습니다.
- 가중치는 단단한 스킨(1.0)이라 천이나 꼬리가 휘는 표현은 없습니다.
- 외피 부품끼리 부딪히는 자세는 이 뼈대가 막지 않습니다. 관절 범위 안이라도 부딪힐 수 있는 자세가 있습니다(`kobi20/shell_rules/README.md`).
- 오른발 충돌 상자가 왼발보다 좌우로 조금 넓은 것(8.35 cm 대 8.09 cm) 같은 시뮬레이션 모델의 값은 이 도구에서 만든 것이 아니라 MJCF에 이미 있는 값을 그대로 옮긴 것입니다.
