# 코비 애니메이션용 스켈레톤(리그) 파일

이 폴더에는 리타게팅과 애니메이션 팀이 Blender, Maya, Unity, MotionBuilder 같은 도구에서 바로 읽을 수 있는 코비의 뼈대(스켈레톤) 파일이 들어 있습니다. 뼈대는 시뮬레이션에 쓰는 MuJoCo 모델 `printed_face/mujoco/kobi_scene.xml`에서 자동으로 뽑았고, 관절 14개가 정책의 행동 번호 0~13과 같은 이름으로 들어 있습니다. 그래서 시뮬레이션의 관절 각도를 그대로 뼈의 회전으로 옮길 수 있습니다.

**먼저 알아 둘 점: Blender 5.2.2에서의 재읽기와 파일 내부 직접 읽기로 검증했습니다. Maya, Unity, MotionBuilder 프로그램에서 직접 열어 본 것은 아닙니다.** 검증 방법과 숫자는 7절에 있습니다.

## 1. 파일 목록

| 파일 | 크기 | 쓰는 도구 | 내용 |
|---|---|---|---|
| `kobi_rig.blend` | 7.9 MB | Blender | 원본입니다. 뼈대, 겉모양 메시, 동작 2개(`rest`, `home`)가 모두 들어 있습니다. 다른 파일은 모두 이 파일에서 만들었습니다. |
| `kobi_rig.fbx` | 8.7 MB | Maya, Unity, MotionBuilder 등 | 뼈대와 스킨 입힌 메시와 동작 2개입니다. |
| `kobi_rig.glb` | 39 MB | 웹(three.js), Godot, Unity 플러그인, Blender 등 | glTF 2.0 바이너리입니다. 스킨과 동작 2개(`rest`, `home`)가 들어 있습니다. |
| `kobi_skeleton.bvh` | 3 KB | MotionBuilder, 리타게팅 도구 | 뼈대만 있고 메시는 없습니다. 1프레임이 기준 자세(`rest`), 2프레임이 홈 자세(`home`)입니다. |
| `skeleton.json` | 76 KB | 직접 만드는 도구 | 뼈와 관절의 위치, 회전축, 범위, 토크 한계, 메시 배치를 숫자로 적은 원본 데이터입니다. 위 파일들은 모두 이 파일에서 만듭니다. |

겉모양 메시는 한 개의 물체(`Kobi_Mesh`)로 합쳐져 있고, 꼭짓점 151,191개와 삼각형 302,061개이며 색 재질 9개입니다. 질감 그림은 없습니다. 파일이 95 MB보다 훨씬 작아 줄이기(데시메이트)는 하지 않았고, 모든 파일에 같은 메시가 들어 있습니다.

## 2. 뼈 구조와 정책 행동 번호

뼈는 15개이고, 관절 하나가 뼈 하나를 움직입니다. 뼈 이름은 그 뼈를 돌리는 관절 이름과 같습니다. 뿌리 뼈 `trunk`만 관절이 없고(바닥 위에서 자유롭게 움직이는 몸통) 몸통 몸체 `trunk_assembly`에 해당합니다.

```
trunk
  left_hip_yaw -> left_hip_roll -> left_hip_pitch -> left_knee -> left_ankle
  right_hip_yaw -> right_hip_roll -> right_hip_pitch -> right_knee -> right_ankle
  neck_yaw -> neck_pitch
  shoulder_left
  shoulder_right
```

| 행동 번호 | 뼈 이름 | 부모 뼈 | 회전축(월드, 기준 자세) | 범위(도) | 홈 자세 값(라디안) | 토크 한계(N·m) |
|---|---|---|---|---|---|---|
| 0 | left_hip_yaw | trunk | +z | -30 ~ 30 | 0.002 | 1.91 |
| 1 | left_hip_roll | left_hip_yaw | -x | -25 ~ 25 | 0.053 | 1.91 |
| 2 | left_hip_pitch | left_hip_roll | -y | -70 ~ 30 | -0.9303 | 1.91 |
| 3 | left_knee | left_hip_pitch | -y | -90 ~ 90 | 1.3812 | 1.91 |
| 4 | left_ankle | left_knee | -y | -90 ~ 90 | -0.4815 | 1.91 |
| 5 | right_hip_yaw | trunk | +z | -30 ~ 30 | -0.003 | 1.91 |
| 6 | right_hip_roll | right_hip_yaw | -x | -25 ~ 25 | -0.065 | 1.91 |
| 7 | right_hip_pitch | right_hip_roll | +y | -30 ~ 70 | 0.9353 | 1.91 |
| 8 | right_knee | right_hip_pitch | -y | -90 ~ 90 | 1.3922 | 1.91 |
| 9 | right_ankle | right_knee | -y | -90 ~ 90 | -0.4871 | 1.91 |
| 10 | neck_yaw | trunk | +z | -30 ~ 30 | 0 | 1.91 |
| 11 | neck_pitch | neck_yaw | +y | -10 ~ 10 | 0 | 1.91 |
| 12 | shoulder_left | trunk | +y | -135 ~ 25 | 0 | 0.20 |
| 13 | shoulder_right | trunk | +y | -135 ~ 25 | 0 | 0.20 |

- 행동 번호는 정책의 행동, 관측, MuJoCo 구동기 순서와 같습니다(저장소 `README.md` 4절). 각 뼈에는 이 번호(`action_index`), 범위(`range_min_rad`, `range_max_rad`), 토크 한계(`torque_limit_nm`)가 Blender 뼈의 사용자 속성으로 붙어 있습니다. 다른 도구에서는 `skeleton.json`의 `joints`를 보세요.
- 토크 한계는 학습할 때 쓴 값입니다(다리와 목 1.91, 어깨 0.20). MJCF 파일 안의 ±3.23 N·m가 아닙니다(`README.md` 4절 참고).
- 회전축 열은 모든 관절 각도가 0일 때의 월드 방향입니다. 왼쪽과 오른쪽 엉덩이 앞뒤(pitch)는 축 방향이 반대(-y와 +y)라서, 같은 양의 각도라도 서로 거울 방향으로 돕니다. 이는 시뮬레이션 모델 그대로입니다.
- 이름으로 찾을 수 있게 `skeleton.json`에는 MuJoCo 몸체 이름(`hip_roll_assembly`, `kobi_head` 등)과 뼈의 대응도 들어 있습니다(`bodies`와 `joints`의 `child_body`).

## 3. 좌표축과 회전 규칙

### 3.1 모든 파일에 공통인 규칙

- **관절은 자기 뼈의 로컬 Z축을 중심으로 돕니다.** 뼈의 머리(시작점)는 관절 위치(MuJoCo의 관절 앵커)에 있고, 뼈를 만들 때 로컬 Z축이 정확히 관절 회전축과 같아지도록 롤을 맞췄습니다. 그래서 뼈의 로컬 Z 회전량이 곧 관절 각도(라디안)입니다. 양수 방향은 MuJoCo와 같습니다(회전축 방향으로 볼 때 오른손 법칙).
- **기준 자세(rest pose)는 모든 관절 각도가 0인 자세**입니다. 이때 모든 뼈의 회전값이 0입니다. 정책이 서 있는 홈 자세와는 다르니 헷갈리지 마세요. 홈 자세는 `home` 동작에 따로 있습니다.
- 뼈의 꼬리 방향(로컬 Y축)은 회전축과 수직이고 가능하면 자식 관절 쪽을 향합니다. 자식이 없는 뼈(발목, 목 끄덕임, 어깨)는 몸체 무게중심 쪽입니다. 꼬리는 모양을 보기 위한 것일 뿐 관절 위치를 정하지 않습니다.
- 뿌리 뼈 `trunk`는 로컬 축이 월드 축과 같도록 만들었습니다. 그래서 위치 이동값을 월드 방향 그대로 읽을 수 있습니다.
- **길이 단위는 미터**입니다(FBX는 3.3절 참고). 각도는 Blender에서 라디안이고, glTF에서는 사원수이며, FBX와 BVH 파일 안에서는 도입니다.
- **로봇이 바라보는 방향은 +X입니다.** MuJoCo 월드를 그대로 썼기 때문입니다(x 앞, y 왼쪽, z 위). 기존 `kobi_v7.blend`처럼 -Y를 보도록 돌리지 않았습니다. 다른 도구의 "앞쪽" 규칙(예: glTF는 +Z)과 다르므로, 보기 방향을 맞추려면 도구 쪽에서 몸 전체를 돌려야 합니다. 시뮬레이션 숫자와 직접 비교하려면 돌리지 않는 편이 안전합니다.

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
- **FBX 안의 뼈 회전은 기준 자세의 방향을 포함한 전체 오일러 각도(XYZ 순서, 도)로 저장됩니다.** 예를 들어 `left_hip_roll`의 기준 자세 회전 값이 0이 아닙니다. 그래서 FBX에서 "Z 회전 값 = 관절 각도"가 성립하지 않습니다. 정확한 식은 `뼈의 기준 자세 회전 × (뼈 로컬 Z축 둘레로 관절 각도만큼 회전)`입니다. 관절 각도를 얻으려면 `rest` 동작을 기준으로 삼아 `rest` 대비 상대 회전을 구하세요.
- 동작 이름은 `Kobi_Armature|rest`, `Kobi_Armature|home`으로 들어갑니다.
- 이 설정의 장점은 Blender 기본값 그대로라는 점입니다. 단점은 위의 배율 100 때문에 일부 도구에서 최상위 물체의 배율이 100으로 보일 수 있다는 점입니다. 그럴 때는 `tools/skeleton/build_rig.py`의 `export_fbx`에서 `apply_scale_options`를 `FBX_SCALE_ALL`로 바꿔 다시 만들어 보세요(이 설정은 시험하지 않았고, 바꾸면 검증을 다시 돌려야 합니다).

### 3.4 BVH 읽는 법

- BVH는 뼈의 롤을 저장하지 못합니다. 모든 관절의 회전 틀이 월드 축과 나란하다고 보고, 기준 자세에서 0도인 상태를 기준으로 월드 축 둘레의 회전(X, Y, Z 오일러, 도)을 적습니다. 그래서 BVH의 값은 "뼈 로컬 Z 회전"이 아니라 월드 축 기준 회전입니다. 예를 들어 `left_knee`의 회전축은 월드 -y이므로 Y 회전 채널에 부호가 반대인 값이 들어갑니다.
- 뿌리 `trunk`만 위치 채널(미터)이 있고, 나머지는 회전 채널만 있습니다. 프레임 간격은 1/30초입니다.
- MotionBuilder처럼 위쪽이 Y이고 단위가 센티미터인 도구에서는 불러온 뒤 좌표축과 배율(100배)을 맞추세요.

## 4. 동작(액션) 두 개

각 동작은 1프레임과 2프레임에 같은 자세가 키로 들어 있습니다(프레임 간격 1/30초). 한 프레임짜리 클립을 싫어하는 도구가 있어서 길이가 0이 아니게 했습니다.

- `rest`: 모든 관절 각도 0, 뿌리 위치 (0, 0, 0.22) m. 기준 자세이며 리타게팅의 기준 틀로 쓰세요.
- `home`: MuJoCo 키프레임 `home`의 14개 관절 값(2절의 "홈 자세 값")과 뿌리 위치 (0.0353725, 0, 0.206397) m, 뿌리 방향은 회전 없음입니다. 정책이 서 있는 자세입니다. `rest`에서 `home`으로 가면서 뿌리가 x로 +0.0353725 m, z로 -0.013603 m 움직입니다. 이 값은 뿌리 뼈의 위치 이동 채널에 들어 있습니다.

Blender에서는 두 동작이 가짜 사용자 표시(F)가 붙어 저장되어 있고 현재 동작은 `rest`입니다. glTF에서는 동작 이름이 `rest`, `home`인 애니메이션 두 개입니다. BVH는 한 클립에 `rest`(1프레임)와 `home`(2프레임)을 이어 붙였습니다.

## 5. 겉모양 메시와 스킨

- 한 물체 `Kobi_Mesh`에 뼈마다 꼭짓점 그룹이 하나씩 있고(15개), 모든 꼭짓점은 한 뼈에만 가중치 1.0으로 묶여 있습니다(몸체 하나가 뼈 하나이므로 단단한 스킨입니다). 부드럽게 휘는 스킨이 아닙니다. 팔다리 부품이 몸체째로 움직입니다.
- 뼈에 붙는 메시: `trunk` 30조각, `neck_pitch` 12조각(머리), 어깨 두 뼈 각 5조각(소매와 장갑), 무릎 두 뼈 각 2조각(보호대와 다리 천), 엉덩이 롤 두 뼈와 발목 두 뼈 각 1조각입니다. `left_hip_yaw`, `right_hip_yaw`, `left_hip_pitch`, `right_hip_pitch`, `neck_yaw`에는 붙는 메시가 없습니다(꼭짓점 그룹은 비어 있습니다). 앞의 네 뼈는 겉모양이 업스트림 메시뿐이라 아래 6절 때문에 뺐고, `neck_yaw`는 MJCF에 겉모양 메시가 따로 없습니다.
- 모양은 기준 자세(모든 각도 0)에서 만들었습니다. MuJoCo가 읽은 겉모양과 위치가 같은지 확인했고(최대 차이 6.9e-9 m), 서로 닿는 꼭짓점은 한 조각 안에서만 합쳤습니다. 면의 음영은 평평(플랫) 셰이딩입니다.
- 색은 MJCF의 재질 색을 단순 색으로 옮겼습니다. 거칠기 0.6, 금속성 없음입니다.

## 6. 업스트림 메시 28개가 들어 있지 않은 이유

코비의 뼈대 부품 중 28개 메시 파일은 라이선스가 없는 Open_Duck_Playground 저장소 것이라 이 저장소에 담지 않았고, 필요하면 `tools/fetch_upstream_meshes.py`로 내려받습니다(저장소 `NOTICE.md` 참고). 라이선스가 없으면 다시 배포할 수 없으므로, 이 폴더의 어떤 파일(.blend, .fbx, .glb, .bvh, .json)에도 그 메시의 모양을 넣지 않았습니다.

- 시각용 메시 86개 중 우리가 만든 60개만 넣었고, 업스트림 26개(파일로는 11종: 트렁크 위아래, 엉덩이 모터 위아래, 다리 스페이서, 무릎-발목 판 2종, 발 4종)는 `skeleton.json`의 `excluded_upstream_geoms`에 이름만 적고 모양은 뺐습니다.
- 뺄지 말지는 두 가지로 판단했습니다. 파일 이름이 업스트림 28개 표에 있는지, 그리고 파일 내용의 SHA-256 값이 그 표에 있는지입니다. 둘 중 하나라도 맞으면 뺍니다.
- 뼈의 위치와 방향, 관절 범위 같은 숫자는 MJCF에서 왔으므로 `NOTICE.md`에 적힌 Open Duck Mini v2(Apache-2.0)의 변경물에 해당합니다.
- 그래서 이 메시는 우리가 만든 코리 외피와 몇몇 부품만 보여 주는 용도입니다. 업스트림 기계 뼈대 부품은 보이지 않으므로, 몸통과 다리 안쪽이 비어 보이는 것은 정상입니다.

## 7. 검증 결과

`tools/skeleton/verify_rig.py`가 만든 파일을 다시 읽어 MuJoCo의 순기구학(`mj_kinematics`) 결과와 비교합니다. 허용 오차는 1e-4 m(0.1 mm)입니다.

- ① 한 일: 관절 범위 안에서 무작위로 고른 20가지 자세와 `rest`, `home`을 파일에 적용해, 모든 뼈 머리의 월드 위치를 MuJoCo의 관절 앵커 위치와 비교했습니다. 자세를 만들 때는 각 뼈의 로컬 관절 축을 중심으로 관절 각도만큼만 돌렸습니다.
- ② 값의 뜻: 아래 표의 오차는 가장 크게 어긋난 뼈 하나의 거리(미터)입니다. 모두 1e-6 m보다 작아서, 허용 오차보다 100배 이상 작습니다. 이 정도는 숫자 반올림 수준입니다.
- ③ 왜 믿을 수 있나: 시험이 제대로 작동하는지 일부러 한 관절의 값을 부호만 뒤집어 넣어 봤고, 그때 오차가 0.14~0.29 m로 커지며 실패로 나왔습니다.

| 검사한 대상 | 방법 | 뼈 머리 위치 최대 오차(m) | 스킨된 메시 꼭짓점 최대 오차(m) |
|---|---|---|---|
| `kobi_rig.blend` | 원본 파일, 로컬 Z 오일러 회전 | 4.6e-8 (무작위 20가지), 1.7e-8 (rest), 3.9e-8 (home) | 1.5e-7 |
| `kobi_rig.fbx` | Blender로 새로 불러옴 | 9.5e-8, 9.2e-8, 9.3e-8 | 1.8e-7 |
| `kobi_rig.fbx` | 파일 내부를 직접 해석(Blender 없이), Y 위 변환과 100배 단위 포함 | 7.2e-8, rest 5.9e-8, home 6.7e-8 | 해당 없음 |
| `kobi_rig.glb` | Blender로 새로 불러옴 | 8.6e-8, 8.9e-8, 4.6e-8 | 1.4e-7 |
| `kobi_rig.glb` | 파일 내부를 직접 해석(JSON과 이진 데이터), (x, y, z) -> (x, z, -y) 포함 | 4.4e-8, rest 6.0e-8, home 4.2e-8 | 해당 없음 |
| `kobi_skeleton.bvh` | 새로 불러와 1프레임(rest)과 2프레임(home) 비교 | 6.5e-7 | 해당 없음 |
| BVH 시험 파일 | 같은 자세 22개를 새로 구운 22프레임 BVH | 6.5e-7 | 해당 없음 |

- 뼈 개수 15개, 뼈 이름, 부모 관계는 `skeleton.json`과 모든 파일(blend, FBX와 GLB의 불러온 결과와 직접 해석, BVH)에서 일치했습니다.
- 모든 꼭짓점이 정확히 한 뼈에 가중치 1.0으로 묶여 있습니다(blend 151,191개 전부). 다시 불러온 FBX와 GLB에서도 꼭짓점마다 가중치가 있는 뼈가 하나뿐입니다(GLB는 불러올 때 꼭짓점이 812,076개로 늘어납니다. 면마다 법선을 쪼개기 때문이며 모양은 같습니다).
- 스킨 검사: 기준 자세에서 읽은 꼭짓점을 자세에 따라 움직인 뒤, MuJoCo의 해당 몸체 움직임(회전과 이동)을 같은 꼭짓점에 직접 적용한 값과 비교했습니다.
- 뼈 로컬 Z가 관절 축이라는 주장도 확인했습니다: blend와 다시 불러온 FBX와 GLB, 그리고 FBX와 GLB의 파일 내부 노드에서 기준 자세의 관절 축을 뼈 로컬 좌표로 바꾸면 모두 (0, 0, 1)입니다(오차 1e-6 이하). BVH는 롤이 없어서 이 검사를 하지 않았습니다.
- 메모리: 가장 큰 Blender 실행(검증)의 최대 상주 메모리는 약 1.2 GB였고, 뼈대 만들기는 약 0.7 GB였습니다.

## 8. 다시 만드는 방법

모든 명령은 저장소 맨 위 폴더에서 실행합니다. Blender 5.2.2 LTS와 `uv`가 필요합니다.

```
# 0. 업스트림 메시 28개 내려받기(MuJoCo 장면이 읽히려면 필요, 한 번만)
python3 tools/fetch_upstream_meshes.py

# 1. MJCF에서 skeleton.json과 검증용 정답(MuJoCo 순기구학)을 뽑기
uv run --no-project --python 3.12 --with mujoco==3.14.0 --with numpy python -I tools/skeleton/extract_skeleton.py \
    --fk-reference out/skeleton_verify/fk_reference.json

# 2. 뼈대, 메시, 동작을 만들고 .blend, .fbx, .glb, .bvh 저장
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/build_rig.py -- \
    --skeleton printed_face/skeleton/skeleton.json --repo-root . --out-dir printed_face/skeleton

# 3. 검증(실패하면 마지막 줄에 FAIL, 성공하면 PASS)
/Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup --python tools/skeleton/verify_rig.py -- \
    --skeleton printed_face/skeleton/skeleton.json --fk-reference out/skeleton_verify/fk_reference.json \
    --rig-dir printed_face/skeleton --work-dir out/skeleton_verify --report out/skeleton_verify/report.json
```

- `out/`는 저장소에서 제외되는 폴더입니다(`.gitignore`). 검증용 정답과 시험 BVH, 보고서 JSON이 여기에 생깁니다.
- 1단계의 `--samples`(기본 20)와 `--seed`로 무작위 자세 수와 난수 씨앗을 바꿀 수 있습니다.
- 메시가 너무 무거울 때는 2단계에 `--max-export-triangles 200000`을 주면 복사본을 줄여 FBX와 GLB에만 쓰고, `.blend`에는 원본을 남깁니다(이번 산출물에는 쓰지 않았습니다).
- **LCD 얼굴 판 코비에도 같은 방법으로 만들 수 있습니다.** LCD 판 MJCF가 생기면 1단계에 `--mjcf lcd_face/.../kobi_scene.xml --out lcd_face/skeleton/skeleton.json`을 주고, 2단계와 3단계의 경로를 그쪽으로 바꾸면 됩니다. 관절 14개와 정책 행동 순서가 같다는 가정이며, 다르면 1단계가 멈춥니다.

## 9. 알려진 한계

- Maya, Unity, MotionBuilder 프로그램에서 직접 열어 본 것은 아닙니다. 위 검증은 Blender와, FBX와 GLB를 직접 읽는 코드로 한 것입니다.
- FBX의 뼈 회전은 기준 자세를 포함한 오일러 값이므로, 도구에서 "관절 각도"를 바로 읽을 수는 없습니다(3.3절). GLB와 Blender는 뼈 로컬 Z 회전이 그대로 관절 각도입니다.
- 오른쪽 무릎은 같은 다리의 엉덩이 앞뒤 관절에서 옆(y)으로 0.037 m 떨어져 있고, 왼쪽은 0.0001 m입니다. 이는 시뮬레이션 모델(MJCF와 URDF)에 이미 있는 값을 그대로 옮긴 것이고, 이 도구에서 만든 오차가 아닙니다.
- 겉모양 메시에 업스트림 부품이 없으므로 다리와 몸통 안쪽이 비어 보입니다(6절).
- 가중치는 단단한 스킨(1.0)이라 천이나 꼬리가 휘는 표현은 없습니다.
