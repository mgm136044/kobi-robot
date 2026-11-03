# NOTICE: 출처, 라이선스, 우리가 바꾼 점

이 저장소는 다른 사람의 오픈소스 작업을 바탕으로 합니다. 아래에 무엇을 어디서 가져왔고, 무엇을 바꿨고, 어떤 권리가 누구에게 있는지 적습니다.

## 1. Open Duck Mini v2 (Apache License 2.0)

- 만든 사람: Antoine Pirrone (GitHub 계정 apirrone)
- 저장소: https://github.com/apirrone/Open_Duck_Mini
- 라이선스: Apache License 2.0. 원문은 이 저장소의 `LICENSES/Apache-2.0.txt`에 있으며, 위 저장소의 `LICENSE` 파일(2026-01-31 판, 커밋 `b23317a`)을 그대로 복사한 것입니다.

코비의 다리와 몸통 뼈대 형상은 Open Duck Mini v2에서 왔습니다. 이 뼈대에서 파생된 다음 파일들도 같은 출처에 기대고 있습니다.

- `printed_face/mujoco/kobi_scene.xml`, `printed_face/mujoco/kobi_scene_backlash.xml`와 `printed_face/mujoco/assets/`의 뼈대 부품 메시(`body_*.stl`, `trunk_*.stl`, `foot_*.stl`, `head_*.stl`, `*_sheet.stl` 등)
- `printed_face/urdf/kobi.urdf`, `printed_face/urdf/kobi_gait.urdf`와 `printed_face/urdf/assets/`의 뼈대 부품 메시
- `printed_face/blender/kobi_v7.blend` 안의 Open Duck 뼈대 물체들
- `lcd_face/mujoco/kobi_scene.xml`, `lcd_face/mujoco/kobi_scene_backlash.xml`와 `lcd_face/mujoco/assets/`의 뼈대 부품 메시, `lcd_face/blender/kobi_v7_lcd_v6.blend` 안의 Open Duck 뼈대 물체들(LCD 얼굴 판도 같은 뼈대 위에 머리만 바꾼 것이며, 아래 "우리가 바꾼 점"이 그대로 적용됩니다. 머리 몸체, 몸통 본체, 목 링크의 질량 값은 LCD 머리 모델에 맞게 인쇄 얼굴 판과 달라져 있습니다)

### 우리가 바꾼 점 (Apache-2.0 제4조 (b)에 따른 변경 표시)

- **엉덩이 간격을 좁혔습니다.** 원본 오리 로봇보다 두 다리 사이가 좁은 "narrowed hips" 형상입니다.
- **목과 머리를 바꿨습니다.** 원본의 4축 목과 머리를 빼고, 좌우 돌리기와 끄덕임 2축 목과 코리 머리 껍데기로 바꿨습니다.
- **어깨와 팔을 더했습니다.** 마이크로 서보(MG90S) 어깨 관절 2개와 팔 소매 등을 새로 달았습니다.
- **외피를 입혔습니다.** 재킷, 꼬리, 날개, 발 덮개, 다리 보호대, 흰 얼굴 등 3D 프린트 외피 부품은 우리가 새로 만들었고, 모델에서는 충돌 없는 겉모양(visual) 메시로 붙였습니다.
- **질량과 관성을 바꿨습니다.** 실제 부품 질량이 한 번씩만 반영되도록 몸체 관성을 다시 계산해 넣었습니다(전체 2.7789782 kg).
- **구동기 설정을 바꿨습니다.** 학습과 URDF에서 다리와 목 서보 토크 한계를 1.91 N·m, 어깨를 0.20 N·m로 했습니다.
- **학습용 장면을 추가했습니다.** 다리 관절마다 백래시용 가짜 관절(±0.5도)을 넣고 서보 위치 제어 세기를 13.37에서 17.11로 올린 `kobi_scene_backlash.xml`입니다.
- **URDF를 새로 만들었습니다.** 원본 URDF를 고친 것이 아니라 코비 MJCF에서 `tools/build_urdf.py`로 생성한 것입니다.

## 2. 라이선스가 없는 업스트림 저장소 (코드를 넣지 않음)

다음 세 저장소에는 라이선스 파일이 없어서, 코드를 수정했든 안 했든 다시 배포할 수 없다고 보고 이 저장소에 넣지 않았습니다.

- Open_Duck_Playground (학습 환경): https://github.com/apirrone/Open_Duck_Playground
- Open_Duck_reference_motion_generator (기준 걸음 생성기): https://github.com/apirrone/Open_Duck_reference_motion_generator
- Open_Duck_Mini_Runtime (실제 로봇용 실행 코드): https://github.com/apirrone/Open_Duck_Mini_Runtime

우리 학습 환경은 Open_Duck_Playground의 커밋 `b9be205`(2025-08-05)를 바탕으로 고친 것입니다. 무엇을 바꿨는지는 `README.md` 11절에 말로 적었습니다. 기준 걸음 데이터(`policy/polynomial_coefficients_kobi.pkl`)는 Open_Duck_reference_motion_generator의 방식으로 코비 모델에서 새로 만든 우리 출력물이며, 그 코드는 들어 있지 않습니다. 학습된 정책(`policy/kobi_v2_final_151M.onnx`)의 가중치는 우리가 학습한 결과물입니다.

## 3. 소유자가 직접 판단해야 할 점

저장소를 공개하기 전에 확인하세요. 아래는 자동으로 결론낼 수 없었던 부분입니다.

- **MJCF 겉틀의 유사성**: `kobi_scene.xml`의 센서 목록, 기본 클래스 이름(`open_duck_mini_v2`, `collision`, `backlash`), 재질 이름과 색, `option` 설정은 Open_Duck_Playground의 `open_duck_mini_v2.xml`과 같은 줄이 약 50줄 있고, Open_Duck_Mini의 `robot_motors.xml`과는 3줄만 같습니다. 뼈대 메시와 몸체 구조는 Apache-2.0인 Open_Duck_Mini에서 왔지만, 이 겉틀 부분이 라이선스 없는 Playground 쪽에서 왔을 가능성이 있습니다. 대부분 MuJoCo의 일반적인 설정 이름이라 위험은 작아 보이지만, 확신하지 못합니다.
- **환경 거울 코드**: `tools/kobi_env_mirror.py`는 학습 환경의 동작을 NumPy와 MuJoCo로 다시 쓴 우리 코드이고, 업스트림 코드와 같은 긴 줄은 거의 없었습니다. 다만 관측 순서, 행동 크기(0.25), 속도 제한(5.24 rad/s), 잡음 크기 같은 인터페이스 값은 업스트림 학습 환경과 같아야 정책이 맞게 돌기 때문에 같은 값을 씁니다. 이는 사실 정보(상수)라고 판단했습니다.
- **업스트림 Playground에서 온 메시 파일은 이 저장소에 넣지 않고 내려받습니다 (가장 중요)**: 뼈대 부품 메시 28개(`printed_face/mujoco/assets/`와 `lcd_face/mujoco/assets/`에 같은 28개, 그 가운데 11개는 `printed_face/urdf/assets/`에도 사용)는 Open_Duck_Playground 저장소(`playground/open_duck_mini_v2/xmls/assets/`)의 파일과 바이트 단위로 똑같습니다. 이 파일들은 Apache-2.0인 Open_Duck_Mini 저장소의 원본 메시(28 KB~495 KB)가 아니라 단순화된 작은 메시(각 5,284바이트)이고, 그 저장소에는 라이선스가 없습니다. 그래서 이 저장소에는 **포함하지 않았고** `.gitignore`로 제외했습니다(`lcd_face/` 쪽 경로도 같습니다). `python3 tools/fetch_upstream_meshes.py`가 고정된 업스트림 커밋 `b9be205ac64488c23504ca42e5ec790337adeec3`에서 GitHub raw 주소로 내려받고, 스크립트에 적힌 SHA-256과 맞을 때만 저장합니다(LCD 얼굴 판 장면이 쓰는 업스트림 메시도 이 28개와 이름과 해시가 같아서, 스크립트가 `lcd_face/mujoco/assets/`에도 같은 파일을 내려받습니다). 내려받은 파일은 사용자가 직접 업스트림에서 받은 것이며 이 저장소의 일부로 재배포되지 않습니다. 다만 우리 MJCF와 URDF가 이 파일들을 가리키므로, 장면을 열려면 내려받기가 필요합니다. 근본 해결을 하려면 소유자가 저작자에게 라이선스를 문의하거나, Open_Duck_Mini의 Apache-2.0 원본 메시를 직접 단순화해 바꾸고 모델을 다시 검증해야 합니다. 내려받는 파일 이름과 해시 표는 `tools/fetch_upstream_meshes.py`에 있습니다.
  - `hfield.png`(지형 높이 그림)도 Playground 파일이지만 코비 장면이 쓰지 않아서 내려받기 대상에서도 뺐고, `.gitignore`로 제외했습니다.
- **블렌더 파일**: `kobi_v7.blend`에는 Open Duck 뼈대 물체 80개 정도가 들어 있습니다. 저장소에 넣기 전에 제작자 컴퓨터의 경로와, 코리 캐릭터 참고 그림을 가리키는 이미지 항목(1개)을 지웠습니다. 그 그림을 보여 주던 빈 물체(`Kori_front_11-09A`)는 그림 없이 남아 있습니다. 나머지 물체, 메시, 재질은 그대로입니다. `lcd_face/blender/kobi_v7_lcd_v6.blend`도 같은 방식으로 코리 참고 그림 이미지 1개와 개인 컴퓨터 경로를 지웠습니다(`lcd_face/README.md` 10절). 이 파일에도 같은 빈 물체(`Kori_front_11-09A`)가 그림 없이 남아 있고, 화면에 띄우는 얼굴 그림(`kori_face_800x480.png`)이 파일 안에 포함되어 있습니다.

## 4. 코리 캐릭터

"코리(Kori)" 캐릭터와 그 모양에 대한 권리는 원 소유자(소속 대학교)에게 있습니다. 이 저장소의 외피 부품(`lcd_face/print/`의 머리 껍데기, 흰 얼굴 포함)과 영상에는 그 캐릭터의 모양이 들어 있으므로, 공개나 재사용 전에 소유자의 허락 범위를 확인해야 합니다. 캐릭터 원본 그림 파일은 이 저장소에 넣지 않았습니다.

## 5. 우리가 만든 파일의 권리

위 1~4에 해당하지 않는 우리가 만든 파일(`tools/`, `policy/`(`policy/lcd_face/` 포함), `lcd_face/`의 부품표와 모델 수정분, 외피 부품의 형상과 프린트 파일, 영상, 정지화면, 이 문서들)에는 별도의 오픈소스 라이선스를 붙이지 않았습니다. 따로 밝히지 않은 한 모든 권리를 소유자가 갖습니다(all rights reserved). 이는 임시 표시이며, 소유자가 나중에 라이선스를 정해 `LICENSE` 파일을 추가할 수 있습니다. 그전까지는 이 저장소를 받은 사람이 복사, 수정, 재배포할 권리를 따로 허락받지 않았습니다.

## 6. 이 저장소가 불러 쓰는 제3자 소프트웨어

이 저장소는 MuJoCo, NumPy, onnxruntime을 내려받아 쓰지만 그 코드를 포함하지 않습니다. 각 프로젝트의 라이선스를 따릅니다.
