from data_generator import SimParams, generate_synthetic_cube
from subtraction_core import SubtractConfig, run_two_stage_subtraction
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 1. 데이터 생성 (논문 스펙)
    p = SimParams(df_mhz=0.05, dt_s=1.0)
    sim = generate_synthetic_cube(p, seed=2025)
    raw = sim['raw']
    freqs = sim['freqs']

    print(f"Data Shape: {raw.shape} (Time x Freq)")

    # 2. [진짜_최종_황금설정] Config
    # 이 설정은 빗살무늬(Comb)를 잡고, 천체 신호를 보호합니다.
    cfg_final = SubtractConfig(
        # ---------------------------
        # (A) 윈도우 및 보호 구역
        # ---------------------------
        sat_window=(25, 35),   # 위성 통과 구간
        edges_margin=15,       # 배경 노이즈 학습 구간 확보
        shoulder=2.0,          # 보호 구역 주변 참조 범위

        protect_bw=0.80,       # [핵심] 보호 구역을 0.8MHz로 넓게 잡음 (폭발 방지)
        science_core_bw=0.15,
        notch_bw=0.20,         # 노치 필터 강화

        # ---------------------------
        # (B) 모델 지능 (Rank) - 여기가 제일 중요함!
        # ---------------------------
        r1=6,                  # [지능 상향] 위성 덩어리 형태를 완벽히 학습
        r2=2,                  # [빗살무늬 킬러] 미세한 빗살무늬 제거용 2단계 PCA 켜기
        
        poly_order=2,          # 끊어진 구간을 곡선으로 부드럽게 연결
        clip_k=5.0,            # 클램핑을 넉넉하게 풀어서 모델이 작동하게 함

        # ---------------------------
        # (C) 안전 장치
        # ---------------------------
        subtract_protect=True, # [절대 끄지 마세요] 천체 신호 보호 모드
        safety_floor_q=0.0,    # 바닥 자르기 금지 (0.0 필수)
        gate_taper_s=2.0,      # 부드러운 진입
    )

    # 3. 알고리즘 실행
    print("\n🚀 알고리즘 실행 중... (Rank 6 + 2단계 PCA)")
    cln, m = run_two_stage_subtraction(
        raw=raw,
        freqs=freqs,
        target_freq=p.science_freq_mhz,
        target_sigma=p.science_sigma_mhz,
        comb_peaks=list(p.comb_peak_freqs_mhz),
        cfg=cfg_final
    )

    # 4. 결과 출력
    print("="*40)
    print(f"📊 최종 결과 리포트")
    print(f"✅ Reduction (노이즈 제거): {m.reduction_db:.2f} dB  (목표: >15)")
    print(f"✅ Preservation (천체 보존): {m.pres_sat*100:.1f} %   (목표: 90~110)")
    print(f"✅ Comb Residual (잔차):    {m.comb_resid:.2f}      (목표: <20)")
    print("="*40)

    # 5. 시각화 (Figure 3 자동 생성)
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    times = np.arange(raw.shape[0])
    extent = [times[0], times[-1], freqs[0], freqs[-1]]
    
    # 컬러 스케일 자동 조정
    vmin, vmax = np.percentile(raw, 1), np.percentile(raw, 99)

    # (a) Raw
    axes[0].imshow(raw.T, origin='lower', aspect='auto', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0].set_title("(a) Raw Data: Satellite RFI Dominant")
    
    # (b) Masking (Baseline)
    masked = raw.copy()
    masked[25:35, :] = np.nan
    axes[1].imshow(masked.T, origin='lower', aspect='auto', extent=extent, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1].set_title("(b) Baseline: Masking (Data Loss)")

    # (c) Proposed
    axes[2].imshow(cln.T, origin='lower', aspect='auto', extent=extent, cmap='inferno', vmin=np.percentile(cln, 1), vmax=np.percentile(cln, 99))
    axes[2].set_title(f"(c) Proposed: RFI Subtracted (Preserved={m.pres_sat*100:.1f}%)")
    
    # 화살표 표시
    axes[2].annotate('Science Target', xy=(30, 60), xytext=(45, 63),
                     arrowprops=dict(facecolor='white', shrink=0.05), color='white', fontweight='bold')

    plt.tight_layout()
    plt.show()
