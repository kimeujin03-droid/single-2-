% !TeX program = xelatex
% !TeX encoding = UTF-8
% Path B full revision draft. Compile with XeLaTeX or LuaLaTeX.
% This draft reframes the paper as a multi-background, null-calibrated
% EoR-window residual-risk assessment for Starlink-like UEMR injection.

\documentclass[11pt,a4paper]{article}

\usepackage{iftex}
\ifPDFTeX
  \errmessage{This manuscript contains Korean UTF-8 text. Compile with XeLaTeX or LuaLaTeX, not pdfLaTeX}
\fi
\usepackage{kotex}
\ifXeTeX
  \IfFontExistsTF{Noto Serif CJK KR}{\setmainhangulfont{Noto Serif CJK KR}}{}
  \IfFontExistsTF{Noto Serif CJK KR}{\setmainfont{Noto Serif CJK KR}}{}
\fi
\usepackage{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage[numbers,sort&compress]{natbib}
\usepackage{xcolor}
\usepackage{enumitem}
\usepackage{siunitx}

\geometry{margin=25mm}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue
}

\title{HERA-like EoR Delay-space Analysis에서 Starlink-like UEMR의\\
Multi-background Forward-injection 기반 잔류 위험 평가}
\author{김유진\\
제주대학교 데이터사이언스학과}
\date{Draft v0.8 Korean \TeX{} Working Manuscript}

% -----------------------------------------------------------------------------
% Placeholder result macros. Replace after final runs.
% -----------------------------------------------------------------------------
\newcommand{\NBackgrounds}{5}
\newcommand{\NBaselineGroups}{3}
\newcommand{\NFluxScales}{5}
\newcommand{\NNullTrialsFinal}{500}
\newcommand{\NNullTrialsPilot}{100}
\newcommand{\FluxGrid}{10, 30, 100, 300, 1000\,Jy}
\newcommand{\RealisticLow}{10\,Jy}
\newcommand{\RealisticHigh}{100\,Jy}
\newcommand{\StressLow}{300\,Jy}
\newcommand{\StressHigh}{1000\,Jy}
\newcommand{\PilotPassShort}{--}
\newcommand{\MainShortMedian}{--}
\newcommand{\MainShortIQR}{--}
\newcommand{\MainObsMinusP}{--}
\newcommand{\InpaintReduction}{--}

\begin{document}
\maketitle

\begin{abstract}
저궤도 위성군의 확장은 우주 재이온화 시대(Epoch of Reionization, EoR) 21\,cm 관측에 대해 새로운 형태의 이동성 전파 간섭(moving-source RFI) 문제를 제기한다. 특히 Starlink-like 위성에서 발생할 수 있는 비의도 전자파 방출(unintended electromagnetic radiation, UEMR)은 원시 시간--주파수 waterfall에서의 검출 가능성만으로 EoR 분석 위험도를 판단하기 어렵다. 본 연구는 TLE-conditioned Starlink-like UEMR을 HERA public-data 기반 visibility background에 주입하고, 명시적으로 정의한 처리 연산자 $P$를 통과한 뒤의 interaction residual,
\[
V_{\mathrm{int}}=P(V_{\mathrm{bg}}+V_{\mathrm{sat}})-P(V_{\mathrm{bg}}),
\]
을 이용하여 EoR-relevant delay window의 operator-conditioned residual perturbation을 평가한다. 분석은 단일 realization에 의존하지 않도록 서로 다른 LST와 flagging 상태를 갖는 \NBackgrounds{}개 background, \NBaselineGroups{}개 baseline geometry group, 그리고 $S_{\rm ref}=\{\FluxGrid\}$의 reference flux grid를 사용한다. 핵심 metric은 interaction residual power ratio $R_{\rm win}^{(\rm int)}$와 absolute window power perturbation score $S_{\rm win}^{(|\Delta P|)}$이며, 각 main case는 phase-randomized window-metric null ensemble과 비교한다. 본 연구는 Starlink UEMR의 확정적 검출이나 full cylindrical power-spectrum contamination measurement를 주장하지 않는다. 대신 published-order flux scale과 stress-test flux scale에서 Starlink-like moving emitter가 HERA-like delay-space processing 후 window-side perturbation을 얼마나 안정적으로 남기는지, 그리고 그 효과가 background realization, baseline geometry, flagging/inpainting 상태, simplified inpainting surrogate에 대해 얼마나 보존되는지를 평가하는 재현 가능한 EoR QA framework를 제시한다.
\end{abstract}

\noindent\textbf{핵심어:} Starlink, UEMR, RFI, HERA, EoR, delay spectrum, fringe rate, TLE, residual morphology, null calibration

% ============================================================
\section{서론}
% ============================================================

저주파 전파천문학, 특히 EoR 21\,cm power spectrum 분석은 전경(foreground), 열잡음, calibration residual, flagging/inpainting artifact, 그리고 RFI가 결합된 고동적범위(high-dynamic-range) 문제이다 \citep{morales2010,pritchard2012,liu2020,deboer2017,abdurashidova2022}. Starlink와 같은 저궤도(low-Earth orbit, LEO) 위성군은 기존의 준정적 지상 RFI와 달리, 시간에 따라 변하는 delay와 fringe-rate를 각 기선(baseline)에 투영하는 moving emitter이다. 최근 LOFAR, NenuFAR, 21CMA 계열 연구는 Starlink-associated UEMR이 100--200\,MHz 부근에서 광대역 및 협대역 성분을 가질 수 있음을 보고하였다 \citep{divruno2023starlink,bassa2024starlink,zhang2025starlink,yang2026starlink21cma}. 따라서 위성형 UEMR의 위험성은 단순히 raw waterfall에서 보이는가가 아니라, EoR delay-space 분석 이후 science-relevant window 영역에 어떤 perturbation을 남기는가로 평가되어야 한다.

본 연구의 핵심 질문은 다음과 같다.

\begin{quote}
Starlink-like moving-emitter UEMR을 HERA-like visibility background에 주입했을 때, HERA-like delay/fringe processing 이후 EoR-window proxy 영역에 background 대비 의미 있는 residual perturbation이 안정적으로 남는가?
\end{quote}

여기서 ``위험''은 절대적이고 보편적인 Starlink contamination claim이 아니라, 특정 background product의 processing history와 특정 post-processing operator $P$에 조건부인 QA risk를 의미한다. 특히 본 논문은 다음을 구분한다.

\begin{enumerate}[leftmargin=2em]
    \item 특정 Starlink UEMR을 실제 관측에서 검출했다는 주장;
    \item full HERA production pipeline을 raw data부터 재현했다는 주장;
    \item full cylindrical power-spectrum contamination을 측정했다는 주장;
    \item HERA public-data product 위에 forward-injection을 수행하여 operator-conditioned EoR-window perturbation을 평가했다는 주장.
\end{enumerate}

본 연구는 네 번째 주장에 한정된다. 초기 개발 과정에서 TLE history entry를 독립 physical satellite처럼 취급하면 multi-emitter coherence metric이 인위적으로 증가하는 문제가 확인되었다. 해당 run은 code/input-catalog provenance artifact로 간주하여 scientific analysis에서 제외하며, 본문에 보고하는 실험은 physical-satellite 기준으로 de-duplicated된 catalog만 사용한다. 이 항목은 본 논문의 결과가 아니라, moving-emitter injection study에서 필요한 입력 카탈로그 QC 조건이다.

본 논문의 기여는 네 가지이다. 첫째, TLE 기반 LEO trajectory, near-field antenna-to-satellite range-difference delay, HERA-like beam response, range attenuation, finite integration/channel smearing, 문헌 기반 UEMR spectral template을 결합한 forward-injection framework를 정리한다. 둘째, $V_{\rm int}$ 기반 interaction residual을 통해 raw detectability가 아니라 processing 이후의 residual morphology를 평가한다. 셋째, EoR-window proxy metric $R_{\rm win}^{(\rm int)}$와 $S_{\rm win}^{(|\Delta P|)}$를 정의하고, phase-randomized window-metric null로 calibration한다. 넷째, multi-background, multi-baseline, flux-prior, inpainting-surrogate grid를 통해 single-realization artifact 가능성을 줄이는 claim ladder를 제시한다.

% ============================================================
\section{선행연구와 위치 설정}
% ============================================================

\subsection{Starlink UEMR 관측과 본 연구의 차이}

LOFAR 기반 연구는 Starlink 위성에서 저주파 UEMR을 검출하며, 위성군 문제가 가시광 streak에 국한되지 않고 저주파 전파천문학의 spectral contamination 문제로 확장됨을 보였다 \citep{divruno2023starlink,bassa2024starlink}. NenuFAR 및 21CMA 계열 결과도 저주파 영역에서 위성-associated emission의 존재 가능성을 강화한다 \citep{zhang2025starlink,yang2026starlink21cma}. 이들 연구는 실제 위성 방출의 검출 및 spectral characterization에 초점을 둔다. 본 연구는 특정 위성 UEMR의 실측 검출 논문이 아니라, published-order spectral morphology와 flux scale을 사용한 controlled forward-injection 연구이다.

\subsection{HERA delay-spectrum 분석과 residual QA}

HERA는 EoR 21\,cm 신호 탐지를 위해 설계된 drift-scan 전파간섭계이며 \citep{deboer2017}, delay-spectrum 접근은 foreground wedge와 EoR window의 분리를 핵심 전제로 한다 \citep{parsons2012delay,liu2014wedge,pober2014}. 그러나 flagging, calibration residual, chromatic beam, cable reflection, mutual coupling, inpainting artifact는 foreground power를 high-delay 영역으로 재분배할 수 있다 \citep{morales2004,thyagarajan2015,abdurashidova2022}. 따라서 moving-emitter RFI는 raw time-frequency detectability뿐 아니라, 처리 이후 window-side perturbation으로도 평가되어야 한다.

\subsection{본 연구의 범위}

본 연구는 HERA production pipeline 전체를 대체하지 않는다. 특히 public data product가 이미 flagging, calibration, LST-binning, inpainting 등의 processing history를 포함할 수 있으므로, 본 연구의 operator $P$는 raw-to-final pipeline이 아니라 선택된 background product 위에 추가로 적용되는 명시적 post-processing operator이다. 이 점은 Section~\ref{sec:data}에서 별도로 명시한다.

% ============================================================
\section{데이터와 processing history}
\label{sec:data}
% ============================================================

\subsection{Background visibility product의 정의}

본 연구는 HERA public-data product에서 추출한 visibility crop을 background $V_{\rm bg}$로 사용한다. Background product는 원시 visibility가 아니라 release-specific preprocessing을 거친 산물일 수 있다. 따라서 각 background는 Table~\ref{tab:background_manifest}와 같이 LST, baseline, polarization, flag fraction, invalid-sample fraction, processing history, inpainting 여부를 manifest로 기록한다.

\begin{table}[h]
\centering
\caption{Background manifest. 실제 run 이후 각 항목을 채운다. ``Processing history''는 release memo와 extraction script에 근거해 기록한다.}
\label{tab:background_manifest}
\begin{tabular}{lllllll}
\toprule
ID & LST range & Product & Baseline/pol & Flag frac. & Inpainting & Selection reason \\
\midrule
bg\_01 & -- & -- & -- & -- & -- & clean/low-flag \\
bg\_02 & -- & -- & -- & -- & -- & moderate-flag \\
bg\_03 & -- & -- & -- & -- & -- & heavily flagged or invalid samples \\
bg\_04 & -- & -- & -- & -- & -- & processing diversity \\
bg\_05 & -- & -- & -- & -- & -- & LST diversity \\
\bottomrule
\end{tabular}
\end{table}

중요한 점은 다음과 같다. 본 연구에서 사용하는 $P$는 HERA raw data부터 power spectrum까지의 production pipeline 전체가 아니다. 선택된 public background product가 이미 flagging, calibration, LST-binning, inpainting 또는 기타 release-specific conditioning을 거쳤다면, $P$는 그 산물 위에 추가로 적용되는 operator이다. 따라서 interaction residual
\begin{equation}
V_{\rm int}=P(V_{\rm bg}+V_{\rm sat})-P(V_{\rm bg})
\label{eq:vint}
\end{equation}
은 background product의 processing history와 본 논문에서 정의한 $P$ 모두에 조건부이다. 이 조건부성을 명시하기 위해 모든 result table에는 background ID, product name, flag fraction, invalid-sample fraction, taper, delay normalization, buffer, fringe-rate filter setting을 함께 기록한다.

\subsection{Baseline group과 redundancy 처리}

Moving emitter는 baseline마다 서로 다른 delay track과 phase evolution을 갖는다. 따라서 redundant baseline group에 속한 visibility를 injection 전 또는 metric 계산 전 단순 평균하지 않는다. 각 baseline $b$에 대해 $S_{\rm win}^{(b)}$를 독립적으로 계산하고, group-level 결과는 median 및 IQR로만 요약한다.

\begin{equation}
S_{\rm win}^{(G)} = \mathrm{median}_{b\in G}\, S_{\rm win}^{(b)}.
\end{equation}

본 연구의 baseline group은 Table~\ref{tab:baseline_groups}와 같이 설정한다.

\begin{table}[h]
\centering
\caption{Baseline geometry groups. Representative baseline은 main grid에 사용하고, redundant baselines는 robustness check에 사용한다.}
\label{tab:baseline_groups}
\begin{tabular}{llll}
\toprule
Group & Nominal length & Orientation & Role \\
\midrule
Short EW & 14.6\,m & EW & short-baseline window response \\
Mid EW & 29.2\,m & EW & intermediate geometry \\
Long EW & 73\,m & EW & representative longer baseline \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Flux-prior tier}

본 연구의 $S_{\rm ref}$는 calibrated HERA-apparent Starlink flux prediction이 아니라, published low-frequency Starlink UEMR measurements의 order of magnitude에 anchoring한 reference scaling parameter이다. Beam response, range attenuation, smearing, spectral template과 결합되어 최종 injected visibility amplitude가 결정된다.

\begin{table}[h]
\centering
\caption{Reference flux scale tier. 최종 수치 범위는 Dí Vruno et al. 및 Bassa et al.의 원문 수치 확인 후 확정한다.}
\label{tab:flux_tiers}
\begin{tabular}{lll}
\toprule
Tier & $S_{\rm ref}$ & Interpretation \\
\midrule
Realistic-low & \RealisticLow & published-order lower/mid scale anchor \\
Realistic-high & \RealisticHigh & conservative bright-component anchor \\
Stress-test & \StressLow--\StressHigh & bright/worst-case sensitivity test \\
\bottomrule
\end{tabular}
\end{table}

본 논문은 LOFAR measured flux를 HERA apparent flux로 정확히 변환했다고 주장하지 않는다. 대신 $S_{\rm ref}$ sweep을 통해 published-order scale과 stress-test scale에서 window-side perturbation이 어떻게 달라지는지를 평가한다.

% ============================================================
\section{Forward-injection model}
% ============================================================

\subsection{TLE propagation과 catalog QC}

TLE는 candidate satellite trajectory를 생성하는 데 사용한다. 같은 physical satellite의 여러 TLE epoch가 history file에 반복될 수 있으므로, 모든 scientific run은 physical satellite identity, 예컨대 NORAD ID, 기준으로 de-duplicate된 catalog를 사용한다. 초기 duplicate-history run은 multi-emitter coherence metric을 인위적으로 증가시킬 수 있으므로, 본문 결과에서 제외한다.

\subsection{Near-field delay}

위성의 topocentric ENU 위치를 $\mathbf{r}_{\rm sat}(t)$, 두 antenna의 위치를 $\mathbf{r}_i$, $\mathbf{r}_j$라 하면, 기하학적 지연은 plane-wave approximation이 아니라 antenna-to-satellite range difference로 계산한다.
\begin{equation}
\tau_{ij}(t)=\frac{|\mathbf{r}_{\rm sat}(t)-\mathbf{r}_j|-|\mathbf{r}_{\rm sat}(t)-\mathbf{r}_i|}{c}.
\label{eq:nearfield_delay}
\end{equation}
이 정의는 HERA-scale baseline과 LEO 거리에서 불필요한 plane-wave assumption을 피하기 위한 conservative implementation이다.

\subsection{Fringe-rate와 smearing}

시간에 따라 변하는 delay는 frequency-dependent fringe-rate를 만든다.
\begin{equation}
f_{\rm fr}(t,\nu) \approx \nu \frac{d\tau(t)}{dt}.
\end{equation}
Finite integration time $\Delta t$와 finite channel width $\Delta \nu$는 각각 다음 attenuation을 준다.
\begin{align}
A_{\rm time}(t,\nu) &= \left|\mathrm{sinc}\left(f_{\rm fr}(t,\nu)\Delta t\right)\right|,\\
A_{\rm chan}(t) &= \left|\mathrm{sinc}\left(\tau(t)\Delta\nu\right)\right|.
\end{align}

\subsection{Amplitude and spectral template}

Injected visibility amplitude는 다음 형태로 둔다.
\begin{equation}
A(t,\nu)=S_{\rm ref}M_{\rm spec}(\nu)B(t,\nu)R(t)A_{\rm time}(t,\nu)A_{\rm chan}(t)D(t),
\label{eq:amp_model}
\end{equation}
여기서 $M_{\rm spec}$은 문헌 기반 Starlink-like UEMR spectral template이며, proprietary waveform reconstruction이 아니다. $B$는 bounded HERA-like beam response, $R(t)$는 reference range에 대한 range attenuation, $D(t)$는 optional duty modulation이다.

Polarization 관련 run은 sensitivity test로만 해석한다. Diagonal Jones approximation 또는 scalar per-pol scaling을 사용할 수 있으나, D-terms, mutual coupling, full-Stokes satellite attitude variability, ionospheric Faraday rotation은 본 모델에 포함하지 않는다.

% ============================================================
\section{Processing operator와 EoR-window metric}
% ============================================================

\subsection{Operator $P$}

본 연구의 $P$는 background-only 데이터와 injected 데이터에 동일하게 적용되는 명시적 post-processing operator이다. 기본 $P_0$는 다음 단계로 구성된다.

\begin{enumerate}[leftmargin=2em]
    \item complex visibility $V(t,\nu)$와 weight/flag mask를 읽는다;
    \item invalid sample의 weight를 0으로 둔다;
    \item frequency taper를 적용한다;
    \item weighted delay transform을 수행한다;
    \item horizon plus buffer 바깥의 delay-window 영역을 정의한다;
    \item 선택적 fringe-rate filtering을 적용한다;
    \item window-side residual metric을 계산한다.
\end{enumerate}

$P$의 수치 결과는 taper, normalization, buffer, weight handling, flagging state, fringe-rate filter에 의존한다. 따라서 모든 결과는 operator-conditioned diagnostic으로 해석한다.

\subsection{Window definition}

Baseline length $|\mathbf{b}|$에 대한 horizon delay는
\begin{equation}
\tau_{\rm hor}=\frac{|\mathbf{b}|}{c}
\end{equation}
로 정의한다. Window proxy는 buffer $\Delta\tau_{\rm buf}$를 포함해 다음과 같이 둔다.
\begin{equation}
W_{\rm win}=\left\{\tau: |\tau|>\tau_{\rm hor}+\Delta\tau_{\rm buf}\right\}.
\end{equation}
이 window는 full $k_\perp$--$k_\parallel$ cylindrical analysis를 대체하지 않는다. 이는 single-baseline 또는 baseline-group level에서 window-side perturbation을 빠르게 점검하기 위한 QA proxy이다.

\subsection{Interaction residual power ratio}

첫 번째 diagnostic은 interaction residual 자체의 window power가 processed background의 window power에 비해 얼마나 큰지를 본다.
\begin{equation}
R_{\rm win}^{(\rm int)}=
10\log_{10}
\left[
\frac{\sum_{\tau\in W_{\rm win}}|\widetilde{V}_{\rm int}(\tau)|^2}
{\sum_{\tau\in W_{\rm win}}|\widetilde{V}_{\rm bg}(\tau)|^2}
\right].
\label{eq:rwin_int}
\end{equation}
여기서 tilde는 동일한 taper/weighting을 사용한 delay transform을 의미한다. 이 값은 residual amplitude diagnostic이며, 최종 power spectrum perturbation과 동일하지 않다.

\subsection{Absolute window power perturbation score}

주요 EoR-risk diagnostic은 dirty processed power와 background processed power의 차이를 직접 비교하는 absolute perturbation score이다.
\begin{equation}
S_{\rm win}^{(|\Delta P|)}=
10\log_{10}
\left[
\frac{\sum_{\tau\in W_{\rm win}}
\left|
|\widetilde{V}_{\rm dirty}(\tau)|^2-|\widetilde{V}_{\rm bg}(\tau)|^2
\right|}
{\sum_{\tau\in W_{\rm win}}|\widetilde{V}_{\rm bg}(\tau)|^2}
\right],
\label{eq:swin_abs}
\end{equation}
where $V_{\rm dirty}=P(V_{\rm bg}+V_{\rm sat})$. 절대값을 사용하는 이유는 signed $\Delta P$가 delay bin 사이에서 상쇄될 수 있기 때문이다. QA 관점에서는 signed total change보다 window-side perturbation magnitude가 중요하다.

보조적으로 signed score도 보고한다.
\begin{equation}
S_{\rm win}^{(\Delta P)}=
10\log_{10}
\left[
\frac{\left|\sum_{\tau\in W_{\rm win}}
\left(|\widetilde{V}_{\rm dirty}(\tau)|^2-|\widetilde{V}_{\rm bg}(\tau)|^2\right)\right|}
{\sum_{\tau\in W_{\rm win}}|\widetilde{V}_{\rm bg}(\tau)|^2}
\right].
\end{equation}

\subsection{Phase-randomized window-metric null}

각 main case에 대해 observed $S_{\rm win}^{(|\Delta P|)}$를 phase-randomized null ensemble과 비교한다. Null trial은 각 emitter 또는 injected component의 amplitude envelope, time/frequency support, delay support는 보존하고, global phase만 randomize한다. 각 trial $m$에 대해
\begin{equation}
V_{\rm sat}^{(m)}(t,\nu)=\sum_k V_k(t,\nu)e^{i\phi_k^{(m)}},\quad \phi_k^{(m)}\sim U(0,2\pi)
\end{equation}
를 만들고 동일한 $P$와 Eq.~\eqref{eq:swin_abs}를 적용한다. Null excess는
\begin{equation}
\Delta_{\rm null}=S_{\rm win,obs}^{(|\Delta P|)}-Q_{0.95}\left(S_{\rm win,null}^{(|\Delta P|)}\right)
\label{eq:null_excess}
\end{equation}
로 정의한다. $\Delta_{\rm null}>0$이면 해당 observed geometry가 randomized-phase support-matched null p95를 초과한다는 뜻이다.

% ============================================================
\section{실험 설계: Path B}
% ============================================================

\subsection{Phase 0: benchmark와 metric freeze}

Main grid를 수행하기 전에 단일 observed run과 단일 null trial의 runtime을 측정한다. 1 null trial이 1초 미만이면 full $75\times500$ null grid를 실행하고, 1--5초이면 pilot은 100 null, final main cases만 500 null로 실행한다. 5초를 넘으면 병렬화 또는 case 축소를 적용한다.

Metric은 pilot 이전에 고정한다. Main claim에는 Eq.~\eqref{eq:swin_abs}와 Eq.~\eqref{eq:null_excess}를 사용하고, Eq.~\eqref{eq:rwin_int}는 secondary diagnostic으로 둔다. 중간에 metric을 변경한 경우 pilot과 main grid를 직접 비교하지 않는다.

\subsection{Phase 1: pilot grid와 통과 기준}

Pilot grid는 다음과 같다.

\begin{table}[h]
\centering
\caption{Pilot grid.}
\label{tab:pilot_grid}
\begin{tabular}{ll}
\toprule
Axis & Values \\
\midrule
Backgrounds & 2 backgrounds with different LST/flag state \\
Baseline groups & Short EW, Mid EW, Long EW \\
Representative baseline & 1 baseline per group \\
Flux grid & $S_{\rm ref}=\{\FluxGrid\}$ \\
Null trials & \NNullTrialsPilot{} or \NNullTrialsFinal{} depending on benchmark \\
\bottomrule
\end{tabular}
\end{table}

Pilot 통과 기준은 사전에 다음과 같이 고정한다.

\begin{enumerate}[leftmargin=2em]
    \item 2개 background 모두에서 동일 baseline group이 $\Delta_{\rm null}>0$을 보인다;
    \item $S_{\rm ref}\le100$\,Jy case에서 최소 하나 이상의 $\Delta_{\rm null}>0$이 나타난다;
    \item baseline ordering이 완전히 무작위가 아니라, 특정 geometry class에서 반복되는 경향을 보인다;
    \item null distribution이 p95 해석 가능한 형태이다. Heavy-tail이면 p95와 p99를 함께 보고하고, multimodal이면 null definition을 재검토한다.
\end{enumerate}

Pilot이 이 기준을 충족하지 못하면 main grid로 확장하지 않는다. 먼저 metric implementation, window definition, buffer choice, spectral template, flux prior를 점검한다. 그럼에도 realistic flux에서 반복적 null excess가 없으면 논문의 claim을 framework/stress-test level로 낮춘다.

\subsection{Phase 2A: main multi-background grid}

Main grid는 \NBackgrounds{}개 background, \NBaselineGroups{}개 baseline group, \NFluxScales{}개 flux scale로 구성한다.
\begin{equation}
N_{\rm obs}=\NBackgrounds\times\NBaselineGroups\times\NFluxScales=75.
\end{equation}
각 final claim case에는 \NNullTrialsFinal{}개의 phase-randomized window-metric null trial을 적용한다. Main grid output은 Table~\ref{tab:main_results_template}에 저장한다.

\begin{table}[h]
\centering
\caption{Main result table template. 실제 run 이후 값을 채운다.}
\label{tab:main_results_template}
\begin{tabular}{llllrrrr}
\toprule
BG & Group & Baseline & $S_{\rm ref}$ & $S_{\rm obs}$ & Null med. & Null p95 & $\Delta_{\rm null}$ \\
\midrule
bg\_01 & Short EW & -- & 10 Jy & -- & -- & -- & -- \\
bg\_01 & Short EW & -- & 30 Jy & -- & -- & -- & -- \\
bg\_01 & Short EW & -- & 100 Jy & -- & -- & -- & -- \\
\multicolumn{8}{c}{\dots} \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Phase 2B: redundant-baseline robustness}

대표 baseline에서 pattern이 확인된 후, 각 group에서 3개 이상의 redundant baseline을 선택한다. Visibility는 평균하지 않고, 각 baseline별 metric을 계산한 뒤 group distribution으로 요약한다.

\begin{equation}
\left\{S_{\rm win}^{(b)}: b\in G\right\}\rightarrow\left(\mathrm{median},\mathrm{IQR},\mathrm{min},\mathrm{max}\right).
\end{equation}

이 단계의 목적은 single-baseline artifact를 줄이는 것이지, moving emitter phase를 visibility level에서 평균해 제거하는 것이 아니다.

\subsection{Phase 2C: simplified inpainting-surrogate survivability}

Mitigation survivability를 보기 위해 두 operator를 비교한다.
\begin{align}
P_0 &= \mathrm{weighted\ delay/fringe\ operator},\\
P_1 &= \mathrm{simplified\ smooth\mbox{-}spectrum\ inpainting\ surrogate}+P_0.
\end{align}
$P_1$은 HERA production inpainting과 동일하다고 주장하지 않는다. 이는 단순한 smooth-spectrum reconstruction surrogate이며, moving/comb-like residual이 such operator 아래에서 얼마나 줄어드는지 확인하기 위한 stress test이다.

Inpainting effect는
\begin{equation}
\Delta S_{\rm inpaint}=S_{\rm win}^{P_1}-S_{\rm win}^{P_0}
\end{equation}
로 보고한다. 음수이면 inpainting surrogate가 perturbation을 줄인 것이고, 0에 가까우면 해당 residual morphology가 surrogate 아래에서도 유지된다는 뜻이다.

\subsection{Phase 2D: null distribution diagnostics}

각 final case에 대해 null distribution histogram을 appendix에 제시한다. 최소 보고값은 null median, p95, p99, standard deviation, IQR, skewness, observed percentile, observed-minus-p95이다.

\begin{table}[h]
\centering
\caption{Null distribution diagnostic table template.}
\label{tab:null_diag}
\begin{tabular}{lrrrrrr}
\toprule
Case & med. & p95 & p99 & IQR & skew & obs perc. \\
\midrule
Short EW, 100 Jy, bg\_01 & -- & -- & -- & -- & -- & -- \\
Short EW, 100 Jy, bg\_02 & -- & -- & -- & -- & -- & -- \\
\bottomrule
\end{tabular}
\end{table}

% ============================================================
\section{결과}
% ============================================================

\subsection{Pilot pass/fail summary}

이 절은 pilot run 이후 작성한다. 통과 기준은 Section~5.2에 이미 고정되어 있으므로, 결과 해석은 사후적으로 바꾸지 않는다. Pilot이 통과한 경우 다음 문장을 사용한다.

\begin{quote}
Pilot grid에서 동일 baseline group이 두 background 모두에서 $\Delta_{\rm null}>0$을 보였고, $S_{\rm ref}\le100$\,Jy에서도 최소 하나의 null-exceeding case가 확인되었다. 따라서 main multi-background grid로 확장하였다.
\end{quote}

Pilot이 실패한 경우 다음 문장을 사용한다.

\begin{quote}
Pilot grid에서 realistic flux scale의 반복적 null excess가 확인되지 않았다. 따라서 본 논문은 robust EoR QA-risk claim 대신, forward-injection metric과 null-calibration framework의 방법론적 제시에 초점을 둔다.
\end{quote}

\subsection{Flux-dependent window perturbation}

Figure~\ref{fig:swin_flux}는 $S_{\rm win}^{(|\Delta P|)}$의 flux dependence를 보여준다. X축은 $S_{\rm ref}$, Y축은 $S_{\rm win}^{(|\Delta P|)}$이며, band는 background-to-background variance를 나타낸다. Realistic flux tier와 stress-test tier를 음영으로 구분한다.

\begin{figure}[h]
\centering
\includegraphics[width=0.90\linewidth]{figures/fig_swin_vs_flux.pdf}
\caption{Flux-dependent absolute EoR-window perturbation score. X축은 reference flux scale $S_{\rm ref}$, Y축은 $S_{\rm win}^{(|\Delta P|)}$이다. 선은 baseline group의 median, band는 \NBackgrounds{}개 background realization의 spread를 나타낸다. Vertical shading은 published-order flux tier와 stress-test tier를 구분한다.}
\label{fig:swin_flux}
\end{figure}

\subsection{Null-calibrated excess}

Figure~\ref{fig:null_excess}는 Eq.~\eqref{eq:null_excess}의 $\Delta_{\rm null}$을 보여준다. $\Delta_{\rm null}=0$은 phase-randomized window-metric null p95 기준선이다. 0보다 큰 값은 observed TLE-conditioned geometry가 support-matched randomized-phase null p95를 초과했다는 뜻이다.

\begin{figure}[h]
\centering
\includegraphics[width=0.90\linewidth]{figures/fig_null_excess_vs_flux.pdf}
\caption{Null-calibrated window-metric excess. $\Delta_{\rm null}=S_{\rm obs}-S_{\rm null,p95}$이며, horizontal zero line은 null p95 threshold를 나타낸다. 각 점은 background realization을 나타내고, 선은 baseline group별 median trend를 나타낸다.}
\label{fig:null_excess}
\end{figure}

\subsection{Baseline-group robustness}

Redundant-baseline robustness 결과는 visibility averaging 없이 baseline-level metric distribution으로 제시한다. 만약 short baseline group이 반복적으로 높은 $S_{\rm win}^{(|\Delta P|)}$ 또는 $\Delta_{\rm null}$을 보이면, 이는 short baseline 자체가 보편적으로 위험하다는 뜻이 아니라, 해당 geometry class에서 delay overlap과 fringe washing 조건이 window-side perturbation을 키울 수 있음을 시사한다.

\subsection{Processing-state and inpainting-surrogate response}

Figure~\ref{fig:inpaint_response}는 $P_0$와 $P_1$의 결과를 비교한다. $P_1$에서 $S_{\rm win}$이 크게 감소하면 smooth-spectrum inpainting surrogate가 해당 perturbation을 완화할 수 있음을 뜻한다. 반대로 partial reduction 또는 no reduction이면 moving/comb-like residual morphology가 simple inpainting operator 아래에서도 QA 대상으로 남는다는 뜻이다.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\linewidth]{figures/fig_inpainting_survivability.pdf}
\caption{Simplified inpainting-surrogate survivability test. $P_0$는 baseline weighted delay/fringe operator이고, $P_1$은 simplified smooth-spectrum inpainting surrogate를 추가한 operator이다. 음의 $\Delta S_{\rm inpaint}$는 inpainting surrogate가 window-side perturbation을 줄였음을 의미한다.}
\label{fig:inpaint_response}
\end{figure}

% ============================================================
\section{논의}
% ============================================================

\subsection{Claim ladder}

본 논문의 claim은 결과에 따라 다음 네 수준 중 하나로 제한한다.

\begin{table}[h]
\centering
\caption{Result-dependent claim ladder.}
\label{tab:claim_ladder}
\begin{tabular}{lll}
\toprule
Level & Condition & Claim \\
\midrule
1 & realistic flux에서 null excess 없음 & reproducible QA framework \\
2 & 300--1000 Jy에서만 반복 & bright/stress-test UEMR risk \\
3 & $\le100$ Jy에서 $\ge3$ backgrounds 반복 & published-order flux QA risk \\
4 & redundancy, processing-state, inpainting 후에도 유지 & robust operator-conditioned EoR QA risk \\
\bottomrule
\end{tabular}
\end{table}

이 ladder의 목적은 결과가 허락하는 것보다 강한 claim을 하지 않는 것이다. 특히 Level 3 이상을 주장하려면 realistic flux tier에서 반복적 $\Delta_{\rm null}>0$, background-to-background stability, baseline group ordering stability가 필요하다.

\subsection{이 연구가 주장하지 않는 것}

본 연구는 다음을 주장하지 않는다.

\begin{enumerate}[leftmargin=2em]
    \item 특정 Starlink 위성의 실제 UEMR을 HERA에서 검출했다;
    \item HERA production pipeline 전체를 raw visibility부터 재현했다;
    \item full cylindrical $k_\perp$--$k_\parallel$ power spectrum contamination을 측정했다;
    \item $S_{\rm ref}$가 HERA에서 관측될 calibrated Starlink apparent flux를 정확히 예측한다;
    \item simplified inpainting surrogate가 HERA production inpainting과 동일하다.
\end{enumerate}

본 연구의 주장은 특정 public-data background product와 명시적 operator $P$ 아래에서 Starlink-like moving emitter가 만드는 window-side residual perturbation의 QA-relevant behavior에 한정된다.

\subsection{해석상 주의점}

$S_{\rm win}^{(|\Delta P|)}$는 full EoR power spectrum bias estimator가 아니다. 이는 delay-window proxy 영역에서 processed dirty power와 processed background power의 차이를 background power로 normalize한 QA metric이다. 또한 phase-randomized null은 실제 위성 방출의 correlated electronics, constellation operation, polarization variability, ionospheric phase fluctuation을 완전히 대표하지 않는다. 따라서 null 초과는 ``실제 우주론 분석 bias가 발생했다''가 아니라, ``support-matched randomized-phase null보다 큰 operator-conditioned window perturbation이 발생했다''로 해석한다.

% ============================================================
\section{결론}
% ============================================================

본 연구는 Starlink-like UEMR의 raw detectability보다, HERA-like EoR delay-space processing 이후 window-side residual perturbation이 얼마나 안정적으로 남는지를 평가하는 multi-background forward-injection framework를 제시한다. 핵심 quantity는 interaction residual
\[
V_{\rm int}=P(V_{\rm bg}+V_{\rm sat})-P(V_{\rm bg})
\]
과 absolute window power perturbation score $S_{\rm win}^{(|\Delta P|)}$이다.

경로 B의 최종 성공 조건은 다음과 같다. 첫째, 최소 3개 이상의 background realization에서 동일 geometry class의 null-exceeding perturbation이 반복되어야 한다. 둘째, $S_{\rm ref}\le100$\,Jy의 published-order flux tier에서 최소 하나 이상의 baseline group이 안정적 $\Delta_{\rm null}>0$을 보여야 한다. 셋째, redundant baseline group에서는 visibility averaging이 아니라 metric-level aggregation으로 패턴이 유지되어야 한다. 넷째, simplified inpainting surrogate 또는 processing-state 변화 이후에도 결과 방향성이 해석 가능해야 한다.

이 조건이 충족되면 본 연구는 Starlink-like UEMR이 HERA-like EoR delay-space analysis에서 baseline- and operator-conditioned QA risk factor가 될 수 있음을 주장할 수 있다. 조건이 충족되지 않으면 본 논문은 strong EoR-risk claim을 포기하고, reproducible forward-injection metric과 null-calibration framework를 제시하는 방법론 논문으로 제한한다.

\section*{데이터 및 코드 공개}

본 연구의 code, configuration files, background manifest, derived metric tables, null-distribution diagnostics는 논문 제출 시 함께 archive한다. Direct UVH5 loader를 사용한 exploratory run과 달리, paper-facing run은 explicit weights/flags를 포함한 NPZ background product를 사용한다.

\section*{감사의 글}

본 연구는 HERA public data와 low-frequency radio astronomy community의 공개 software ecosystem에 기반한다. 모든 수치 결과는 final run 이후 재현 가능한 configuration과 함께 보고한다.

\bibliographystyle{plainnat}
\bibliography{starlink_uemr_refs}

\end{document}
