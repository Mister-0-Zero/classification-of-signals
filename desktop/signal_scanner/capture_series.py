"""Reproducible, one-window collection for signal-presence datasets."""
from dataclasses import asdict, dataclass, replace
import math
import random
import secrets
import time

from .domain import CaptureLabels, utc_now
from .instruments import Cancelled
from .storage import write_json


@dataclass(frozen=True)
class SeriesPlan:
    variants: int = 15
    rbw_span_min: float = .002
    rbw_span_max: float = .2
    vbw_rbw_min: float = .1
    vbw_rbw_max: float = 1.
    seed: int | None = None

    def validate(self):
        if not 10 <= self.variants <= 20:
            raise ValueError("Серия должна содержать 10–20 случайных вариантов.")
        if not 0 < self.rbw_span_min <= self.rbw_span_max <= 1:
            raise ValueError("Некорректный диапазон RBW/span.")
        if not 0 < self.vbw_rbw_min <= self.vbw_rbw_max <= 1:
            raise ValueError("Некорректный диапазон VBW/RBW.")


def bandwidth_steps(minimum):
    # FSL normal filters: 1/3 decade steps, capped to the base instrument range.
    return tuple(sorted({float(m * 10**e) for e in range(8) for m in (1, 3)
                         if minimum <= m * 10**e <= 10_000_000}))


def variants_for(settings, state, plan):
    """Uniform sampling without replacement across discrete logarithmic steps."""
    plan.validate()
    rng = random.Random(plan.seed)
    rbws = [v for v in bandwidth_steps(10) if
            plan.rbw_span_min <= v/settings.span_hz <= plan.rbw_span_max]
    candidates = []
    original_count = int(float(state.get("sweep_count", 0)))
    for rbw in rbws:
        for vbw in bandwidth_steps(1):
            if not plan.vbw_rbw_min <= vbw/rbw <= plan.vbw_rbw_max:
                continue
            for trace in ("WRIT", "AVER", "MAXH"):
                count = original_count if trace == "WRIT" else (original_count or 5)
                for detector in ("POS", "RMS", "SAMPLE"):
                    cfg = replace(settings, rbw_hz=rbw, vbw_hz=vbw, detector=detector)
                    cfg.validate()
                    initial_trace = str(state.get("trace_mode", "WRIT")).upper()
                    initial_trace = {"WRITE": "WRIT", "AVERAGE": "AVER", "MAXHOLD": "MAXH"}.get(initial_trace, initial_trace)
                    if cfg == settings and trace == initial_trace and count == original_count:
                        continue
                    candidates.append({"settings": asdict(cfg), "trace_mode": trace, "sweep_count": count})
    if len(candidates) < plan.variants:
        raise ValueError("Для этого span слишком мало допустимых сочетаний RBW/VBW. Частотное окно не изменено.")
    # Each bandwidth step occupies roughly equal log space. Do not use uniform Hz.
    return rng.sample(candidates, plan.variants)


def assert_frozen(settings, initial):
    for field in ("center_hz", "span_hz", "points"):
        if not math.isclose(getattr(settings, field), getattr(initial, field), rel_tol=1e-10, abs_tol=1e-6):
            raise ValueError(f"Прибор изменил {field}; серия остановлена, чтобы не смешивать разные окна.")


def collect_series(analyzer, repository, control, emit, labels: CaptureLabels,
                   sweeps=1, plan: SeriesPlan | None = None, scene="Эфир"):
    labels.validate()
    if not 1 <= sweeps <= 50:
        raise ValueError("Количество срезов: 1–50.")
    plan = plan or SeriesPlan()
    plan.validate()
    labels = replace(labels, settings_source="simulated_front_panel" if analyzer.simulated else "instrument_front_panel")
    seed = secrets.randbits(63) if plan.seed is None else plan.seed
    signal_present = labels.signal_present
    plan = replace(plan, seed=seed)
    initial = analyzer.read_settings()
    initial.validate()
    state = analyzer.series_snapshot()
    configurations = variants_for(initial, state, plan)
    run = repository.new_session("captures", simulated=analyzer.simulated)
    total = len(configurations) + 2
    session = {"schema_version": 3, "status": "running", "simulated": analyzer.simulated,
               "purpose": "signal_presence_and_settings_comparison", "started_at": utc_now(),
               "initial_settings": asdict(initial), "initial_state": state,
               "settings_source": "instrument_front_panel" if not analyzer.simulated else "simulated_front_panel",
               "plan": asdict(plan), "planned_variants": configurations, "captured": 0,
               "expected_captures": total, "labels": asdict(labels),
               "signal_present": signal_present, "training_eligible": signal_present is not None,
               "quality_target": None,
               "quality_target_status": "not_scored", "quality_notes":
               "Model confidence alone is not ground truth. Compare correctness on independent labelled observations and acquisition time.",
               "frozen_fields": ["center_hz", "span_hz", "points"]}
    changed, capture_started, primary = False, False, None
    started = time.perf_counter()
    write_json(run / "session.json", session)

    def save(index, role, requested=None, setting_apply_s=0):
        nonlocal capture_started
        control.checkpoint()
        assert_frozen(analyzer.read_settings(), initial)
        begin = time.perf_counter()
        capture_started = True
        spectrum = analyzer.capture(sweeps, control.cancel)
        elapsed = time.perf_counter() - begin
        assert_frozen(spectrum.settings, initial)
        spectrum.measurement_metadata["series"] = {
            "role": role, "seed": seed, "variant_index": index,
            "requested": requested, "capture_call_s": elapsed,
            "setting_apply_s": setting_apply_s, "setting_and_capture_s": setting_apply_s + elapsed,
            "signal_present": signal_present, "quality_score": None,
            "label_scope": "physical signal count in the same frequency window; weaker visibility is not relabelled as absence",
        }
        if analyzer.simulated:
            spectrum.measurement_metadata["actual_settings"] = {
                **asdict(spectrum.settings), "trace_mode": getattr(analyzer, "series_trace_mode", "WRIT"),
                "sweep_count": getattr(analyzer, "series_sweep_count", 0)}
            spectrum.measurement_metadata["simulation_trace_effects_modeled"] = False
        repository.save_capture(run, index, spectrum, labels, variant=role == "random_variant", scene=scene)
        session["captured"] += 1
        emit({"type": "capture", "spectrum": spectrum, "index": index+1, "total": total,
              "run": str(run), "variant": role == "random_variant", "role": role})
        write_json(run / "session.json", session)

    try:
        save(0, "baseline_start")
        for index, variant in enumerate(configurations, 1):
            control.checkpoint()
            changed = True  # even a partially rejected write requires restoration
            apply_started = time.perf_counter()
            actual = analyzer.apply_variant(variant, initial)
            apply_elapsed = time.perf_counter() - apply_started
            assert_frozen(actual, initial)
            save(index, "random_variant", variant, apply_elapsed)
        if changed:
            analyzer.restore_series_state(state)
            restored = analyzer.read_settings()
            for field in asdict(initial):
                old, new = getattr(initial, field), getattr(restored, field)
                if (isinstance(old, str) and old != new) or (not isinstance(old, str) and not math.isclose(old,new,rel_tol=1e-8,abs_tol=1e-6)):
                    raise ValueError(f"Исходная настройка {field} не восстановлена перед повторным снимком")
            changed = False
        save(total-1, "baseline_end")
        session["status"] = "complete"
    except Cancelled:
        session["status"] = "stopped"
    except Exception as error:
        primary = error
        session.update(status="error", error=str(error),
                       diagnostic_context=getattr(error, "context", None), error_notes=getattr(error, "__notes__", []))
        raise
    finally:
        cleanup = []
        if changed or capture_started:
            try:
                analyzer.restore_series_state(state, spectral=changed)
            except Exception as error:
                cleanup.append(str(error))
        try:
            analyzer.release_local()
        except Exception as error:
            cleanup.append(str(error))
        if cleanup:
            session.update(status="error", cleanup_errors=cleanup)
            if primary is not None:
                primary.add_note("Восстановление серии: " + "; ".join(cleanup))
        session.update(finished_at=utc_now(), elapsed_s=time.perf_counter()-started,
                       settings_restored=not cleanup)
        write_json(run / "session.json", session)
        emit({"type": "capture_finished", "status": session["status"], "run": str(run), "captured": session["captured"]})
        if cleanup and primary is None:
            raise RuntimeError("Не восстановлены настройки серии: " + "; ".join(cleanup))
