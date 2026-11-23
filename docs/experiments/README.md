# Дополнительные отчёты по классификации сигналов

[Итоговые схемы и ключевые результаты](../modulation_pipeline.md)

## От выбранных режимов к деталям

- [Ансамбль MLP и GB на одной развёртке](single_sweep_ensemble.md) — веса моделей, пороги и компромисс между пропусками и ложными сигналами.
- [Компактная MLP](fast_mlp.md) — аугментация, архитектуры, уверенность и время.
- [Сравнение отдельных моделей на одной развёртке](single_sweep_models.md) — GB, MLP, CNN, матрица ошибок и каскад на одном спектре.
- [Уверенность GB и каскад с Deep Sets](gb_confidence.md) — выбор порога и сопоставление одного и семи спектров.
- [Время измерения FSL3](fsl3_timing.md) — паспорт, журналы и задержка получения трассы.

## Усреднение, сдвиги и архитектуры

- **GB** — [усреднение и проверка на смещённых FM](gb_sweep_averaging.md), [фиксированные и случайные сдвиги](gb_shift_augmentation.md).
- **Сети на семи развёртках** — [Deep Sets и 1D CNN](modulation_deep_sets.md), [случайные сдвиги для MLP, CNN и Deep Sets и уточнение центра для GB](neural_shift_augmentation.md).
- **MLP** — [базовые варианты](modulation_mlp_baseline.md), [ошибки на смещённых спектрах](modulation_mlp_shifted.md), [архитектуры](modulation_mlp_architectures.md), [dropout](modulation_mlp_wide_dropout.md), [ранний опыт со сдвигами](modulation_mlp_shift_augmentation.md).
- **1D CNN** — [базовое сравнение](modulation_cnn1d_baseline.md), [архитектуры](modulation_cnn1d_architectures.md), [дилатация и многомасштабный вариант](modulation_cnn1d_combined.md).
