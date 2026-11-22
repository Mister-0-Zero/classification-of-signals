"""FSL manual 1300.2519.12-12 section 4.46: ASCII trace file parser."""
import math
import re


def parse_trace_export(data):
    if len(data) > 8_000_000:
        raise ValueError("Файл трассы слишком велик")
    text = data.decode("latin-1") if isinstance(data, bytes) else data
    header, xs, ys, minima = {}, [], [], []
    in_trace = False
    for line in text.splitlines():
        cells = [v.strip() for v in line.split(";")]
        if not cells or not cells[0]:
            continue
        key = cells[0].lower().replace(" ", "").replace("–", "-")
        if re.fullmatch(r"trace[1-6]:?", key):
            if in_trace or key not in ("trace1", "trace1:"):
                raise ValueError("Ожидался файл только Trace 1")
            in_trace = True
            continue
        try:
            x = float(cells[0].replace(",", "."))
        except ValueError:
            if len(cells) >= 2:
                header[key] = cells[1]
            continue
        if not in_trace or len(cells) < 2:
            raise ValueError("Неверная строка файла трассы")
        y = float(cells[1].replace(",", "."))
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Неконечные значения в файле трассы")
        xs.append(x)
        ys.append(y)
        extra = cells[2:] if len(cells) > 2 else []
        extra = [cell for cell in extra if cell]
        if len(extra) > 1:
            raise ValueError("Неизвестные дополнительные столбцы файла трассы")
        if extra:
            value = float(extra[0].replace(",", "."))
            if not math.isfinite(value):
                raise ValueError("Неконечное значение AUTOPEAK")
            minima.append(value)
    if header.get("x-unit", "").upper() != "HZ":
        raise ValueError("Трасса не содержит частот в Hz")
    if len(xs) < 2 or int(header.get("values", "0")) != len(xs):
        raise ValueError("Число строк файла не соответствует Values")
    if not all(b > a for a, b in zip(xs, xs[1:])):
        raise ValueError("Частотная ось должна возрастать")
    if minima and len(minima) != len(xs):
        raise ValueError("Неполный второй столбец AUTOPEAK")
    unit = header.get("y-unit", "").upper().replace("µ", "U").replace("Μ", "U")
    if unit not in ("DBM", "DBUV", "DBMV", "DBUA", "DBPW"):
        raise ValueError("Неподдерживаемая единица файла трассы: " + unit)
    spacing = header.get("x-axis", "").upper()
    if spacing not in ("LIN", "LOG"):
        raise ValueError("В файле не указан тип частотной шкалы")
    return dict(header=header, frequency_hz=tuple(xs), amplitudes=tuple(ys),
                frequency_axis=spacing, amplitude_unit=unit, autopeak_min=tuple(minima))
