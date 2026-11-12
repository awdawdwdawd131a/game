# Источники звуков

Все звуки в этой папке — **публичное достояние (CC0 1.0 Universal)**.
Можно свободно использовать, изменять и распространять, атрибуция не требуется
(но мы её указываем из уважения к авторам).

| Игровое имя | Источник | Оригинал | Лицензия |
|---|---|---|---|
| `turret`, `shotgun`, `explosion`, `blast_small`, `truck_hit`, `drone_hit`, `drone_down`, `reload`, `drone_start`, `land`, `step` | npm `@netforge/asset-library-files` (каталог `library/audio/sfx`) | `shoot.ogg`, `impact.ogg`, `big_kill.ogg`, `hit.ogg`, `hurt.ogg`, `downed.ogg`, `dash.ogg`, `revive.ogg`, `land.ogg`, `step.ogg` | CC0 1.0 |
| `ui_*`, `round_start`, `victory`, `defeat`, `ew_on`, `ew_loop`, `smoke` | npm `uisfx` (папки `mechanical`, `minimal`, `scifi`) | `click.ogg`, `hover.ogg`, `confirm.ogg`, `error.ogg`, `toggle-on.ogg`, `notification.ogg`, `warning.ogg`, `start.ogg`, `success.ogg`, `blocked.ogg`, `connect.ogg`, `processing.ogg`, `expand.ogg` | CC0 1.0 |

Пересборка: `python tools/build_sfx.py --netforge <каталог> --uisfx <каталог>`
Обработка: обрезка тишины, нормализация громкости, «дальний» вариант (ФНЧ + хвост),
16-битное моно 22050 Гц.

Процедурные (синтезируются в `src/audio.py`, файлов на диске нет):
ветер, шум дождя/снега, «свист» РЭБ-помех, гул двигателя V12, визг моторов FPV.
